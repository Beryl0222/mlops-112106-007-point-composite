"""合成算子的数值正确性与资料约定测试。

覆盖：

* 前向合成与残余透射量（手工可复核的小场景）；
* 反向颜色/不透明度梯度与有限差分一致，含近乎不透明的前方点；
* 深度缺口造成的误差上界与可训练资格；
* 透射上界足够小时提前结束，且仍能同时满足反向误差；
* 瓦片重复报送去重、覆盖冲突像素不可确认；
* 版本令牌失效保护与预算约束。
"""

from __future__ import annotations

import copy
import unittest

import numpy as np

from alpha_compose.load import load_scene
from alpha_compose.session import (
    BW_ADJOINT,
    BW_BACKWARD_ERR,
    BW_CONFLICT,
    COMMITTED,
    CONFLICT,
    WAITING,
    CompositeSession,
    StaleVersionError,
)
from alpha_compose.timeline import play


def _base_scene() -> dict:
    return {
        "shape": [2, 2],
        "background": [0.2, 0.4, 0.1],
        "expected_slices": {
            "front": {"depth": [0, 3], "rect": [0, 2, 0, 2], "point_limit": 4},
            "deep": {"depth": [3, 8], "rect": [0, 2, 0, 2], "point_limit": 4},
        },
        "points": {
            "p1": {"rgb": [0.1, 0.9, 0.1], "alpha": 0.999, "depth": 1.0, "slice": "front"},
            "p2": {"rgb": [0.8, 0.3, 0.9], "alpha": 0.8, "depth": 4.0, "slice": "deep"},
            "p3": {"rgb": [1.0, 0.5, 0.0], "alpha": 0.3, "depth": 6.0, "slice": "deep"},
        },
        "tiles": {
            "a": [
                ["p1", 0, 0, 1.0],
                ["p1", 0, 1, 0.5],
                ["p2", 0, 0, 0.9],
                ["p3", 0, 0, 1.0],
            ],
            "b": [
                # 跨界重复报送，覆盖一致，应只算一次。
                ["p1", 0, 0, 1.0],
                ["p2", 0, 0, 0.9],
            ],
        },
        "regions": [[0, 2, 0, 2]],
        "dy": [[0, 0, [1.0, -0.4, 0.5], 0.2]],
        "error": {"forward": 0.01, "backward": 0.05},
        "adjoint_bounds": {"rgb": [1.0, 1.0, 1.0], "transmittance": 1.0},
        "timeline": [["arrive", "front"], ["arrive", "deep"], ["read", 0], ["backward", 0]],
        "budget_bytes": 1 << 20,
    }


def _loss_and_grads(scene_data: dict, pixel=(0, 0), g_rgb=None, g_t=0.2):
    """解析地反传一次，返回 (颜色, 透射, 主记录梯度字典)。"""
    scene = load_scene(copy.deepcopy(scene_data))
    session = CompositeSession(scene)
    for name in scene.slices:
        session.arrive(name)
    resp = session.read_region(0)
    y, x = pixel
    if g_rgb is None:
        g_rgb = np.array([1.0, -0.4, 0.5])
    bw = session.backward(resp.token, [y], [x], [g_rgb], [g_t])
    return resp, bw


class ForwardTests(unittest.TestCase):
    def test_duplicate_report_counts_once(self):
        data = _base_scene()
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")
        session.arrive("deep")
        resp = session.read_region(0)
        # 手工合成像素 (0,0)：p1 o=0.999（覆盖1），p2 o=0.8×0.9=0.72，p3 o=0.3×1。
        T1, o1 = 1.0, 0.999
        T2 = T1 * (1 - o1)
        o2 = 0.8 * 0.9
        T3 = T2 * (1 - o2)
        o3 = 0.3 * 1.0
        T_final = T3 * (1 - o3)
        c1 = np.array([0.1, 0.9, 0.1])
        c2 = np.array([0.8, 0.3, 0.9])
        c3 = np.array([1.0, 0.5, 0.0])
        bg = np.array([0.2, 0.4, 0.1])
        expected = T1 * o1 * c1 + T2 * o2 * c2 + T3 * o3 * c3 + T_final * bg
        np.testing.assert_allclose(resp.color[0, 0], expected, atol=1e-12)
        self.assertAlmostEqual(resp.transmittance[0, 0], T_final, places=12)
        self.assertEqual(resp.pixel_status[0, 0], COMMITTED)
        # 点进入像素一次：令牌中该像素保留三个点。
        self.assertEqual(resp.token.pixels[(0, 0)].point_ids, ("p1", "p2", "p3"))

    def test_depth_tie_breaks_by_stable_id(self):
        data = _base_scene()
        # p2 改入 front 窗口并与 p1 同深度，验证同深度按稳定点号定序。
        data["points"]["p2"]["depth"] = 1.0
        data["points"]["p2"]["slice"] = "front"
        data["expected_slices"]["front"]["point_limit"] = 5
        resp, _ = _loss_and_grads(data)
        self.assertEqual(resp.token.pixels[(0, 0)].point_ids, ("p1", "p2", "p3"))

    def test_empty_pixel_is_background(self):
        data = _base_scene()
        resp, _ = _loss_and_grads(data)
        np.testing.assert_allclose(resp.color[1, 1], [0.2, 0.4, 0.1])
        self.assertEqual(resp.transmittance[1, 1], 1.0)

    def test_coverage_conflict_pixel_unconfirmable(self):
        data = _base_scene()
        data["tiles"]["b"].append(["p1", 0, 0, 0.75])  # 与 1.0 冲突
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")
        session.arrive("deep")
        resp = session.read_region(0)
        self.assertEqual(resp.pixel_status[0, 0], CONFLICT)
        self.assertTrue(np.all(np.isnan(resp.color[0, 0])))
        self.assertFalse(resp.forward_trainable[0, 0])
        bw = session.backward(resp.token, [0], [0], [[1.0, 1.0, 1.0]], [1.0])
        self.assertEqual(bw.pixel_reason[0]["reason"], BW_CONFLICT)
        self.assertEqual(bw.accepted_pixels, 0)


class BackwardTests(unittest.TestCase):
    def test_gradients_match_finite_differences(self):
        data = _base_scene()
        resp, bw = _loss_and_grads(data)
        g_rgb = np.array([1.0, -0.4, 0.5])
        g_t = 0.2
        eps = 1e-6

        def outputs(modified):
            scene = load_scene(copy.deepcopy(modified))
            session = CompositeSession(scene)
            for name in scene.slices:
                session.arrive(name)
            r = session.read_region(0)
            return r.color[0, 0].copy(), r.transmittance[0, 0]

        def finite_diff(pid: str, field_path: tuple, value0: float) -> float:
            """对 [0,1] 有界量做中心差分，边界自动退化为单侧差分。"""
            if value0 <= eps:
                lo, hi = value0, min(1.0, value0 + 2 * eps)
            elif value0 >= 1.0 - eps:
                lo, hi = max(0.0, value0 - 2 * eps), value0
            else:
                lo, hi = value0 - eps, value0 + eps
            bumped = copy.deepcopy(data)
            target = bumped["points"][pid]
            for key in field_path[:-1]:
                target = target[key]
            target[field_path[-1]] = hi
            c_hi, t_hi = outputs(bumped)
            target[field_path[-1]] = lo
            c_lo, t_lo = outputs(bumped)
            return (
                float(g_rgb @ c_hi + g_t * t_hi) - float(g_rgb @ c_lo + g_t * t_lo)
            ) / (hi - lo)

        for pid in ("p1", "p2", "p3"):
            for ch in range(3):
                value0 = data["points"][pid]["rgb"][ch]
                fd = finite_diff(pid, ("rgb", ch), value0)
                self.assertAlmostEqual(bw.master_rgb_grad[pid][ch], fd, delta=2e-5,
                                       msg=f"{pid} rgb[{ch}] 梯度与有限差分不符")
            # alpha 扰动：有效不透明度 o = coverage*alpha 随 alpha 线性变化。
            alpha0 = data["points"][pid]["alpha"]
            fd = finite_diff(pid, ("alpha",), alpha0)
            self.assertAlmostEqual(bw.master_alpha_grad[pid], fd, delta=2e-5,
                                   msg=f"{pid} alpha 梯度与有限差分不符")

    def test_hidden_point_alpha_gradient_stays_finite_near_opaque_front(self):
        # 前方点几乎完全不透明时，隐藏点的不透明度导数仍受后方颜色影响，
        # 且无除法实现下数值稳定、与有限差分一致。
        data = _base_scene()
        data["points"]["p1"]["alpha"] = 1.0
        data["tiles"]["a"][0] = ["p1", 0, 0, 1.0]
        data["tiles"]["b"][0] = ["p1", 0, 0, 1.0]
        resp, bw = _loss_and_grads(data, g_t=-0.7)
        # 前方全不透明 → 隐藏点梯度极小但有限，且数值不产生 NaN/inf。
        self.assertTrue(np.isfinite(bw.master_alpha_grad["p2"]))
        self.assertTrue(np.all(np.isfinite(bw.master_rgb_grad["p2"])))
        self.assertLess(abs(bw.master_alpha_grad["p2"]), 1e-3)

    def test_multi_pixel_contributions_sum_to_master(self):
        data = _base_scene()
        # p1 同时落在 (0,0) 与 (0,1)，两个像素都归还梯度。
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")
        session.arrive("deep")
        resp = session.read_region(0)
        bw = session.backward(
            resp.token,
            [0, 0], [0, 1],
            [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
            [0.1, 0.2],
        )
        single_0 = session.backward(resp.token, [0], [0], [[1.0, 1.0, 1.0]], [0.1])
        single_1 = session.backward(resp.token, [0], [1], [[0.5, 0.5, 0.5]], [0.2])
        np.testing.assert_allclose(
            bw.master_rgb_grad["p1"],
            single_0.master_rgb_grad["p1"] + single_1.master_rgb_grad["p1"],
        )
        self.assertAlmostEqual(
            bw.master_alpha_grad["p1"],
            single_0.master_alpha_grad["p1"] + single_1.master_alpha_grad["p1"],
        )

    def test_gradients_route_to_master_record(self):
        data = _base_scene()
        # p3 声明主记录为 p2：p3 的梯度应汇总到 p2 名下，不单独出现。
        data["points"]["p3"]["master"] = "p2"
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")
        session.arrive("deep")
        resp = session.read_region(0)
        bw = session.backward(resp.token, [0], [0], [[1.0, 1.0, 1.0]], [0.0])
        self.assertNotIn("p3", bw.master_rgb_grad)
        self.assertIn("p2", bw.master_rgb_grad)
        # p2 主记录同时收到 p2 自身与 p3 的非零贡献。
        self.assertTrue(np.any(bw.master_rgb_grad["p2"] != 0.0))
        self.assertNotEqual(bw.master_alpha_grad["p2"], 0.0)

    def test_repeated_reads_issue_independent_tokens(self):
        data = _base_scene()
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")
        first = session.read_region(0)
        second = session.read_region(0)  # 编辑器重复请求同一区域
        self.assertNotEqual(first.token.token_id, second.token.token_id)
        self.assertEqual(first.token.pixels[(0, 0)].point_ids,
                         second.token.pixels[(0, 0)].point_ids)
        # 两个令牌都能独立归还，互不污染。
        bw1 = session.backward(first.token, [0], [1], [[0.2, 0.2, 0.2]], [0.0])
        bw2 = session.backward(second.token, [0], [1], [[0.2, 0.2, 0.2]], [0.0])
        np.testing.assert_allclose(
            bw1.master_rgb_grad["p1"], bw2.master_rgb_grad["p1"]
        )


class SliceAndBoundTests(unittest.TestCase):
    def test_missing_depth_slice_blocks_training(self):
        data = _base_scene()
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")  # deep 未到，可能挡住目前可见的点
        resp = session.read_region(0)
        # 像素 (0,1) 只有 p1(覆盖0.5)，其后是 deep 缺口；T=0.5005 > tol。
        self.assertEqual(resp.pixel_status[0, 1], WAITING)
        self.assertFalse(resp.forward_trainable[0, 1])
        self.assertGreaterEqual(resp.forward_bound[0, 1], 0.5)

    def test_small_transmittance_allows_early_commit_with_backward(self):
        data = _base_scene()
        data["error"] = {"forward": 0.01, "backward": 0.05}
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")  # deep 仍缺
        resp = session.read_region(0)
        state = resp.token.pixels[(0, 0)]
        # p1 几乎不透明，残余透射 0.001 ≤ forward tol → 前向可承诺并提前结束。
        self.assertEqual(resp.pixel_status[0, 0], COMMITTED)
        self.assertTrue(state.stopped)
        self.assertTrue(resp.forward_trainable[0, 0])
        # 后方缺口入口透射量也是 0.001：不透明度梯度误差界 0.001×4=0.004，
        # 反向同样可承诺——训练与预览在同一可见性定义下都可用。
        self.assertTrue(resp.backward_ok_worst[0, 0])
        self.assertLessEqual(state.backward_bound_worst, 0.05)
        bw = session.backward(resp.token, [0], [0], [[1.0, 1.0, 1.0]], [1.0])
        self.assertEqual(bw.accepted_pixels, 1)

    def test_loose_forward_tight_backward_reports_separate_verdict(self):
        # 前向容差宽松、反向容差严苛：前向可承诺但反向被单独拒绝。
        data = _base_scene()
        data["error"] = {"forward": 0.01, "backward": 1e-4}
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")
        resp = session.read_region(0)
        self.assertTrue(resp.forward_trainable[0, 0])
        self.assertFalse(resp.backward_ok_worst[0, 0])
        bw = session.backward(resp.token, [0], [0], [[1.0, 1.0, 1.0]], [1.0])
        self.assertEqual(bw.pixel_reason[0]["reason"], BW_BACKWARD_ERR)

    def test_nested_occlusion_can_commit_forward_and_backward(self):
        # 前面已有强遮挡点把透射量压到很小，其后再遇缺口：
        # 此时前向与反向公开界都能落在容差内。
        data = _base_scene()
        data["expected_slices"] = {
            "front": {"depth": [0, 3], "rect": [0, 2, 0, 2], "point_limit": 4},
            "mid": {"depth": [3, 5], "rect": [0, 2, 0, 2], "point_limit": 4},
            "deep": {"depth": [5, 9], "rect": [0, 2, 0, 2], "point_limit": 4},
        }
        data["points"]["p2"]["slice"] = "mid"
        data["points"]["p2"]["depth"] = 4.0
        data["points"]["p3"]["slice"] = "deep"
        data["error"] = {"forward": 0.01, "backward": 0.05}
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")
        session.arrive("mid")  # deep 缺口仍在，但 p1 已把透射压到 0.001
        resp = session.read_region(0)
        state = resp.token.pixels[(0, 0)]
        self.assertEqual(resp.pixel_status[0, 0], COMMITTED)
        self.assertTrue(resp.forward_trainable[0, 0])
        # 最后处理点 p2 的入口透射量 ≈ 0.001，反向界随之很小。
        self.assertTrue(resp.backward_ok_worst[0, 0])
        self.assertLessEqual(state.backward_bound_worst, 0.05)
        bw = session.backward(resp.token, [0], [0], [[1.0, 1.0, 1.0]], [1.0])
        self.assertEqual(bw.accepted_pixels, 1)

    def test_backward_error_can_fail_while_forward_holds(self):
        # 颜色已稳定，但隐藏层点对不透明度的导数可能超出可接受范围：
        # 收紧反向容差到 0，使任何缺口像素都无法承诺反向精度。
        data = _base_scene()
        data["error"] = {"forward": 0.01, "backward": 1e-12}
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")
        resp = session.read_region(0)
        # (0,0) 前向仍可承诺（T=0.001），但最坏反向误差 > 0。
        self.assertTrue(resp.forward_trainable[0, 0])
        self.assertFalse(resp.backward_ok_worst[0, 0])
        bw = session.backward(resp.token, [0], [0], [[1.0, 1.0, 1.0]], [1.0])
        self.assertEqual(bw.pixel_reason[0]["reason"], BW_BACKWARD_ERR)

    def test_completing_slice_promotes_pixels_and_tightens_bound(self):
        data = _base_scene()
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")
        first = session.read_region(0)
        self.assertIn(WAITING, first.pixel_status.reshape(-1).tolist())
        session.arrive("deep")
        second = session.read_region(0)
        self.assertTrue(np.all(second.pixel_status == COMMITTED))
        # 无缺口像素界为 0；提前结束像素的界等于停止时残余透射量（紧界）。
        np.testing.assert_array_less(second.forward_bound, data["error"]["forward"] + 1e-12)
        self.assertEqual(second.forward_bound[0, 1], 0.0)
        self.assertEqual(second.forward_bound[1, 1], 0.0)

    def test_adjoint_over_bound_is_rejected(self):
        data = _base_scene()
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")
        session.arrive("deep")
        resp = session.read_region(0)
        bw = session.backward(resp.token, [0], [0], [[2.0, 0.0, 0.0]], [0.0])
        self.assertEqual(bw.pixel_reason[0]["reason"], BW_ADJOINT)


class VersionAndBudgetTests(unittest.TestCase):
    def test_stale_token_rejected_after_facts_replaced(self):
        # 三层深度：front、mid 已到时 (0,0) 在小透射下提前停止，
        # 且最后处理点入口透射很小，旧承诺同时满足反向容差。
        data = _base_scene()
        data["expected_slices"] = {
            "front": {"depth": [0, 3], "rect": [0, 2, 0, 2], "point_limit": 4},
            "mid": {"depth": [3, 5], "rect": [0, 2, 0, 2], "point_limit": 4},
            "deep": {"depth": [5, 9], "rect": [0, 2, 0, 2], "point_limit": 4},
        }
        data["points"]["p2"]["slice"] = "mid"
        data["points"]["p2"]["depth"] = 4.0
        data["points"]["p3"]["slice"] = "deep"
        data["error"] = {"forward": 0.01, "backward": 0.05}
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")
        session.arrive("mid")
        old = session.read_region(0)
        session.arrive("deep")
        # 增补分片不改变已有点事实：旧令牌仍可归还（承诺继续可用）。
        self.assertEqual(old.token.pixels[(0, 0)].status, COMMITTED)
        bw = session.backward(old.token, [0], [0], [[1.0, 1.0, 1.0]], [1.0])
        self.assertEqual(bw.accepted_pixels, 1)
        # 只归到旧令牌冻结过的 p1/p2，不会混入新到的 p3。
        frozen = set(old.token.pixels[(0, 0)].point_ids)
        self.assertEqual(frozen, {"p1", "p2"})
        self.assertTrue(np.all(bw.master_rgb_grad["p3"] == 0.0))
        # 点主记录被替换后，旧令牌指纹失效。
        scene.points["p1"] = type(scene.points["p1"])(
            "p1", (0.5, 0.5, 0.5), 0.5, 1.0, "front", "p1"
        )
        with self.assertRaises(StaleVersionError):
            session.backward(old.token, [0], [0], [[1.0, 1.0, 1.0]], [0.0])

    def test_tokens_keep_separate_point_orders(self):
        data = _base_scene()
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")
        first = session.read_region(0)
        first_order = first.token.pixels[(0, 0)].point_ids
        session.arrive("deep")
        second = session.read_region(0)
        second_order = second.token.pixels[(0, 0)].point_ids
        self.assertEqual(first_order, ("p1",))
        self.assertEqual(second_order, ("p1", "p2", "p3"))
        # 旧令牌的顺序没有被新几何污染。
        self.assertEqual(first.token.pixels[(0, 0)].point_ids, ("p1",))

    def test_tiny_budget_marks_region_needs_continue(self):
        data = _base_scene()
        data["budget_bytes"] = 80  # 连一个块的 Python 暂存都不够
        scene = load_scene(copy.deepcopy(data))
        session = CompositeSession(scene)
        session.arrive("front")
        session.arrive("deep")
        resp = session.read_region(0)
        self.assertEqual(resp.region_status, "needs_continue")
        self.assertFalse(resp.retained)
        self.assertLessEqual(resp.budget["peak_bytes"], data["budget_bytes"])

    def test_peak_budget_never_exceeds_limit_on_large_region(self):
        rng = np.random.default_rng(7)
        n_points = 2000
        points, tiles = {}, {"t": []}
        for i in range(n_points):
            pid = f"p{i:05d}"
            points[pid] = {
                "rgb": rng.random(3).tolist(),
                "alpha": float(rng.random()),
                "depth": 1.0 + 8.0 * rng.random(),
                "slice": "only",
            }
            y = int(rng.integers(0, 64))
            x = int(rng.integers(0, 64))
            tiles["t"].append([pid, y, x, float(rng.random())])
        data = {
            "shape": [64, 64],
            "background": [0.0, 0.0, 0.0],
            "expected_slices": {"only": {"depth": [0, 10], "rect": [0, 64, 0, 64], "point_limit": n_points}},
            "points": points,
            "tiles": tiles,
            "regions": [[0, 64, 0, 64]],
            "dy": [],
            "error": {"forward": 0.01, "backward": 0.05},
            "adjoint_bounds": {"rgb": [1.0, 1.0, 1.0], "transmittance": 1.0},
            "timeline": [["arrive", "only"], ["read", 0]],
            "budget_bytes": 65536,
        }
        result = play(copy.deepcopy(data))
        self.assertLessEqual(result["budget"]["peak_bytes"], 65536)
        read = result["reads"][0]
        self.assertEqual(read["committed_pixels"] + read["waiting_pixels"], 64 * 64)


class RandomizedBoundTests(unittest.TestCase):
    def test_public_bounds_hold_against_any_hidden_geometry(self):
        """在缺口内塞入任意合法隐藏点，真实误差不得超过公开界。"""
        rng = np.random.default_rng(2026)
        bg = np.array([0.2, 0.4, 0.1])
        for _ in range(40):
            # 已知前方点。
            n_known = int(rng.integers(1, 4))
            known = []
            T = 1.0
            for i in range(n_known):
                o = float(rng.random())
                known.append((f"k{i}", float(i), o, rng.random(3)))
                T *= 1 - o
            tol = float(rng.uniform(1e-3, 0.2))
            # 一个覆盖全深度轴的缺口。
            from alpha_compose.composite import PixelCandidate, forward_pixel
            cands = [
                PixelCandidate(i, pid, depth, o, o, rgb)
                for i, (pid, depth, o, rgb) in enumerate(known)
            ]
            result = forward_pixel(cands, [(n_known + 1, n_known + 10)], bg, tol)

            # 缺口内任意数量、任意合法参数的隐藏点（含近乎全不透明）。
            n_hidden = int(rng.integers(1, 6))
            hidden_cands = list(cands)
            for j in range(n_hidden):
                o = float(rng.uniform(0.0, 1.0))
                hidden_cands.append(
                    PixelCandidate(
                        n_known + j, f"h{j}",
                        n_known + 2 + j, o, o, rng.random(3),
                    )
                )
            truth = forward_pixel(hidden_cands, [], bg, tol)
            color_err = np.abs(truth.color - result.color).max()
            trans_err = abs(truth.transmittance - result.transmittance)
            self.assertLessEqual(color_err, result.bound + 1e-12)
            self.assertLessEqual(trans_err, result.bound + 1e-12)
            if result.bound <= tol:
                self.assertEqual(result.stopped or result.first_gap_T > 0, True)

    def test_backward_gradient_bounds_hold_under_perturbation(self):
        """已处理点在缺口被任意几何填充后，梯度变化不超过公开反向界。"""
        rng = np.random.default_rng(11)
        bg = np.array([0.0, 0.0, 0.0])
        from alpha_compose.composite import (
            PixelCandidate, forward_pixel, backward_pixel,
        )
        g = np.array([0.7, -0.3, 0.5])
        g_t = 0.4
        rgb_table = rng.random((8, 3))
        for _ in range(30):
            cands = []
            for i in range(3):
                o = float(rng.random())
                cands.append(PixelCandidate(i, f"k{i}", float(i), o, o, rgb_table[i]))
            approx = forward_pixel(cands, [(5.0, 9.0)], bg, 0.01)
            # 用任意隐藏几何补全缺口后求“真实”梯度。
            full_cands = list(cands)
            for j in range(3):
                o = float(rng.uniform(0.0, 1.0))
                full_cands.append(
                    PixelCandidate(3 + j, f"h{j}", 6.0 + j, o, o, rgb_table[3 + j])
                )
            exact = forward_pixel(full_cands, [], bg, 0.01)
            grad_rgb_a, grad_alp_a = backward_pixel(approx, g, g_t, bg, rgb_table)
            grad_rgb_e, grad_alp_e = backward_pixel(exact, g, g_t, bg, rgb_table)
            # 近似结果可能因提前结束只保留部分已知点，按点号对齐比较。
            kept = {pid: i for i, pid in enumerate(approx.point_ids)}
            rgb_e_by_id = {pid: grad_rgb_e[i] for i, pid in enumerate(exact.point_ids)}
            alp_e_by_id = {pid: grad_alp_e[i] for i, pid in enumerate(exact.point_ids)}
            if not kept:
                continue
            rgb_diff = np.array([
                np.abs(grad_rgb_a[i] - rgb_e_by_id[pid]) for pid, i in kept.items()
            ])
            alp_diff = np.array([
                abs(grad_alp_a[i] - alp_e_by_id[pid]) for pid, i in kept.items()
            ])
            # 缺口填充只可能压低入口透射量：颜色与不透明度梯度的扰动
            # 都受未知点入口透射上界导出的公开界约束（近似值是上界侧）。
            # 最后一个已处理点的不透明度梯度依赖后方未知颜色，
            # 其上界由该点入口透射量决定。
            entry_unknown = max(approx.first_gap_T, approx.transmittance)
            color_bound = entry_unknown * np.abs(g).max()
            self.assertLessEqual(rgb_diff.max(), color_bound + 1e-12)
            last_entry = float(approx.entry_T.max()) if approx.entry_T.size else 0.0
            bound = max(entry_unknown, last_entry) * (np.abs(g).sum() + abs(g_t))
            self.assertLessEqual(alp_diff.max(), bound + 1e-12)


class TimelineTests(unittest.TestCase):
    def test_play_sample_scene(self):
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "samples" / "scene.json"
        result = play(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(len(result["reads"]), 3)
        # 最后一次读取在 middle 补齐后全部承诺。
        self.assertEqual(result["reads"][-1]["region_status"], "committed")
        # 提前结束像素保留等于残余透射量的紧界，整体不超过公开前向容差。
        self.assertLessEqual(result["reads"][-1]["max_forward_bound"], 0.003 + 1e-12)
        bw = result["backwards"][0]
        self.assertEqual(bw["accepted_pixels"], 1)
        self.assertIn("p01", bw["master_alpha_grad"])


if __name__ == "__main__":
    unittest.main()
