"""有状态的可微 alpha 合成会话。

会话接住解析后的 :class:`~alpha_compose.load.Scene`，按几何分片的到达顺序
维护可见性事实。同一会话可随编辑器更换兴趣区域与瓦片方式重复请求；
每次读取颁发不可混用的版本令牌，反向归还时核对该次交付所依据的
深度与覆盖事实，拒绝向已被取代的点应用新几何的梯度。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .budget import BudgetLedger, BudgetExceeded, _PY_OBJECT_BYTES
from .composite import (
    RGB_DIM,
    ForwardResult,
    backward_pixel,
    forward_pixel,
    gap_windows_for_pixel,
    merge_pixel_hits,
    unseen_gradient_bounds,
)
from .load import Scene

# 像素状态：已承诺（公开界内可确定）/ 等待几何 / 覆盖冲突不可确认。
COMMITTED = "committed"
WAITING = "waiting"
CONFLICT = "conflict"

# 区域状态汇总。
REGION_COMMITTED = "committed"
REGION_WAITING = "waiting"
REGION_CONTINUE = "needs_continue"

# 反向逐像素裁决原因。
BW_OK = "accepted"
BW_OUT = "out_of_region"
BW_CONFLICT = "conflict"
BW_NOT_COMMITTED = "not_committed"
BW_ADJOINT = "adjoint_bound_exceeded"
BW_BACKWARD_ERR = "backward_error_exceeded"
BW_NO_STATE = "state_not_retained"

_CHUNK = 64
_NAN = float("nan")


class StaleVersionError(RuntimeError):
    """反向梯度所依据的版本已被取代，不能应用。"""


@dataclass
class PixelState:
    """一次读取中单个像素所冻结的事实（反向归还的唯一依据）。"""

    status: str
    color: np.ndarray | None
    transmittance: float | None
    forward_bound: float
    backward_bound_worst: float
    first_gap_T: float
    retained: bool
    # 以下仅在 retained 且为 committed 时有效。
    entry_T: np.ndarray | None = None
    opacities: np.ndarray | None = None
    coverages: np.ndarray | None = None
    unknown_entry_T: np.ndarray | None = None
    point_indices: np.ndarray | None = None
    point_ids: tuple[str, ...] = ()
    stopped: bool = False


@dataclass
class VersionToken:
    """一次区域读取的版本令牌。

    数据版由帧和几何版共同组成，并附当前已到分片集合与点内容指纹；
    几何补齐后旧令牌仍可用于核对旧承诺，但与新事实不符的归还会被拒绝。
    """

    token_id: int
    region_index: int
    rect: tuple[int, int, int, int]
    frame_version: int
    geometry_version: int
    arrived: frozenset[str]
    point_fingerprint: str
    pixels: dict[tuple[int, int], PixelState]
    superseded: bool = False

    @property
    def version(self) -> str:
        return f"frame={self.frame_version};geometry={self.geometry_version}"


@dataclass
class ReadResponse:
    region_index: int
    rect: tuple[int, int, int, int]
    color: np.ndarray                     # (h,w,3)，冲突像素为 NaN
    transmittance: np.ndarray             # (h,w)，冲突像素为 NaN
    pixel_status: np.ndarray              # (h,w) 字符串数组
    forward_bound: np.ndarray             # (h,w)
    backward_bound_worst: np.ndarray      # (h,w)，按 adjoint_bounds 最坏情况
    forward_trainable: np.ndarray         # (h,w) bool
    backward_ok_worst: np.ndarray         # (h,w) bool
    region_status: str
    retained: bool
    token: VersionToken
    budget: dict


@dataclass
class BackwardResponse:
    token_id: int
    version: str
    master_rgb_grad: dict[str, np.ndarray]
    master_alpha_grad: dict[str, float]
    pixel_reason: list[dict]
    accepted_pixels: int
    rejected_pixels: int
    budget: dict


class CompositeSession:
    def __init__(self, scene: Scene) -> None:
        self.scene = scene
        self.arrived: set[str] = set()
        self.geometry_version = scene.geometry_version
        self.frame_version = scene.frame_version
        self.ledger = BudgetLedger(scene.budget_bytes)
        self._tokens: dict[int, VersionToken] = {}
        self._next_token_id = 0
        self._rgb_table = np.asarray(
            [p.rgb for p in sorted(scene.points.values(), key=lambda p: p.point_id)],
            dtype=np.float64,
        )
        self._id_to_index = {
            p.point_id: i for i, p in enumerate(sorted(scene.points.values(), key=lambda p: p.point_id))
        }
        # 按像素索引瓦片报送，区域读取只扫描与区域相交的报送。
        self._hits_by_pixel: dict[tuple[int, int], list] = {}
        for hit in scene.tiles:
            self._hits_by_pixel.setdefault((hit.y, hit.x), []).append(hit)

    # ------------------------------------------------------------------ 到达

    def arrive(self, slice_name: str) -> None:
        """登记一个几何分片到达；未知分片直接报错。"""
        if slice_name not in self.scene.slices:
            raise KeyError(f"未知几何分片: {slice_name}")
        if slice_name not in self.arrived:
            self.arrived.add(slice_name)
            # 新几何会取代此前的读取承诺：旧令牌标记为已取代，
            # 但仍然保留，供训练侧核对旧梯度；不能再用于新的归还。
            for token in self._tokens.values():
                if not token.superseded:
                    token.superseded = True
            self.geometry_version += 1

    @property
    def arrived_slices(self) -> frozenset[str]:
        return frozenset(self.arrived)

    def _fingerprint(self) -> str:
        """点事实指纹（RGB/透明度/深度/主记录与每像素覆盖）。

        新分片到达只增补点，不改写已有点事实，因此旧令牌冻结的导数
        仍然自洽；若点主记录本身被替换，指纹变化会令旧令牌归还失效。
        """
        h = hashlib.sha256()
        for point_id in sorted(self.scene.points):
            p = self.scene.points[point_id]
            h.update(point_id.encode())
            h.update(repr((p.rgb, p.alpha, p.depth, p.slice_name, p.master)).encode())
        for (y, x), hits in sorted(self._hits_by_pixel.items()):
            for hit in sorted(hits, key=lambda z: (z.tile, z.point_id)):
                h.update(repr((y, x, hit.tile, hit.point_id, hit.coverage)).encode())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------ 读取

    def read_region(self, region_index: int, *, retain: bool = True) -> ReadResponse:
        scene = self.scene
        if not (0 <= region_index < len(scene.regions)):
            raise IndexError(f"区域编号越界: {region_index}")
        y0, y1, x0, x1 = scene.regions[region_index]
        h, w = y1 - y0, x1 - x0

        color = np.full((h, w, RGB_DIM), _NAN, dtype=np.float64)
        trans = np.full((h, w), _NAN, dtype=np.float64)
        status = np.empty((h, w), dtype=object)
        fbound = np.zeros((h, w), dtype=np.float64)
        bbound = np.zeros((h, w), dtype=np.float64)
        ftrain = np.zeros((h, w), dtype=bool)
        bok = np.zeros((h, w), dtype=bool)
        pixel_states: dict[tuple[int, int], PixelState] = {}

        region_status = REGION_COMMITTED
        retained = retain
        worst_g = np.asarray(scene.rgb_bound, dtype=np.float64)
        worst_gt = scene.trans_bound

        pixel_coords = [(y, x) for y in range(y0, y1) for x in range(x0, x1)]
        for chunk_start in range(0, len(pixel_coords), _CHUNK):
            chunk = pixel_coords[chunk_start:chunk_start + _CHUNK]
            # 本块的 Python 临时报送列表，按预算保守记账。
            try:
                self.ledger.check(len(chunk) * _PY_OBJECT_BYTES)
            except BudgetExceeded:
                region_status = REGION_CONTINUE
                retained = False
                for y, x in chunk:
                    sy, sx = y - y0, x - x0
                    status[sy, sx] = WAITING
                    state = PixelState(
                        status=WAITING,
                        color=None,
                        transmittance=None,
                        forward_bound=_NAN,
                        backward_bound_worst=_NAN,
                        first_gap_T=_NAN,
                        retained=False,
                    )
                    pixel_states[(y, x)] = state
                continue

            chunk_states: list[tuple[tuple[int, int], PixelState]] = []
            for y, x in chunk:
                state, result = self._read_one_pixel(y, x, retain, worst_g, worst_gt)
                sy, sx = y - y0, x - x0
                status[sy, sx] = state.status
                fbound[sy, sx] = state.forward_bound
                bbound[sy, sx] = state.backward_bound_worst
                if state.status != CONFLICT:
                    color[sy, sx] = state.color
                    trans[sy, sx] = state.transmittance
                    ftrain[sy, sx] = (
                        state.status == COMMITTED and state.forward_bound <= scene.forward_tol
                    )
                    bok[sy, sx] = (
                        state.status == COMMITTED
                        and state.backward_bound_worst <= scene.backward_tol
                    )
                if state.status == WAITING and region_status == REGION_COMMITTED:
                    region_status = REGION_WAITING
                chunk_states.append(((y, x), state))

            if retain:
                # 暂存反向所需事实；预算不允许时，本块只作前向预览。
                need = sum(
                    (
                        (s.entry_T.nbytes if s.entry_T is not None else 0)
                        + (s.opacities.nbytes if s.opacities is not None else 0)
                        + (s.coverages.nbytes if s.coverages is not None else 0)
                        + (s.unknown_entry_T.nbytes if s.unknown_entry_T is not None else 0)
                        + (s.point_indices.nbytes if s.point_indices is not None else 0)
                    )
                    for _, s in chunk_states
                ) + len(chunk_states) * _PY_OBJECT_BYTES
                try:
                    self.ledger.check(need)
                except BudgetExceeded:
                    region_status = REGION_CONTINUE
                    retained = False
                    for _, s in chunk_states:
                        s.retained = False
                        s.entry_T = None
                        s.opacities = None
                        s.coverages = None
                        s.unknown_entry_T = None
                        s.point_indices = None
                        s.point_ids = ()
                else:
                    for _, s in chunk_states:
                        self.ledger.reserve(
                            [
                                a
                                for a in (
                                    s.entry_T, s.opacities, s.coverages,
                                    s.unknown_entry_T, s.point_indices,
                                )
                                if a is not None
                            ],
                            py_objects=1,
                        )
            for coord, s in chunk_states:
                pixel_states[coord] = s

        token = VersionToken(
            token_id=self._next_token_id,
            region_index=region_index,
            rect=(y0, y1, x0, x1),
            frame_version=self.frame_version,
            geometry_version=self.geometry_version,
            arrived=frozenset(self.arrived),
            point_fingerprint=self._fingerprint(),
            pixels=pixel_states,
        )
        self._tokens[self._next_token_id] = token
        self._next_token_id += 1

        return ReadResponse(
            region_index=region_index,
            rect=(y0, y1, x0, x1),
            color=color,
            transmittance=trans,
            pixel_status=status,
            forward_bound=fbound,
            backward_bound_worst=bbound,
            forward_trainable=ftrain,
            backward_ok_worst=bok,
            region_status=region_status,
            retained=retained,
            token=token,
            budget=self.ledger.snapshot(),
        )

    def _read_one_pixel(
        self,
        y: int,
        x: int,
        retain: bool,
        worst_g: np.ndarray,
        worst_gt: float,
    ) -> tuple[PixelState, ForwardResult | None]:
        scene = self.scene
        raw_hits = self._hits_by_pixel.get((y, x), [])

        prepared = []
        for hit in raw_hits:
            point = scene.points[hit.point_id]
            if point.slice_name not in self.arrived:
                # 资料约定：事件按几何片解释，未到分片的报送不参与本版可见性。
                continue
            prepared.append(
                (point.point_id, point.depth, hit.coverage, point.alpha, point.rgb)
            )

        candidates, conflict = merge_pixel_hits(prepared)
        if conflict:
            return (
                PixelState(
                    status=CONFLICT,
                    color=None,
                    transmittance=None,
                    forward_bound=_NAN,
                    backward_bound_worst=_NAN,
                    first_gap_T=_NAN,
                    retained=False,
                ),
                None,
            )
        for cand in candidates:
            cand.point_index = self._id_to_index[cand.point_id]

        gaps = gap_windows_for_pixel(y, x, frozenset(self.arrived), scene.slices)
        result = forward_pixel(candidates, gaps, np.asarray(scene.background), scene.forward_tol)

        # 最坏情况下的反向误差界（用 adjoint_bounds 作为可能上游梯度的公开上界）。
        bw_bound = self._backward_bound(result, worst_g, worst_gt)

        committed = result.bound <= scene.forward_tol
        status = COMMITTED if committed else WAITING
        keep = retain and committed
        state = PixelState(
            status=status,
            color=result.color,
            transmittance=result.transmittance,
            forward_bound=result.bound,
            backward_bound_worst=bw_bound,
            first_gap_T=result.first_gap_T,
            retained=keep,
            entry_T=result.entry_T.copy() if keep else None,
            opacities=result.opacities.copy() if keep else None,
            coverages=result.coverages.copy() if keep else None,
            unknown_entry_T=result.unknown_entry_T.copy() if keep else None,
            point_indices=result.point_indices.copy() if keep else None,
            point_ids=tuple(result.point_ids) if keep else (),
            stopped=result.stopped,
        )
        return state, result

    @staticmethod
    def _backward_bound(result: ForwardResult, g: np.ndarray, g_t: float) -> float:
        """给定上游梯度绝对值时，本像素全部点（含尚未出现的真实点）的最大梯度误差。

        * 未到达/未处理点：入口透射量不超过首个缺口入口，或提前停止时的
          残余透射量，颜色与不透明度梯度分别有线性界；
        * 已处理点：颜色梯度精确；不透明度梯度依赖后方合成色，逐点的误差
          上界由该点后方未知区域的入口透射量 ``unknown_entry_T`` 决定，
          后方几何已确定的点该值为 0，梯度精确。
        """
        g_abs = np.abs(np.asarray(g, dtype=np.float64))
        g_sum = float(g_abs.sum()) + abs(float(g_t))
        if result.opacities.shape[0] == 0:
            entry_unknown = result.first_gap_T if result.first_gap_T > 0 else 0.0
            if entry_unknown == 0.0:
                return 0.0
            color_unknown, opacity_unknown = unseen_gradient_bounds(entry_unknown, g_abs, g_t)
            return float(max(color_unknown.max(initial=0.0), opacity_unknown))

        # 逐点的不透明度梯度误差上界（后方无未知几何的点为 0）。
        processed_alpha_err = float((result.unknown_entry_T * g_sum).max())
        # 未到达点的入口透射量上界。
        entry_unknown = result.first_gap_T
        if result.stopped:
            entry_unknown = max(entry_unknown, result.transmittance)
        color_unknown, opacity_unknown = unseen_gradient_bounds(max(entry_unknown, 0.0), g_abs, g_t)
        return float(max(
            color_unknown.max(initial=0.0),
            opacity_unknown,
            processed_alpha_err,
        ))

    # ------------------------------------------------------------------ 反向

    def backward(
        self,
        token: VersionToken,
        ys: Iterable[int],
        xs: Iterable[int],
        rgb_grad: np.ndarray,
        t_grad: Iterable[float],
    ) -> BackwardResponse:
        """按令牌冻结的深度与覆盖事实归还梯度，按主记录求和。

        梯度超出 ``adjoint_bounds`` 的像素不会被应用（需要重新取得更高精度
        的结果）；已在允许误差内的训练承诺继续可用。
        """

        if not isinstance(token, VersionToken) or token.token_id not in self._tokens:
            raise StaleVersionError("令牌不属于本会话")
        if token.point_fingerprint != self._fingerprint():
            raise StaleVersionError(
                "点主记录或覆盖事实已被替换，不能向已被取代的点应用新几何的梯度"
            )

        ys = list(ys)
        xs = list(xs)
        rgb_grad = np.asarray(rgb_grad, dtype=np.float64).reshape(-1, RGB_DIM)
        t_grad = np.asarray(list(t_grad), dtype=np.float64)
        if not (rgb_grad.shape[0] == len(ys) == len(xs) == t_grad.shape[0]):
            raise ValueError("反向输入长度不一致")

        rgb_bound = np.asarray(self.scene.rgb_bound, dtype=np.float64)
        # 先按点建账（每个点一份暂存），输出前再按主记录汇总。
        point_rgb: dict[str, np.ndarray] = {
            pid: np.zeros(RGB_DIM, dtype=np.float64) for pid in self.scene.points
        }
        point_alpha: dict[str, float] = {pid: 0.0 for pid in self.scene.points}
        # 归并暂存计入临时区预算（最终交付的字典不计）。
        self.ledger.reserve([], py_objects=2 * len(point_rgb))

        reasons: list[dict] = []
        accepted = 0
        rejected = 0
        y0, y1, x0, x1 = token.rect
        index_to_id = {v: k for k, v in self._id_to_index.items()}

        try:
            for y, x, g, g_t in zip(ys, xs, rgb_grad, t_grad):
                y, x = int(y), int(x)
                base = {"y": y, "x": x}
                if not (y0 <= y < y1 and x0 <= x < x1):
                    reasons.append({**base, "reason": BW_OUT})
                    rejected += 1
                    continue
                state = token.pixels.get((y, x))
                if state is None:
                    reasons.append({**base, "reason": BW_OUT})
                    rejected += 1
                    continue
                if state.status == CONFLICT:
                    reasons.append({**base, "reason": BW_CONFLICT})
                    rejected += 1
                    continue
                if state.status != COMMITTED:
                    reasons.append({**base, "reason": BW_NOT_COMMITTED})
                    rejected += 1
                    continue
                if not state.retained:
                    reasons.append({**base, "reason": BW_NO_STATE})
                    rejected += 1
                    continue
                if np.any(np.abs(g) > rgb_bound + 1e-12) or abs(g_t) > self.scene.trans_bound + 1e-12:
                    reasons.append({**base, "reason": BW_ADJOINT})
                    rejected += 1
                    continue
                # 用本次真实上游梯度重新证明反向误差。
                bw_bound = self._backward_bound_from_state(state, g, g_t)
                if bw_bound > self.scene.backward_tol + 1e-12:
                    reasons.append({**base, "reason": BW_BACKWARD_ERR, "bound": bw_bound})
                    rejected += 1
                    continue

                result = ForwardResult(
                    color=state.color,
                    transmittance=state.transmittance,
                    bound=state.forward_bound,
                    first_gap_T=state.first_gap_T,
                    entry_T=state.entry_T,
                    opacities=state.opacities,
                    coverages=state.coverages,
                    unknown_entry_T=state.unknown_entry_T,
                    point_ids=list(state.point_ids),
                    point_indices=state.point_indices,
                    stopped=state.stopped,
                )
                grad_rgb, grad_alpha = backward_pixel(
                    result, g, float(g_t),
                    np.asarray(self.scene.background), self._rgb_table,
                )
                for row, pi in zip(range(grad_rgb.shape[0]), state.point_indices):
                    point_id = index_to_id[int(pi)]
                    point_rgb[point_id] += grad_rgb[row]
                    point_alpha[point_id] += grad_alpha[row]
                reasons.append({**base, "reason": BW_OK})
                accepted += 1
        finally:
            self.ledger.release_py(2 * len(point_rgb))

        # 空间上相同点的多像素贡献按主记录求和。
        master_rgb: dict[str, np.ndarray] = {}
        master_alpha: dict[str, float] = {}
        for pid, grad in point_rgb.items():
            master = self.scene.points[pid].master
            master_rgb[master] = master_rgb.get(master, np.zeros(RGB_DIM)) + grad
            master_alpha[master] = master_alpha.get(master, 0.0) + point_alpha[pid]

        return BackwardResponse(
            token_id=token.token_id,
            version=token.version,
            master_rgb_grad=master_rgb,
            master_alpha_grad=master_alpha,
            pixel_reason=reasons,
            accepted_pixels=accepted,
            rejected_pixels=rejected,
            budget=self.ledger.snapshot(),
        )

    def _backward_bound_from_state(self, state: PixelState, g: np.ndarray, g_t: float) -> float:
        result = ForwardResult(
            color=state.color,
            transmittance=state.transmittance,
            bound=state.forward_bound,
            first_gap_T=state.first_gap_T,
            entry_T=state.entry_T if state.entry_T is not None else np.zeros(0),
            opacities=state.opacities if state.opacities is not None else np.zeros(0),
            coverages=state.coverages if state.coverages is not None else np.zeros(0),
            unknown_entry_T=(
                state.unknown_entry_T if state.unknown_entry_T is not None else np.zeros(0)
            ),
            point_ids=list(state.point_ids),
            point_indices=state.point_indices if state.point_indices is not None else np.zeros(0, dtype=np.int32),
            stopped=state.stopped,
        )
        return self._backward_bound(result, g, g_t)

    def release_token(self, token: VersionToken) -> None:
        """释放令牌占用的临时区。"""
        if token.token_id not in self._tokens:
            return
        arrays = []
        for state in token.pixels.values():
            arrays.extend(
                a
                for a in (
                    state.entry_T, state.opacities, state.coverages,
                    state.unknown_entry_T, state.point_indices,
                )
                if a is not None
            )
        self.ledger.release(*arrays)
        self.ledger.release_py(len(token.pixels))
        del self._tokens[token.token_id]
