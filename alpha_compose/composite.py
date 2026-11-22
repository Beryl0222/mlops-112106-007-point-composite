"""单像素前向 alpha 合成与反向梯度的纯函数实现。

约定（exchange.md）：

* 有效不透明度 ``o = 覆盖系数 coverage × 点透明度 alpha``；
* 从前向后透射乘积合成：``C = Σ T_i o_i c_i``，``T_i`` 为进入点 i 前的透射量，
  残余透射 ``T_final`` 乘背景色；
* 深度小者在前，同深度按稳定点号排序，同一点在同一像素只进入一次；
* 尚未到达的几何分片构成深度缺口（gap）。

误差界（只用公开事实：量值在 [0,1]、候选上界 point_limit）：

* 任意缺口之后的颜色对每通道影响不超过缺口入口透射量 ``T_gap``，
  对残余透射量的影响同理；
* 利用 ``dC/do_i = T_i (c_i - C_behind_i)``、``C_behind ∈ [0,1]``，
  任意入口透射量不超过 ``T_e`` 的点，其不透明度梯度绝对值不超过
  ``T_e (Σ|g_rgb| + |g_t|``)，颜色梯度每通道不超过 ``|g_ch| T_e``——
  即使该点不透明度趋近 1，该界依然有限。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RGB_DIM = 3


@dataclass
class PixelCandidate:
    """合并去重后进入某像素的一个点（一次）。"""

    point_index: int
    point_id: str
    depth: float
    opacity: float
    coverage: float
    rgb: np.ndarray


def merge_pixel_hits(
    hits: list[tuple[str, float, float, float, np.ndarray]],
) -> tuple[list[PixelCandidate], bool]:
    """合并同一像素的瓦片报送。

    ``hits`` 元素为 ``(point_id, depth, coverage, alpha, rgb)``：
    有效不透明度 ``o = coverage × alpha``。本函数只负责同版去重与冲突判定：

    * 同一点重复报送且覆盖一致 → 只算一次（光栅跨界重复报送）；
    * 同一点覆盖冲突（或深度冲突，深度来自同一点主记录时理论上一致，
      仍做防御性检查）→ 该像素不可确认，返回 ``conflict=True``。
    """

    merged: dict[str, tuple[float, float, float, np.ndarray]] = {}
    conflict = False
    for point_id, depth, coverage, alpha, rgb in hits:
        if point_id in merged:
            old_depth, old_cov, _, _ = merged[point_id]
            # 覆盖系数由浮点乘积得到，重复报送按容差视为同版一致；
            # 真正不同的申报（如 0.9 与 0.35）仍远超容差，会被判冲突。
            if not (np.isclose(old_cov, coverage, rtol=0.0, atol=1e-12)
                    and old_depth == depth):
                conflict = True
                continue
        else:
            merged[point_id] = (depth, coverage, alpha, rgb)
    if conflict:
        return [], True

    candidates = [
        PixelCandidate(
            point_index=-1,  # 由调用方在打包时填入点表下标
            point_id=pid,
            depth=item_depth,
            opacity=float(item_coverage * item_alpha),
            coverage=float(item_coverage),
            rgb=np.asarray(item_rgb, dtype=np.float64),
        )
        for pid, (item_depth, item_coverage, item_alpha, item_rgb) in merged.items()
    ]
    # 深度小到大，同深度按稳定点号定序（点号为字符串，按码点稳定排序）。
    candidates.sort(key=lambda c: (c.depth, c.point_id))
    return candidates, False


def gap_windows_for_pixel(
    pixel_y: int,
    pixel_x: int,
    arrived: frozenset[str],
    slices: dict,
) -> list[tuple[float, float]]:
    """返回覆盖该像素、且对应分片尚未到达的深度窗口（按深度排序）。"""

    gaps = []
    for spec in slices.values():
        if spec.name in arrived:
            continue
        y0, y1, x0, x1 = spec.rect
        if y0 <= pixel_y < y1 and x0 <= pixel_x < x1:
            gaps.append(spec.depth)
    gaps.sort()
    return gaps


@dataclass
class ForwardResult:
    color: np.ndarray              # (3,) 合成颜色
    transmittance: float           # 残余透射量（乘背景前）
    bound: float                   # 每通道颜色与透射量的统一绝对误差上界
    first_gap_T: float             # 首个深度缺口入口透射量；无缺口为 0.0
    entry_T: np.ndarray            # 每个已处理候选的入口透射量
    opacities: np.ndarray          # 每个已处理候选的有效不透明度 o=cov*alpha
    coverages: np.ndarray          # 每个已处理候选的像素覆盖系数（alpha 反传用）
    # 每个已处理候选后方未知区域的入口透射量上界：该候选之后若没有
    # 未到达几何则为 0（梯度精确），否则取其后方第一个缺口入口透射量。
    unknown_entry_T: np.ndarray
    point_ids: list[str]           # 已处理候选点号（与数组对齐）
    point_indices: np.ndarray      # 点表下标
    stopped: bool                  # 是否因透射上界足够小而提前结束


def forward_pixel(
    candidates: list[PixelCandidate],
    gaps: list[tuple[float, float]],
    background: np.ndarray,
    forward_tol: float,
) -> ForwardResult:
    """前向合成一个像素，并证明公开误差上界。

    把深度轴上的两类事件按顺序处理：

    * 缺口事件（尚未到达的几何分片）：缺口内任意点每通道最多改变当时的
      透射量 ``T``，对残余透射量同理，因此把 ``T`` 累计进误差界；
    * 候选点事件：标准前向透射乘积合成。

    当运行透射量已不超过 ``forward_tol`` 且后面还挂着未处理事件时，
    在公开误差界内提前结束（``stopped=True``）；几何已齐时必须走完
    全部候选，给出精确结果与零误差界。
    """

    background = np.asarray(background, dtype=np.float64)
    T = 1.0
    color = np.zeros(RGB_DIM, dtype=np.float64)
    bound = 0.0
    first_gap_T = 0.0
    gap_index = 0
    n_gaps = len(gaps)
    # 已结算缺口的 (深度下界, 入口透射量)，用于逐点反向误差界。
    gap_entries: list[tuple[float, float]] = []
    entry_T: list[float] = []
    opacities: list[float] = []
    coverages: list[float] = []
    point_depths: list[float] = []
    point_ids: list[str] = []
    point_indices: list[int] = []
    stopped = False

    def _settle_gap(current_T: float) -> None:
        nonlocal first_gap_T, bound, gap_index
        if first_gap_T == 0.0:
            first_gap_T = current_T
        gap_entries.append((gaps[gap_index][0], current_T))
        bound += current_T
        gap_index += 1

    n_candidates = len(candidates)
    for ci, cand in enumerate(candidates):
        # 先结算严格位于本候选之前的缺口（同深度缺口与本候选一起延后，
        # 保证“候选之后是否还有未到达几何”的信号不丢失）。
        while gap_index < n_gaps and gaps[gap_index][1] <= cand.depth:
            _settle_gap(T)

        entry_T.append(T)
        opacities.append(cand.opacity)
        coverages.append(cand.coverage)
        point_depths.append(cand.depth)
        point_ids.append(cand.point_id)
        point_indices.append(cand.point_index)
        color += T * cand.opacity * cand.rgb
        T *= 1.0 - cand.opacity

        # 本候选处理完后，若已没有更深的已到达候选、深度轴上还挂着
        # 未结算缺口，且当前透射量足够小，则在公开误差界内提前结束。
        # 后面若还有已到达候选，必须继续精确处理，不能用界跳过。
        is_last_candidate = ci == n_candidates - 1
        gaps_remain = gap_index < n_gaps
        if T <= forward_tol and is_last_candidate and gaps_remain:
            while gap_index < n_gaps:
                _settle_gap(T)
            stopped = True
            break

    if not stopped:
        # 候选走完，像素尾部可能还挂着缺口（含像素无候选的情形），
        # 按各缺口入口处的实际透射量逐笔记账。
        while gap_index < n_gaps:
            _settle_gap(T)

    # 逐点求后方最近缺口的入口透射量：该点不透明度梯度的误差上界。
    # 后方无未到达几何的点取 0，梯度精确。
    unknown_entry_T = np.zeros(len(entry_T), dtype=np.float64)
    if gap_entries:
        gap_depths = np.asarray([d for d, _ in gap_entries])
        gap_Ts = np.asarray([t for _, t in gap_entries])
        depths_arr = np.asarray(point_depths)
        for i, depth in enumerate(depths_arr):
            later = np.nonzero(gap_depths > depth)[0]
            if later.size:
                unknown_entry_T[i] = gap_Ts[int(later[0])]

    color = color + T * background
    return ForwardResult(
        color=color,
        transmittance=T,
        bound=bound,
        first_gap_T=first_gap_T,
        entry_T=np.asarray(entry_T, dtype=np.float64),
        opacities=np.asarray(opacities, dtype=np.float64),
        coverages=np.asarray(coverages, dtype=np.float64),
        unknown_entry_T=unknown_entry_T,
        point_ids=point_ids,
        point_indices=np.asarray(point_indices, dtype=np.int32),
        stopped=stopped,
    )


def backward_pixel(
    result: ForwardResult,
    rgb_adjoint: np.ndarray,
    t_adjoint: float,
    background: np.ndarray,
    rgb_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """对已处理候选反传，返回 ``(颜色梯度 (n,3), 点透明度梯度 (n,))``。

    采用无除法的后方合成色形式，``o → 1`` 时数值稳定：

    ``dL/do_i = T_i [ g_rgb·(c_i - C_behind_i) - g_t τ_behind_i ]``，

    其中 ``C_behind`` 是点 i 后方（含背景）的归一化合成色，
    ``τ_behind`` 是点 i 后方的相对透射量，二者都在 [0,1]。
    点透明度梯度按链式法则补上覆盖系数：
    ``dL/dalpha_i = coverage_i · dL/do_i``。
    """

    n = result.opacities.shape[0]
    if n == 0:
        return np.zeros((0, RGB_DIM)), np.zeros(0)

    g = np.asarray(rgb_adjoint, dtype=np.float64)
    g_t = float(t_adjoint)
    background = np.asarray(background, dtype=np.float64)
    colors = rgb_table[result.point_indices]

    # 从后向前累积后方合成色 C_behind 与后方相对透射 tau_behind。
    grad_rgb = np.empty((n, RGB_DIM), dtype=np.float64)
    grad_o = np.empty(n, dtype=np.float64)
    behind_color = background.copy()
    behind_tau = 1.0
    for i in range(n - 1, -1, -1):
        o = result.opacities[i]
        grad_rgb[i] = g * result.entry_T[i] * o
        grad_o[i] = result.entry_T[i] * (
            float(np.dot(g, colors[i] - behind_color)) - g_t * behind_tau
        )
        # 把本点并入“后方”：C' = o c + (1-o) C，τ' = (1-o) τ。
        behind_color = o * colors[i] + (1.0 - o) * behind_color
        behind_tau = (1.0 - o) * behind_tau
    grad_alpha = grad_o * result.coverages
    return grad_rgb, grad_alpha


def unseen_gradient_bounds(
    entry_T_bound: float,
    rgb_adjoint: np.ndarray,
    t_adjoint: float,
) -> tuple[np.ndarray, float]:
    """入口透射量不超过 ``entry_T_bound`` 的未处理点的梯度公开上界。

    返回每通道颜色梯度上界数组与不透明度梯度标量上界。
    """

    g = np.abs(np.asarray(rgb_adjoint, dtype=np.float64))
    color_bound = g * entry_T_bound
    opacity_bound = entry_T_bound * (float(g.sum()) + abs(float(t_adjoint)))
    return color_bound, opacity_bound
