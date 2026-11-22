"""交换资料的加载与校验。

资料约定（见 exchange.md）：

* 像素坐标为 (y, x)，矩形 ``[y0, y1, x0, x1)`` 右/下边界不含；
* 每个点携带深度、稳定编号、主记录、RGB、标量透明度与像素覆盖系数；
* 有效深度为正，颜色分量、点透明度、覆盖系数均在 [0, 1]；
* 同版同一点在同一像素的重复报送只算一次，深度或覆盖冲突则该像素不可确认；
* 数据版由帧和几何版共同组成。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SliceSpec:
    """一个几何分片声明：深度窗口、可能覆盖的矩形、候选点上界。"""

    name: str
    depth: tuple[float, float]
    rect: tuple[int, int, int, int]
    point_limit: int


@dataclass(frozen=True)
class PointRecord:
    """点主记录。``master`` 为梯度归并所指向的主记录编号。"""

    point_id: str
    rgb: tuple[float, float, float]
    alpha: float
    depth: float
    slice_name: str
    master: str


@dataclass
class TileHit:
    """一次瓦片报送：点、像素 (y, x) 与覆盖系数。"""

    point_id: str
    y: int
    x: int
    coverage: float
    tile: str


@dataclass
class Scene:
    """解析后的合成场景。

    ``tiles`` 保留原始报送（含瓦片来源），去重与冲突判定推迟到会话读取时，
    这样同一份场景可以在不同到达顺序下反复回放。
    """

    shape: tuple[int, int]
    background: tuple[float, float, float]
    slices: dict[str, SliceSpec]
    points: dict[str, PointRecord]
    tiles: list[TileHit]
    regions: list[tuple[int, int, int, int]]
    dy: tuple[np.ndarray, np.ndarray, np.ndarray]
    forward_tol: float
    backward_tol: float
    rgb_bound: tuple[float, float, float]
    trans_bound: float
    timeline: list[tuple[str, Any]]
    budget_bytes: int
    frame_version: int = 0
    geometry_version: int = 0
    raw: dict = field(default_factory=dict)


def _require(mapping: dict, key: str, source: str) -> Any:
    if key not in mapping:
        raise KeyError(f"{source} 缺少关键字段: {key}")
    return mapping[key]


def _bounded(value: float, name: str) -> float:
    value = float(value)
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} 超出 [0,1]: {value}")
    return value


def load_scene(source: str | Path | dict) -> Scene:
    """从 JSON 路径或已解析的字典构造 :class:`Scene`，并做资料边界校验。"""

    if isinstance(source, dict):
        data = source
        source_name = "<dict>"
    else:
        path = Path(source)
        data = json.loads(path.read_text(encoding="utf-8"))
        source_name = str(path)

    height, width = (int(v) for v in _require(data, "shape", source_name))
    if height <= 0 or width <= 0:
        raise ValueError("shape 必须为正")

    background = tuple(_bounded(v, "background") for v in _require(data, "background", source_name))
    if len(background) != 3:
        raise ValueError("background 必须是三通道")

    slices: dict[str, SliceSpec] = {}
    for name, spec in _require(data, "expected_slices", source_name).items():
        d0, d1 = (float(v) for v in _require(spec, "depth", name))
        if not (d0 >= 0 and d1 > d0):
            raise ValueError(f"分片 {name} 的深度窗口非法: {(d0, d1)}")
        y0, y1, x0, x1 = (int(v) for v in _require(spec, "rect", name))
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError(f"分片 {name} 的矩形越界")
        limit = int(_require(spec, "point_limit", name))
        if limit <= 0:
            raise ValueError(f"分片 {name} 的 point_limit 必须为正")
        slices[name] = SliceSpec(name, (d0, d1), (y0, y1, x0, x1), limit)

    points: dict[str, PointRecord] = {}
    for point_id, desc in _require(data, "points", source_name).items():
        rgb = tuple(_bounded(v, f"{point_id}.rgb[{i}]") for i, v in enumerate(_require(desc, "rgb", point_id)))
        if len(rgb) != 3:
            raise ValueError(f"{point_id}.rgb 必须是三通道")
        alpha = _bounded(_require(desc, "alpha", point_id), f"{point_id}.alpha")
        depth = float(_require(desc, "depth", point_id))
        if not depth > 0:
            raise ValueError(f"{point_id}.depth 必须为正")
        slice_name = _require(desc, "slice", point_id)
        if slice_name not in slices:
            raise ValueError(f"{point_id} 引用了未知分片 {slice_name}")
        spec = slices[slice_name]
        if not (spec.depth[0] <= depth < spec.depth[1]):
            raise ValueError(f"{point_id} 的深度 {depth} 不属于分片 {slice_name} 的窗口 {spec.depth}")
        master = str(desc.get("master", point_id))
        if master not in data["points"]:
            raise ValueError(f"{point_id} 的主记录 {master} 不存在")
        points[point_id] = PointRecord(point_id, rgb, alpha, depth, slice_name, master)

    tiles: list[TileHit] = []
    for tile_name, hits in _require(data, "tiles", source_name).items():
        for entry in hits:
            point_id, y, x, coverage = entry
            if point_id not in points:
                raise ValueError(f"瓦片 {tile_name} 报送了未知点 {point_id}")
            y, x = int(y), int(x)
            if not (0 <= y < height and 0 <= x < width):
                raise ValueError(f"瓦片 {tile_name} 的像素 ({y},{x}) 越界")
            coverage = _bounded(coverage, f"{tile_name}:{point_id} 覆盖系数")
            desc = points[point_id]
            y0, y1, x0, x1 = slices[desc.slice_name].rect
            if not (y0 <= y < y1 and x0 <= x < x1):
                raise ValueError(
                    f"瓦片 {tile_name} 报送 {point_id} 到 ({y},{x})，超出其分片矩形"
                )
            tiles.append(TileHit(point_id, y, x, coverage, tile_name))

    regions: list[tuple[int, int, int, int]] = []
    for rect in _require(data, "regions", source_name):
        y0, y1, x0, x1 = (int(v) for v in rect)
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError(f"兴趣区域越界: {rect}")
        regions.append((y0, y1, x0, x1))

    dy_rows = _require(data, "dy", source_name)
    dy_y = np.asarray([int(r[0]) for r in dy_rows], dtype=np.int64)
    dy_x = np.asarray([int(r[1]) for r in dy_rows], dtype=np.int64)
    dy_rgb = np.asarray([r[2] for r in dy_rows], dtype=np.float64)
    dy_t = np.asarray([float(r[3]) for r in dy_rows], dtype=np.float64)

    error = _require(data, "error", source_name)
    forward_tol = float(_require(error, "forward", "error"))
    backward_tol = float(_require(error, "backward", "error"))
    if forward_tol <= 0 or backward_tol <= 0:
        raise ValueError("误差界必须为正")

    bounds = _require(data, "adjoint_bounds", source_name)
    rgb_bound = tuple(float(v) for v in _require(bounds, "rgb", "adjoint_bounds"))
    trans_bound = float(_require(bounds, "transmittance", "adjoint_bounds"))

    timeline = [(event, payload) for event, payload in _require(data, "timeline", source_name)]
    budget_bytes = int(_require(data, "budget_bytes", source_name))
    if budget_bytes <= 0:
        raise ValueError("budget_bytes 必须为正")

    return Scene(
        shape=(height, width),
        background=background,
        slices=slices,
        points=points,
        tiles=tiles,
        regions=regions,
        dy=(dy_y, dy_x, dy_rgb, dy_t),
        forward_tol=forward_tol,
        backward_tol=backward_tol,
        rgb_bound=rgb_bound,
        trans_bound=trans_bound,
        timeline=timeline,
        budget_bytes=budget_bytes,
        frame_version=int(data.get("frame_version", 0)),
        geometry_version=int(data.get("geometry_version", 0)),
        raw=data,
    )
