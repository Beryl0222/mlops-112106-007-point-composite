"""按交换资料中的时间线回放会话。

时间序列条目表示调用方送达或发起读取的顺序，数值为输入：

* ``["arrive", 分片名]`` —— 几何分片送达；
* ``["read", 区域编号]`` —— 对兴趣区域发起一次前向读取；
* ``["backward", 区域编号]`` —— 用场景中的 ``dy`` 对该区域最近一次
  读取所冻结的事实归还梯度。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .load import Scene, load_scene
from .session import (
    COMMITTED,
    CompositeSession,
    ReadResponse,
    BackwardResponse,
)


def play(scene: Scene | str) -> dict[str, Any]:
    """回放完整时间线，返回每次读取与反向的结构化结果。"""

    if not isinstance(scene, Scene):
        scene = load_scene(scene)
    session = CompositeSession(scene)
    reads: list[dict[str, Any]] = []
    backwards: list[dict[str, Any]] = []
    latest_token: dict[int, Any] = {}
    dy_y, dy_x, dy_rgb, dy_t = scene.dy

    for event, payload in scene.timeline:
        if event == "arrive":
            session.arrive(payload)
        elif event == "read":
            response = session.read_region(int(payload))
            latest_token[int(payload)] = response.token
            reads.append(_summarize_read(response))
        elif event == "backward":
            region_index = int(payload)
            token = latest_token.get(region_index)
            if token is None:
                backwards.append({"region": region_index, "error": "该区域尚无读取记录"})
                continue
            y0, y1, x0, x1 = token.rect
            mask = (dy_y >= y0) & (dy_y < y1) & (dy_x >= x0) & (dy_x < x1)
            response = session.backward(
                token,
                dy_y[mask],
                dy_x[mask],
                dy_rgb[mask],
                dy_t[mask],
            )
            backwards.append(_summarize_backward(response))
        else:
            raise ValueError(f"未知时间线事件: {event}")

    return {
        "arrived": sorted(session.arrived),
        "data_version": f"frame={session.frame_version};geometry={session.geometry_version}",
        "reads": reads,
        "backwards": backwards,
        "budget": session.ledger.snapshot(),
        "session": session,
    }


def _summarize_read(response: ReadResponse) -> dict[str, Any]:
    trainable = response.forward_trainable
    backward_ok = response.backward_ok_worst
    return {
        "region": response.region_index,
        "rect": list(response.rect),
        "token_id": response.token.token_id,
        "version": response.token.version,
        "region_status": response.region_status,
        "retained": response.retained,
        "pixel_status": np.where(
            response.pixel_status == COMMITTED, "committed", response.pixel_status
        ).tolist(),
        "committed_pixels": int(np.sum(response.pixel_status == COMMITTED)),
        "waiting_pixels": int(np.sum(response.pixel_status == "waiting")),
        "conflict_pixels": int(np.sum(response.pixel_status == "conflict")),
        "forward_trainable_pixels": int(trainable.sum()),
        "backward_ok_worst_pixels": int(backward_ok.sum()),
        "max_forward_bound": float(np.nanmax(response.forward_bound)),
        "max_backward_bound": float(np.nanmax(response.backward_bound_worst)),
        "color": np.where(np.isnan(response.color), None, response.color).tolist(),
        "transmittance": np.where(np.isnan(response.transmittance), None, response.transmittance).tolist(),
        "budget": response.budget,
    }


def _summarize_backward(response: BackwardResponse) -> dict[str, Any]:
    return {
        "token_id": response.token_id,
        "version": response.version,
        "accepted_pixels": response.accepted_pixels,
        "rejected_pixels": response.rejected_pixels,
        "pixel_reason": response.pixel_reason,
        "master_rgb_grad": {
            master: np.round(grad, 9).tolist()
            for master, grad in response.master_rgb_grad.items()
            if np.any(grad != 0.0)
        },
        "master_alpha_grad": {
            master: round(grad, 9)
            for master, grad in response.master_alpha_grad.items()
            if grad != 0.0
        },
        "budget": response.budget,
    }
