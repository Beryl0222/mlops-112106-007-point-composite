"""临时区预算记账。

``budget_bytes`` 计入补齐候选及执行、梯度归并过程中实际暂存；
外部区域输入（场景、瓦片报送）和调用者收到的最终输出不计。

记账按 NumPy 数组的真实字节数累计（``nbytes`` 含 dtype 宽度），
Python 容器与标量按固定保守估计计入，保证不把整幅图的点-像素
相交矩阵塞进临时区。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


class BudgetExceeded(RuntimeError):
    """临时区预算不足，当前区域只能部分交付，需要继续处理。"""


# Python 端每个候选条目（元组/字典/列表节点）的保守字节估计。
_PY_OBJECT_BYTES = 64


class BudgetLedger:
    def __init__(self, limit_bytes: int) -> None:
        self.limit_bytes = int(limit_bytes)
        self.peak_bytes = 0
        self._live = 0
        self._allocations: dict[int, int] = {}

    @property
    def live_bytes(self) -> int:
        return self._live

    def check(self, extra: int) -> None:
        """预判再申请 ``extra`` 字节是否越界，越界即抛 :class:`BudgetExceeded`。"""
        if self._live + int(extra) > self.limit_bytes:
            raise BudgetExceeded(
                f"临时区预算不足: 需要 {self._live + int(extra)} 字节，"
                f"上限 {self.limit_bytes} 字节"
            )

    def reserve(self, arrays: Iterable[np.ndarray], py_objects: int = 0) -> None:
        """登记一组 NumPy 暂存与若干 Python 临时对象。"""
        total = sum(int(a.nbytes) for a in arrays) + py_objects * _PY_OBJECT_BYTES
        self.check(total)
        for a in arrays:
            self._allocations[id(a)] = int(a.nbytes)
        self._live += total
        self.peak_bytes = max(self.peak_bytes, self._live)

    def release(self, *arrays: np.ndarray) -> None:
        released = 0
        for a in arrays:
            released += self._allocations.pop(id(a), 0)
        self._live -= released

    def release_py(self, py_objects: int) -> None:
        self._live -= py_objects * _PY_OBJECT_BYTES
        if self._live < 0:
            self._live = 0

    def snapshot(self) -> dict:
        return {
            "limit_bytes": self.limit_bytes,
            "live_bytes": self._live,
            "peak_bytes": self.peak_bytes,
        }
