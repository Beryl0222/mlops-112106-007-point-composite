"""点投影深度合成的可微 alpha 合成算子（CPU / NumPy）。

公共入口：

* :class:`~alpha_compose.session.CompositeSession` —— 有状态合成会话，
  按几何分片到达顺序维护可见性事实与数据版；
* :func:`~alpha_compose.session.CompositeSession.read_region` —— 区域前向读取，
  返回像素合成结果、剩余透射量、误差界与可训练资格；
* :func:`~alpha_compose.session.CompositeSession.backward` —— 按某次读取所依据的
  深度与覆盖事实归还颜色/不透明度梯度；
* :func:`~alpha_compose.timeline.play` / :func:`~alpha_compose.load.load_scene` ——
  接住交换资料并按时间线回放。
"""

from .load import Scene, SliceSpec, PointRecord, load_scene
from .session import CompositeSession, ReadResponse, BackwardResponse, StaleVersionError
from .budget import BudgetExceeded

__all__ = [
    "Scene",
    "SliceSpec",
    "PointRecord",
    "load_scene",
    "CompositeSession",
    "ReadResponse",
    "BackwardResponse",
    "StaleVersionError",
    "BudgetExceeded",
]
