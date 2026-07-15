"""
su_compass.virtual.algorithms — 三算法虚拟时间调度控制器。

提供 FedAvg / FedAsync / FedCompass 的虚拟时间版本调度语义。
每个控制器只负责"事件到达后怎么聚合"，真实训练与虚拟时间生成
由上层 VirtualClientRuntime 统一管理。
"""

from .base import VirtualAlgorithmController  # noqa: F401
from .fedavg import VirtualFedAvgController  # noqa: F401
from .fedasync import VirtualFedAsyncController  # noqa: F401
from .fedcompass import VirtualFedCompassController  # noqa: F401
from .state_compass import VirtualStateCompassController  # noqa: F401
from .oort_compass import VirtualOortCompassController  # noqa: F401
from .utility import OortConfig  # noqa: F401
from .rup_compass import VirtualRUPCompassController  # noqa: F401
