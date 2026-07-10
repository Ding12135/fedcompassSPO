"""Runtime-side client modeling utilities for SU-Compass."""  # 声明 runtime 子包用于端侧运行状态建模。

from .profile import ClientRuntimeProfile, build_default_profiles  # 导出端侧运行画像和默认场景生成函数。
from .state import ClientRoundReport, ClientRuntimeState, RuntimeStateTracker  # 导出每轮反馈、运行状态和状态追踪器。
from .virtual_runtime import VirtualRuntimeModel, VirtualRoundResult  # 导出虚拟运行时间模型和单轮模拟结果。

__all__ = [  # 定义该包对外暴露的公开接口列表。
    "ClientRuntimeProfile",  # 暴露客户端端侧运行画像。
    "build_default_profiles",  # 暴露默认端侧场景生成函数。
    "ClientRoundReport",  # 暴露客户端每轮反馈记录。
    "ClientRuntimeState",  # 暴露客户端运行状态快照。
    "RuntimeStateTracker",  # 暴露运行状态追踪器。
    "VirtualRuntimeModel",  # 暴露虚拟运行时间模型。
    "VirtualRoundResult",  # 暴露虚拟单轮结果。
]  # 结束公开接口列表。

