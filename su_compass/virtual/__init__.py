"""
su_compass.virtual — 虚拟时间实验核心层。

提供统一的事件驱动虚拟时间框架，使 FedAvg / FedAsync / FedCompass
三种联邦学习算法共享同一套虚拟客户端画像与到达时间标准。

子模块：
    event           虚拟事件与最小堆事件队列
    client_runtime  真实训练 + 虚拟时间桥接
    trace           统一 trace / CSV / JSON 输出
    algorithms/     三算法虚拟时间调度控制器
"""

from .event import VirtualEvent, EventQueue, EventType  # noqa: F401


def __getattr__(name):
    # Keep lightweight scheduling/trace modules importable in environments
    # without the optional torch training runtime (notably policy unit tests).
    if name == "VirtualClientRuntime":
        from .client_runtime import VirtualClientRuntime
        return VirtualClientRuntime
    raise AttributeError(name)
