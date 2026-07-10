"""
su_compass.virtual.event — 虚拟事件定义与最小堆事件队列。

所有虚拟时间实验的调度事件都通过本模块管理。
事件按 virtual_time 排序，保证 server 处理顺序由虚拟完成时间决定，
而非真实 GPU 训练完成顺序。

事件队列是整个 virtual 包的“时钟”：runner 不主动推进时间，
而是不断 pop 最早事件，并把 event.time 当成新的 virtual_now。

同一虚拟时间下，CLIENT_UPLOAD 优先于 FEDCOMPASS_GROUP_DEADLINE。
这与原始 FedCompass 的迟到判断 `curr_time > latest_arrival_time` 对齐：
客户端正好在 deadline 时刻到达时，不应被判为迟到。
"""

import heapq
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ──────────────────────────── 事件类型枚举 ────────────────────────────

class EventType(Enum):
    """虚拟事件类型。

    CLIENT_UPLOAD               客户端虚拟完成训练并上传本地模型。
    FEDCOMPASS_GROUP_DEADLINE    FedCompass arrival group 最晚到达时间触发。
    """
    CLIENT_UPLOAD = auto()
    FEDCOMPASS_GROUP_DEADLINE = auto()


# ──────────────────────────── 虚拟事件 ────────────────────────────────

@dataclass(order=True)
class VirtualEvent:
    """单个虚拟事件。

    事件按 time 排序（dataclass order=True），time 相同时按 _priority 排序，
    再按插入序号 _seq 保证 FIFO 稳定性，payload 不参与比较。

    Attributes:
        time:       事件发生的虚拟时间戳（秒）。runner pop 事件后通常令 virtual_now = time。
        _priority:  同一时间下的事件优先级，CLIENT_UPLOAD 优先于 deadline。
                    这是 FedCompass 边界条件的一部分，不是普通排序细节。
        _seq:       插入序号，由 EventQueue 自动填充，用于同时间稳定排序。
                    两个客户端同一秒上传时，按入队顺序处理，避免堆比较 payload。
        event_type: 事件类型枚举，决定 runner 调 controller.on_client_upload 还是 on_timer_event。
        client_id:  关联的客户端编号。CLIENT_UPLOAD 必填；GROUP_DEADLINE 通常为 None。
        payload:    事件携带的附加数据：
                    CLIENT_UPLOAD 含 local_model/report/runtime_state/profile_type；
                    GROUP_DEADLINE 含 group_idx。
    """
    time: float
    _priority: int = field(default=0, init=False, compare=True)
    _seq: int = field(default=0, compare=True)
    event_type: EventType = field(default=EventType.CLIENT_UPLOAD, compare=False)
    client_id: Optional[str] = field(default=None, compare=False)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)


# ──────────────────────────── 事件队列 ────────────────────────────────

class EventQueue:
    """基于最小堆的虚拟事件队列。

    保证每次 pop 返回 virtual_time 最小的事件。
    同时间先处理 CLIENT_UPLOAD，再处理 FEDCOMPASS_GROUP_DEADLINE；
    同类型事件按插入顺序 FIFO。
    """

    def __init__(self) -> None:
        self._heap: List[VirtualEvent] = []  # 最小堆存储
        self._counter: int = 0               # 自增序号，保证同时间 FIFO

    def push(self, event: VirtualEvent) -> None:
        """将事件入队，自动分配序号。"""
        # priority 与 seq 都在入队时写入，避免调用方构造事件时需要理解堆排序细节。
        event._priority = _event_priority(event.event_type)
        event._seq = self._counter  # 写入全局自增序号
        self._counter += 1
        heapq.heappush(self._heap, event)

    def pop(self) -> VirtualEvent:
        """弹出 virtual_time 最小的事件。

        Raises:
            IndexError: 队列为空时抛出。
        """
        if not self._heap:
            raise IndexError("pop from empty EventQueue")
        return heapq.heappop(self._heap)

    def peek(self) -> VirtualEvent:
        """查看（不弹出）队头事件。"""
        if not self._heap:
            raise IndexError("peek at empty EventQueue")
        return self._heap[0]

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return len(self._heap) > 0


def _event_priority(event_type: EventType) -> int:
    """返回同一虚拟时间下的事件优先级。

    FedCompass 原始代码中，客户端到达是否迟到使用 `curr_time > latest_arrival_time`。
    因此当 CLIENT_UPLOAD.time == GROUP_DEADLINE.time 时，应先处理上传事件，
    让该客户端仍然走“按时到达”分支。
    """
    if event_type == EventType.CLIENT_UPLOAD:
        return 0
    if event_type == EventType.FEDCOMPASS_GROUP_DEADLINE:
        return 1
    return 10
