"""
su_compass.virtual.algorithms.base — 虚拟算法控制器抽象接口。

三算法（FedAvg / FedAsync / FedCompass）都实现本接口。
Runner 主循环通过本接口与算法交互，不感知具体调度语义。

调用流程：
    controller.initialize(...)
    while not controller.training_finished():
        dispatches = controller.next_dispatches()
        ...真实训练 + 虚拟事件入队...
        event = queue.pop()
        new_events = controller.on_client_upload(event, virtual_now)
        或 new_events = controller.on_timer_event(event, virtual_now)
        ...controller.pop_aggregation_traces() 等写出 trace...

这个接口刻意很薄：控制器不直接训练、不写文件、不操作事件队列。
它只维护算法状态，并把“下一批要训练谁”通过 Dispatch 交回 runner。
"""

import abc
from typing import Any, Dict, List, Optional

from su_compass.virtual.event import VirtualEvent


class Dispatch:
    """描述一次客户端派发任务。

    Runner 根据此对象执行真实训练 + 虚拟时间生成。
    可以把 Dispatch 看成 server 发给 client 的一张“训练工单”：
    包含模型快照、派发虚拟时间、本轮 Q，以及 trace 需要的版本信息。

    Attributes:
        client_id:           客户端编号。
        global_model:        本次训练使用的全局模型快照。必须 deepcopy，
                             否则后续聚合更新全局模型时会污染已派发任务。
        dispatch_time:       server 在虚拟时间轴上把任务发给客户端的时间。
                             VirtualRuntimeModel 会从这个时间开始累加下载/训练/上传耗时。
        local_steps:         本轮本地训练步数 Q。FedAvg/FedAsync 固定，FedCompass 动态分配。
        target_arrival_time: 目标 group 的期望到达时间（仅 FedCompass）。
                             用于模拟 report.early_margin，也帮助 trace 判断是否贴合 group。
        latest_arrival_time: 目标 group 的最晚到达时间（仅 FedCompass）。
                             CLIENT_UPLOAD.time > 该值时，FedCompass 认为客户端迟到。
        staleness:           派发时模型陈旧度（global_ts - client_ts）。
                             这是“开始训练时已经落后多少”，写入 round_reports。
        model_version_at_dispatch: 派发给客户端的全局模型版本号。
                             这是客户端本轮训练真正基于的版本，聚合 trace 会用它还原 staleness。
    """
    __slots__ = (
        "client_id", "global_model", "dispatch_time",
        "local_steps", "target_arrival_time", "latest_arrival_time",
        "staleness", "model_version_at_dispatch",
        "decision_id",
    )

    def __init__(
        self,
        client_id: str,
        global_model: Dict,
        dispatch_time: float,
        local_steps: int,
        target_arrival_time: Optional[float] = None,
        latest_arrival_time: Optional[float] = None,
        staleness: int = 0,
        model_version_at_dispatch: int = 0,
        decision_id: str = "",
    ) -> None:
        self.client_id = client_id
        self.global_model = global_model
        self.dispatch_time = dispatch_time
        self.local_steps = local_steps
        self.target_arrival_time = target_arrival_time   # FedCompass：目标 group 期望到达时间
        self.latest_arrival_time = latest_arrival_time   # FedCompass：迟到判据阈值
        self.staleness = staleness                       # 派发瞬间的版本差，不等同于聚合瞬间 staleness
        self.model_version_at_dispatch = model_version_at_dispatch  # 本地训练起点版本
        self.decision_id = decision_id


class VirtualAlgorithmController(abc.ABC):
    """虚拟算法控制器抽象基类。

    子类需实现以下方法：
        initialize          初始化控制器状态与首批 dispatch。
        on_client_upload    处理 CLIENT_UPLOAD 事件。
        on_timer_event      处理定时事件（如 FedCompass group deadline）。
        next_dispatches     返回等待派发的任务队列。
        training_finished   判断是否达到终止条件。
        get_global_timestamp 返回当前全局模型版本号。

    可选接口（trace 对齐，Runner 通过 hasattr 检测）：
        pop_aggregation_traces      弹出聚合 trace 缓冲
        pop_group_traces            弹出 FedCompass group trace
        pop_dispatch_decision_traces 弹出调度决策 trace
        pop_upload_result           弹出单次 upload 元数据
        get_current_global_model    返回当前全局模型 state_dict
        get_num_client_update_budget_used  FedCompass 客户端更新贡献计数
    """

    @abc.abstractmethod
    def initialize(
        self,
        client_ids: List[str],
        initial_global_model: Dict,
    ) -> None:
        """初始化控制器，设置首批 dispatch。

        Args:
            client_ids:          全部客户端编号列表。
            initial_global_model: 初始全局模型 state_dict。
        """

    @abc.abstractmethod
    def on_client_upload(
        self,
        event: VirtualEvent,
        virtual_now: float,
    ) -> List[VirtualEvent]:
        """处理 CLIENT_UPLOAD 事件，返回新产生的事件列表。

        Args:
            event:       CLIENT_UPLOAD 虚拟事件。
            virtual_now: 当前虚拟时间（= event.time）。

        Returns:
            新产生的虚拟事件列表（如 GROUP_DEADLINE），可为空。
        """

    @abc.abstractmethod
    def on_timer_event(
        self,
        event: VirtualEvent,
        virtual_now: float,
    ) -> List[VirtualEvent]:
        """处理定时事件（如 group deadline），返回新产生的事件列表。"""

    @abc.abstractmethod
    def next_dispatches(self) -> List[Dispatch]:
        """返回并清空待派发队列。

        Runner 会依次执行这些 dispatch 的真实训练和虚拟事件入队。
        """

    @abc.abstractmethod
    def training_finished(self) -> bool:
        """判断训练是否达到终止条件（如 num_global_epochs）。"""

    @abc.abstractmethod
    def get_global_timestamp(self) -> int:
        """返回当前全局模型版本号。"""

    @property
    @abc.abstractmethod
    def algorithm_name(self) -> str:
        """返回算法名（fedavg / fedasync / fedcompass）。"""
