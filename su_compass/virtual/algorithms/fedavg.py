"""
su_compass.virtual.algorithms.fedavg — FedAvg 虚拟时间调度控制器。

语义：
    1. 每轮所有 8 个客户端从同一 global_model 出发训练。
    2. 等收齐本轮全部 CLIENT_UPLOAD 事件后，在 virtual_now = max(finish_time) 处聚合。
    3. 聚合使用 FedAvgAggregator.aggregate(local_models)。
    4. 聚合后下一轮从该虚拟时间再派发所有客户端。

读代码抓住一个状态即可：_pending_models。
它代表“本同步轮已经回来的客户端”。只有它收满 num_clients，FedAvg 才推进一次
global_timestamp，并从同一个新模型重新派发所有客户端。

预期：network_poor / availability_limited 会显著拉长每轮虚拟总时间。

复用：直接调用 appfl.aggregator.FedAvgAggregator，不改主代码。

Trace：同步聚合时写入 aggregation_trace（staleness 全为 0，全员同版本）。
"""

import copy
import json
from typing import Any, Dict, List

from .base import Dispatch, VirtualAlgorithmController
from su_compass.virtual.event import VirtualEvent


class VirtualFedAvgController(VirtualAlgorithmController):
    """FedAvg 虚拟时间调度控制器。

    Attributes:
        aggregator:           FedAvgAggregator 实例（直接复用 appfl 主代码）。
        num_clients:          客户端总数。
        local_steps:          每轮固定本地训练步数（FedAvg 无动态 Q 分配）。
        num_global_epochs:    目标全局聚合轮次。
        _global_timestamp:    当前全局模型版本号。每次同步聚合 +1。
        _num_updates:         已完成的全局聚合次数，用于 training_finished()。
        _current_global_model: 当前全局模型 state_dict。下一轮所有客户端都会拿这份模型。
        _pending_models:      本轮已收集但尚未聚合的客户端模型。收满 num_clients 才聚合。
        _pending_reports:     本轮已收集的 ClientRoundReport。当前主要保留给 trace/调试扩展。
        _dispatch_queue:      待派发任务队列。controller 只入队，runner 通过 next_dispatches() 取走。
        _virtual_now:         当前虚拟时间。FedAvg 聚合时间等于本轮最后一个上传的 finish_time。
        _client_ids:          固定客户端顺序。每轮聚合后按这个列表重新派发所有客户端。
        _aggregation_trace_buffer: 聚合 trace 临时缓冲。runner 处理完事件后 pop 并写 CSV。
    """

    def __init__(
        self,
        aggregator: Any,
        num_clients: int,
        local_steps: int,
        num_global_epochs: int,
    ) -> None:
        """初始化 FedAvg 虚拟控制器。

        Args:
            aggregator:       FedAvgAggregator 实例。
            num_clients:      客户端总数。
            local_steps:      每轮固定 local steps。
            num_global_epochs: 目标全局聚合轮次。
        """
        self.aggregator = aggregator
        self.num_clients = num_clients
        self.local_steps = local_steps
        self.num_global_epochs = num_global_epochs

        self._global_timestamp: int = 0   # 全局模型版本；同步聚合一次推进一个版本
        self._num_updates: int = 0        # 已完成同步轮数
        self._current_global_model: Dict = {}  # 当前 server 端模型快照
        self._pending_models: Dict[str, Dict] = {}   # 本轮已到达的 local_model，收满才聚合
        self._pending_reports: Dict[str, Any] = {}    # 本轮已到达的 report，仅用于 trace/调试扩展
        self._dispatch_queue: List[Dispatch] = []      # 等待 runner 执行真实训练的工单
        self._virtual_now: float = 0.0                 # 当前处理到的虚拟时间
        self._client_ids: List[str] = []               # 所有客户端，聚合后全量重派
        self._aggregation_trace_buffer: List[Dict] = []  # 同步聚合 trace，供 Runner pop

    # ──────────── 接口实现 ────────────

    def initialize(self, client_ids: List[str], initial_global_model: Dict) -> None:
        """初始化并为所有客户端创建首轮 dispatch。"""
        self._client_ids = list(client_ids)
        self._current_global_model = copy.deepcopy(initial_global_model)
        self._virtual_now = 0.0

        # 首轮：为所有客户端派发同一份初始模型。FedAvg 没有 staleness，
        # 因为同步轮内所有客户端都基于同一个 global_timestamp。
        for cid in self._client_ids:
            self._dispatch_queue.append(Dispatch(
                client_id=cid,
                global_model=copy.deepcopy(self._current_global_model),
                dispatch_time=0.0,
                local_steps=self.local_steps,
                staleness=0,
                model_version_at_dispatch=self._global_timestamp,
            ))

    def on_client_upload(self, event: VirtualEvent, virtual_now: float) -> List[VirtualEvent]:
        """收集一个客户端上传；等齐所有客户端后聚合。

        FedAvg 同步语义：收到全部 num_clients 个上传后，
        在 virtual_now = max(finish_time) 时聚合。
        """
        self._virtual_now = virtual_now
        client_id = event.client_id
        self._pending_models[client_id] = event.payload["local_model"]
        self._pending_reports[client_id] = event.payload["report"]

        # 未收齐，等待后续 CLIENT_UPLOAD；虚拟时间由事件队列继续往前推进。
        if len(self._pending_models) < self.num_clients:
            return []

        # ──── 已收齐全部客户端，执行同步聚合 ────
        ts_before = self._global_timestamp
        global_model = self.aggregator.aggregate(self._pending_models)
        self._current_global_model = copy.deepcopy(global_model)
        self._global_timestamp += 1
        self._num_updates += 1

        # 同步 FedAvg：所有客户端基于同一 global_timestamp，staleness = 0
        participating = list(self._pending_models.keys())
        staleness = {cid: 0 for cid in participating}
        local_steps_map = {cid: self.local_steps for cid in participating}
        model_versions = {cid: ts_before for cid in participating}
        self._aggregation_trace_buffer.append({
            "global_timestamp_before": ts_before,
            "global_timestamp_after": self._global_timestamp,
            "virtual_time": virtual_now,
            "trigger": "fedavg_sync",
            "participating_clients": ",".join(sorted(participating)),
            "per_client_staleness": json.dumps(staleness),
            "per_client_local_steps": json.dumps(local_steps_map),
            "per_client_model_version": json.dumps(model_versions),
            "group_id": -1,
            "num_clients": len(participating),
            "client_update_budget_delta": len(participating),
        })

        # 下一轮必须从聚合发生的 virtual_now 同时派发，而不是从各自上传时间派发；
        # 这正是同步 FedAvg 的“慢客户端决定轮长”。
        for cid in self._client_ids:
            self._dispatch_queue.append(Dispatch(
                client_id=cid,
                global_model=copy.deepcopy(self._current_global_model),
                dispatch_time=virtual_now,
                local_steps=self.local_steps,
                staleness=0,
                model_version_at_dispatch=self._global_timestamp,
            ))

        # 清空本轮缓冲
        self._pending_models.clear()
        self._pending_reports.clear()

        return []  # FedAvg 不产生额外事件

    def on_timer_event(self, event: VirtualEvent, virtual_now: float) -> List[VirtualEvent]:
        """FedAvg 无定时事件，直接返回空列表。"""
        return []

    def next_dispatches(self) -> List[Dispatch]:
        """返回并清空待派发队列。"""
        dispatches = list(self._dispatch_queue)
        self._dispatch_queue.clear()
        return dispatches

    def training_finished(self) -> bool:
        """判断是否达到目标聚合轮次。"""
        return self._num_updates >= self.num_global_epochs

    def get_global_timestamp(self) -> int:
        return self._global_timestamp

    def pop_aggregation_traces(self) -> List[Dict]:
        """弹出并清空 aggregation trace 缓冲。"""
        traces = list(self._aggregation_trace_buffer)
        self._aggregation_trace_buffer.clear()
        return traces

    def get_current_global_model(self) -> Dict:
        """返回当前全局模型 state_dict，供 global_eval 使用。"""
        return self._current_global_model

    @property
    def algorithm_name(self) -> str:
        return "fedavg"
