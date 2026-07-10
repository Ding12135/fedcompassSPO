"""
su_compass.virtual.algorithms.fedasync — FedAsync 虚拟时间调度控制器。

语义：
    1. 事件队列按 virtual_finish_time 排序处理 CLIENT_UPLOAD。
    2. 每到一个客户端，立即调用 FedAsyncAggregator.aggregate(client_id, local_model)。
    3. 该客户端马上拿到新 global_model，进入下一轮训练。

读代码抓住一个状态即可：_client_timestamp。
它记录每个客户端训练所基于的全局版本。上传到达时，
global_timestamp - _client_timestamp[client_id] 就是本次聚合真实 staleness。

预期：快客户端更新次数更多，慢客户端 staleness 更高。

复用：直接调用 appfl.aggregator.FedAsyncAggregator，不改主代码。

Trace：每次即到即聚合写入 aggregation_trace，并 pop_upload_result 供 scheduler_trace 使用。
"""

import copy
import json
from typing import Any, Dict, List

from .base import Dispatch, VirtualAlgorithmController
from su_compass.virtual.event import VirtualEvent


class VirtualFedAsyncController(VirtualAlgorithmController):
    """FedAsync 虚拟时间调度控制器。

    Attributes:
        aggregator:           FedAsyncAggregator 实例（直接复用 appfl 主代码）。
        num_clients:          客户端总数。
        local_steps:          每轮固定本地训练步数。
        num_global_epochs:    目标全局聚合次数（每个客户端到达算一次）。
        _global_timestamp:    当前全局模型版本号。每个 CLIENT_UPLOAD 到达并聚合后 +1。
        _num_updates:         已完成的全局聚合次数，用于 training_finished()。
        _current_global_model: 当前全局模型 state_dict。上传者下一轮会立即拿到它。
        _client_timestamp:    每客户端当前训练所基于的模型版本号，用于计算 aggregation staleness。
        _dispatch_queue:      待派发任务队列。FedAsync 每处理一个上传，只给该客户端追加下一轮。
        _virtual_now:         当前虚拟时间，等于正在处理的上传 finish_time。
        _last_upload_results: 单次上传后的 trace 补充信息。runner pop 后写 scheduler_trace/round_reports。
    """

    def __init__(
        self,
        aggregator: Any,
        num_clients: int,
        local_steps: int,
        num_global_epochs: int,
    ) -> None:
        """初始化 FedAsync 虚拟控制器。

        Args:
            aggregator:        FedAsyncAggregator 实例。
            num_clients:       客户端总数。
            local_steps:       每轮固定 local steps。
            num_global_epochs: 目标全局聚合次数。
        """
        self.aggregator = aggregator
        self.num_clients = num_clients
        self.local_steps = local_steps
        self.num_global_epochs = num_global_epochs

        self._global_timestamp: int = 0   # server 端全局版本号
        self._num_updates: int = 0        # 已处理的异步上传数
        self._current_global_model: Dict = {}  # 最新全局模型
        self._client_timestamp: Dict[str, int] = {}  # 每客户端当前本地模型基于的全局版本
        self._dispatch_queue: List[Dispatch] = []     # 下一批要训练的客户端工单
        self._virtual_now: float = 0.0                # 当前事件时间
        self._client_ids: List[str] = []              # 初始化首轮全量派发时使用
        self._aggregation_trace_buffer: List[Dict] = []  # 即到即聚合 trace
        self._last_upload_results: Dict[str, Dict] = {}  # upload 元数据，供 Runner pop

    # ──────────── 接口实现 ────────────

    def initialize(self, client_ids: List[str], initial_global_model: Dict) -> None:
        """初始化并为所有客户端创建首轮 dispatch。"""
        self._client_ids = list(client_ids)
        self._current_global_model = copy.deepcopy(initial_global_model)
        self._virtual_now = 0.0

        # 首轮：所有客户端都基于版本 0；之后只有上传完成的客户端会立即拿到新版本。
        for cid in self._client_ids:
            self._client_timestamp[cid] = 0
            self._dispatch_queue.append(Dispatch(
                client_id=cid,
                global_model=copy.deepcopy(self._current_global_model),
                dispatch_time=0.0,
                local_steps=self.local_steps,
                staleness=0,
                model_version_at_dispatch=self._global_timestamp,
            ))

    def on_client_upload(self, event: VirtualEvent, virtual_now: float) -> List[VirtualEvent]:
        """处理单客户端上传：即到即聚合，立即派发下一轮。

        FedAsync 异步语义：每到一个客户端立即聚合并更新全局模型，
        然后该客户端拿到最新 global_model 进入下一轮训练。
        """
        self._virtual_now = virtual_now
        client_id = event.client_id
        local_model = event.payload["local_model"]
        report = event.payload["report"]

        # 聚合前计算真实 staleness（与 FedAsyncAggregator 内部逻辑一致）
        agg_staleness = self._global_timestamp - self._client_timestamp.get(client_id, 0)
        model_version_at_upload = self._client_timestamp.get(client_id, 0)
        ts_before = self._global_timestamp

        # FedAsync 的核心：一个上传就是一次全局更新，不等待其它客户端。
        global_model = self.aggregator.aggregate(client_id, local_model)
        self._current_global_model = copy.deepcopy(global_model)
        self._global_timestamp += 1
        self._num_updates += 1

        # 写入 aggregation trace
        self._aggregation_trace_buffer.append({
            "global_timestamp_before": ts_before,
            "global_timestamp_after": self._global_timestamp,
            "virtual_time": virtual_now,
            "trigger": "fedasync_single",
            "participating_clients": client_id,
            "per_client_staleness": json.dumps({client_id: agg_staleness}),
            "per_client_local_steps": json.dumps({client_id: self.local_steps}),
            "per_client_model_version": json.dumps({client_id: model_version_at_upload}),
            "group_id": -1,
            "num_clients": 1,
            "client_update_budget_delta": 1,
        })

        # 聚合完成后，只有当前客户端拿到了最新全局模型；其它正在训练的客户端
        # 仍保留旧 timestamp，等它们上传时自然体现为 staleness。
        self._client_timestamp[client_id] = self._global_timestamp

        # 当前客户端立即进入下一轮训练，dispatch_time 就是本次上传的 virtual_now。
        self._dispatch_queue.append(Dispatch(
            client_id=client_id,
            global_model=copy.deepcopy(self._current_global_model),
            dispatch_time=virtual_now,
            local_steps=self.local_steps,
            staleness=0,
            model_version_at_dispatch=self._global_timestamp,
        ))

        speed_raw = report.round_time / max(report.local_steps, 1)
        # FedAsync 无 speed_momentum，smoothed 与 raw 相同
        self._last_upload_results[client_id] = {
            "upload_group_id": -1,
            "next_group_id": -1,
            "aggregation_staleness": agg_staleness,
            "model_version_at_upload": model_version_at_upload,
            "speed_raw": speed_raw,
            "speed_smoothed": speed_raw,
            "dispatch_staleness": report.staleness,
        }

        return []

    def on_timer_event(self, event: VirtualEvent, virtual_now: float) -> List[VirtualEvent]:
        """FedAsync 无定时事件。"""
        return []

    def next_dispatches(self) -> List[Dispatch]:
        """返回并清空待派发队列。"""
        dispatches = list(self._dispatch_queue)
        self._dispatch_queue.clear()
        return dispatches

    def training_finished(self) -> bool:
        """判断是否达到目标全局聚合次数。"""
        return self._num_updates >= self.num_global_epochs

    def get_global_timestamp(self) -> int:
        return self._global_timestamp

    def get_client_staleness(self, client_id: str) -> int:
        """返回指定客户端当前的模型陈旧度。"""
        return self._global_timestamp - self._client_timestamp.get(client_id, 0)

    def pop_aggregation_traces(self) -> List[Dict]:
        """弹出并清空 aggregation trace 缓冲。"""
        traces = list(self._aggregation_trace_buffer)
        self._aggregation_trace_buffer.clear()
        return traces

    def pop_upload_result(self, client_id: str):
        """取出并清除指定客户端最近一次 upload 元数据。"""
        return self._last_upload_results.pop(client_id, None)

    def get_current_global_model(self) -> Dict:
        """返回当前全局模型 state_dict，供 global_eval 使用。"""
        return self._current_global_model

    @property
    def algorithm_name(self) -> str:
        return "fedasync"
