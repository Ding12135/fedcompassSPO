"""
su_compass.virtual.algorithms.fedcompass — FedCompass 虚拟时间调度控制器。

从 src/appfl/scheduler/compass_scheduler.py 复刻核心调度逻辑，
将 time.time() 替换为 virtual_now，将 threading.Timer 替换为事件队列。
聚合仍复用 appfl.aggregator.FedCompassAggregator。

对应关系：
    原始 CompassScheduler               本虚拟控制器
    ─────────────────────────────────    ─────────────────────────
    time.time() - self.start_time       self._virtual_now
    threading.Timer(delay, callback)    VirtualEvent(GROUP_DEADLINE)
    client_update_time / local_steps    report.round_time / local_steps
    threading.Lock                      不需要（单线程事件循环）

读代码时可以把本文件理解成一个「单线程虚拟版 CompassScheduler」：
    - 外层 runner 只负责从事件队列取 CLIENT_UPLOAD / GROUP_DEADLINE。
    - 本控制器只负责回答两个问题：这次上传要不要聚合？客户端下一轮进哪个 group、做多少 Q？
    - 所有真实训练已经在 client_runtime.py 完成，这里看到的 local_model 都是训练结果。

语义保留：
    - arrival group 半异步分组
    - speed_momentum 指数平滑速度估计
    - Q ∈ [min_local_steps, max_local_steps]
    - late update 处理
    - general_buffer + group_buffer 合并聚合

Trace 输出（供 Runner 通过 pop_* 接口写出）：
    - aggregation_trace：每次聚合的真实 per-client staleness
    - dispatch_decision_trace：join/create/first_group 决策及 runtime 快照
    - group_trace：arrival group 完整生命周期
    - upload_result：单次 CLIENT_UPLOAD 的 upload_group / aggregation_staleness 等
"""

import copy
import json
import math
from typing import Any, Dict, List, Optional, Union

from .base import Dispatch, VirtualAlgorithmController
from su_compass.virtual.event import VirtualEvent, EventType


class VirtualFedCompassController(VirtualAlgorithmController):
    """FedCompass 虚拟时间调度控制器。

    Attributes:
        aggregator:         FedCompassAggregator 实例（复用 appfl 主代码）。
        num_clients:        客户端总数。
        min_local_steps:    最小本地训练步数 Qmin。
        max_local_steps:    最大本地训练步数 Qmax。
        speed_momentum:     速度估计动量系数。
        latest_time_factor: 最晚到达时间放大系数。
        num_global_epochs:  目标全局聚合次数。
        client_info:        每客户端调度账本，字段包括：
                            timestamp：该客户端当前本地模型基于的全局版本；
                            speed：平滑后的单 step 虚拟耗时，Q 分配的核心依据；
                            local_steps：下一轮计划执行的 Q；
                            goa：group of arrival，当前被安排到的 arrival group，-1 表示无组；
                            start_time：本轮被派发的虚拟时间。
        arrival_group:      活跃 arrival group 表，key 为 group_id，value 保存：
                            clients：仍未按时到达、还在等待的客户端；
                            arrived_clients：已按时到达并进入 group_buffer 的客户端；
                            expected_arrival_time：该组理想到齐/聚合时间；
                            latest_arrival_time：超过该时间视为迟到；
                            created_time：创建该组的虚拟时间。
        group_buffer:       按时到达的 group 内模型缓冲，等组聚合时消费。
        general_buffer:     迟到模型缓冲；迟到更新不丢弃，会并入下一次 group 聚合。
        global_timestamp:   server 端全局模型版本号，每次实际聚合 +1。
        _num_global_epochs: 已消耗的客户端更新贡献数，而不是全局版本数。
    """

    def __init__(
        self,
        aggregator: Any,
        num_clients: int,
        min_local_steps: int,
        max_local_steps: int,
        speed_momentum: float,
        latest_time_factor: float,
        num_global_epochs: int,
    ) -> None:
        self.aggregator = aggregator
        self.num_clients = num_clients
        self.min_local_steps = min_local_steps
        self.max_local_steps = max_local_steps
        self.speed_momentum = speed_momentum
        self.latest_time_factor = latest_time_factor
        self.num_global_epochs = num_global_epochs

        # ──── 调度状态（对齐原始 CompassScheduler.__init__）────
        # client_info 是 FedCompass 的“客户端账本”：记录客户端手里模型版本、
        # 平滑速度、下一轮 Q、以及当前被安排到的 group(goa)。
        self.client_info: Dict[str, Dict[str, Any]] = {}   # client_id -> 调度账本，见类注释
        self.group_counter: int = 0                        # group 自增编号，创建 group 后立即 +1
        self.arrival_group: Dict[int, Dict[str, Any]] = {} # 活跃 arrival group：还在等谁、deadline 是多少
        self.group_buffer: Dict[int, Dict[str, Any]] = {}  # group 内已按时到达、等待一起聚合的模型
        self.general_buffer: Dict[str, Dict] = {           # 迟到客户端模型缓冲
            "local_models": {},
            "local_steps": {},
            "timestamp": {},
        }
        self.global_timestamp: int = 0                     # 全局模型版本号，只按真实聚合次数递增
        self._num_global_epochs: int = 0                   # 客户端更新贡献预算，单更 +1，组更 +len(local_models)

        # ──── 虚拟时间特有状态 ────
        self._virtual_now: float = 0.0                     # 当前处理到的事件时间
        self._current_global_model: Dict = {}              # 最新全局模型，派发时会 deepcopy
        self._dispatch_queue: List[Dispatch] = []          # 等待 runner 执行的训练工单
        self._pending_results: Dict[str, Dict] = {}         # 客户端已到达但聚合未发生，用于 group 聚合后清理
        self._deadline_events: set = set()                   # 已入队的 group deadline，避免重复 timer 事件
        self._group_trace_buffer: List[Dict] = []            # group trace 记录
        self._aggregation_trace_buffer: List[Dict] = []      # 每次全局聚合的 trace
        self._dispatch_decision_trace_buffer: List[Dict] = []  # Q/group 调度决策 trace
        self._group_late_clients: Dict[int, List[str]] = {}  # group_id -> 迟到客户端列表，仅用于 group_trace 还原生命周期
        self._last_upload_results: Dict[str, Dict[str, Any]] = {}  # 供 Runner pop 的 upload 元数据
        self._pending_dispatch_after_upload: Dict[str, Dict[str, Any]] = {}  # 组聚合会一次重派多人，先暂存每人的 next_group
        self._current_runtime_state: Any = None              # 当前 upload 的 RuntimeStateTracker 快照，只用于 trace 决策上下文
        self._client_ids: List[str] = []

    # ═══════════════════════ 接口实现 ═══════════════════════

    def initialize(self, client_ids: List[str], initial_global_model: Dict) -> None:
        """初始化并为所有客户端创建首轮 dispatch。"""
        self._client_ids = list(client_ids)
        self._current_global_model = copy.deepcopy(initial_global_model)
        self._virtual_now = 0.0

        # 首轮尚无速度反馈，先给所有客户端最大 Q；首轮回传后才有 speed_momentum 可用。
        for cid in self._client_ids:
            self._dispatch_queue.append(Dispatch(
                client_id=cid,
                global_model=copy.deepcopy(self._current_global_model),
                dispatch_time=0.0,
                local_steps=self.max_local_steps,
                staleness=0,
                model_version_at_dispatch=self.global_timestamp,
            ))

    def on_client_upload(self, event: VirtualEvent, virtual_now: float) -> List[VirtualEvent]:
        """处理 CLIENT_UPLOAD 事件。

        对齐原始 CompassScheduler.schedule() 流程：
            1. _record_info：更新速度估计。
            2. 判断客户端属于哪个 group。
            3. 若无 group（首次到达）：_single_update + _assign_group。
            4. 若有 group：_group_update。
        """
        self._virtual_now = virtual_now
        client_id = event.client_id
        local_model = event.payload["local_model"]
        report = event.payload["report"]
        new_events: List[VirtualEvent] = []

        # ──── 在 _record_info 之前快照：upload 所属 group 与训练起始全局版本 ────
        # _record_info / 聚合逻辑会修改 client_info，因此 trace 需要的“上传前状态”
        # 必须在任何状态变更前取出来。
        upload_group_id = self.client_info.get(client_id, {}).get("goa", -1)
        model_version_at_upload = self.client_info.get(client_id, {}).get("timestamp", 0)
        dispatch_staleness = report.staleness  # 来自 Dispatch，即派发时陈旧度
        self._current_runtime_state = event.payload.get("runtime_state")  # 供决策 trace 使用

        # ──── 步骤 1：更新速度估计（对齐 _record_info）────
        self._record_info(client_id, report)
        speed_raw = report.round_time / max(report.local_steps, 1)
        speed_smoothed = self.client_info[client_id]["speed"]  # 平滑后，调度器实际使用的 speed
        aggregation_staleness: Optional[int] = None
        next_group_id = -1

        # ──── 步骤 2：判断 group 归属 ────
        arrival_group_idx = (
            self.client_info[client_id].get("goa", -1)
        )

        if arrival_group_idx == -1:
            # 首次到达或无组：单客户端聚合 + 立即派发下一轮
            aggregation_staleness = self._single_update(client_id, local_model, buffer=False)
            new_events.extend(self._assign_group(client_id))

            local_steps_assigned = self.client_info[client_id]["local_steps"]
            next_group_id = self.client_info[client_id].get("goa", -1)
            target_arrival = None
            latest_arrival = None
            if next_group_id >= 0 and next_group_id in self.arrival_group:
                target_arrival = self.arrival_group[next_group_id]["expected_arrival_time"]
                latest_arrival = self.arrival_group[next_group_id]["latest_arrival_time"]

            self._append_dispatch(
                client_id=client_id,
                virtual_now=virtual_now,
                local_steps_assigned=local_steps_assigned,
                target_arrival=target_arrival,
                latest_arrival=latest_arrival,
            )
        else:
            # 有目标 group：尝试组内更新
            upload_events, agg_staleness = self._group_update(
                client_id, local_model, arrival_group_idx,
            )
            new_events.extend(upload_events)
            if agg_staleness is not None:
                aggregation_staleness = agg_staleness

            # 如果这次上传触发了 group 聚合，_group_aggregation 已经为参与者安排了
            # 下一轮 dispatch；这里从暂存区取出对应 next_group 写 trace。
            if client_id in self._pending_dispatch_after_upload:
                pending = self._pending_dispatch_after_upload.pop(client_id)
                next_group_id = pending["next_group_id"]
            else:
                next_group_id = self.client_info[client_id].get("goa", -1)

        # Runner 处理完事件后会 pop 这份结果，用同一份事实同时补
        # scheduler_trace 和该客户端 round_reports 的 upload 字段。
        self._last_upload_results[client_id] = {
            "upload_group_id": upload_group_id,
            "next_group_id": next_group_id,
            "aggregation_staleness": aggregation_staleness,
            "model_version_at_upload": model_version_at_upload,
            "speed_raw": speed_raw,
            "speed_smoothed": speed_smoothed,
            "dispatch_staleness": dispatch_staleness,
        }  # Runner 通过 pop_upload_result() 取出并写入 scheduler_trace / round_reports

        return new_events

    def on_timer_event(self, event: VirtualEvent, virtual_now: float) -> List[VirtualEvent]:
        """处理 GROUP_DEADLINE 事件。

        对齐原始 CompassScheduler 中 threading.Timer 触发的 _group_aggregation。
        当 group deadline 到达时，强制对已到达的客户端执行聚合。
        """
        self._virtual_now = virtual_now
        group_idx = event.payload.get("group_idx")
        if group_idx is not None:
            # deadline 事件可能在 group 已经 all_arrived 聚合后才从队列中弹出；
            # _group_aggregation 内部会检查 group/buffer 是否仍存在，空事件自然忽略。
            self._deadline_events.discard(group_idx)
            events, _ = self._group_aggregation(group_idx, trigger="deadline")
            return events
        return []

    def next_dispatches(self) -> List[Dispatch]:
        """返回并清空待派发队列。"""
        dispatches = list(self._dispatch_queue)
        self._dispatch_queue.clear()
        return dispatches

    def training_finished(self) -> bool:
        return self._num_global_epochs >= self.num_global_epochs

    def get_global_timestamp(self) -> int:
        return self.global_timestamp

    def get_num_client_update_budget_used(self) -> int:
        """返回累计客户端更新贡献数。

        该值对齐原始 CompassScheduler.get_num_global_epochs() 的口径：
            - 单客户端更新贡献 +1
            - group 聚合贡献 +len(local_models)

        名字显式写成 client update budget，避免和 global_timestamp
        （真实全局模型版本数）混淆。
        """
        return self._num_global_epochs

    def pop_group_traces(self) -> List[Dict]:
        """弹出并清空 group trace 缓冲，供 Runner 写出。"""
        traces = list(self._group_trace_buffer)
        self._group_trace_buffer.clear()
        return traces

    def pop_aggregation_traces(self) -> List[Dict]:
        """弹出并清空 aggregation trace 缓冲。

        Runner 在每次事件处理结束后调用，与 training_metrics / global_eval 同步写出。
        """
        traces = list(self._aggregation_trace_buffer)
        self._aggregation_trace_buffer.clear()
        return traces

    def pop_dispatch_decision_traces(self) -> List[Dict]:
        """弹出并清空 dispatch decision trace 缓冲。"""
        traces = list(self._dispatch_decision_trace_buffer)
        self._dispatch_decision_trace_buffer.clear()
        return traces

    def pop_upload_result(self, client_id: str) -> Optional[Dict[str, Any]]:
        """取出并清除指定客户端最近一次 upload 元数据。"""
        return self._last_upload_results.pop(client_id, None)

    def get_current_global_model(self) -> Dict:
        """返回当前全局模型 state_dict。"""
        return self._current_global_model

    @property
    def algorithm_name(self) -> str:
        return "fedcompass"

    # ═══════════════════════ 内部方法（对齐 compass_scheduler.py）═══════════════════════

    def _record_info(self, client_id: str, report) -> None:
        """更新客户端速度估计。

        对齐原始 _record_info()，把 time.time()-start_time 替换为
        report.round_time / local_steps 直接获取虚拟速度。
        """
        # 从 report 直接计算虚拟速度
        client_steps = (
            self.client_info[client_id]["local_steps"]
            if client_id in self.client_info
            else self.max_local_steps
        )
        client_speed = report.round_time / max(client_steps, 1)

        if client_id not in self.client_info:
            # 首次上报：初始化 client_info
            self.client_info[client_id] = {
                "timestamp": 0,                 # 客户端本地模型基于的全局版本
                "speed": client_speed,          # 平滑前的首个单 step 虚拟耗时
                "local_steps": self.max_local_steps,  # 下一轮默认 Q，后续由 group 分配覆盖
            }
        else:
            # 指数平滑速度估计（对齐原始 speed_momentum 逻辑）
            self.client_info[client_id]["speed"] = (
                (1 - self.speed_momentum) * self.client_info[client_id]["speed"]
                + self.speed_momentum * client_speed
            )

    def _single_update(
        self, client_id: str, local_model: Dict, buffer: bool = True,
    ) -> Optional[int]:
        """单客户端聚合。返回该客户端的 aggregation_staleness（立即聚合时）。"""
        agg_staleness: Optional[int] = None
        if not buffer:
            # buffer=False 是“真正立即聚合”：首次到达或无组客户端直接贡献一次全局更新。
            agg_staleness = self.global_timestamp - self.client_info[client_id]["timestamp"]
            global_model = self.aggregator.aggregate(
                client_id,
                local_model,
                staleness=agg_staleness,
                local_steps=self.client_info[client_id]["local_steps"],
            )
            self._current_global_model = copy.deepcopy(global_model)
            ts_before = self.global_timestamp
            self.global_timestamp += 1
            self._record_aggregation_trace(
                trigger="single",
                participating={client_id: agg_staleness},
                local_steps={client_id: self.client_info[client_id]["local_steps"]},
                model_versions={client_id: self.client_info[client_id]["timestamp"]},
                group_id=-1,
                budget_delta=1,
                ts_before=ts_before,
            )
        else:
            # buffer=True 表示客户端已经迟到：不立即聚合，先放进 general_buffer，
            # 等下一次 group 聚合时和按时客户端一起合并。
            self.general_buffer["local_models"][client_id] = local_model
            self.general_buffer["local_steps"][client_id] = self.client_info[client_id]["local_steps"]
            self.general_buffer["timestamp"][client_id] = self.client_info[client_id]["timestamp"]

        self.client_info[client_id]["timestamp"] = self.global_timestamp
        self._num_global_epochs += 1
        return agg_staleness

    def _group_update(
        self, client_id: str, local_model: Dict, group_idx: int,
    ) -> tuple:
        """组内更新。返回 (new_events, aggregation_staleness_for_this_client)。"""
        new_events: List[VirtualEvent] = []
        curr_time = self._virtual_now
        client_agg_staleness: Optional[int] = None

        if curr_time > self.arrival_group[group_idx]["latest_arrival_time"]:
            # ──── 迟到：移出 group，写入 general_buffer，立即派发下一轮 ────
            self.arrival_group[group_idx]["clients"].remove(client_id)
            if group_idx not in self._group_late_clients:
                self._group_late_clients[group_idx] = []
            self._group_late_clients[group_idx].append(client_id)
            if len(self.arrival_group[group_idx]["clients"]) == 0:
                del self.arrival_group[group_idx]
            self._single_update(client_id, local_model, buffer=True)
            new_events.extend(self._assign_group(client_id))

            local_steps_assigned = self.client_info[client_id]["local_steps"]
            next_group_id = self.client_info[client_id].get("goa", -1)
            target_arrival = None
            latest_arrival = None
            if next_group_id >= 0 and next_group_id in self.arrival_group:
                target_arrival = self.arrival_group[next_group_id]["expected_arrival_time"]
                latest_arrival = self.arrival_group[next_group_id]["latest_arrival_time"]

            self._pending_dispatch_after_upload[client_id] = {"next_group_id": next_group_id}
            self._append_dispatch(
                client_id=client_id,
                virtual_now=curr_time,
                local_steps_assigned=local_steps_assigned,
                target_arrival=target_arrival,
                latest_arrival=latest_arrival,
            )
        else:
            # ──── 按时到达：放入 group_buffer，全员到齐则触发 group_all_arrived 聚合 ────
            self.arrival_group[group_idx]["clients"].remove(client_id)
            self.arrival_group[group_idx]["arrived_clients"].append(client_id)

            if group_idx not in self.group_buffer:
                self.group_buffer[group_idx] = {
                    "local_models": {},
                    "local_steps": {},
                    "timestamp": {},
                }
            self.group_buffer[group_idx]["local_models"][client_id] = local_model
            self.group_buffer[group_idx]["local_steps"][client_id] = self.client_info[client_id]["local_steps"]
            self.group_buffer[group_idx]["timestamp"][client_id] = self.client_info[client_id]["timestamp"]

            self._pending_results[client_id] = {"group_idx": group_idx}

            if len(self.arrival_group[group_idx]["clients"]) == 0:
                # group 内所有“仍被期待的客户端”都到了，deadline 之前也可以提前聚合。
                agg_events, per_staleness = self._group_aggregation(group_idx, trigger="all_arrived")
                new_events.extend(agg_events)
                client_agg_staleness = per_staleness.get(client_id)

        return new_events, client_agg_staleness

    def _group_aggregation(self, group_idx: int, trigger: str = "all_arrived") -> tuple:
        """组聚合。返回 (new_events, per_client_staleness_dict)。"""
        new_events: List[VirtualEvent] = []
        empty_staleness: Dict[str, int] = {}

        if group_idx not in self.arrival_group or group_idx not in self.group_buffer:
            # 可能是过期 deadline，或 group 还没有任何按时到达者；此时没有可聚合内容。
            return new_events, empty_staleness

        # 是否合并了迟到客户端的 general_buffer（影响 group_trace.merged_general_buffer）
        merged_general = bool(self.general_buffer["local_models"])

        # 聚合输入 = 全局迟到缓冲 general_buffer + 当前 group 的按时到达缓冲。
        # 这正是 FedCompass 的半异步语义：迟到更新不丢弃，而是在后续组聚合中消费。
        local_models = {
            **self.general_buffer["local_models"],
            **self.group_buffer[group_idx]["local_models"],
        }
        local_steps_map = {
            **self.general_buffer["local_steps"],
            **self.group_buffer[group_idx]["local_steps"],
        }
        timestamp = {
            **self.general_buffer["timestamp"],
            **self.group_buffer[group_idx]["timestamp"],
        }
        staleness = {
            cid: self.global_timestamp - timestamp[cid]
            for cid in timestamp
        }

        # general_buffer 一旦被本次 group 聚合消费，就必须清空，避免迟到更新重复贡献。
        self.general_buffer = {
            "local_models": {},
            "local_steps": {},
            "timestamp": {},
        }

        global_model = self.aggregator.aggregate(
            local_models=local_models,
            staleness=staleness,
            local_steps=local_steps_map,
        )
        self._current_global_model = copy.deepcopy(global_model)
        ts_before = self.global_timestamp
        self.global_timestamp += 1
        self._num_global_epochs += len(local_models)

        arrived = list(self.arrival_group[group_idx]["arrived_clients"])
        pending = list(self.arrival_group[group_idx]["clients"])
        late_clients = list(self._group_late_clients.get(group_idx, []))
        initial_ids = sorted(set(arrived + pending + late_clients))

        self._record_aggregation_trace(
            trigger=f"group_{trigger}",
            participating=staleness,
            local_steps=local_steps_map,
            model_versions=timestamp,
            group_id=group_idx,
            budget_delta=len(local_models),
            ts_before=ts_before,
        )

        self._group_trace_buffer.append({
            "group_id": group_idx,
            "created_time": self.arrival_group[group_idx].get("created_time", 0.0),
            "expected_arrival_time": self.arrival_group[group_idx]["expected_arrival_time"],
            "latest_arrival_time": self.arrival_group[group_idx]["latest_arrival_time"],
            "initial_client_ids": ",".join(initial_ids),
            "arrived_client_ids": ",".join(arrived),
            "pending_client_ids": ",".join(pending),
            "aggregation_time": self._virtual_now,
            "group_size": len(local_models),
            "late_clients": ",".join(late_clients),
            "trigger": trigger,
            "merged_general_buffer": int(merged_general),
            "per_client_staleness": json.dumps(staleness),
        })
        self._group_late_clients.pop(group_idx, None)

        # 聚合后只给本 group 中按时到达的客户端重派；pending/late 客户端已经不在这个
        # dispatch 链路中，late 客户端在迟到分支已单独安排下一轮。
        client_speeds = []
        for cid in arrived:
            self.client_info[cid]["timestamp"] = self.global_timestamp
            client_speeds.append((cid, self.client_info[cid]["speed"]))
        # 先给快客户端分组，使它们更可能作为新 group 的时间锚点。
        sorted_client_speeds = sorted(client_speeds, key=lambda x: x[1], reverse=False)

        self.arrival_group[group_idx]["expected_arrival_time"] = 0
        self.arrival_group[group_idx]["latest_arrival_time"] = 0

        for cid, _ in sorted_client_speeds:
            assign_events = self._assign_group(cid)
            new_events.extend(assign_events)

            local_steps_assigned = self.client_info[cid]["local_steps"]
            next_group_id = self.client_info[cid].get("goa", -1)
            target_arrival = None
            latest_arrival = None
            if next_group_id >= 0 and next_group_id in self.arrival_group:
                target_arrival = self.arrival_group[next_group_id]["expected_arrival_time"]
                latest_arrival = self.arrival_group[next_group_id]["latest_arrival_time"]

            self._pending_dispatch_after_upload[cid] = {"next_group_id": next_group_id}
            self._append_dispatch(
                client_id=cid,
                virtual_now=self._virtual_now,
                local_steps_assigned=local_steps_assigned,
                target_arrival=target_arrival,
                latest_arrival=latest_arrival,
            )
            self._pending_results.pop(cid, None)

        if len(self.arrival_group[group_idx]["clients"]) == 0:
            del self.arrival_group[group_idx]
        if group_idx in self.group_buffer:
            del self.group_buffer[group_idx]

        return new_events, staleness

    def _assign_group(self, client_id: str) -> List[VirtualEvent]:
        """为客户端分配 arrival group。

        对齐原始 _assign_group()：
            - 无活跃 group → 创建新 group
            - 有活跃 group → 尝试 _join_group，失败则 _create_group
        """
        curr_time = self._virtual_now
        new_events: List[VirtualEvent] = []

        if len(self.arrival_group) == 0:
            # 没有任何活跃 group 时，当前客户端自然成为新 group 的锚点；
            # 这个 group 的 expected/latest 由当前客户端 max Q 的预计完成时间决定。
            expected = curr_time + self.max_local_steps * self.client_info[client_id]["speed"]
            latest = curr_time + self.max_local_steps * self.client_info[client_id]["speed"] * self.latest_time_factor

            self.arrival_group[self.group_counter] = {
                "clients": [client_id],              # 还没到、仍在等待的客户端
                "arrived_clients": [],               # 已按时到达并进入 group_buffer 的客户端
                "expected_arrival_time": expected,   # 目标到齐时间，join_group 用它反推 Q
                "latest_arrival_time": latest,       # 迟到判据，deadline 事件也排在这个时间
                "created_time": curr_time,           # group 创建时间，用于 trace 和 Oort slack/span
            }

            if self.group_counter not in self._deadline_events:
                new_events.append(VirtualEvent(
                    time=latest,
                    event_type=EventType.FEDCOMPASS_GROUP_DEADLINE,
                    payload={"group_idx": self.group_counter},
                ))
                self._deadline_events.add(self.group_counter)

            self.client_info[client_id]["goa"] = self.group_counter
            self.client_info[client_id]["local_steps"] = self.max_local_steps
            self.client_info[client_id]["start_time"] = curr_time

            self._record_dispatch_decision(
                client_id=client_id,
                decision="first_group",
                assigned_group=self.group_counter,
                assigned_steps=self.max_local_steps,
                speed_raw=self.client_info[client_id]["speed"],
                remaining_time=None,
                target_arrival=expected,
                latest_arrival=latest,
            )
            self.group_counter += 1
        else:
            joined = self._join_group(client_id)
            if not joined:
                new_events.extend(self._create_group(client_id))

        return new_events

    def _join_group(self, client_id: str) -> bool:
        """尝试让客户端加入一个已有的 arrival group。

        对齐原始 _join_group()：
            遍历所有活跃 group，计算 remaining_time / speed 得到 local_steps，
            选择满足 [Qmin, Qmax] 且 local_steps 最大的 group。
        """
        curr_time = self._virtual_now
        assigned_group = -1
        assigned_steps = -1

        for group in self.arrival_group:
            remaining_time = self.arrival_group[group]["expected_arrival_time"] - curr_time
            if remaining_time <= 0:
                continue
            local_steps = math.floor(remaining_time / self.client_info[client_id]["speed"])
            # 目标是“尽量贴近已有 group 的 expected_arrival_time”，所以选择满足范围内
            # local_steps 最大的 group；Q 太小/太大都会破坏 FedCompass 的训练步数约束。
            if (
                local_steps < self.min_local_steps
                or local_steps < assigned_steps
                or local_steps > self.max_local_steps
            ):
                continue
            assigned_group = group
            assigned_steps = local_steps

        if assigned_group == -1:
            return False

        self.arrival_group[assigned_group]["clients"].append(client_id)
        self.client_info[client_id]["goa"] = assigned_group
        self.client_info[client_id]["local_steps"] = assigned_steps
        self.client_info[client_id]["start_time"] = curr_time

        remaining = self.arrival_group[assigned_group]["expected_arrival_time"] - curr_time
        self._record_dispatch_decision(
            client_id=client_id,
            decision="join_group",
            assigned_group=assigned_group,
            assigned_steps=assigned_steps,
            speed_raw=self.client_info[client_id]["speed"],
            remaining_time=remaining,
            target_arrival=self.arrival_group[assigned_group]["expected_arrival_time"],
            latest_arrival=self.arrival_group[assigned_group]["latest_arrival_time"],
        )
        return True

    def _create_group(self, client_id: str) -> List[VirtualEvent]:
        """为客户端创建新的 arrival group。

        对齐原始 _create_group()：
            遍历活跃 group 估计新 group 的到达时间，
            计算当前客户端在该到达时间下能做多少步，
            取满足 Qmax 的最大 local_steps。
        """
        curr_time = self._virtual_now
        assigned_steps = -1
        new_events: List[VirtualEvent] = []

        # 遍历现有 group 估计下一个 group 的期望到达时间：
        # 让新 group 尽量接在已有 group 的 latest 之后，避免所有 group 的 deadline 堆在一起。
        for group in self.arrival_group:
            if curr_time < self.arrival_group[group]["latest_arrival_time"]:
                # 找该 group 中最快的客户端速度，用它估计“这个 group 结束后最快多久能再形成下一组”。
                fastest_speed = float("inf")
                group_clients = (
                    self.arrival_group[group]["clients"]
                    + self.arrival_group[group]["arrived_clients"]
                )
                for client in group_clients:
                    fastest_speed = min(fastest_speed, self.client_info[client]["speed"])

                # 估计新 group 的到达时间，再反推当前客户端在这段时间里能做多少 Q。
                est_arrival_time = (
                    self.arrival_group[group]["latest_arrival_time"]
                    + fastest_speed * self.max_local_steps
                )
                local_steps = math.floor(
                    (est_arrival_time - curr_time) / self.client_info[client_id]["speed"]
                )
                if local_steps <= self.max_local_steps:
                    assigned_steps = max(assigned_steps, local_steps)

        # 截断到 [Qmin, Qmax]
        if assigned_steps >= 0 and assigned_steps < self.min_local_steps:
            assigned_steps = self.min_local_steps
        if assigned_steps < 0:
            assigned_steps = self.max_local_steps

        # 创建新 group
        expected = curr_time + assigned_steps * self.client_info[client_id]["speed"]
        latest = curr_time + assigned_steps * self.client_info[client_id]["speed"] * self.latest_time_factor

        self.arrival_group[self.group_counter] = {
            "clients": [client_id],              # 新 group 的首个等待客户端
            "arrived_clients": [],               # 创建时还没有客户端上传完成
            "expected_arrival_time": expected,   # 当前客户端按 assigned_steps 预计完成的时间
            "latest_arrival_time": latest,       # expected 按 latest_time_factor 放宽后的迟到线
            "created_time": curr_time,           # 用于后续 group_trace / Oort 风险余量计算
        }

        # 用事件队列替代 threading.Timer
        if self.group_counter not in self._deadline_events:
            new_events.append(VirtualEvent(
                time=latest,
                event_type=EventType.FEDCOMPASS_GROUP_DEADLINE,
                payload={"group_idx": self.group_counter},
            ))
            self._deadline_events.add(self.group_counter)

        self.client_info[client_id]["goa"] = self.group_counter
        self.client_info[client_id]["local_steps"] = assigned_steps
        self.client_info[client_id]["start_time"] = curr_time

        self._record_dispatch_decision(
            client_id=client_id,
            decision="create_group",
            assigned_group=self.group_counter,
            assigned_steps=assigned_steps,
            speed_raw=self.client_info[client_id]["speed"],
            remaining_time=None,
            target_arrival=expected,
            latest_arrival=latest,
        )
        self.group_counter += 1

        return new_events

    def _append_dispatch(
        self,
        client_id: str,
        virtual_now: float,
        local_steps_assigned: int,
        target_arrival: Optional[float],
        latest_arrival: Optional[float],
    ) -> None:
        """创建 Dispatch 并写入正确的 staleness 与 model_version。

        dispatch_staleness = global_timestamp - client_info.timestamp：
            刚聚合后通常为 0；若客户端长期未同步则可能 > 0。
        model_version_at_dispatch = 当前 global_timestamp（客户端拿到的全局版本）。
        """
        dispatch_staleness = self.global_timestamp - self.client_info[client_id]["timestamp"]
        self._dispatch_queue.append(Dispatch(
            client_id=client_id,
            global_model=copy.deepcopy(self._current_global_model),
            dispatch_time=virtual_now,
            local_steps=local_steps_assigned,
            target_arrival_time=target_arrival,
            latest_arrival_time=latest_arrival,
            staleness=dispatch_staleness,
            model_version_at_dispatch=self.global_timestamp,
        ))

    def _record_aggregation_trace(
        self,
        trigger: str,
        participating: Dict[str, int],
        local_steps: Dict[str, int],
        model_versions: Dict[str, int],
        group_id: int,
        budget_delta: int,
        ts_before: int,
    ) -> None:
        """记录一次全局聚合到 aggregation_trace 缓冲。

        participating 字典的 value 即各客户端 aggregation_staleness，
        与 FedCompassAggregator.aggregate(staleness=...) 入参一致。
        """
        self._aggregation_trace_buffer.append({
            "global_timestamp_before": ts_before,
            "global_timestamp_after": ts_before + 1,
            "virtual_time": self._virtual_now,
            "trigger": trigger,
            "participating_clients": ",".join(sorted(participating.keys())),
            "per_client_staleness": json.dumps(participating),
            "per_client_local_steps": json.dumps(local_steps),
            "per_client_model_version": json.dumps(model_versions),
            "group_id": group_id,
            "num_clients": len(participating),
            "client_update_budget_delta": budget_delta,
        })

    def _record_dispatch_decision(
        self,
        client_id: str,
        decision: str,
        assigned_group: int,
        assigned_steps: int,
        speed_raw: float,
        remaining_time: Optional[float],
        target_arrival: float,
        latest_arrival: float,
    ) -> None:
        """记录一次 Q/group 调度决策到 dispatch_decision_trace 缓冲。

        在 _assign_group / _join_group / _create_group 末尾调用；
        附带当前 upload 的 RuntimeStateTracker 快照，供后续 reason-aware 方法分析。
        """
        state = self._current_runtime_state
        comm_ratio_mean = state.communication_ratio_mean if state is not None else ""
        late_rate = state.late_rate if state is not None else ""
        availability_rate = state.availability_rate if state is not None else ""

        self._dispatch_decision_trace_buffer.append({
            "virtual_time": self._virtual_now,
            "client_id": client_id,
            "decision": decision,
            "assigned_group": assigned_group,
            "assigned_local_steps": assigned_steps,
            "speed_smoothed": speed_raw,
            "speed_raw": speed_raw,
            "remaining_time": remaining_time if remaining_time is not None else "",
            "target_arrival_time": target_arrival,
            "latest_arrival_time": latest_arrival,
            "qmin": self.min_local_steps,
            "qmax": self.max_local_steps,
            "late_threshold_factor": self.latest_time_factor,
            "communication_ratio_mean": comm_ratio_mean,
            "late_rate": late_rate,
            "availability_rate": availability_rate,
        })
