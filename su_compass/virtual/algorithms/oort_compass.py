"""
su_compass.virtual.algorithms.oort_compass — Oort-Compass 虚拟调度控制器。

在**完全不修改**原始 FedCompass 控制器（fedcompass.py）的前提下，通过继承
`VirtualFedCompassController` 并只重写两处 Q 计算入口（`_join_group` /
`_create_group`），把 Oort 的 reason-aware 效用引入 FedCompass 的半异步调度。
所有效用计算委托给纯函数模块 `utility.py`，本文件只负责「在调度决策点上
决定是否采用 Oort 的 Q / 是否按风险过滤分组」。

────────────────────────────────────────────────────────────────────────
解耦设计
────────────────────────────────────────────────────────────────────────
    - 不改父类：父类 fedcompass.py 一行不动，本类通过继承 + override 复用其
      全部聚合 / group 生命周期 / trace 逻辑。
    - 单点改动：Oort 仅替换 Q 公式的分母（speed → effective_step_time），
      arrival group 的到达时间估计仍用真实 speed，语义不被破坏。
    - 独立观测：Oort 分数写入独立的 oort_trace.csv（record_oort_decision），
      不污染 baseline 的 dispatch_decision_trace，便于 shadow 模式逐行对比。

────────────────────────────────────────────────────────────────────────
四段式安全引入（由 OortConfig.mode 控制，与文档 §14 gate 对应）
────────────────────────────────────────────────────────────────────────
    off          完全等价原始 FedCompass（分数都不算）。
    shadow       计算 Oort 分数并写 oort_trace，但 Q/分组仍用原始 speed；
                 用于一致性回归：shadow 的聚合/分组结果必须与 baseline 逐行一致。
    q_only       用 effective_step_time 替换 speed 影响 Q，分组逻辑不变。
    q_and_group  在 q_only 基础上，叠加对高风险客户端的分组迟到风险过滤。

────────────────────────────────────────────────────────────────────────
参考 Oort 开源实现的适配点
────────────────────────────────────────────────────────────────────────
Oort（github.com/SymbioticLab/Oort，`oort.py._training_selector`）为每个
客户端维护 utility 与 duration，并对 round_duration 超过偏好值的 straggler
施加惩罚后排序挑选。本类沿用「用多维状态惩罚高代价客户端」的思想，但因
FedCompass 不做 client selection，改为：惩罚越高 → effective_step_time 越大
→ 分到的 Q 越小 / 越难挤进时间紧的 group，从而降低迟到与陈旧度。
"""

import math
from typing import Any, Dict, List, Optional

from .fedcompass import VirtualFedCompassController
from .utility import (
    OortConfig,
    effective_step_time,
    oort_score,
    risk_score,
    system_penalty,
)
from su_compass.virtual.event import VirtualEvent, EventType


class VirtualOortCompassController(VirtualFedCompassController):
    """Oort-Compass 虚拟调度控制器（继承并复用 FedCompass 全部逻辑）。

    Attributes:
        oort_cfg:               Oort 效用与 mode 配置。
                                mode 决定“只观测”还是“真正影响 Q/分组”。
        _client_runtime_states: 每客户端最近一次 RuntimeStateTracker 快照缓存。
                                组聚合批量重派时，父类只持有上传者的快照，
                                本缓存保证为每个被重派客户端取到其**自身**状态。
        _oort_trace_buffer:     oort_trace 记录缓冲，供 Runner pop 写出。
                                里面保留 baseline Q、Oort Q、实际采用 Q，便于判断改动是否生效。
    """

    def __init__(self, *args, oort_config: Optional[OortConfig] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # oort_cfg 是本类唯一新增的行为开关；mode=off/shadow 时应保证调度结果
        # 与父类 FedCompass 可逐行对齐，便于做回归验证。
        self.oort_cfg = oort_config or OortConfig()
        # 父类处理 group 聚合时会批量给多个客户端重派任务，而 event.payload 只属于
        # 当前上传者；这里缓存“每个客户端最近状态”，避免给 A 调度时误用 B 的状态。
        self._client_runtime_states: Dict[str, Any] = {}
        self._oort_trace_buffer: List[Dict[str, Any]] = []

    @property
    def algorithm_name(self) -> str:
        return "oort_compass"

    # ═══════════════════════ 重写：缓存各客户端状态 ═══════════════════════

    def on_client_upload(self, event: VirtualEvent, virtual_now: float) -> List[VirtualEvent]:
        """在父类处理前先缓存上传者的 runtime 状态。

        缓存必须发生在 super() 之前：父类在处理组聚合时会循环为多个客户端
        调用 _assign_group → _join_group/_create_group，届时需要按 client_id
        取到各自最近的状态，而不仅是本次上传者的状态。
        """
        state = event.payload.get("runtime_state")
        if state is not None:
            self._client_runtime_states[event.client_id] = state
        return super().on_client_upload(event, virtual_now)

    # ═══════════════════════ 重写：Q 计算（唯一改动点）═══════════════════════

    def _join_group(self, client_id: str) -> bool:
        """尝试加入已有 arrival group。

        与父类逻辑逐行对齐，唯一差异：
            1. Q 的分母 speed → q_divisor（q_only/q_and_group 下为 effective_step_time）；
            2. q_and_group 模式下对高风险客户端做迟到风险过滤（跳过时间紧的 group）；
            3. 决策后额外写一条 oort_trace（记录 q_baseline vs q_after_oort）。
        分组的到达时间（expected/latest）是 group 固有属性，此处不重算，故不受影响。
        """
        curr_time = self._virtual_now
        speed = self.client_info[client_id]["speed"]
        state = self._client_runtime_states.get(client_id)
        # effective_step_time = 原始平滑速度 × 系统惩罚。
        # 它只改变“同样剩余时间能做多少步”，不改变 group 本身的 expected/latest 时间。
        eff_step = effective_step_time(speed, state, self.oort_cfg)
        # shadow/off 模式下用真实 speed → 保证 Q 与 baseline 完全一致
        q_divisor = eff_step if self.oort_cfg.affects_q else speed

        assigned_group = -1
        assigned_steps = -1
        chosen_remaining: Optional[float] = None

        for group in self.arrival_group:
            remaining_time = self.arrival_group[group]["expected_arrival_time"] - curr_time
            if remaining_time <= 0:
                continue
            # q_and_group：高风险客户端不进时间余量太小的 group。
            # 这一步只过滤候选 group，不会改变已有 group 的 deadline。
            if self.oort_cfg.affects_group and self._group_too_risky(group, state):
                continue
            local_steps = math.floor(remaining_time / q_divisor)
            if (
                local_steps < self.min_local_steps
                or local_steps < assigned_steps
                or local_steps > self.max_local_steps
            ):
                continue
            assigned_group = group
            assigned_steps = local_steps
            chosen_remaining = remaining_time

        if assigned_group == -1:
            # 记录一次未加入决策，便于分析 Oort 是否因风险过滤/惩罚导致 join 失败
            self._record_oort(
                client_id, "join_failed", -1, speed, eff_step, state,
                q_baseline=-1, q_after_oort=-1, q_applied=-1, remaining=chosen_remaining,
            )
            return False

        self.arrival_group[assigned_group]["clients"].append(client_id)
        self.client_info[client_id]["goa"] = assigned_group
        self.client_info[client_id]["local_steps"] = assigned_steps
        self.client_info[client_id]["start_time"] = curr_time

        remaining = self.arrival_group[assigned_group]["expected_arrival_time"] - curr_time
        # 复用父类 dispatch_decision_trace 记录（字段与 baseline 完全一致）
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
        # 额外的 Oort 观测：同一 remaining 下 baseline 与 oort 分别会给多少 Q
        # 这两个 Q 是同一候选 group 下的反事实对比：
        # q_base 表示原始 FedCompass 会给多少步，q_oort 表示启用惩罚后会给多少步。
        q_base = math.floor(chosen_remaining / speed) if chosen_remaining else assigned_steps
        q_oort = math.floor(chosen_remaining / eff_step) if chosen_remaining else assigned_steps
        self._record_oort(
            client_id, "join_group", assigned_group, speed, eff_step, state,
            q_baseline=q_base, q_after_oort=q_oort, q_applied=assigned_steps,
            remaining=chosen_remaining,
        )
        return True

    def _create_group(self, client_id: str) -> List[VirtualEvent]:
        """创建新 arrival group。

        与父类逐行对齐，唯一差异：估计 local_steps 时的分母 speed → q_divisor。
        新 group 的 expected/latest 到达时间仍用真实 speed 计算
        （assigned_steps 步实际需要多少时间），保证时间语义正确。
        """
        curr_time = self._virtual_now
        speed = self.client_info[client_id]["speed"]
        state = self._client_runtime_states.get(client_id)
        eff_step = effective_step_time(speed, state, self.oort_cfg)
        q_divisor = eff_step if self.oort_cfg.affects_q else speed

        assigned_steps = -1
        new_events: List[VirtualEvent] = []

        for group in self.arrival_group:
            if curr_time < self.arrival_group[group]["latest_arrival_time"]:
                # 新建 group 不是凭空取 max Q：它参考已有 group 的节奏，
                # 推算一个“接在当前 group 后面”的合理到达时间，再反推 Q。
                fastest_speed = float("inf")
                group_clients = (
                    self.arrival_group[group]["clients"]
                    + self.arrival_group[group]["arrived_clients"]
                )
                for client in group_clients:
                    fastest_speed = min(fastest_speed, self.client_info[client]["speed"])

                est_arrival_time = (
                    self.arrival_group[group]["latest_arrival_time"]
                    + fastest_speed * self.max_local_steps
                )
                local_steps = math.floor((est_arrival_time - curr_time) / q_divisor)
                if local_steps <= self.max_local_steps:
                    assigned_steps = max(assigned_steps, local_steps)

        # q_baseline 供 oort_trace 对比：用真实 speed 复算一遍（此处只取近似）
        q_base_est = assigned_steps
        if self.oort_cfg.affects_q and assigned_steps >= 0 and eff_step > 0:
            # 反推 baseline 会给的步数（分母换回 speed），仅用于观测
            q_base_est = math.floor(assigned_steps * eff_step / max(speed, 1e-8))

        if assigned_steps >= 0 and assigned_steps < self.min_local_steps:
            assigned_steps = self.min_local_steps
        if assigned_steps < 0:
            # 没有任何可参考 group，或估计结果不可用时，回到 FedCompass 的保守默认：max Q。
            assigned_steps = self.max_local_steps

        # 注意这里仍用真实 speed 计算 expected/latest。
        # Oort 只影响给多少 Q；完成这些 Q 需要的虚拟时间仍由真实速度决定。
        expected = curr_time + assigned_steps * speed
        latest = curr_time + assigned_steps * speed * self.latest_time_factor

        self.arrival_group[self.group_counter] = {
            "clients": [client_id],
            "arrived_clients": [],
            "expected_arrival_time": expected,
            "latest_arrival_time": latest,
            "created_time": curr_time,
        }

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
        self._record_oort(
            client_id, "create_group", self.group_counter, speed, eff_step, state,
            q_baseline=q_base_est, q_after_oort=assigned_steps, q_applied=assigned_steps,
            remaining=None,
        )
        self.group_counter += 1
        return new_events

    # ═══════════════════════ 辅助方法 ═══════════════════════

    def _group_too_risky(self, group_idx: int, state: Any) -> bool:
        """判断高风险客户端是否应被挡在某个 group 之外（q_and_group 模式）。

        风险低于门槛的客户端一律放行；高风险客户端仅当目标 group 的时间余量
        （slack = latest-expected，相对 group 时间跨度）足够宽时才允许加入。
        """
        risk = risk_score(state, self.oort_cfg)
        if risk <= self.oort_cfg.risk_threshold:
            return False
        g = self.arrival_group[group_idx]
        span = g["expected_arrival_time"] - g.get("created_time", 0.0)
        if span <= 0:
            return False
        slack = g["latest_arrival_time"] - g["expected_arrival_time"]
        # slack/span 越小说明 deadline 越紧。高风险客户端只有在余量足够宽时才放行。
        return (slack / span) < self.oort_cfg.slack_min_ratio

    def _record_oort(
        self,
        client_id: str,
        decision: str,
        assigned_group: int,
        speed: float,
        eff_step: float,
        state: Any,
        q_baseline: int,
        q_after_oort: int,
        q_applied: int,
        remaining: Optional[float],
    ) -> None:
        """写一条 oort_trace 记录（shadow 及各 mode 均记录，便于离线分析）。"""
        # oort_trace 是调度干预的“黑匣子记录”：即使 shadow 模式不改变行为，
        # 也会保留惩罚、风险和反事实 Q，方便离线比较是否值得打开 q_only/q_and_group。
        self._oort_trace_buffer.append({
            "virtual_time": self._virtual_now,       # 做出本次 Q/group 决策的虚拟时间
            "client_id": client_id,
            "decision": decision,                    # join/create/failed，和 dispatch_decision_trace 对齐
            "assigned_group": assigned_group,        # 实际分到的 group；join_failed 为 -1
            "oort_mode": self.oort_cfg.mode,         # shadow/q_only/q_and_group，解释 q_applied 是否受 Oort 影响
            "speed_smoothed": speed,                 # 原始 FedCompass 会使用的 Q 分母
            "effective_step_time": eff_step,         # Oort 惩罚后的 Q 分母
            "system_penalty": system_penalty(state, self.oort_cfg),  # eff_step / speed 的主要来源
            "risk_score": risk_score(state, self.oort_cfg),          # q_and_group 过滤 group 的依据
            "oort_score": oort_score(speed, state, self.oort_cfg),   # 越高表示越值得给资源，当前仅观测
            "q_baseline": q_baseline,                # 若按原始 speed，本次会给多少 Q
            "q_after_oort": q_after_oort,            # 若按 effective_step_time，本次会给多少 Q
            "q_applied": q_applied,                  # 实际写入 Dispatch 的 Q，受 mode 控制
            "remaining_time": remaining if remaining is not None else "",  # join 时距离 group expected 的剩余时间
            "communication_ratio_mean": getattr(state, "communication_ratio_mean", "") if state else "",
            "late_rate": getattr(state, "late_rate", "") if state else "",
            "step_time_cv": getattr(state, "step_time_cv", "") if state else "",
            "availability_rate": getattr(state, "availability_rate", "") if state else "",
            "num_reports": getattr(state, "num_reports", 0) if state else 0,
        })

    def pop_oort_traces(self) -> List[Dict[str, Any]]:
        """弹出并清空 oort_trace 缓冲，供 Runner 写出。"""
        traces = list(self._oort_trace_buffer)
        self._oort_trace_buffer.clear()
        return traces
