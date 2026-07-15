"""StateCompass Q-only虚拟调度控制器。

本控制器在VirtualFedCompassController上进行最小增量接入：
    1. FedCompass仍按原公式选择已有arrival group；
    2. group选定后，使用状态预测器在同一group时间窗内重新推荐Q；
    3. 若不存在安全Q，Shadow保持原Q；Apply拒绝入组并调用原始建组路径；
    4. 首组和新建group完全沿用FedCompass，避免本轮混入分组策略收益；
    5. deadline、general buffer、聚合和staleness逻辑全部复用父类。

RCP-GS目前以旁路Shadow形式枚举其他原合法已有group，只输出候选和推荐trace，
不把推荐写回调度器。因此真实运行仍只回答“状态感知Q分配是否带来收益”；
分组模块当前回答“是否存在安全、保训练量且到达偏差不劣的换组机会”。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from su_compass.scheduling.policies import (
    AllGroupFeasibilityShadowPolicy,
    CommunicationTailRiskShadowPolicy,
    CommunicationRobustQShadowPolicy,
    GroupCreationCounterfactualShadowPolicy,
    ParetoGroupShadowPolicy,
    RiskConstrainedGroupAdmissionPolicy,
    ShadowQPolicy,
    StateGroupCreationQShadowPolicy,
    StateGroupWindowShadowPolicy,
    StateWindowAdmissionShadowPolicy,
)
from su_compass.scheduling.predictors import AdaptiveLatencyPredictor
from su_compass.virtual.algorithms.fedcompass import VirtualFedCompassController
from su_compass.virtual.event import EventType, VirtualEvent


class VirtualStateCompassController(VirtualFedCompassController):
    """只在加入既有group时应用状态感知Q的FedCompass控制器。"""

    def __init__(
        self,
        *args,
        q_increase_ratio: float = 0.20,
        group_admission_mode: str = "shadow",
        all_group_feasibility_mode: str = "off",
        group_creation_counterfactual_mode: str = "off",
        state_group_creation_q_mode: str = "off",
        state_group_window_mode: str = "off",
        state_window_admission_mode: str = "off",
        communication_tail_risk_mode: str = "off",
        communication_robust_q_mode: str = "off",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 <= q_increase_ratio <= 1.0:
            raise ValueError("q_increase_ratio must be in [0, 1]")
        # V2只限制增Q幅度。安全预测要求的减Q不能被强行拉回，否则可能重新超过
        # deadline；增Q则最多20%，用于控制聚合间隔和TTA退化风险。
        self.q_increase_ratio = q_increase_ratio
        self._state_predictor = AdaptiveLatencyPredictor()
        self._q_policy = ShadowQPolicy(self._state_predictor)
        self._group_shadow_policy = ParetoGroupShadowPolicy(
            self._state_predictor,
            q_increase_ratio=q_increase_ratio,
        )
        self._group_admission_policy = RiskConstrainedGroupAdmissionPolicy(
            group_admission_mode
        )
        if all_group_feasibility_mode not in ("off", "shadow"):
            raise ValueError("all_group_feasibility_mode must be off or shadow")
        self._all_group_feasibility_mode = all_group_feasibility_mode
        self._all_group_feasibility_policy = AllGroupFeasibilityShadowPolicy(
            self._q_policy
        )
        if group_creation_counterfactual_mode not in ("off", "shadow"):
            raise ValueError("group_creation_counterfactual_mode must be off or shadow")
        self._group_creation_counterfactual_mode = group_creation_counterfactual_mode
        self._group_creation_counterfactual_policy = (
            GroupCreationCounterfactualShadowPolicy(self._state_predictor)
        )
        if state_group_creation_q_mode not in ("off", "shadow"):
            raise ValueError("state_group_creation_q_mode must be off or shadow")
        self._state_group_creation_q_mode = state_group_creation_q_mode
        self._state_group_creation_q_policy = StateGroupCreationQShadowPolicy(
            self._state_predictor
        )
        if state_group_window_mode not in ("off", "shadow"):
            raise ValueError("state_group_window_mode must be off or shadow")
        self._state_group_window_mode = state_group_window_mode
        self._state_group_window_policy = StateGroupWindowShadowPolicy(
            self._state_predictor
        )
        if state_window_admission_mode not in ("off", "shadow", "apply"):
            raise ValueError("state_window_admission_mode must be off, shadow or apply")
        if state_window_admission_mode == "apply" and group_admission_mode != "shadow":
            raise ValueError("state-window apply requires legacy group_admission_mode=shadow")
        self._state_window_admission_mode = state_window_admission_mode
        self._state_window_admission_policy = StateWindowAdmissionShadowPolicy()
        if communication_tail_risk_mode not in ("off", "shadow"):
            raise ValueError("communication_tail_risk_mode must be off or shadow")
        self._communication_tail_risk_mode = communication_tail_risk_mode
        self._communication_tail_risk_policy = CommunicationTailRiskShadowPolicy()
        if communication_robust_q_mode not in ("off", "shadow"):
            raise ValueError("communication_robust_q_mode must be off or shadow")
        self._communication_robust_q_mode = communication_robust_q_mode
        self._communication_robust_q_policy = CommunicationRobustQShadowPolicy(
            self._state_predictor
        )
        # 组聚合会批量重派多个客户端，因此必须按client_id缓存各自最后一次已上传
        # 的状态，不能复用父类仅代表当前上传者的_current_runtime_state。
        self._client_runtime_states: Dict[str, Any] = {}
        self._state_q_trace_buffer: List[Dict[str, Any]] = []
        self._group_candidate_trace_buffer: List[Dict[str, Any]] = []
        self._group_recommendation_trace_buffer: List[Dict[str, Any]] = []
        self._group_admission_trace_buffer: List[Dict[str, Any]] = []
        self._all_group_feasibility_trace_buffer: List[Dict[str, Any]] = []
        self._all_group_recommendation_trace_buffer: List[Dict[str, Any]] = []
        self._group_creation_counterfactual_trace_buffer: List[Dict[str, Any]] = []
        self._state_group_creation_q_trace_buffer: List[Dict[str, Any]] = []
        self._state_group_window_trace_buffer: List[Dict[str, Any]] = []
        self._state_window_admission_trace_buffer: List[Dict[str, Any]] = []
        self._communication_tail_risk_trace_buffer: List[Dict[str, Any]] = []
        self._communication_robust_q_trace_buffer: List[Dict[str, Any]] = []
        # Apply拒绝后，真实Q要等父类_create_group计算完才能确定。这里保存同一个
        # trace字典引用，建组完成后原地回填，避免把“候选Q”误写成“实际Q”。
        self._pending_admission_trace_by_client: Dict[str, Dict[str, Any]] = {}
        self._pending_state_q_trace_by_client: Dict[str, Dict[str, Any]] = {}
        # 仅Gate认可的客户端保存状态时间窗计划；父类建组完成后消费并删除。
        self._pending_state_window_plan_by_client: Dict[str, Any] = {}

    @property
    def algorithm_name(self) -> str:
        return "state_compass_trust_q_v2"

    def on_client_upload(self, event: VirtualEvent, virtual_now: float):
        """客户端上传真正到达后缓存状态，再交给父类完成原始处理。"""
        state = event.payload.get("runtime_state")
        if state is not None:
            self._client_runtime_states[event.client_id] = state
        return super().on_client_upload(event, virtual_now)

    def _join_group(self, client_id: str) -> bool:
        """保持FedCompass选组结果，只替换该组内实际执行的Q。"""
        curr_time = self._virtual_now
        speed = self.client_info[client_id]["speed"]

        # 第一步严格复现父类选组逻辑。这样本轮实验不会因为选择了不同group而
        # 获得收益，Q-only与后续Q+Group实验可以清晰区分。
        assigned_group = -1
        baseline_steps = -1
        for group_idx in self.arrival_group:
            remaining_time = (
                self.arrival_group[group_idx]["expected_arrival_time"] - curr_time
            )
            if remaining_time <= 0:
                continue
            local_steps = math.floor(remaining_time / speed)
            if (
                local_steps < self.min_local_steps
                or local_steps < baseline_steps
                or local_steps > self.max_local_steps
            ):
                continue
            assigned_group = group_idx
            baseline_steps = local_steps

        if assigned_group == -1:
            return False

        group = self.arrival_group[assigned_group]
        state = self._client_runtime_states.get(client_id)
        recommendation = self._q_policy.recommend(
            client_id=client_id,
            dispatch_time=curr_time,
            speed_smoothed=speed,
            runtime_state=state,
            expected_arrival_time=group["expected_arrival_time"],
            latest_arrival_time=group["latest_arrival_time"],
            qmin=self.min_local_steps,
            qmax=self.max_local_steps,
            current_q=baseline_steps,
        )
        raw_state_steps = recommendation.recommended_q
        max_increase_q = min(
            self.max_local_steps,
            int(math.floor(baseline_steps * (1.0 + self.q_increase_ratio))),
        )
        applied_steps = min(raw_state_steps, max_increase_q)

        # Availability点预测针对多数正常轮次优化MAE，但真实不可用事件发生时，
        # 增大的Q会与额外等待叠加。已有实验观察到availability迟到由1增至4，
        # 因此在至少发生2次不可用事件后保留减Q能力，但禁止高于FedCompass原Q。
        unavailable_event_count = (
            int(getattr(state, "unavailable_event_count", 0)) if state else 0
        )
        availability_increase_guard = unavailable_event_count >= 2
        if availability_increase_guard:
            applied_steps = min(applied_steps, baseline_steps)

        if not recommendation.safe_feasible:
            # 无安全Q说明是group mismatch。即使经过信任域也必须保持父类原Q，
            # 不让Q-only模块为无法解决的分组问题损失本地训练量。
            applied_steps = baseline_steps

        # Admission不重复预测：它严格消费上面的Trust-Q结论。apply模式下仅对
        # group_mismatch返回False，父类_assign_group随后会调用原始_create_group。
        admission = self._group_admission_policy.decide(
            recommendation.safe_feasible
        )
        all_groups = None
        if (
            self._all_group_feasibility_mode == "shadow"
            or self._state_window_admission_mode in ("shadow", "apply")
        ):
            all_groups = self._all_group_feasibility_policy.recommend(
                client_id=client_id, dispatch_time=curr_time,
                speed_smoothed=speed, runtime_state=state,
                groups=self.arrival_group, current_group_id=assigned_group,
                qmin=self.min_local_steps, qmax=self.max_local_steps,
            )
            if self._all_group_feasibility_mode == "shadow":
                self._record_all_group_feasibility(client_id, all_groups)
        # 反事实只在Trust-Q确认mismatch后运行。结果仅写Trace，绝不能改变
        # Admission决定或_join_group返回值，否则本轮将不再是纯Shadow。
        state_window = None
        if not recommendation.safe_feasible and (
            self._group_creation_counterfactual_mode == "shadow"
            or self._state_group_creation_q_mode == "shadow"
            or self._state_group_window_mode == "shadow"
            or self._state_window_admission_mode in ("shadow", "apply")
        ):
            counterfactual = self._group_creation_counterfactual_policy.evaluate(
                client_id=client_id, dispatch_time=curr_time,
                speed_smoothed=speed, runtime_state=state,
                groups=self.arrival_group, client_info=self.client_info,
                current_group_id=assigned_group, current_q=applied_steps,
                current_predicted_finish_time=recommendation.predicted_finish_time,
                current_safe_finish_time=recommendation.safe_finish_time,
                qmin=self.min_local_steps, qmax=self.max_local_steps,
                latest_time_factor=self.latest_time_factor,
            )
            if self._group_creation_counterfactual_mode == "shadow":
                self._record_group_creation_counterfactual(client_id, counterfactual)
            if self._state_group_creation_q_mode == "shadow":
                # 只读枚举原新组公式下的所有Q，不覆盖counterfactual或真实建组结果。
                state_new_group_q = self._state_group_creation_q_policy.recommend(
                    client_id=client_id, dispatch_time=curr_time,
                    speed_smoothed=speed, runtime_state=state,
                    original_q=counterfactual.counterfactual_q,
                    original_expected_arrival_time=(
                        counterfactual.counterfactual_expected_arrival_time
                    ),
                    original_safe_feasible=(
                        counterfactual.counterfactual_safe_feasible
                    ),
                    qmin=self.min_local_steps, qmax=self.max_local_steps,
                    latest_time_factor=self.latest_time_factor,
                )
                self._record_state_group_creation_q(
                    client_id, counterfactual, state_new_group_q
                )
            if (
                self._state_group_window_mode == "shadow"
                or self._state_window_admission_mode in ("shadow", "apply")
            ):
                # 固定FedCompass原始新组Q，只替换时间预测基准，隔离时间窗因素。
                state_window = self._state_group_window_policy.evaluate(
                    client_id=client_id, dispatch_time=curr_time,
                    speed_smoothed=speed, runtime_state=state,
                    fixed_q=counterfactual.counterfactual_q,
                    speed_expected_arrival_time=(
                        counterfactual.counterfactual_expected_arrival_time
                    ),
                    speed_latest_arrival_time=(
                        counterfactual.counterfactual_latest_arrival_time
                    ),
                    qmin=self.min_local_steps, qmax=self.max_local_steps,
                    latest_time_factor=self.latest_time_factor,
                )
                if self._state_group_window_mode == "shadow":
                    self._record_state_group_window(client_id, counterfactual, state_window)
                if self._communication_tail_risk_mode == "shadow":
                    # 只评估新增通信尾部Gate，不覆盖state_window或真实Admission。
                    tail_risk = self._communication_tail_risk_policy.evaluate(
                        runtime_state=state,
                        original_safe_slack=state_window.state_safe_slack,
                        existing_uncertainty=state_window.uncertainty,
                    )
                    self._record_communication_tail_risk(
                        client_id, assigned_group, state_window, tail_risk
                    )
                if self._communication_robust_q_mode == "shadow":
                    robust_q = self._communication_robust_q_policy.recommend(
                        client_id=client_id,
                        dispatch_time=curr_time,
                        speed_smoothed=speed,
                        runtime_state=state,
                        original_q=state_window.fixed_q,
                        group_latest_time=state_window.state_latest_arrival_time,
                        qmin=self.min_local_steps,
                        qmax=self.max_local_steps,
                    )
                    self._record_communication_robust_q(
                        client_id, assigned_group, state_window, robust_q
                    )

        state_window_apply = False
        gate = None
        if self._state_window_admission_mode in ("shadow", "apply"):
            # Gate只组合上游事实。即使建议create/switch，也不会改变真实Admission。
            other_safe = bool(
                all_groups is not None
                and any(
                    c.state_safe_feasible and not c.is_current_group
                    for c in all_groups.candidates
                )
            )
            new_window_safe = bool(
                not recommendation.safe_feasible
                and state_window is not None
                and state_window.state_window_safe_feasible
            )
            gate = self._state_window_admission_policy.decide(
                current_group_safe=recommendation.safe_feasible,
                other_existing_group_safe=other_safe,
                state_new_group_safe=new_window_safe,
            )
            state_window_apply = (
                self._state_window_admission_mode == "apply"
                and gate.action == "create_state_window_group"
            )
            if state_window_apply:
                self._pending_state_window_plan_by_client[client_id] = state_window
            self._record_state_window_admission(
                client_id=client_id, assigned_group=assigned_group,
                current_safe_slack=(
                    group["latest_arrival_time"] - recommendation.safe_finish_time
                ),
                all_groups=all_groups, state_window=state_window, decision=gate,
                applied=state_window_apply,
            )
        final_admitted = admission.admitted and not state_window_apply
        admission_trace = {
            "virtual_time": curr_time,
            "client_id": client_id,
            "mode": admission.mode,
            "candidate_group_id": assigned_group,
            "fedcompass_q": baseline_steps,
            "candidate_trust_q": applied_steps,
            "trust_q_safe_feasible": int(recommendation.safe_feasible),
            "group_mismatch": int(not recommendation.safe_feasible),
            "shadow_action": admission.shadow_action,
            "applied_action": (
                "create_state_window_group" if state_window_apply
                else admission.applied_action
            ),
            "admitted": int(final_admitted),
            "fedcompass_create_group_id": (
                self.group_counter if not final_admitted else -1
            ),
            "actual_group_id": assigned_group if final_admitted else -1,
            "actual_dispatched_q": applied_steps if final_admitted else -1,
            "reason": (
                "state_window_gate_apply" if state_window_apply else admission.reason
            ),
        }
        self._group_admission_trace_buffer.append(admission_trace)
        # Q事实必须在准入返回前记录；否则Apply拒绝的关键mismatch样本会从
        # state_q_trace消失，无法与Shadow做逐决策核对。
        state_q_trace = {
            "virtual_time": curr_time,
            "client_id": client_id,
            "assigned_group": assigned_group,
            "fedcompass_q": baseline_steps,
            "raw_state_q": raw_state_steps,
            "state_q": applied_steps,
            "q_difference": applied_steps - baseline_steps,
            "max_increase_q": max_increase_q,
            "q_increase_clipped": int(raw_state_steps > max_increase_q),
            "availability_increase_guard": int(availability_increase_guard),
            "unavailable_event_count": unavailable_event_count,
            "state_safe_feasible": int(recommendation.safe_feasible),
            "group_mismatch": int(not recommendation.safe_feasible),
            "predicted_finish_time": recommendation.predicted_finish_time,
            "safe_finish_time": recommendation.safe_finish_time,
            "expected_arrival_time": group["expected_arrival_time"],
            "latest_arrival_time": group["latest_arrival_time"],
            "recommendation_reason": recommendation.reason,
            "applied_reason": _applied_reason(
                recommendation.safe_feasible,
                raw_state_steps,
                applied_steps,
                max_increase_q,
                availability_increase_guard,
            ),
            "q_applied_to_dispatch": int(final_admitted),
            "actual_dispatched_q": applied_steps if final_admitted else -1,
            "actual_group_id": assigned_group if final_admitted else -1,
            "num_state_reports": getattr(state, "num_reports", 0) if state else 0,
        }
        self._state_q_trace_buffer.append(state_q_trace)

        if not final_admitted:
            # 尚未写入group/client_info，拒绝无副作用；新组、Q、deadline事件均由
            # 父类先按原公式生成Q和组；State-Window Apply随后仅替换时间窗。
            self._pending_admission_trace_by_client[client_id] = admission_trace
            self._pending_state_q_trace_by_client[client_id] = state_q_trace
            return False

        # RCP-GS当前只做Shadow：以Trust-Q V2实际会采用的(group,Q)为基线，
        # 枚举其他FedCompass原合法已有group。此调用只读取arrival_group快照，
        # 推荐结果不会写回group或client_info。
        group_shadow = self._group_shadow_policy.recommend(
            client_id=client_id,
            dispatch_time=curr_time,
            speed_smoothed=speed,
            runtime_state=state,
            groups=self.arrival_group,
            baseline_group_id=assigned_group,
            baseline_q=applied_steps,
            qmin=self.min_local_steps,
            qmax=self.max_local_steps,
        )
        self._record_group_shadow(client_id, group_shadow)

        group["clients"].append(client_id)
        self.client_info[client_id]["goa"] = assigned_group
        self.client_info[client_id]["local_steps"] = applied_steps
        self.client_info[client_id]["start_time"] = curr_time

        remaining = group["expected_arrival_time"] - curr_time
        self._record_dispatch_decision(
            client_id=client_id,
            decision="join_group",
            assigned_group=assigned_group,
            assigned_steps=applied_steps,
            speed_raw=speed,
            remaining_time=remaining,
            target_arrival=group["expected_arrival_time"],
            latest_arrival=group["latest_arrival_time"],
        )
        return True

    def _create_group(self, client_id: str):
        """复用父类建组；Gate Apply时仅替换新组时间窗与deadline事件。

        本方法不改变任何建组公式；父类返回后只读取已经写好的client_info，更新
        Trace字段。这样候选Q与真实执行Q可被严格区分。
        """
        new_events = super()._create_group(client_id)
        actual_group_id = int(self.client_info[client_id]["goa"])
        actual_q = int(self.client_info[client_id]["local_steps"])

        state_window = self._pending_state_window_plan_by_client.pop(client_id, None)
        if state_window is not None:
            if actual_q != int(state_window.fixed_q):
                raise RuntimeError("state-window plan Q differs from original create-group Q")
            group = self.arrival_group[actual_group_id]
            group["expected_arrival_time"] = state_window.state_expected_arrival_time
            group["latest_arrival_time"] = state_window.state_latest_arrival_time
            # 父类已经创建唯一deadline事件；原地改时刻可避免重复事件和计数。
            deadline_events = [
                event for event in new_events
                if event.event_type == EventType.FEDCOMPASS_GROUP_DEADLINE
                and int(event.payload.get("group_idx", -1)) == actual_group_id
            ]
            if len(deadline_events) != 1:
                raise RuntimeError("expected exactly one deadline event for new group")
            deadline_events[0].time = state_window.state_latest_arrival_time
            # 父类Trace已记录原始时间窗；同步回填实际执行事实，避免候选/真实混淆。
            dispatch_trace = self._dispatch_decision_trace_buffer[-1]
            dispatch_trace["target_arrival_time"] = state_window.state_expected_arrival_time
            dispatch_trace["latest_arrival_time"] = state_window.state_latest_arrival_time

        admission_trace = self._pending_admission_trace_by_client.pop(client_id, None)
        if admission_trace is not None:
            admission_trace["actual_group_id"] = actual_group_id
            admission_trace["actual_dispatched_q"] = actual_q

        state_q_trace = self._pending_state_q_trace_by_client.pop(client_id, None)
        if state_q_trace is not None:
            state_q_trace["actual_group_id"] = actual_group_id
            state_q_trace["actual_dispatched_q"] = actual_q
            # 当前state_q是被拒绝的已有组候选；实际派发Q来自原始建组逻辑。
            state_q_trace["q_applied_to_dispatch"] = 0
            state_q_trace["applied_reason"] = (
                "group_mismatch_then_state_window_create"
                if state_window is not None
                else "group_mismatch_rejected_then_original_create"
            )
        return new_events

    def pop_state_q_traces(self) -> List[Dict[str, Any]]:
        """弹出Q-only决策trace，供统一运行器写出。"""
        rows = list(self._state_q_trace_buffer)
        self._state_q_trace_buffer.clear()
        return rows

    def _record_group_shadow(self, client_id: str, recommendation) -> None:
        """将每个候选group和最终Shadow推荐拆成两类trace。"""
        for candidate in recommendation.candidates:
            self._group_candidate_trace_buffer.append({
                "virtual_time": self._virtual_now,
                "client_id": client_id,
                "baseline_group_id": recommendation.baseline_group_id,
                "baseline_q": recommendation.baseline_q,
                "baseline_safe_feasible": int(recommendation.baseline_safe_feasible),
                "candidate_group_id": candidate.group_id,
                "fedcompass_reference_q": candidate.fedcompass_reference_q,
                "candidate_q": candidate.candidate_q,
                "expected_arrival_time": candidate.expected_arrival_time,
                "latest_arrival_time": candidate.latest_arrival_time,
                "predicted_finish_time": candidate.predicted_finish_time,
                "safe_finish_time": candidate.safe_finish_time,
                "arrival_deviation": candidate.arrival_deviation,
                "safe_slack": candidate.safe_slack,
                "original_group_feasible": int(candidate.original_group_feasible),
                "safe_feasible": int(candidate.safe_feasible),
                "arrival_window_feasible": int(candidate.arrival_window_feasible),
                "work_preserved": int(candidate.work_preserved),
                "pareto_dominates_baseline": int(candidate.pareto_dominates_baseline),
                "selected_by_shadow": int(
                    recommendation.action == "switch_existing_group"
                    and candidate.group_id == recommendation.recommended_group_id
                ),
                "rejection_reason": candidate.rejection_reason,
            })
        self._group_recommendation_trace_buffer.append({
            "virtual_time": self._virtual_now,
            "client_id": client_id,
            "baseline_group_id": recommendation.baseline_group_id,
            "baseline_q": recommendation.baseline_q,
            "baseline_safe_feasible": int(recommendation.baseline_safe_feasible),
            "baseline_arrival_deviation": recommendation.baseline_arrival_deviation,
            "recommended_group_id": recommendation.recommended_group_id,
            "recommended_q": recommendation.recommended_q,
            "action": recommendation.action,
            "group_changed": int(recommendation.group_changed),
            "mismatch_repaired": int(recommendation.mismatch_repaired),
            "num_candidates": len(recommendation.candidates),
            "num_pareto_candidates": sum(
                int(c.pareto_dominates_baseline) for c in recommendation.candidates
            ),
            "reason": recommendation.reason,
        })

    def pop_group_candidate_traces(self) -> List[Dict[str, Any]]:
        """弹出RCP-GS逐候选Shadow记录。"""
        rows = list(self._group_candidate_trace_buffer)
        self._group_candidate_trace_buffer.clear()
        return rows

    def pop_group_recommendation_traces(self) -> List[Dict[str, Any]]:
        """弹出RCP-GS每次dispatch的最终Shadow推荐。"""
        rows = list(self._group_recommendation_trace_buffer)
        self._group_recommendation_trace_buffer.clear()
        return rows

    def pop_group_admission_traces(self) -> List[Dict[str, Any]]:
        """弹出风险约束入组记录，独立于Q和RCP-GS trace。"""
        rows = list(self._group_admission_trace_buffer)
        self._group_admission_trace_buffer.clear()
        return rows

    def _record_all_group_feasibility(self, client_id: str, recommendation) -> None:
        """记录全量已有组Shadow事实；推荐值不会反馈给真实调度。"""
        for candidate in recommendation.candidates:
            self._all_group_feasibility_trace_buffer.append({
                "virtual_time": self._virtual_now, "client_id": client_id,
                "current_group_id": recommendation.current_group_id,
                "group_id": candidate.group_id,
                "is_current_group": int(candidate.is_current_group),
                "expected_arrival_time": candidate.expected_arrival_time,
                "latest_arrival_time": candidate.latest_arrival_time,
                "fedcompass_reference_q": candidate.fedcompass_reference_q,
                "fedcompass_q_feasible": int(candidate.fedcompass_q_feasible),
                "state_recommended_q": candidate.state_recommended_q,
                "state_safe_feasible": int(candidate.state_safe_feasible),
                "predicted_finish_time": candidate.predicted_finish_time,
                "safe_finish_time": candidate.safe_finish_time,
                "safe_slack": candidate.safe_slack,
                "arrival_deviation": candidate.arrival_deviation,
                "feasibility_class": candidate.feasibility_class,
                "selected_by_shadow": int(candidate.selected_by_shadow),
                "rejection_reason": candidate.rejection_reason,
            })
        candidates = recommendation.candidates
        self._all_group_recommendation_trace_buffer.append({
            "virtual_time": self._virtual_now, "client_id": client_id,
            "current_group_id": recommendation.current_group_id,
            "current_group_safe": int(recommendation.current_group_safe),
            "num_unexpired_groups": len(candidates),
            "num_fedcompass_feasible_groups": sum(c.fedcompass_q_feasible for c in candidates),
            "num_state_safe_groups": sum(c.state_safe_feasible for c in candidates),
            "num_state_only_groups": sum(c.feasibility_class == "state_only" for c in candidates),
            "recommended_group_id": recommendation.recommended_group_id,
            "recommended_q": recommendation.recommended_q,
            "group_changed": int(recommendation.group_changed),
            "mismatch_repaired": int(recommendation.mismatch_repaired),
            "shadow_action": recommendation.shadow_action,
            "reason": recommendation.reason,
        })

    def pop_all_group_feasibility_traces(self) -> List[Dict[str, Any]]:
        rows = list(self._all_group_feasibility_trace_buffer)
        self._all_group_feasibility_trace_buffer.clear()
        return rows

    def pop_all_group_recommendation_traces(self) -> List[Dict[str, Any]]:
        rows = list(self._all_group_recommendation_trace_buffer)
        self._all_group_recommendation_trace_buffer.clear()
        return rows

    def _record_group_creation_counterfactual(self, client_id: str, result) -> None:
        """记录原始新建组反事实；该行不代表真实执行了create_group。"""
        self._group_creation_counterfactual_trace_buffer.append({
            "virtual_time": self._virtual_now, "client_id": client_id,
            "current_group_id": result.current_group_id,
            "current_q": result.current_q,
            "current_expected_arrival_time": result.current_expected_arrival_time,
            "current_latest_arrival_time": result.current_latest_arrival_time,
            "current_predicted_finish_time": result.current_predicted_finish_time,
            "current_safe_finish_time": result.current_safe_finish_time,
            "current_safe_slack": result.current_safe_slack,
            "counterfactual_q": result.counterfactual_q,
            "counterfactual_expected_arrival_time": result.counterfactual_expected_arrival_time,
            "counterfactual_latest_arrival_time": result.counterfactual_latest_arrival_time,
            "counterfactual_predicted_finish_time": result.counterfactual_predicted_finish_time,
            "counterfactual_safe_finish_time": result.counterfactual_safe_finish_time,
            "counterfactual_safe_slack": result.counterfactual_safe_slack,
            "counterfactual_safe_feasible": int(result.counterfactual_safe_feasible),
            "safe_slack_improvement": result.safe_slack_improvement,
            "expected_delay_increase": result.expected_delay_increase,
            "q_difference": result.q_difference,
            "shadow_action": result.shadow_action, "reason": result.reason,
        })

    def pop_group_creation_counterfactual_traces(self) -> List[Dict[str, Any]]:
        rows = list(self._group_creation_counterfactual_trace_buffer)
        self._group_creation_counterfactual_trace_buffer.clear()
        return rows

    def _record_state_group_creation_q(self, client_id: str, original, result) -> None:
        """记录新组Q枚举事实；推荐Q不会写入client_info或Dispatch。"""
        self._state_group_creation_q_trace_buffer.append({
            "virtual_time": self._virtual_now, "client_id": client_id,
            "current_group_id": original.current_group_id,
            "current_safe_slack": original.current_safe_slack,
            "original_new_group_q": result.original_q,
            "original_new_group_safe_slack": original.counterfactual_safe_slack,
            "original_new_group_safe_feasible": int(result.original_safe_feasible),
            "num_safe_q_candidates": result.num_safe_q_candidates,
            "min_safe_q": result.min_safe_q, "max_safe_q": result.max_safe_q,
            "recommended_q": result.recommended_q,
            "recommended_expected_arrival_time": result.recommended_expected_arrival_time,
            "recommended_latest_arrival_time": result.recommended_latest_arrival_time,
            "recommended_predicted_finish_time": result.recommended_predicted_finish_time,
            "recommended_safe_finish_time": result.recommended_safe_finish_time,
            "recommended_safe_slack": result.recommended_safe_slack,
            "q_difference": result.q_difference,
            "work_retention_ratio": result.work_retention_ratio,
            "expected_delay_difference": result.expected_delay_difference,
            "shadow_action": result.shadow_action, "reason": result.reason,
        })

    def pop_state_group_creation_q_traces(self) -> List[Dict[str, Any]]:
        rows = list(self._state_group_creation_q_trace_buffer)
        self._state_group_creation_q_trace_buffer.clear()
        return rows

    def _record_state_group_window(self, client_id: str, original, result) -> None:
        """记录固定Q时间窗对照；状态时间窗不会写入真实arrival_group。"""
        self._state_group_window_trace_buffer.append({
            "virtual_time": self._virtual_now, "client_id": client_id,
            "current_group_id": original.current_group_id,
            "current_safe_slack": original.current_safe_slack,
            "fixed_q": result.fixed_q,
            "speed_expected_arrival_time": result.speed_expected_arrival_time,
            "speed_latest_arrival_time": result.speed_latest_arrival_time,
            "speed_safe_slack": result.speed_safe_slack,
            "state_expected_arrival_time": result.state_expected_arrival_time,
            "state_latest_arrival_time": result.state_latest_arrival_time,
            "state_predicted_finish_time": result.state_predicted_finish_time,
            "state_safe_finish_time": result.state_safe_finish_time,
            "state_safe_slack": result.state_safe_slack,
            "predicted_duration": result.predicted_duration,
            "uncertainty": result.uncertainty,
            "predicted_compute_duration": result.predicted_compute_duration,
            "predicted_communication_duration": (
                result.predicted_communication_duration
            ),
            "predicted_spike_duration": result.predicted_spike_duration,
            "predicted_availability_duration": (
                result.predicted_availability_duration
            ),
            "predicted_availability_risk_duration": (
                result.predicted_availability_risk_duration
            ),
            "availability_event_rate": result.availability_event_rate,
            "availability_event_count": result.availability_event_count,
            "predictor_used_fallback": int(result.predictor_used_fallback),
            "predictor_num_reports": result.predictor_num_reports,
            "expected_shift": result.expected_shift,
            "latest_shift": result.latest_shift,
            "safe_slack_improvement": result.safe_slack_improvement,
            "state_window_safe_feasible": int(result.state_window_safe_feasible),
            "shadow_action": result.shadow_action, "reason": result.reason,
        })

    def pop_state_group_window_traces(self) -> List[Dict[str, Any]]:
        rows = list(self._state_group_window_trace_buffer)
        self._state_group_window_trace_buffer.clear()
        return rows

    def _record_state_window_admission(
        self, *, client_id: str, assigned_group: int, current_safe_slack: float,
        all_groups, state_window, decision, applied: bool,
    ) -> None:
        """记录组合Gate建议；applied_action固定为keep，证明Shadow无写入。"""
        self._state_window_admission_trace_buffer.append({
            "virtual_time": self._virtual_now, "client_id": client_id,
            "current_group_id": assigned_group,
            "current_group_safe": int(decision.current_group_safe),
            "current_safe_slack": current_safe_slack,
            "other_existing_group_safe": int(decision.other_existing_group_safe),
            "num_other_safe_groups": sum(
                c.state_safe_feasible and not c.is_current_group
                for c in (all_groups.candidates if all_groups is not None else ())
            ),
            "state_new_group_safe": int(decision.state_new_group_safe),
            "state_new_group_q": state_window.fixed_q if state_window else -1,
            "state_new_group_expected_arrival_time": (
                state_window.state_expected_arrival_time if state_window else ""
            ),
            "state_new_group_latest_arrival_time": (
                state_window.state_latest_arrival_time if state_window else ""
            ),
            "state_new_group_safe_slack": (
                state_window.state_safe_slack if state_window else ""
            ),
            "shadow_action": decision.action,
            "applied_action": (
                "create_state_window_group" if applied
                else "keep_trust_q_v2_schedule"
            ),
            "reason": decision.reason,
        })

    def pop_state_window_admission_traces(self) -> List[Dict[str, Any]]:
        rows = list(self._state_window_admission_trace_buffer)
        self._state_window_admission_trace_buffer.clear()
        return rows

    def _record_communication_tail_risk(
        self, client_id: str, current_group_id: int, state_window, result,
    ) -> None:
        """记录通信尾部风险Shadow；不修改状态时间窗或Apply动作。"""
        self._communication_tail_risk_trace_buffer.append({
            "virtual_time": self._virtual_now,
            "client_id": client_id,
            "current_group_id": current_group_id,
            "state_new_group_q": state_window.fixed_q,
            "num_reports": result.num_reports,
            "communication_mean": result.communication_mean,
            "communication_std": result.communication_std,
            "communication_p90": result.communication_p90,
            "communication_recent_max": result.communication_recent_max,
            "existing_uncertainty": result.existing_uncertainty,
            "p90_tail_excess": result.p90_tail_excess,
            "max_tail_excess": result.max_tail_excess,
            "incremental_p90_margin": result.incremental_p90_margin,
            "incremental_max_margin": result.incremental_max_margin,
            "original_safe_slack": result.original_safe_slack,
            "p90_calibrated_safe_slack": result.p90_calibrated_safe_slack,
            "max_calibrated_safe_slack": result.max_calibrated_safe_slack,
            "p90_safe_feasible": int(result.p90_safe_feasible),
            "max_safe_feasible": int(result.max_safe_feasible),
            "shadow_action": result.shadow_action,
            "reason": result.reason,
        })

    def pop_communication_tail_risk_traces(self) -> List[Dict[str, Any]]:
        rows = list(self._communication_tail_risk_trace_buffer)
        self._communication_tail_risk_trace_buffer.clear()
        return rows

    def _record_communication_robust_q(
        self, client_id: str, current_group_id: int, state_window, result,
    ) -> None:
        """记录通信稳健Q Shadow；推荐Q绝不写回真实Dispatch。"""
        self._communication_robust_q_trace_buffer.append({
            "virtual_time": self._virtual_now,
            "client_id": client_id,
            "current_group_id": current_group_id,
            "group_latest_time": state_window.state_latest_arrival_time,
            "original_q": result.original_q,
            "recommended_q": result.recommended_q,
            "q_reduction": result.q_reduction,
            "work_retention_ratio": result.work_retention_ratio,
            "communication_std": result.communication_std,
            "risk_beta": result.risk_beta,
            "target_communication_risk": result.target_communication_risk,
            "existing_uncertainty": result.existing_uncertainty,
            "incremental_communication_reserve": (
                result.incremental_communication_reserve
            ),
            "original_robust_slack": result.original_robust_slack,
            "recommended_predicted_finish_time": (
                result.recommended_predicted_finish_time
            ),
            "recommended_safe_finish_time": result.recommended_safe_finish_time,
            "recommended_robust_finish_time": result.recommended_robust_finish_time,
            "recommended_robust_slack": result.recommended_robust_slack,
            "robust_safe_feasible": int(result.robust_safe_feasible),
            "num_q_candidates": result.num_q_candidates,
            "shadow_action": result.shadow_action,
            "reason": result.reason,
        })

    def pop_communication_robust_q_traces(self) -> List[Dict[str, Any]]:
        rows = list(self._communication_robust_q_trace_buffer)
        self._communication_robust_q_trace_buffer.clear()
        return rows


def _applied_reason(
    safe_feasible: bool,
    raw_q: int,
    applied_q: int,
    max_increase_q: int,
    availability_guard: bool,
) -> str:
    """生成可解释的Q裁剪原因，便于实验逐轮排查。"""
    if not safe_feasible:
        return "group_mismatch_keep_fedcompass"
    if availability_guard and applied_q < raw_q:
        return "availability_guard_no_increase"
    if raw_q > max_increase_q:
        return "increase_trust_region_clipped"
    return "raw_state_q_applied"
