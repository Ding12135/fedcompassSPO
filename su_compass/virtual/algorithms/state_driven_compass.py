"""State-native FedCompass scheduler with modular ablation modes."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from su_compass.scheduling.fedcompass_reference import (
    existing_group_reference,
    new_group_reference_q,
    new_group_reference_window,
)
from su_compass.scheduling.policies.joint_group_q import JointGroupQPolicy
from su_compass.scheduling.policies.lyapunov_group_q import (
    LyapunovAction,
    LyapunovGroupQPolicy,
)
from su_compass.scheduling.predictors.regime_calibrated import RegimeCalibratedPredictor
from su_compass.scheduling.state_time_model import StateTimeModel, state_group_window
from su_compass.scheduling.state_driven_config import StateDrivenConfig
from su_compass.virtual.algorithms.fedcompass import VirtualFedCompassController
from su_compass.virtual.event import EventType, VirtualEvent


class VirtualStateDrivenCompassController(VirtualFedCompassController):
    def __init__(self, *args, state_driven_config: StateDrivenConfig, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state_driven_config = state_driven_config
        self._state_time_model = StateTimeModel()
        self._joint_policy = JointGroupQPolicy(
            state_driven_config.target_band_ratio,
            state_driven_config.alignment_equivalence_band,
        )
        self._runtime_states: Dict[str, Any] = {}
        self._state_time_trace_buffer: List[dict] = []
        self._joint_trace_buffer: List[dict] = []
        self._candidate_trace_buffer: List[dict] = []
        self._state_group_creation_trace_buffer: List[dict] = []
        self._calibrated_shadow = RegimeCalibratedPredictor(
            target_coverage=state_driven_config.calibrated_shadow_target_coverage,
        )
        self._calibrated_shadow_trace_buffer: List[dict] = []
        self._native_group_shadow_trace_buffer: List[dict] = []
        self._pending_calibrated_shadow: Dict[str, dict] = {}
        self._pending_native_group_shadow: Dict[str, dict] = {}
        self._lyapunov_policy = LyapunovGroupQPolicy(
            rhythm_target=state_driven_config.lyapunov_rhythm_target,
            tradeoff_v=state_driven_config.lyapunov_v,
            max_holding_wait=state_driven_config.lyapunov_max_holding_wait,
            q_trust_eta=state_driven_config.lyapunov_q_trust_eta,
            create_penalty=state_driven_config.lyapunov_create_penalty,
            enable_rhythm_queue=state_driven_config.lyapunov_enable_rhythm_queue,
            enable_workload_queue=state_driven_config.lyapunov_enable_workload_queue,
        )
        self._lyapunov_rhythm_debt = 0.0
        self._lyapunov_workload_debt: Dict[str, float] = {}
        self._lyapunov_target_rates = self._parse_target_rates(
            state_driven_config.lyapunov_client_target_rates
        )
        self._lyapunov_last_aggregation_time = 0.0
        self._lyapunov_decision_trace_buffer: List[dict] = []
        self._lyapunov_queue_trace_buffer: List[dict] = []

    @staticmethod
    def _parse_target_rates(spec: str) -> Dict[str, float]:
        rates: Dict[str, float] = {}
        if not spec.strip():
            return rates
        for item in spec.split(","):
            client_id, value = item.split("=", 1)
            rate = float(value)
            if rate < 0:
                raise ValueError("Lyapunov client target rates must be non-negative")
            rates[client_id.strip()] = rate
        return rates

    @property
    def algorithm_name(self) -> str:
        return "state_driven_compass"

    @property
    def predictor_name(self) -> str:
        return self._state_time_model.name

    @property
    def predictor_version(self) -> str:
        return self._state_time_model.version

    def on_client_upload(self, event: VirtualEvent, virtual_now: float):
        self._close_shadow_outcomes(event)
        state = event.payload.get("runtime_state")
        if state is not None:
            self._runtime_states[event.client_id] = state
        return super().on_client_upload(event, virtual_now)

    @staticmethod
    def _pinball(actual: float, quantile_prediction: float, quantile: float) -> float:
        residual = actual - quantile_prediction
        return quantile * residual if residual >= 0.0 else (quantile - 1.0) * residual

    def _close_shadow_outcomes(self, event: VirtualEvent) -> None:
        report = event.payload.get("report")
        if report is None:
            return
        decision_id = str(event.payload.get("decision_id", ""))
        profile_type = str(event.payload.get("profile_type", ""))
        actual = float(report.round_time)
        calibrated = self._pending_calibrated_shadow.pop(decision_id, None)
        if calibrated is not None:
            target = self.state_driven_config.calibrated_shadow_target_coverage
            baseline_error = abs(actual - calibrated["baseline_duration"])
            shadow_error = abs(actual - calibrated["shadow_duration"])
            baseline_pinball = self._pinball(actual, calibrated["baseline_safe_duration"], target)
            shadow_pinball = self._pinball(actual, calibrated["shadow_safe_duration"], target)
            calibrated.update({
                "profile_type": profile_type,
                "actual_duration": actual,
                "baseline_abs_error": baseline_error,
                "shadow_abs_error": shadow_error,
                "baseline_safe_hit": int(actual <= calibrated["baseline_safe_duration"]),
                "shadow_safe_hit": int(actual <= calibrated["shadow_safe_duration"]),
                "baseline_pinball": baseline_pinball,
                "shadow_pinball": shadow_pinball,
                "point_prediction_better": int(shadow_error < baseline_error),
                "safe_prediction_better": int(shadow_pinball < baseline_pinball),
            })
            self._calibrated_shadow_trace_buffer.append(calibrated)

        native = self._pending_native_group_shadow.pop(decision_id, None)
        if native is not None:
            applied_q = max(1, int(native["applied_q"]))
            native_q = max(1, int(native["native_q"]))
            compute = float(report.train_time or 0.0) / applied_q * native_q
            fixed = float(report.communication_time)
            fixed += float(report.spike_delay or 0.0) + float(report.availability_wait or 0.0)
            counterfactual_actual = compute + fixed
            applied_error = abs(actual - float(native["applied_predicted_duration"]))
            native_error = abs(counterfactual_actual - float(native["native_predicted_duration"]))
            native.update({
                "profile_type": profile_type,
                "actual_applied_duration": actual,
                "counterfactual_actual_duration": counterfactual_actual,
                "applied_abs_error": applied_error,
                "native_abs_error": native_error,
                "native_safe_hit": int(
                    counterfactual_actual <= float(native["native_safe_duration"])
                ),
                "native_prediction_better": int(native_error < applied_error),
                "native_reduces_qmax": int(
                    int(native["applied_q"]) == self.max_local_steps
                    and native_q < self.max_local_steps
                ),
            })
            self._native_group_shadow_trace_buffer.append(native)

        self._calibrated_shadow.observe(
            client_id=event.client_id,
            local_steps=int(report.local_steps), actual_duration=actual,
            compute_duration=float(report.train_time or 0.0),
            communication_duration=float(report.communication_time),
            spike_duration=float(report.spike_delay or 0.0),
            availability_duration=float(report.availability_wait or 0.0),
        )

    def _append_dispatch(
        self, client_id: str, virtual_now: float, local_steps_assigned: int,
        target_arrival, latest_arrival,
    ) -> None:
        if self.state_driven_config.calibrated_predictor_shadow:
            point = self._state_time_model.predict_q(
                client_id=client_id, dispatch_time=virtual_now,
                q=local_steps_assigned,
                runtime_state=self._runtime_states.get(client_id),
                speed_fallback=float(self.client_info[client_id]["speed"]),
            )
            shadow = self._calibrated_shadow.predict(
                client_id=client_id, local_steps=local_steps_assigned,
                baseline_duration=point.predicted_duration,
                baseline_safe_duration=point.safe_duration,
            )
            decision_id = self._decision_id(client_id)
            self._pending_calibrated_shadow[decision_id] = {
                "decision_id": decision_id,
                "client_id": client_id,
                "dispatch_time": virtual_now,
                "q": local_steps_assigned,
                "predictor_source": point.predictor_source,
                "num_reports": point.num_reports,
                "baseline_duration": point.predicted_duration,
                "baseline_safe_duration": point.safe_duration,
                "shadow_duration": shadow.predicted_duration,
                "shadow_safe_duration": shadow.safe_duration,
                "shadow_conformal_margin": shadow.conformal_margin,
                "shadow_used_candidate": int(shadow.used_candidate),
                "shadow_burst_probability": shadow.burst_probability,
            }
        return super()._append_dispatch(
            client_id, virtual_now, local_steps_assigned,
            target_arrival, latest_arrival,
        )

    def _decision_id(self, client_id: str) -> str:
        return str(self.client_info.get(client_id, {}).get("decision_id", ""))

    def _curve(self, client_id: str):
        return self._state_time_model.predict_curve(
            client_id=client_id, dispatch_time=self._virtual_now,
            qs=range(self.min_local_steps, self.max_local_steps + 1),
            runtime_state=self._runtime_states.get(client_id),
            speed_fallback=float(self.client_info[client_id]["speed"]),
        )

    def _evaluate_state_join(self, client_id: str):
        curve, monotonic = self._curve(client_id)
        candidates = self._joint_policy.enumerate_candidates(
            now=self._virtual_now, groups=self.arrival_group, curve=curve,
        )
        decision = self._joint_policy.choose(candidates)
        return curve, candidates, decision, monotonic

    def _record_state_time_points(self, client_id: str, curve, selected_q: int, fed_q: int, monotonic: bool) -> None:
        level = self.state_driven_config.candidate_trace_level
        if level == "none":
            return
        keep = {self.min_local_steps, self.max_local_steps, selected_q, fed_q}
        for point in curve:
            if level != "full" and point.q not in keep:
                continue
            row = point.to_trace()
            row.update({
                "decision_id": self._decision_id(client_id),
                "virtual_time": self._virtual_now,
                "client_id": client_id,
                "curve_monotonic": int(monotonic),
                "is_fedcompass_q": int(point.q == fed_q),
                "is_state_selected_q": int(point.q == selected_q),
                "is_qmin": int(point.q == self.min_local_steps),
                "is_qmax": int(point.q == self.max_local_steps),
            })
            self._state_time_trace_buffer.append(row)

    def _record_single_state_time(self, client_id: str, point, *, selected: bool, fed_q: int) -> None:
        if self.state_driven_config.candidate_trace_level == "none":
            return
        row = point.to_trace()
        row.update({
            "decision_id": self._decision_id(client_id),
            "virtual_time": self._virtual_now, "client_id": client_id,
            "curve_monotonic": 1, "is_fedcompass_q": int(point.q == fed_q),
            "is_state_selected_q": int(selected),
            "is_qmin": int(point.q == self.min_local_steps),
            "is_qmax": int(point.q == self.max_local_steps),
        })
        self._state_time_trace_buffer.append(row)

    def _record_joint(
        self, client_id: str, fed, decision, candidates, applied_group: int,
        applied_q: int, applied: bool, *, monotonic: bool,
        fallback_to_fedcompass: bool = False,
    ) -> None:
        fed_candidate = next((c for c in candidates if c.group_id == fed.group_id and c.q == fed.q), None)
        state_candidate = next((c for c in candidates if c.group_id == decision.group_id and c.q == decision.q), None)
        safe = [c for c in candidates if c.deadline_safe]
        aligned = [c for c in safe if c.target_aligned]
        row = {
            "decision_id": self._decision_id(client_id),
            "virtual_time": self._virtual_now,
            "client_id": client_id,
            "mode": self.state_driven_config.existing_group_mode,
            "applied": int(applied),
            "num_active_groups": len(self.arrival_group),
            "num_total_candidates": len(candidates),
            "num_deadline_safe_candidates": len(safe),
            "num_target_aligned_candidates": len(aligned),
            "fedcompass_group_id": fed.group_id,
            "fedcompass_q": fed.q,
            "fedcompass_predicted_finish": fed_candidate.predicted_finish_time if fed_candidate else "",
            "fedcompass_safe_finish": fed_candidate.safe_finish_time if fed_candidate else "",
            "fedcompass_state_safe": int(fed_candidate.deadline_safe) if fed_candidate else 0,
            "fedcompass_alignment_error": fed_candidate.alignment_error if fed_candidate else "",
            "state_group_id": decision.group_id,
            "state_q": decision.q,
            "state_predicted_finish": decision.predicted_finish_time if decision.feasible else "",
            "state_safe_finish": decision.safe_finish_time if decision.feasible else "",
            "state_safe": int(decision.feasible),
            "state_alignment_error": decision.alignment_error if decision.feasible else "",
            "state_safe_slack": decision.safe_slack if decision.feasible else "",
            "state_group_size_before": decision.group_size if decision.feasible else 0,
            "applied_group_id": applied_group,
            "applied_q": applied_q,
            "group_changed": int(applied_group != fed.group_id),
            "q_changed": int(applied_q != fed.q),
            "state_only_feasible_group_found": int(
                decision.feasible and decision.group_id != fed.group_id
            ),
            "all_existing_groups_infeasible": int(not decision.feasible),
            "selection_reason": decision.reason,
            "predictor_source": state_candidate.predictor_source if state_candidate else "",
            "num_reports": state_candidate.num_reports if state_candidate else 0,
            "curve_monotonic": int(monotonic),
            "state_control_active": int(applied and not fallback_to_fedcompass),
            "fallback_to_fedcompass": int(fallback_to_fedcompass),
        }
        self._joint_trace_buffer.append(row)
        if self.state_driven_config.candidate_trace_level == "full":
            for candidate in candidates:
                item = candidate.to_trace()
                item.update({
                    "decision_id": self._decision_id(client_id),
                    "client_id": client_id,
                    "selected_by_state": int(
                        decision.feasible and candidate.group_id == decision.group_id
                        and candidate.q == decision.q
                    ),
                })
                self._candidate_trace_buffer.append(item)

    def _join_group(self, client_id: str) -> bool:
        cfg = self.state_driven_config
        if cfg.existing_group_mode == "fedcompass":
            return super()._join_group(client_id)

        speed = float(self.client_info[client_id]["speed"])
        fed = existing_group_reference(
            now=self._virtual_now, speed=speed, groups=self.arrival_group,
            qmin=self.min_local_steps, qmax=self.max_local_steps,
        )
        curve, candidates, decision, monotonic = self._evaluate_state_join(client_id)

        # A fallback curve is a FedCompass reference, not a state prediction.
        # Never feed Q*speed values through the formal state decision path.
        # Until the predictor is ready, execute the parent policy explicitly.
        curve_uses_fallback = bool(curve) and all(point.used_fallback for point in curve)

        if cfg.existing_group_mode == "state_shadow":
            joined = super()._join_group(client_id)
            applied_group = int(self.client_info[client_id].get("goa", -1)) if joined else -1
            applied_q = int(self.client_info[client_id].get("local_steps", -1)) if joined else -1
            self._record_joint(
                client_id, fed, decision, candidates, applied_group, applied_q,
                False, monotonic=monotonic,
                fallback_to_fedcompass=curve_uses_fallback,
            )
            self._record_state_time_points(client_id, curve, decision.q, fed.q, monotonic)
            return joined

        if curve_uses_fallback:
            joined = super()._join_group(client_id)
            applied_group = int(self.client_info[client_id].get("goa", -1)) if joined else -1
            applied_q = int(self.client_info[client_id].get("local_steps", -1)) if joined else -1
            self._record_joint(
                client_id, fed, decision, candidates, applied_group, applied_q,
                False, monotonic=monotonic, fallback_to_fedcompass=True,
            )
            self._record_state_time_points(client_id, curve, -1, fed.q, monotonic)
            return joined

        if not decision.feasible:
            self._record_joint(
                client_id, fed, decision, candidates, -1, -1, True,
                monotonic=monotonic,
            )
            self._record_state_time_points(client_id, curve, -1, fed.q, monotonic)
            return False

        group = self.arrival_group[decision.group_id]
        group["clients"].append(client_id)
        self.client_info[client_id]["goa"] = decision.group_id
        self.client_info[client_id]["local_steps"] = decision.q
        self.client_info[client_id]["start_time"] = self._virtual_now
        self._record_dispatch_decision(
            client_id=client_id, decision="state_join_group",
            assigned_group=decision.group_id, assigned_steps=decision.q,
            speed_raw=speed,
            remaining_time=group["expected_arrival_time"] - self._virtual_now,
            target_arrival=group["expected_arrival_time"],
            latest_arrival=group["latest_arrival_time"],
        )
        self._record_joint(
            client_id, fed, decision, candidates,
            decision.group_id, decision.q, True, monotonic=monotonic,
        )
        self._record_state_time_points(client_id, curve, decision.q, fed.q, monotonic)
        return True

    def _assign_group(self, client_id: str) -> List[VirtualEvent]:
        self._prepare_next_decision(client_id)
        if self.state_driven_config.lyapunov_mode != "off":
            return self._assign_group_lyapunov(client_id)
        return self._assign_group_safe_reuse(client_id)

    def _assign_group_safe_reuse(self, client_id: str) -> List[VirtualEvent]:
        cfg = self.state_driven_config
        if not self.arrival_group:
            if cfg.new_group_window_mode == "state_shadow_fixed_q":
                self._record_state_group_shadow(client_id, self.max_local_steps)
                return self._create_fedcompass_first_group(client_id)
            if cfg.new_group_window_mode in {"state_apply_fixed_q", "state_apply"}:
                q = self.max_local_steps
                return self._create_state_group(client_id, q)
            return self._create_fedcompass_first_group(client_id)
        if self._join_group(client_id):
            return []
        return self._create_group(client_id)

    def _lyapunov_actions(self, client_id: str):
        curve, monotonic = self._curve(client_id)
        speed = float(self.client_info[client_id]["speed"])
        fed_join = existing_group_reference(
            now=self._virtual_now, speed=speed, groups=self.arrival_group,
            qmin=self.min_local_steps, qmax=self.max_local_steps,
        )
        joint = self._joint_policy.enumerate_candidates(
            now=self._virtual_now, groups=self.arrival_group, curve=curve,
        )
        actions = []
        for candidate in joint:
            frontier = candidate.expected_arrival_time
            holding = max(0.0, frontier - candidate.predicted_finish_time)
            external = max(0.0, candidate.predicted_finish_time - frontier)
            actions.append(LyapunovAction(
                mode="join", group_id=candidate.group_id, q=candidate.q,
                predicted_finish_time=candidate.predicted_finish_time,
                safe_finish_time=candidate.safe_finish_time,
                group_frontier_time=frontier,
                latest_arrival_time=candidate.latest_arrival_time,
                deadline_safe=candidate.deadline_safe,
                holding_wait=holding, external_wait=external,
                predicted_sojourn=max(frontier, candidate.predicted_finish_time) - self._virtual_now,
                effective_work=candidate.q / self.max_local_steps,
                utility=math.log1p(candidate.q / max(self.min_local_steps, 1)),
            ))
        fed_new_q = new_group_reference_q(
            now=self._virtual_now, client_id=client_id, speed=speed,
            groups=self.arrival_group, client_info=self.client_info,
            qmin=self.min_local_steps, qmax=self.max_local_steps,
        ) if self.arrival_group else self.max_local_steps
        create_point = next((point for point in curve if point.q == fed_new_q), curve[-1])
        actions.append(LyapunovAction(
            mode="create", group_id=-1, q=fed_new_q,
            predicted_finish_time=create_point.predicted_finish_time,
            safe_finish_time=create_point.safe_finish_time,
            group_frontier_time=create_point.predicted_finish_time,
            latest_arrival_time=create_point.safe_finish_time,
            deadline_safe=True, holding_wait=0.0, external_wait=0.0,
            predicted_sojourn=create_point.predicted_duration,
            effective_work=fed_new_q / self.max_local_steps,
            utility=math.log1p(fed_new_q / max(self.min_local_steps, 1)),
        ))
        scored = self._lyapunov_policy.score(
            actions,
            rhythm_debt=self._lyapunov_rhythm_debt,
            workload_debt=self._lyapunov_workload_debt.get(client_id, 0.0),
            qmax=self.max_local_steps, qmin=self.min_local_steps,
            fedcompass_join_q=fed_join.q if fed_join.feasible else -1,
        )
        return curve, monotonic, fed_join, fed_new_q, scored, self._lyapunov_policy.choose(scored)

    def _record_lyapunov_decision(
        self, client_id: str, scored, decision, *, applied_mode: str,
        applied_group: int, applied_q: int,
    ) -> None:
        selected = decision.action if decision.feasible else None
        legal = [action for action in scored if action.legal]
        rejected_holding = sum(
            action.rejection_reason == "holding_wait_exceeds_cap" for action in scored
        )
        rejected_q = sum(action.rejection_reason == "q_exceeds_trust_region" for action in scored)
        self._lyapunov_decision_trace_buffer.append({
            "decision_id": self._decision_id(client_id),
            "virtual_time": self._virtual_now,
            "client_id": client_id,
            "mode": self.state_driven_config.lyapunov_mode,
            "rhythm_debt": self._lyapunov_rhythm_debt,
            "workload_debt": self._lyapunov_workload_debt.get(client_id, 0.0),
            "num_actions": len(scored), "num_legal_actions": len(legal),
            "rejected_holding_actions": rejected_holding,
            "rejected_q_actions": rejected_q,
            "recommended_mode": selected.mode if selected else "",
            "recommended_group_id": selected.group_id if selected else -1,
            "recommended_q": selected.q if selected else -1,
            "recommended_score": selected.score if selected else "",
            "recommended_sojourn": selected.predicted_sojourn if selected else "",
            "recommended_holding_wait": selected.holding_wait if selected else "",
            "recommended_external_wait": selected.external_wait if selected else "",
            "applied_mode": applied_mode, "applied_group_id": applied_group,
            "applied_q": applied_q,
            "recommendation_applied": int(
                selected is not None and selected.mode == applied_mode
                and selected.q == applied_q
                and (selected.mode == "create" or selected.group_id == applied_group)
            ),
        })

    def _apply_lyapunov_join(self, client_id: str, action: LyapunovAction) -> None:
        group = self.arrival_group[action.group_id]
        group["clients"].append(client_id)
        self.client_info[client_id]["goa"] = action.group_id
        self.client_info[client_id]["local_steps"] = action.q
        self.client_info[client_id]["start_time"] = self._virtual_now
        self._record_dispatch_decision(
            client_id=client_id, decision="lyapunov_join_group",
            assigned_group=action.group_id, assigned_steps=action.q,
            speed_raw=float(self.client_info[client_id]["speed"]),
            remaining_time=group["expected_arrival_time"] - self._virtual_now,
            target_arrival=group["expected_arrival_time"],
            latest_arrival=group["latest_arrival_time"],
        )

    def _assign_group_lyapunov(self, client_id: str) -> List[VirtualEvent]:
        curve, monotonic, fed_join, fed_new_q, scored, decision = self._lyapunov_actions(client_id)
        fallback = bool(curve) and all(point.used_fallback for point in curve)
        if fallback or not decision.feasible:
            events = self._assign_group_safe_reuse(client_id)
            group = int(self.client_info[client_id].get("goa", -1))
            q = int(self.client_info[client_id].get("local_steps", -1))
            self._record_lyapunov_decision(
                client_id, scored, decision, applied_mode="fallback",
                applied_group=group, applied_q=q,
            )
            return events

        if self.state_driven_config.lyapunov_mode == "shadow":
            events = self._assign_group_safe_reuse(client_id)
            group = int(self.client_info[client_id].get("goa", -1))
            q = int(self.client_info[client_id].get("local_steps", -1))
            applied_mode = "join" if not events else "create"
            self._record_lyapunov_decision(
                client_id, scored, decision, applied_mode=applied_mode,
                applied_group=group, applied_q=q,
            )
            return events

        action = decision.action
        assert action is not None
        if action.mode == "join":
            self._apply_lyapunov_join(client_id, action)
            events: List[VirtualEvent] = []
            group = action.group_id
        else:
            events = self._create_state_group(
                client_id, action.q, fed_reference_q=fed_new_q,
            )
            group = int(self.client_info[client_id]["goa"])
        self._record_state_time_points(
            client_id, curve, action.q, fed_join.q, monotonic,
        )
        self._record_lyapunov_decision(
            client_id, scored, decision, applied_mode=action.mode,
            applied_group=group, applied_q=action.q,
        )
        return events

    def _record_aggregation_trace(
        self, trigger, participating, local_steps, model_versions,
        decision_ids, group_id, budget_delta, ts_before,
    ) -> None:
        super()._record_aggregation_trace(
            trigger, participating, local_steps, model_versions,
            decision_ids, group_id, budget_delta, ts_before,
        )
        if self.state_driven_config.lyapunov_mode == "off":
            return
        delta_t = max(0.0, self._virtual_now - self._lyapunov_last_aggregation_time)
        before_h = self._lyapunov_rhythm_debt
        if self.state_driven_config.lyapunov_enable_rhythm_queue:
            self._lyapunov_rhythm_debt = max(
                0.0, before_h + delta_t - self.state_driven_config.lyapunov_rhythm_target
            )
        for client_id in getattr(self, "_client_ids", []):
            before_z = self._lyapunov_workload_debt.get(client_id, 0.0)
            target = self._lyapunov_target_rates.get(client_id, 0.0) * delta_t
            service = 0.0
            if client_id in local_steps:
                service = (
                    float(local_steps[client_id]) / self.max_local_steps
                    / (1.0 + float(participating[client_id]))
                )
            after_z = max(0.0, before_z + target - service) if (
                self.state_driven_config.lyapunov_enable_workload_queue
            ) else before_z
            self._lyapunov_workload_debt[client_id] = after_z
            self._lyapunov_queue_trace_buffer.append({
                "aggregation_id": ts_before + 1,
                "virtual_time": self._virtual_now, "delta_t": delta_t,
                "client_id": client_id, "rhythm_debt_before": before_h,
                "rhythm_debt_after": self._lyapunov_rhythm_debt,
                "workload_debt_before": before_z,
                "target_workload_arrival": target,
                "effective_work_service": service,
                "workload_debt_after": after_z,
                "participated": int(client_id in local_steps),
            })
        self._lyapunov_last_aggregation_time = self._virtual_now

    def _create_group(self, client_id: str) -> List[VirtualEvent]:
        cfg = self.state_driven_config
        if cfg.new_group_window_mode == "fedcompass":
            return super()._create_group(client_id)
        speed = float(self.client_info[client_id]["speed"])
        fed_q = new_group_reference_q(
            now=self._virtual_now, client_id=client_id, speed=speed,
            groups=self.arrival_group, client_info=self.client_info,
            qmin=self.min_local_steps, qmax=self.max_local_steps,
        )
        if cfg.new_group_window_mode == "state_shadow_fixed_q":
            self._record_state_group_shadow(client_id, fed_q)
            return super()._create_group(client_id)
        q = fed_q if cfg.new_group_window_mode == "state_apply_fixed_q" else self.max_local_steps
        if cfg.predictor_native_new_group_shadow:
            self._record_predictor_native_group_shadow(
                client_id=client_id, applied_q=q, fed_reference_q=fed_q,
            )
        return self._create_state_group(client_id, q, fed_reference_q=fed_q)

    def _record_predictor_native_group_shadow(
        self, *, client_id: str, applied_q: int, fed_reference_q: int,
    ) -> None:
        plans = []
        for group_id, group in self.arrival_group.items():
            if self._virtual_now >= float(group["latest_arrival_time"]):
                continue
            group_clients = list(group.get("clients", [])) + list(
                group.get("arrived_clients", [])
            )
            fastest = None
            for member in group_clients:
                if member not in self.client_info:
                    continue
                point = self._state_time_model.predict_q(
                    client_id=member,
                    dispatch_time=float(group["latest_arrival_time"]),
                    q=self.max_local_steps,
                    runtime_state=self._runtime_states.get(member),
                    speed_fallback=float(self.client_info[member]["speed"]),
                )
                if fastest is None or point.predicted_duration < fastest.predicted_duration:
                    fastest = point
            if fastest is None:
                continue
            next_target = float(group["latest_arrival_time"]) + fastest.predicted_duration
            plans.append((group_id, next_target, fastest))

        curve, monotonic = self._curve(client_id)
        candidates = []
        for group_id, next_target, fastest in plans:
            legal = [point for point in curve if point.predicted_finish_time <= next_target]
            if legal:
                point = max(legal, key=lambda item: item.q)
                candidates.append((point.q, next_target, group_id, fastest, point))
        if candidates:
            _, next_target, source_group_id, fastest, native_point = max(
                candidates, key=lambda item: (item[0], -item[1], -item[2])
            )
            native_q = native_point.q
            native_expected, native_latest = state_group_window(
                native_point, self.state_driven_config.min_group_slack,
            )
            fastest_source = fastest.predictor_source
        else:
            native_q = self.max_local_steps
            native_point = next(
                (point for point in curve if point.q == native_q), curve[-1]
            )
            native_expected, native_latest = state_group_window(
                native_point, self.state_driven_config.min_group_slack,
            )
            next_target, source_group_id, fastest_source = "", -1, ""

        applied_point = next(
            (point for point in curve if point.q == applied_q),
            self._state_time_model.predict_q(
                client_id=client_id, dispatch_time=self._virtual_now, q=applied_q,
                runtime_state=self._runtime_states.get(client_id),
                speed_fallback=float(self.client_info[client_id]["speed"]),
            ),
        )
        applied_expected, applied_latest = state_group_window(
            applied_point, self.state_driven_config.min_group_slack,
        )
        decision_id = self._decision_id(client_id)
        self._pending_native_group_shadow[decision_id] = {
            "decision_id": decision_id,
            "client_id": client_id,
            "dispatch_time": self._virtual_now,
            "num_active_groups": len(self.arrival_group),
            "source_group_id": source_group_id,
            "next_target_time": next_target,
            "fastest_predictor_source": fastest_source,
            "curve_monotonic": int(monotonic),
            "fedcompass_reference_q": fed_reference_q,
            "applied_q": applied_q,
            "applied_predicted_duration": applied_point.predicted_duration,
            "applied_safe_duration": applied_point.safe_duration,
            "applied_expected_time": applied_expected,
            "applied_latest_time": applied_latest,
            "native_q": native_q,
            "native_predicted_duration": native_point.predicted_duration,
            "native_safe_duration": native_point.safe_duration,
            "native_expected_time": native_expected,
            "native_latest_time": native_latest,
            "native_predictor_source": native_point.predictor_source,
            "native_num_reports": native_point.num_reports,
            "native_used_fallback": int(native_point.used_fallback),
            "native_q_changed": int(native_q != applied_q),
            "native_qmax": int(native_q == self.max_local_steps),
        }

    def _create_fedcompass_first_group(self, client_id: str) -> List[VirtualEvent]:
        # Reuse the parent first-group implementation without recursively
        # invoking this override.
        return VirtualFedCompassController._assign_group(self, client_id)

    def _state_group_plan(self, client_id: str, q: int):
        point = self._state_time_model.predict_q(
            client_id=client_id, dispatch_time=self._virtual_now, q=q,
            runtime_state=self._runtime_states.get(client_id),
            speed_fallback=float(self.client_info[client_id]["speed"]),
        )
        expected, latest = state_group_window(
            point, self.state_driven_config.min_group_slack
        )
        exceeds = latest - expected > self.state_driven_config.max_group_slack
        return point, expected, latest, exceeds

    def _record_state_group_shadow(self, client_id: str, fed_q: int) -> None:
        point, expected, latest, exceeds = self._state_group_plan(client_id, fed_q)
        speed = float(self.client_info[client_id]["speed"])
        fed_expected, fed_latest = new_group_reference_window(
            now=self._virtual_now, q=fed_q, speed=speed,
            latest_time_factor=self.latest_time_factor,
        )
        if point.used_fallback:
            expected, latest, exceeds = fed_expected, fed_latest, False
        self._state_group_creation_trace_buffer.append({
            "decision_id": self._decision_id(client_id), "virtual_time": self._virtual_now,
            "client_id": client_id, "new_group_id": self.group_counter,
            "new_group_window_mode": "state_shadow_fixed_q",
            "new_group_q_mode": "fedcompass", "applied": 0,
            "fedcompass_reference_q": fed_q,
            "fedcompass_expected_time": fed_expected,
            "fedcompass_latest_time": fed_latest,
            "state_assigned_q": fed_q, "state_expected_time": expected,
            "state_latest_time": latest, "state_safe_finish": point.safe_finish_time,
            "state_uncertainty": point.uncertainty,
            "expected_shift": expected - fed_expected,
            "latest_shift": latest - fed_latest,
            "predictor_source": point.predictor_source,
            "num_reports": point.num_reports, "used_fallback": int(point.used_fallback),
            "fallback_reason": point.fallback_reason,
            "safe_window_exceeds_cap": int(exceeds),
        })
        self._record_single_state_time(client_id, point, selected=False, fed_q=fed_q)

    def _create_state_group(self, client_id: str, q: int, fed_reference_q: int | None = None) -> List[VirtualEvent]:
        point, expected, latest, exceeds = self._state_group_plan(client_id, q)
        speed = float(self.client_info[client_id]["speed"])
        ref_q = q if fed_reference_q is None else fed_reference_q
        fed_expected, fed_latest = new_group_reference_window(
            now=self._virtual_now, q=ref_q, speed=speed,
            latest_time_factor=self.latest_time_factor,
        )
        # No preflight/probe is introduced in this work.  When the unified
        # predictor explicitly reports cold-start fallback, preserve the
        # original FedCompass window rather than creating an unrealistically
        # narrow min-slack-only deadline.
        fallback_to_reference_window = point.used_fallback
        if fallback_to_reference_window:
            expected, latest = fed_expected, fed_latest
            exceeds = False
        group_id = self.group_counter
        time_source = (
            "fedcompass_speed" if fallback_to_reference_window
            else "state_fixed_q" if self.state_driven_config.new_group_window_mode == "state_apply_fixed_q"
            else "state_qmax_anchor"
        )
        self.arrival_group[group_id] = {
            "clients": [client_id], "arrived_clients": [],
            "expected_arrival_time": expected, "latest_arrival_time": latest,
            "created_time": self._virtual_now, "time_source": time_source,
            "anchor_client_id": client_id, "anchor_q": q,
        }
        events = []
        if group_id not in self._deadline_events:
            events.append(VirtualEvent(
                time=latest, event_type=EventType.FEDCOMPASS_GROUP_DEADLINE,
                payload={"group_idx": group_id},
            ))
            self._deadline_events.add(group_id)
        self.client_info[client_id]["goa"] = group_id
        self.client_info[client_id]["local_steps"] = q
        self.client_info[client_id]["start_time"] = self._virtual_now
        self._record_dispatch_decision(
            client_id=client_id, decision="state_create_group",
            assigned_group=group_id, assigned_steps=q, speed_raw=speed,
            remaining_time=None, target_arrival=expected, latest_arrival=latest,
        )
        self._state_group_creation_trace_buffer.append({
            "decision_id": self._decision_id(client_id), "virtual_time": self._virtual_now,
            "client_id": client_id, "new_group_id": group_id,
            "new_group_window_mode": self.state_driven_config.new_group_window_mode,
            "new_group_q_mode": self.state_driven_config.new_group_q_mode,
            "applied": 1, "fedcompass_reference_q": ref_q,
            "fedcompass_expected_time": fed_expected, "fedcompass_latest_time": fed_latest,
            "state_assigned_q": q, "state_expected_time": expected,
            "state_latest_time": latest, "state_safe_finish": point.safe_finish_time,
            "state_uncertainty": point.uncertainty,
            "expected_shift": expected - fed_expected, "latest_shift": latest - fed_latest,
            "predictor_source": point.predictor_source, "num_reports": point.num_reports,
            "used_fallback": int(point.used_fallback),
            "fallback_reason": point.fallback_reason,
            "safe_window_exceeds_cap": int(exceeds),
        })
        self._record_single_state_time(client_id, point, selected=True, fed_q=ref_q)
        self.group_counter += 1
        return events

    def pop_state_time_traces(self):
        rows, self._state_time_trace_buffer = self._state_time_trace_buffer, []
        return rows

    def pop_joint_group_q_traces(self):
        rows, self._joint_trace_buffer = self._joint_trace_buffer, []
        return rows

    def pop_joint_group_q_candidate_traces(self):
        rows, self._candidate_trace_buffer = self._candidate_trace_buffer, []
        return rows

    def pop_state_group_creation_traces(self):
        rows, self._state_group_creation_trace_buffer = self._state_group_creation_trace_buffer, []
        return rows

    def pop_calibrated_predictor_shadow_traces(self):
        rows, self._calibrated_shadow_trace_buffer = self._calibrated_shadow_trace_buffer, []
        return rows

    def pop_predictor_native_group_shadow_traces(self):
        rows, self._native_group_shadow_trace_buffer = self._native_group_shadow_trace_buffer, []
        return rows

    def pop_lyapunov_decision_traces(self):
        rows, self._lyapunov_decision_trace_buffer = self._lyapunov_decision_trace_buffer, []
        return rows

    def pop_lyapunov_queue_traces(self):
        rows, self._lyapunov_queue_trace_buffer = self._lyapunov_queue_trace_buffer, []
        return rows
