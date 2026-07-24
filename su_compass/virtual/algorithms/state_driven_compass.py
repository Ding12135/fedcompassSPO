"""State-native FedCompass scheduler with modular ablation modes."""

from __future__ import annotations

import copy
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
    LyapunovDecision,
    LyapunovGroupQPolicy,
    choose_effective_service_v2,
)
from su_compass.scheduling.policies.controlled_q_candidates import (
    controlled_create_qs,
    controlled_join_qs,
)
from su_compass.scheduling.policies.reason_aware_routing import (
    ReasonAwareRoute,
    SlowCause,
    classify_slow_cause,
    recommend_reason_aware_route,
)
from su_compass.scheduling.policies.one_report_structural import (
    predict_one_report_structural,
)
from su_compass.scheduling.policies.fair_contribution_state import (
    FairContributionState,
)
from su_compass.scheduling.policies.unified_batch_dispatch import (
    rank_unified_batch,
)
from su_compass.scheduling.policies.quality_gated_contribution import (
    recommend_contribution_restoration,
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
            finite_sample_pooling=state_driven_config.finite_sample_safety_calibration,
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
            action_scope=state_driven_config.lyapunov_action_scope,
            holding_weight=state_driven_config.lyapunov_holding_weight,
            max_holding_ratio=state_driven_config.lyapunov_max_holding_ratio,
            join_cadence_weight=state_driven_config.lyapunov_join_cadence_weight,
        )
        self._lyapunov_rhythm_debt = 0.0
        self._lyapunov_workload_debt: Dict[str, float] = {}
        self._lyapunov_target_rates = self._parse_target_rates(
            state_driven_config.lyapunov_client_target_rates
        )
        self._lyapunov_q_references = self._parse_q_references(
            state_driven_config.lyapunov_q_reference_spec
        )
        self._lyapunov_last_aggregation_time = 0.0
        self._lyapunov_decision_trace_buffer: List[dict] = []
        self._lyapunov_queue_trace_buffer: List[dict] = []
        self._effective_service_q_trace_buffer: List[dict] = []
        self._effective_service_region_trace_buffer: List[dict] = []
        self._lyapunov_recent_intervals: List[float] = []
        self._lyapunov_recent_group_sizes: List[int] = []
        self._effective_service_shadow_groups: Dict[int, dict] | None = None
        self._effective_service_shadow_next_group_id = -1
        self._effective_service_shadow_outcome_trace_buffer: List[dict] = []
        self._reason_aware_routing_trace_buffer: List[dict] = []
        self._reason_aware_last_service_time: Dict[str, float] = {}
        self._reason_aware_structural_group_windows: Dict[int, tuple[float, float]] = {}
        self._fair_contribution_state: FairContributionState | None = None
        self._fair_contribution_trace_buffer: List[dict] = []
        self._contribution_restoration_trace_buffer: List[dict] = []
        self._micro_hold_trace_buffer: List[dict] = []
        self._unified_batch_trace_buffer: List[dict] = []

    def initialize(self, client_ids: List[str], initial_global_model: Dict) -> None:
        super().initialize(client_ids, initial_global_model)
        if (
            self.state_driven_config.unified_batch_dispatch_mode != "off"
            or self.state_driven_config.fair_contribution_shadow
            or self.state_driven_config.contribution_restoration_shadow
            or self.state_driven_config.micro_hold_shadow
        ):
            self._fair_contribution_state = FairContributionState(
                client_ids,
                score_cap=self.state_driven_config.fair_contribution_score_cap,
            )

    def _on_aggregation_inputs(
        self,
        *,
        group_idx: int,
        staleness: Dict[str, int],
        local_steps: Dict[str, int],
        model_versions: Dict[str, int],
    ) -> None:
        if self._fair_contribution_state is None:
            return
        alpha = float(getattr(self.aggregator, "alpha", 1.0))
        staleness_fn = getattr(self.aggregator, "staleness_fn", lambda _: 1.0)
        epoch = self._fair_contribution_state.aggregation_epochs
        records = self._fair_contribution_state.update(
            staleness, alpha=alpha, staleness_fn=staleness_fn,
        )
        jain = self._fair_contribution_state.jain_index()
        restoration = {
            row.client_id: row
            for row in recommend_contribution_restoration(
                records,
                local_steps=local_steps,
                qmax=self.max_local_steps,
                rhythm_debt=self._lyapunov_rhythm_debt,
                rhythm_stop=(
                    self.state_driven_config
                    .contribution_restoration_rhythm_stop
                ),
                debt_score_cap=(
                    self.state_driven_config.fair_contribution_score_cap
                ),
                bonus_mass_cap=(
                    self.state_driven_config
                    .contribution_restoration_bonus_cap
                ),
                staleness_hard_cap=(
                    self.state_driven_config
                    .contribution_restoration_staleness_cap
                ),
            )
        }
        for record in records:
            self._fair_contribution_trace_buffer.append({
                "aggregation_epoch": epoch,
                "virtual_time": self._virtual_now,
                "group_id": group_idx,
                "client_id": record.client_id,
                "target_share": record.target_share,
                "participated": int(record.client_id in staleness),
                "local_steps": local_steps.get(record.client_id, 0),
                "model_version": model_versions.get(record.client_id, -1),
                "staleness": record.staleness,
                "raw_aggregation_weight": record.raw_weight,
                "normalized_aggregation_share": record.normalized_weight,
                "target_effective_contribution": (
                    record.target_effective_contribution
                ),
                "effective_contribution": record.effective_contribution,
                "fair_debt_before": record.fair_debt_before,
                "fair_debt_raw": record.fair_debt_raw,
                "fair_debt_score": record.fair_debt_score,
                "fair_debt_overflow": record.fair_debt_overflow,
                "cumulative_jain": jain,
            })
            if self.state_driven_config.contribution_restoration_shadow:
                proposed = restoration[record.client_id]
                self._contribution_restoration_trace_buffer.append({
                    "aggregation_epoch": epoch,
                    "virtual_time": self._virtual_now,
                    "group_id": group_idx,
                    "client_id": record.client_id,
                    "participated": int(record.client_id in staleness),
                    "local_steps": local_steps.get(record.client_id, 0),
                    "staleness": record.staleness,
                    "rhythm_debt": self._lyapunov_rhythm_debt,
                    "fair_debt_before": record.fair_debt_before,
                    "fair_debt_score": proposed.debt_score,
                    "base_share": proposed.base_share,
                    "quality_score": proposed.quality_score,
                    "eligible": int(proposed.eligible),
                    "reason": proposed.reason,
                    "requested_bonus": proposed.requested_bonus,
                    "allocated_bonus": proposed.allocated_bonus,
                    "proposed_share": proposed.proposed_share,
                    "share_delta": (
                        proposed.proposed_share - proposed.base_share
                    ),
                    "applied": 0,
                })

    def _order_arrived_clients_for_redispatch(
        self, client_ids: List[str],
    ) -> List[str]:
        baseline = super()._order_arrived_clients_for_redispatch(client_ids)
        state = self._fair_contribution_state
        mode = self.state_driven_config.unified_batch_dispatch_mode
        if state is None or mode == "off" or len(client_ids) <= 1:
            return baseline

        safe_duration: Dict[str, float] = {}
        for cid in client_ids:
            curve, _ = self._state_time_model.predict_curve(
                client_id=cid,
                dispatch_time=self._virtual_now,
                qs=[self.max_local_steps],
                runtime_state=self._runtime_states.get(cid),
                speed_fallback=float(self.client_info[cid]["speed"]),
            )
            point = curve[0] if curve else None
            safe_duration[cid] = (
                float(point.safe_duration)
                if point is not None
                else float(self.client_info[cid]["speed"]) * self.max_local_steps
            )
        ranked = rank_unified_batch(
            client_ids,
            fair_debt={cid: state.score(cid) for cid in client_ids},
            safe_duration=safe_duration,
            rhythm_target=self.state_driven_config.lyapunov_rhythm_target,
        )
        proposed = [row.client_id for row in ranked]
        batch_id = (
            f"agg{state.aggregation_epochs - 1}"
            f"_t{self._virtual_now:.9f}"
        )
        baseline_rank = {cid: index for index, cid in enumerate(baseline)}
        proposed_rank = {cid: index for index, cid in enumerate(proposed)}
        by_client = {row.client_id: row for row in ranked}
        for cid in client_ids:
            row = by_client[cid]
            self._unified_batch_trace_buffer.append({
                "batch_id": batch_id,
                "virtual_time": self._virtual_now,
                "client_id": cid,
                "mode": mode,
                "baseline_speed_rank": baseline_rank[cid],
                "proposed_rank": proposed_rank[cid],
                "baseline_anchor": int(baseline_rank[cid] == 0),
                "proposed_anchor": int(proposed_rank[cid] == 0),
                "fair_debt_score": row.fair_debt,
                "predicted_safe_duration": row.predicted_safe_duration,
                "freshness": row.freshness,
                "service_probability": row.service_probability,
                "fair_benefit": row.benefit,
                "time_cost": row.time_cost,
                "priority": row.priority,
                "order_changed": int(baseline != proposed),
            })
        return proposed if mode == "apply" else baseline

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

    @staticmethod
    def _parse_q_references(spec: str) -> Dict[str, int]:
        references: Dict[str, int] = {}
        if not spec.strip():
            return references
        for item in spec.split(","):
            client_id, value = item.split("=", 1)
            q = int(value)
            if q <= 0:
                raise ValueError("Lyapunov Q references must be positive")
            references[client_id.strip()] = q
        return references

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

    def on_timer_event(self, event: VirtualEvent, virtual_now: float):
        """Record a bounded deadline-hold counterfactual without applying it."""
        self._virtual_now = virtual_now
        cfg = self.state_driven_config
        group_idx = event.payload.get("group_idx")
        if (
            cfg.micro_hold_shadow
            and group_idx is not None
            and group_idx in self.arrival_group
        ):
            group = self.arrival_group[group_idx]
            wait_cap = (
                cfg.micro_hold_time_ratio * cfg.lyapunov_rhythm_target
            )
            predicted = group.get("predicted_finish_times", {})
            safe = group.get("safe_finish_times", {})
            for client_id in list(group.get("clients", [])):
                predicted_finish = float(predicted.get(client_id, math.inf))
                safe_finish = float(safe.get(client_id, math.inf))
                predicted_wait = max(0.0, predicted_finish - virtual_now)
                safe_wait = max(0.0, safe_finish - virtual_now)
                debt = (
                    self._fair_contribution_state.score(client_id)
                    if self._fair_contribution_state is not None else 0.0
                )
                if not math.isfinite(safe_finish):
                    eligible, reason = False, "missing_safe_prediction"
                elif predicted_finish <= virtual_now or safe_finish <= virtual_now:
                    eligible, reason = False, "pending_after_predicted_finish"
                elif debt <= 0:
                    eligible, reason = False, "no_contribution_deficit"
                elif self._lyapunov_rhythm_debt >= (
                    cfg.contribution_restoration_rhythm_stop
                ):
                    eligible, reason = False, "rhythm_queue_stop"
                elif predicted_wait > wait_cap:
                    eligible, reason = False, "predicted_wait_cap"
                elif safe_wait > wait_cap:
                    eligible, reason = False, "safe_wait_cap"
                else:
                    eligible, reason = True, "near_complete_bounded_hold"
                self._micro_hold_trace_buffer.append({
                    "virtual_time": virtual_now,
                    "group_id": group_idx,
                    "client_id": client_id,
                    "assigned_group_id": self.client_info[client_id].get(
                        "goa", -1
                    ),
                    "predicted_finish_time": (
                        predicted_finish
                        if math.isfinite(predicted_finish) else ""
                    ),
                    "safe_finish_time": (
                        safe_finish if math.isfinite(safe_finish) else ""
                    ),
                    "predicted_wait": (
                        predicted_wait
                        if math.isfinite(predicted_wait) else ""
                    ),
                    "safe_wait": safe_wait if math.isfinite(safe_wait) else "",
                    "wait_cap": wait_cap,
                    "fair_debt_score": debt,
                    "rhythm_debt": self._lyapunov_rhythm_debt,
                    "eligible": int(eligible),
                    "reason": reason,
                    "recommended": int(eligible),
                    "applied": 0,
                })
        return super().on_timer_event(event, virtual_now)

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
                "calibration_source": shadow.calibration_source,
                "calibration_n": shadow.calibration_n,
                "calibration_rank": shadow.calibration_rank,
                "analytical_margin": shadow.analytical_margin,
                "client_margin": shadow.client_margin,
                "pooled_margin": shadow.pooled_margin,
                "selected_margin": shadow.safe_duration - shadow.predicted_duration,
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
        group.setdefault("predicted_finish_times", {})[client_id] = (
            decision.predicted_finish_time
        )
        group.setdefault("safe_finish_times", {})[client_id] = decision.safe_finish_time
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

    def _dynamic_group_frontier(self, group: dict) -> float:
        """Best current finish frontier for clients still pending in a group."""
        pending = list(group.get("clients", []))
        predicted = group.get("predicted_finish_times", {})
        known = [float(predicted[cid]) for cid in pending if cid in predicted]
        if len(known) < len(pending):
            # Old/FedCompass-created groups lack per-member predictions.  Their
            # immutable anchor expectation is the conservative compatible fallback.
            known.append(float(group["expected_arrival_time"]))
        if not known:
            return max(self._virtual_now, float(group["expected_arrival_time"]))
        return min(float(group["latest_arrival_time"]), max(known))

    def _dynamic_group_safe_frontier(self, group: dict) -> float:
        pending = list(group.get("clients", []))
        predicted = group.get("safe_finish_times", {})
        known = [float(predicted[cid]) for cid in pending if cid in predicted]
        if len(known) < len(pending):
            known.append(float(group["latest_arrival_time"]))
        if not known:
            return self._virtual_now
        return max(known)

    def _effective_service_v1_join_actions(self, client_id: str, curve, *, groups=None):
        """Build controlled join candidates for effective-service shadows."""
        calibration_source = "analytical_baseline"
        calibration_n = calibration_rank = 0
        selected_margin = ""
        if (
            self.state_driven_config.lyapunov_action_scope in {
                "effective_service_v2", "effective_service_v2_1",
            }
            and self.state_driven_config.finite_sample_safety_calibration
        ):
            adjusted = []
            for point in curve:
                margin, source, n, rank, _, _ = self._calibrated_shadow.preview_safety_margin(
                    client_id=client_id,
                    analytical_margin=point.safe_duration - point.predicted_duration,
                )
                adjusted.append(type(point)(**{
                    **point.__dict__,
                    "safe_duration": point.predicted_duration + margin,
                    "safe_finish_time": point.predicted_finish_time + margin,
                    "uncertainty": margin,
                }))
                calibration_source = source
                calibration_n, calibration_rank = n, rank
                selected_margin = margin
            curve = adjusted
        reference_q = self._lyapunov_q_references.get(client_id)
        if reference_q is None:
            return []
        by_q = {point.q: point for point in curve}
        actions = []
        active_groups = self.arrival_group if groups is None else groups
        for group_id, group in active_groups.items():
            frontier = self._dynamic_group_frontier(group)
            safe_frontier = self._dynamic_group_safe_frontier(group)
            qset = controlled_join_qs(
                curve=curve, target_time=frontier,
                deadline=float(group["latest_arrival_time"]),
                group_safe_frontier=safe_frontier, reference_q=reference_q,
                qmin=self.min_local_steps, qmax=self.max_local_steps,
                trust_eta=self.state_driven_config.lyapunov_q_trust_eta,
            )
            self._effective_service_q_trace_buffer.append({
                "decision_id": self._decision_id(client_id),
                "virtual_time": self._virtual_now, "client_id": client_id,
                "group_id": int(group_id), "reference_q": qset.reference_q,
                "raw_align_q": qset.raw_align_q,
                "controlled_align_q": qset.controlled_align_q,
                "trust_upper_q": qset.trust_upper_q,
                "candidate_qs": "|".join(map(str, qset.candidate_qs)),
                "predictor_reliable": int(qset.reliable), "reason": qset.reason,
                "expected_frontier": frontier, "safe_frontier": safe_frontier,
                "deadline": float(group["latest_arrival_time"]),
                "group_already_at_risk": int(
                    safe_frontier > float(group["latest_arrival_time"])
                ),
                "safety_calibration_source": calibration_source,
                "safety_calibration_n": calibration_n,
                "safety_calibration_rank": calibration_rank,
                "selected_safety_margin": selected_margin,
            })
            for q in qset.candidate_qs:
                point = by_q[q]
                new_frontier = max(frontier, point.predicted_finish_time)
                actions.append(LyapunovAction(
                    mode="join", group_id=int(group_id), q=q,
                    predicted_finish_time=point.predicted_finish_time,
                    predicted_duration=point.predicted_duration,
                    safe_finish_time=point.safe_finish_time,
                    group_frontier_time=frontier,
                    latest_arrival_time=float(group["latest_arrival_time"]),
                    deadline_safe=(
                        max(safe_frontier, point.safe_finish_time)
                        <= float(group["latest_arrival_time"])
                    ),
                    holding_wait=max(0.0, new_frontier - point.predicted_finish_time),
                    external_wait=max(0.0, new_frontier - frontier),
                    affected_pending_clients=len(group.get("clients", [])),
                    predicted_sojourn=new_frontier - self._virtual_now,
                    effective_work=q / self.max_local_steps,
                    utility=math.log1p(q / max(self.min_local_steps, 1)),
                ))
        return actions

    def _effective_service_v2_create_actions(
        self, client_id: str, curve, *, required_q: int | None = None,
    ):
        reference_q = self._lyapunov_q_references.get(client_id)
        if reference_q is None:
            return []
        adjusted = []
        for point in curve:
            margin, *_ = self._calibrated_shadow.preview_safety_margin(
                client_id=client_id,
                analytical_margin=point.safe_duration - point.predicted_duration,
            )
            adjusted.append(type(point)(**{
                **point.__dict__,
                "safe_duration": point.predicted_duration + margin,
                "safe_finish_time": point.predicted_finish_time + margin,
                "uncertainty": margin,
            }))
        curve = adjusted
        by_q = {point.q: point for point in curve}
        qs = controlled_create_qs(
            curve=curve, reference_q=reference_q,
            qmin=self.min_local_steps, qmax=self.max_local_steps,
            trust_eta=self.state_driven_config.lyapunov_q_trust_eta,
        )
        if required_q is not None:
            trust_upper = min(
                self.max_local_steps,
                math.ceil(
                    self.state_driven_config.lyapunov_q_trust_eta * reference_q
                ),
            )
            if required_q in by_q and self.min_local_steps <= required_q <= trust_upper:
                qs = sorted(set(qs) | {required_q})
        recent = sorted(self._lyapunov_recent_intervals[-12:])
        if recent:
            recruit_expected_raw = recent[len(recent) // 2]
            rank = min(len(recent) - 1, math.ceil(0.85 * (len(recent) + 1)) - 1)
            recruit_safe_raw = recent[rank]
            source = "recent_aggregation_intervals"
        else:
            recruit_expected_raw = self.state_driven_config.lyapunov_rhythm_target
            recruit_safe_raw = 1.2 * recruit_expected_raw
            source = "frozen_rhythm_fallback"
        recruit_safe_cap = (
            self.state_driven_config.lyapunov_recruit_safe_cap_ratio
            * self.state_driven_config.lyapunov_rhythm_target
        )
        recruit_expected = min(recruit_expected_raw, recruit_safe_cap)
        recruit_safe = max(recruit_expected, min(recruit_safe_raw, recruit_safe_cap))
        if (
            recruit_expected < recruit_expected_raw
            or recruit_safe < recruit_safe_raw
        ):
            source += "_rhythm_trust_clipped"
        actions = []
        for q in qs:
            point = by_q[q]
            actions.append(LyapunovAction(
                mode="create", group_id=-1, q=q,
                predicted_finish_time=point.predicted_finish_time,
                predicted_duration=point.predicted_duration,
                safe_finish_time=point.safe_finish_time + recruit_safe,
                group_frontier_time=point.predicted_finish_time + recruit_expected,
                latest_arrival_time=point.safe_finish_time + recruit_safe,
                deadline_safe=True, holding_wait=recruit_expected,
                external_wait=recruit_expected, affected_pending_clients=0,
                predicted_sojourn=point.predicted_duration + (
                    recruit_safe
                    if self.state_driven_config.lyapunov_create_safe_cost
                    else recruit_expected
                ),
                effective_work=q / self.max_local_steps,
                utility=math.log1p(q / max(self.min_local_steps, 1)),
            ))
        recent_sizes = sorted(self._lyapunov_recent_group_sizes[-12:])
        predicted_group_size = recent_sizes[len(recent_sizes) // 2] if recent_sizes else 1
        return (
            actions, recruit_expected, recruit_safe, source, predicted_group_size,
            recruit_expected_raw, recruit_safe_raw, recruit_safe_cap,
        )

    def _choose_effective_service_v2(
        self, client_id: str, scored, *, recruit_expected: float,
        recruit_safe: float, recruitment_source: str, predicted_group_size: int,
        recruit_expected_raw: float = 0.0,
        recruit_safe_raw: float = 0.0, recruit_safe_cap: float = math.inf,
    ) -> LyapunovDecision:
        obvious_limit = (
            self.state_driven_config.lyapunov_region_extension_ratio
            * self.state_driven_config.lyapunov_rhythm_target
        )
        selection = choose_effective_service_v2(
            scored, obvious_extension_limit=obvious_limit,
            obvious_holding_limit=self.state_driven_config.lyapunov_rhythm_target,
            create_hysteresis=self.state_driven_config.lyapunov_create_hysteresis,
        )
        best_join, best_create = selection.best_join, selection.best_create
        calibration_source = ""
        calibration_n = calibration_rank = 0
        if self.state_driven_config.finite_sample_safety_calibration:
            _, calibration_source, calibration_n, calibration_rank, _, _ = (
                self._calibrated_shadow.preview_safety_margin(
                    client_id=client_id, analytical_margin=0.0,
                )
            )
        selected = selection.decision.action
        region, reason = selection.region, selection.reason
        self._effective_service_region_trace_buffer.append({
            "decision_id": self._decision_id(client_id),
            "virtual_time": self._virtual_now, "client_id": client_id,
            "region": region, "reason": reason,
            "num_legal_join": selection.joins, "num_legal_create": selection.creates,
            "best_join_score": best_join.score if best_join else "",
            "best_create_score": best_create.score if best_create else "",
            "score_gap": (
                best_join.score - best_create.score
                if best_join is not None and best_create is not None else ""
            ),
            "hysteresis": self.state_driven_config.lyapunov_create_hysteresis,
            "recommended_mode": selected.mode if selected else "",
            "recommended_group_id": selected.group_id if selected else -1,
            "recommended_q": selected.q if selected else -1,
            "recruitment_expected": recruit_expected,
            "recruitment_expected_raw": recruit_expected_raw,
            "recruitment_safe": recruit_safe,
            "recruitment_safe_raw": recruit_safe_raw,
            "recruitment_safe_cap": recruit_safe_cap,
            "create_safe_cost_enabled": int(
                self.state_driven_config.lyapunov_create_safe_cost
            ),
            "recruitment_source": recruitment_source,
            "predicted_group_size": predicted_group_size,
            "counterfactual_outcome_observable": int(
                self.state_driven_config.lyapunov_action_scope
                == "effective_service_v2_1"
            ),
            "calibration_source": calibration_source,
            "calibration_n": calibration_n,
            "calibration_rank": calibration_rank,
        })
        return selection.decision

    def _settle_effective_service_shadow_groups(self) -> None:
        if self._effective_service_shadow_groups is None:
            return
        expired = [
            group_id for group_id, group in self._effective_service_shadow_groups.items()
            if float(group["latest_arrival_time"]) <= self._virtual_now
        ]
        for group_id in expired:
            group = self._effective_service_shadow_groups.pop(group_id)
            members = list(group.get("clients", []))
            safe_finishes = list(group.get("safe_finish_times", {}).values())
            self._effective_service_shadow_outcome_trace_buffer.append({
                "shadow_group_id": group_id,
                "created_decision_id": group.get("created_decision_id", "seed_real_group"),
                "created_time": group.get("created_time", ""),
                "settled_time": self._virtual_now,
                "deadline": group["latest_arrival_time"],
                "projected_group_size": len(members),
                "projected_singleton": int(len(members) == 1),
                "projected_safe_hit": int(
                    bool(safe_finishes)
                    and max(map(float, safe_finishes)) <= float(group["latest_arrival_time"])
                ),
                "projected_work": sum(group.get("member_q", {}).values()),
                "member_clients": "|".join(members),
                "member_decisions": "|".join(group.get("member_decisions", [])),
                "status": "settled",
            })

    def _ensure_effective_service_shadow_groups(self) -> Dict[int, dict]:
        if self._effective_service_shadow_groups is None:
            # Seed only groups that can still accept/settle pending work.  A real
            # group may remain visible during the dispatch callback that closes
            # it; copying that expired/empty shell into the counterfactual state
            # produces a fake unsafe singleton (or empty group) at first settle.
            self._effective_service_shadow_groups = copy.deepcopy({
                group_id: group
                for group_id, group in self.arrival_group.items()
                if group.get("clients")
                and float(group.get("latest_arrival_time", -math.inf))
                > self._virtual_now
            })
            for group_id, group in self._effective_service_shadow_groups.items():
                group.setdefault("created_decision_id", "seed_real_group")
                group.setdefault("member_decisions", [])
                group.setdefault("member_q", {
                    client_id: int(self.client_info.get(client_id, {}).get("local_steps", 0))
                    for client_id in group.get("clients", [])
                })
            self._effective_service_shadow_next_group_id = -1
        self._settle_effective_service_shadow_groups()
        return self._effective_service_shadow_groups

    def _apply_effective_service_shadow_action(
        self, client_id: str, action: LyapunovAction,
    ) -> None:
        groups = self._ensure_effective_service_shadow_groups()
        decision_id = self._decision_id(client_id)
        if action.mode == "create":
            group_id = self._effective_service_shadow_next_group_id
            self._effective_service_shadow_next_group_id -= 1
            groups[group_id] = {
                "clients": [client_id], "arrived_clients": [],
                "expected_arrival_time": action.group_frontier_time,
                "latest_arrival_time": action.latest_arrival_time,
                "created_time": self._virtual_now,
                "created_decision_id": decision_id,
                "predicted_finish_times": {client_id: action.predicted_finish_time},
                "safe_finish_times": {client_id: action.safe_finish_time},
                "member_q": {client_id: action.q},
                "member_decisions": [decision_id],
            }
            action_group_id = group_id
        else:
            group = groups[action.group_id]
            group.setdefault("clients", []).append(client_id)
            group.setdefault("predicted_finish_times", {})[client_id] = action.predicted_finish_time
            group.setdefault("safe_finish_times", {})[client_id] = action.safe_finish_time
            group.setdefault("member_q", {})[client_id] = action.q
            group.setdefault("member_decisions", []).append(decision_id)
            action_group_id = action.group_id
        self._effective_service_shadow_outcome_trace_buffer.append({
            "shadow_group_id": action_group_id,
            "created_decision_id": groups[action_group_id].get("created_decision_id", ""),
            "created_time": groups[action_group_id].get("created_time", ""),
            "settled_time": "", "deadline": groups[action_group_id]["latest_arrival_time"],
            "projected_group_size": len(groups[action_group_id].get("clients", [])),
            "projected_singleton": "", "projected_safe_hit": "",
            "projected_work": sum(groups[action_group_id].get("member_q", {}).values()),
            "member_clients": "|".join(groups[action_group_id].get("clients", [])),
            "member_decisions": "|".join(groups[action_group_id].get("member_decisions", [])),
            "status": "create" if action.mode == "create" else "join",
        })

    def _lyapunov_actions(self, client_id: str):
        curve, monotonic = self._curve(client_id)
        speed = float(self.client_info[client_id]["speed"])
        fed_join = existing_group_reference(
            now=self._virtual_now, speed=speed, groups=self.arrival_group,
            qmin=self.min_local_steps, qmax=self.max_local_steps,
        )
        scope = self.state_driven_config.lyapunov_action_scope
        if scope == "effective_service_v2_1":
            analytical = curve[0].safe_duration - curve[0].predicted_duration
            _, calibration_source, calibration_n, calibration_rank, _, _ = (
                self._calibrated_shadow.preview_safety_margin(
                    client_id=client_id, analytical_margin=analytical,
                )
            )
            if calibration_source == "analytical":
                self._effective_service_region_trace_buffer.append({
                    "decision_id": self._decision_id(client_id),
                    "virtual_time": self._virtual_now, "client_id": client_id,
                    "region": "cold_start", "reason": "cold_start_defer_to_safe_reuse",
                    "num_legal_join": 0, "num_legal_create": 0,
                    "best_join_score": "", "best_create_score": "",
                    "score_gap": "", "hysteresis": self.state_driven_config.lyapunov_create_hysteresis,
                    "recommended_mode": "defer", "recommended_group_id": -1,
                    "recommended_q": -1, "recruitment_expected": "",
                    "recruitment_expected_raw": "", "recruitment_safe": "",
                    "recruitment_safe_raw": "", "recruitment_safe_cap": "",
                    "create_safe_cost_enabled": int(
                        self.state_driven_config.lyapunov_create_safe_cost
                    ),
                    "recruitment_source": "",
                    "predicted_group_size": "", "counterfactual_outcome_observable": 0,
                    "calibration_source": calibration_source,
                    "calibration_n": calibration_n, "calibration_rank": calibration_rank,
                })
                return (
                    curve, monotonic, fed_join, self.max_local_steps, [],
                    LyapunovDecision(False, reason="cold_start_defer_to_safe_reuse"),
                )
            shadow_mode = self.state_driven_config.lyapunov_mode == "shadow"
            effective_groups = (
                self._ensure_effective_service_shadow_groups()
                if shadow_mode else self.arrival_group
            )
            if shadow_mode and any(
                client_id in group.get("clients", [])
                for group in effective_groups.values()
            ):
                self._effective_service_region_trace_buffer.append({
                    "decision_id": self._decision_id(client_id),
                    "virtual_time": self._virtual_now, "client_id": client_id,
                    "region": "blocked", "reason": "shadow_dispatch_blocked",
                    "num_legal_join": 0, "num_legal_create": 0,
                    "best_join_score": "", "best_create_score": "",
                    "score_gap": "", "hysteresis": self.state_driven_config.lyapunov_create_hysteresis,
                    "recommended_mode": "defer", "recommended_group_id": -1,
                    "recommended_q": -1, "recruitment_expected": "",
                    "recruitment_expected_raw": "", "recruitment_safe": "",
                    "recruitment_safe_raw": "", "recruitment_safe_cap": "",
                    "create_safe_cost_enabled": int(
                        self.state_driven_config.lyapunov_create_safe_cost
                    ),
                    "recruitment_source": "",
                    "predicted_group_size": "", "counterfactual_outcome_observable": 0,
                    "calibration_source": calibration_source,
                    "calibration_n": calibration_n, "calibration_rank": calibration_rank,
                })
                return (
                    curve, monotonic, fed_join, self.max_local_steps, [],
                    LyapunovDecision(False, reason="shadow_dispatch_blocked"),
                )
        else:
            effective_groups = self.arrival_group
        joint = self._joint_policy.enumerate_candidates(
            now=self._virtual_now, groups=self.arrival_group, curve=curve,
        )
        effective_scope = self.state_driven_config.lyapunov_action_scope in {
            "effective_service_v1", "effective_service_v2", "effective_service_v2_1",
        }
        if effective_scope:
            actions = self._effective_service_v1_join_actions(
                client_id, curve, groups=effective_groups,
            )
        else:
            actions = []
        for candidate in joint:
            if effective_scope:
                break
            group = self.arrival_group[candidate.group_id]
            frontier = self._dynamic_group_frontier(group)
            new_frontier = max(frontier, candidate.predicted_finish_time)
            holding = max(0.0, frontier - candidate.predicted_finish_time)
            external = max(0.0, candidate.predicted_finish_time - frontier)
            actions.append(LyapunovAction(
                mode="join", group_id=candidate.group_id, q=candidate.q,
                predicted_finish_time=candidate.predicted_finish_time,
                predicted_duration=candidate.predicted_finish_time - self._virtual_now,
                safe_finish_time=candidate.safe_finish_time,
                group_frontier_time=frontier,
                latest_arrival_time=candidate.latest_arrival_time,
                deadline_safe=candidate.deadline_safe,
                holding_wait=holding, external_wait=external,
                affected_pending_clients=len(group.get("clients", [])),
                predicted_sojourn=new_frontier - self._virtual_now,
                effective_work=candidate.q / self.max_local_steps,
                utility=math.log1p(candidate.q / max(self.min_local_steps, 1)),
            ))
        fed_new_q = new_group_reference_q(
            now=self._virtual_now, client_id=client_id, speed=speed,
            groups=self.arrival_group, client_info=self.client_info,
            qmin=self.min_local_steps, qmax=self.max_local_steps,
        ) if self.arrival_group else self.max_local_steps
        create_point = next((point for point in curve if point.q == fed_new_q), curve[-1])
        recruit_expected = recruit_safe = 0.0
        recruit_expected_raw = recruit_safe_raw = 0.0
        recruit_safe_cap = math.inf
        recruitment_source = ""
        predicted_group_size = 0
        if scope in {"effective_service_v2", "effective_service_v2_1"}:
            (
                create_actions, recruit_expected, recruit_safe,
                recruitment_source, predicted_group_size,
                recruit_expected_raw, recruit_safe_raw, recruit_safe_cap,
            ) = self._effective_service_v2_create_actions(client_id, curve)
            actions.extend(create_actions)
        if self.state_driven_config.lyapunov_action_scope == "joint_v1":
            actions.append(LyapunovAction(
                mode="create", group_id=-1, q=fed_new_q,
                predicted_finish_time=create_point.predicted_finish_time,
                predicted_duration=create_point.predicted_duration,
                safe_finish_time=create_point.safe_finish_time,
                group_frontier_time=create_point.predicted_finish_time,
                latest_arrival_time=create_point.safe_finish_time,
                deadline_safe=True, holding_wait=0.0, external_wait=0.0,
                affected_pending_clients=0,
                predicted_sojourn=create_point.predicted_duration,
                effective_work=fed_new_q / self.max_local_steps,
                utility=math.log1p(fed_new_q / max(self.min_local_steps, 1)),
            ))
        score_reference_q = (
            self._lyapunov_q_references.get(client_id, -1)
            if effective_scope
            else fed_join.q if fed_join.feasible else -1
        )
        scored = self._lyapunov_policy.score(
            actions,
            rhythm_debt=self._lyapunov_rhythm_debt,
            workload_debt=self._lyapunov_workload_debt.get(client_id, 0.0),
            qmax=self.max_local_steps, qmin=self.min_local_steps,
            fedcompass_join_q=score_reference_q,
        )
        decision = (
            self._choose_effective_service_v2(
                client_id, scored, recruit_expected=recruit_expected,
                recruit_safe=recruit_safe, recruitment_source=recruitment_source,
                predicted_group_size=predicted_group_size,
                recruit_expected_raw=recruit_expected_raw,
                recruit_safe_raw=recruit_safe_raw,
                recruit_safe_cap=recruit_safe_cap,
            )
            if scope in {"effective_service_v2", "effective_service_v2_1"}
            else self._lyapunov_policy.choose(scored)
        )
        if (
            scope == "effective_service_v2_1"
            and self.state_driven_config.lyapunov_mode == "shadow"
            and decision.action is not None
        ):
            self._apply_effective_service_shadow_action(client_id, decision.action)
        return curve, monotonic, fed_join, fed_new_q, scored, decision

    def _record_lyapunov_decision(
        self, client_id: str, scored, decision, *, applied_mode: str,
        applied_group: int, applied_q: int,
    ) -> None:
        selected = decision.action if decision.feasible else None
        legal = [action for action in scored if action.legal]
        rejected_holding = sum(
            action.rejection_reason in {"holding_wait_exceeds_cap", "extreme_holding_wait"}
            for action in scored
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
            "recommended_group_frontier": selected.group_frontier_time if selected else "",
            "recommended_affected_pending": (
                selected.affected_pending_clients if selected else ""
            ),
            "recommended_cadence_excess": (
                max(
                    0.0,
                    selected.predicted_sojourn
                    - self.state_driven_config.lyapunov_rhythm_target,
                ) if selected and selected.mode == "join" else 0.0
            ),
            "join_cadence_weight": (
                self.state_driven_config.lyapunov_join_cadence_weight
            ),
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
        group.setdefault("predicted_finish_times", {})[client_id] = (
            action.predicted_finish_time
        )
        group.setdefault("safe_finish_times", {})[client_id] = action.safe_finish_time
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

    def _apply_effective_service_create(
        self, client_id: str, action: LyapunovAction, fed_reference_q: int,
    ) -> List[VirtualEvent]:
        """Create the exact group window selected by Effective-Service V2.1."""
        group_id = self.group_counter
        self.arrival_group[group_id] = {
            "clients": [client_id], "arrived_clients": [],
            "expected_arrival_time": action.group_frontier_time,
            "latest_arrival_time": action.latest_arrival_time,
            "created_time": self._virtual_now,
            "time_source": "effective_service_v2_1_apply",
            "anchor_client_id": client_id, "anchor_q": action.q,
            "predicted_finish_times": {client_id: action.predicted_finish_time},
            "safe_finish_times": {client_id: action.safe_finish_time},
        }
        events = [VirtualEvent(
            time=action.latest_arrival_time,
            event_type=EventType.FEDCOMPASS_GROUP_DEADLINE,
            payload={"group_idx": group_id},
        )]
        self._deadline_events.add(group_id)
        self.client_info[client_id]["goa"] = group_id
        self.client_info[client_id]["local_steps"] = action.q
        self.client_info[client_id]["start_time"] = self._virtual_now
        self._record_dispatch_decision(
            client_id=client_id, decision="effective_service_create_group",
            assigned_group=group_id, assigned_steps=action.q,
            speed_raw=float(self.client_info[client_id]["speed"]),
            remaining_time=None, target_arrival=action.group_frontier_time,
            latest_arrival=action.latest_arrival_time,
        )
        self._state_group_creation_trace_buffer.append({
            "decision_id": self._decision_id(client_id),
            "virtual_time": self._virtual_now, "client_id": client_id,
            "new_group_id": group_id,
            "new_group_window_mode": "effective_service_v2_1_apply",
            "new_group_q_mode": "controlled_dual_anchor",
            "applied": 1, "fedcompass_reference_q": fed_reference_q,
            "fedcompass_expected_time": "", "fedcompass_latest_time": "",
            "state_assigned_q": action.q,
            "state_expected_time": action.group_frontier_time,
            "state_latest_time": action.latest_arrival_time,
            "state_safe_finish": action.safe_finish_time,
            "state_uncertainty": action.safe_finish_time - action.predicted_finish_time,
            "expected_shift": "", "latest_shift": "",
            "predictor_source": "effective_service_v2_1",
            "num_reports": "", "used_fallback": 0, "fallback_reason": "",
            "safe_window_exceeds_cap": 0,
        })
        self.group_counter += 1
        return events

    def _assign_group_lyapunov(self, client_id: str) -> List[VirtualEvent]:
        curve, monotonic, fed_join, fed_new_q, scored, decision = self._lyapunov_actions(client_id)
        fallback = bool(curve) and all(point.used_fallback for point in curve)
        if fallback:
            events = self._assign_group_safe_reuse(client_id)
            group = int(self.client_info[client_id].get("goa", -1))
            q = int(self.client_info[client_id].get("local_steps", -1))
            applied_mode = "join" if not events else "create"
            self._record_reason_aware_routing_shadow(
                client_id, curve=curve, scored=scored, decision=decision,
                applied_mode=applied_mode, applied_group=group, applied_q=q,
            )
            self._record_lyapunov_decision(
                client_id, scored, decision, applied_mode="fallback",
                applied_group=group, applied_q=q,
            )
            return events

        if (
            not decision.feasible
            and self.state_driven_config.lyapunov_action_scope == "join_only_v2"
            and self.state_driven_config.lyapunov_mode == "apply"
        ):
            events = self._create_group(client_id)
            group = int(self.client_info[client_id].get("goa", -1))
            q = int(self.client_info[client_id].get("local_steps", -1))
            self._record_reason_aware_routing_shadow(
                client_id, curve=curve, scored=scored, decision=decision,
                applied_mode="create", applied_group=group, applied_q=q,
            )
            self._record_lyapunov_decision(
                client_id, scored, decision, applied_mode="create_fallback",
                applied_group=group, applied_q=q,
            )
            return events

        if not decision.feasible:
            events = self._assign_group_safe_reuse(client_id)
            group = int(self.client_info[client_id].get("goa", -1))
            q = int(self.client_info[client_id].get("local_steps", -1))
            applied_mode = "join" if not events else "create"
            self._record_reason_aware_routing_shadow(
                client_id, curve=curve, scored=scored, decision=decision,
                applied_mode=applied_mode, applied_group=group, applied_q=q,
            )
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
            self._record_reason_aware_routing_shadow(
                client_id, curve=curve, scored=scored, decision=decision,
                applied_mode=applied_mode, applied_group=group, applied_q=q,
            )
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
            events = (
                self._apply_effective_service_create(client_id, action, fed_new_q)
                if self.state_driven_config.lyapunov_action_scope
                == "effective_service_v2_1"
                else self._create_state_group(
                    client_id, action.q, fed_reference_q=fed_new_q,
                )
            )
            group = int(self.client_info[client_id]["goa"])
        self._record_state_time_points(
            client_id, curve, action.q, fed_join.q, monotonic,
        )
        self._record_reason_aware_routing_shadow(
            client_id, curve=curve, scored=scored, decision=decision,
            applied_mode=action.mode, applied_group=group, applied_q=action.q,
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
        if int(group_id) >= 0:
            self._lyapunov_recent_intervals.append(delta_t)
            self._lyapunov_recent_group_sizes.append(len(local_steps))
            del self._lyapunov_recent_intervals[:-12]
            del self._lyapunov_recent_group_sizes[:-12]
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
            if client_id in local_steps:
                self._reason_aware_last_service_time[client_id] = self._virtual_now
        self._lyapunov_last_aggregation_time = self._virtual_now

    def _record_reason_aware_routing_shadow(
        self, client_id: str, *, curve, scored, decision,
        applied_mode: str, applied_group: int, applied_q: int,
    ) -> None:
        cfg = self.state_driven_config
        if not cfg.reason_aware_routing_shadow:
            return
        self._finalize_reason_aware_batch(self._virtual_now)
        # Shadow comparisons are always anchored to the action that V2.3
        # actually dispatched.  A feasible but unapplied internal candidate
        # must never leak its Q/group into the counterfactual baseline.
        selected = None
        if applied_q > 0:
            applied_point = next(
                (item for item in curve if item.q == applied_q),
                min(curve, key=lambda item: abs(item.q - applied_q))
                if curve else None,
            )
            group_state = self.arrival_group.get(applied_group)
            expected = (
                float(group_state["expected_arrival_time"])
                if group_state is not None else (
                    applied_point.predicted_finish_time
                    if applied_point is not None else self._virtual_now
                )
            )
            latest = (
                float(group_state["latest_arrival_time"])
                if group_state is not None else (
                    applied_point.safe_finish_time
                    if applied_point is not None else expected
                )
            )
            predicted_finish = (
                applied_point.predicted_finish_time
                if applied_point is not None else expected
            )
            safe_finish = (
                applied_point.safe_finish_time
                if applied_point is not None else latest
            )
            predicted_duration = (
                applied_point.predicted_duration
                if applied_point is not None
                else max(0.0, expected - self._virtual_now)
            )
            selected = LyapunovAction(
                mode=applied_mode,
                group_id=applied_group,
                q=applied_q,
                predicted_finish_time=predicted_finish,
                predicted_duration=predicted_duration,
                safe_finish_time=safe_finish,
                group_frontier_time=expected,
                latest_arrival_time=latest,
                deadline_safe=safe_finish <= latest,
                holding_wait=max(0.0, expected - predicted_finish),
                external_wait=0.0,
                affected_pending_clients=0,
                predicted_sojourn=max(0.0, expected - self._virtual_now),
                effective_work=applied_q / self.max_local_steps,
                utility=1.0,
                score=0.0,
                legal=True,
            )
        point = None
        if selected is not None:
            point = next((item for item in curve if item.q == selected.q), None)
        if point is None and curve:
            point = min(curve, key=lambda item: item.q)
        cause = classify_slow_cause(point)
        structural = None
        structural_qmax = None
        state = self._runtime_states.get(client_id)
        if (
            cfg.reason_aware_one_report_structural_shadow
            and selected is not None
            and state is not None
            and int(getattr(state, "num_reports", 0)) == 1
        ):
            observed_q = max(
                1, int(round(
                    float(getattr(state, "round_time_mean", 0.0))
                    / max(float(getattr(state, "step_time_mean", 0.0)), 1e-12)
                )),
            )
            structural = predict_one_report_structural(
                q=selected.q,
                observed_q=observed_q,
                observed_round_duration=float(
                    getattr(state, "round_time_mean", 0.0)
                ),
                observed_compute_duration=(
                    float(getattr(state, "compute_step_time_mean", 0.0))
                    * observed_q
                ),
                observed_communication_duration=float(
                    getattr(state, "communication_time_mean", 0.0)
                ),
                observed_spike_duration=float(
                    getattr(state, "spike_delay_mean", 0.0)
                ),
                observed_availability_duration=float(
                    getattr(state, "availability_wait_mean", 0.0)
                ),
                num_reports=1,
                communication_ratio_gate=(
                    cfg.reason_aware_one_report_communication_gate
                ),
                safety_fraction=(
                    cfg.reason_aware_one_report_safety_fraction
                ),
            )
            if structural.eligible:
                structural_q_cap = min(
                    self.max_local_steps,
                    max(
                        selected.q,
                        int(
                            math.floor(
                                selected.q
                                * cfg.communication_amortized_q_max_ratio
                            )
                        ),
                    ),
                )
                structural_qmax = predict_one_report_structural(
                    q=structural_q_cap,
                    observed_q=observed_q,
                    observed_round_duration=float(
                        getattr(state, "round_time_mean", 0.0)
                    ),
                    observed_compute_duration=(
                        float(
                            getattr(
                                state, "compute_step_time_mean", 0.0
                            )
                        )
                        * observed_q
                    ),
                    observed_communication_duration=float(
                        getattr(state, "communication_time_mean", 0.0)
                    ),
                    observed_spike_duration=float(
                        getattr(state, "spike_delay_mean", 0.0)
                    ),
                    observed_availability_duration=float(
                        getattr(state, "availability_wait_mean", 0.0)
                    ),
                    num_reports=1,
                    communication_ratio_gate=(
                        cfg.reason_aware_one_report_communication_gate
                    ),
                    safety_fraction=(
                        cfg.reason_aware_one_report_safety_fraction
                    ),
                )
                total = max(structural.predicted_duration, 1e-12)
                cause = SlowCause(
                    label="extreme_communication_bound",
                    confidence=structural.communication_ratio,
                    compute_ratio=structural.compute_duration / total,
                    communication_ratio=(
                        structural.fixed_duration / total
                    ),
                    availability_ratio=0.0,
                    spike_ratio=0.0,
                    mature=False,
                )
        last_service = self._reason_aware_last_service_time.get(client_id, 0.0)
        service_age = max(0.0, self._virtual_now - last_service)
        service_age_periods = (
            service_age / cfg.lyapunov_rhythm_target
            if cfg.lyapunov_rhythm_target > 0 else 0.0
        )
        recent = self._lyapunov_recent_intervals[-4:]
        recent_median = (
            sorted(recent)[len(recent) // 2] if recent else math.inf
        )
        recent_max = max(recent) if recent else math.inf
        at_risk_groups = sum(
            self._dynamic_group_safe_frontier(group)
            > float(group["latest_arrival_time"])
            for group in self.arrival_group.values()
            if group.get("clients")
        )
        system_healthy = bool(
            recent
            and recent_median
            <= cfg.reason_aware_cadence_median_ratio * cfg.lyapunov_rhythm_target
            and recent_max
            <= cfg.reason_aware_cadence_max_ratio * cfg.lyapunov_rhythm_target
            and at_risk_groups == 0
        )
        fair_debt_score = (
            self._fair_contribution_state.score(client_id)
            if self._fair_contribution_state is not None else 0.0
        )
        amortized_q_point = point
        amortized_q_eligible = bool(
            cfg.communication_amortized_q_shadow
            and selected is not None
            and point is not None
            and cause.label == "extreme_communication_bound"
            and service_age_periods
            >= cfg.communication_amortized_q_age_periods
            and cause.communication_ratio
            >= cfg.communication_amortized_q_ratio_gate
            and system_healthy
            and fair_debt_score > 0.0
        )
        if amortized_q_eligible:
            marginal_budget = (
                cfg.communication_amortized_q_time_ratio
                * cfg.lyapunov_rhythm_target
            )
            candidates = []
            q_cap = min(
                self.max_local_steps,
                max(
                    applied_q,
                    int(
                        math.floor(
                            applied_q
                            * cfg.communication_amortized_q_max_ratio
                        )
                    ),
                ),
            )
            for candidate in curve:
                if (
                    candidate.q < applied_q
                    or candidate.q > q_cap
                    or candidate.used_fallback
                ):
                    continue
                if (
                    candidate.predicted_duration - point.predicted_duration
                    > marginal_budget
                    or candidate.safe_duration - point.safe_duration
                    > marginal_budget
                ):
                    continue
                if (
                    selected.mode == "join"
                    and candidate.safe_finish_time
                    > selected.latest_arrival_time
                ):
                    continue
                candidates.append(candidate)
            if candidates:
                amortized_q_point = max(
                    candidates, key=lambda candidate: candidate.q
                )
        structural_amortized_q = False
        if (
            cfg.communication_amortized_q_shadow
            and structural is not None
            and structural.eligible
            and structural_qmax is not None
            and selected is not None
            and fair_debt_score > 0.0
            and system_healthy
            and service_age_periods
            >= cfg.communication_amortized_q_age_periods
        ):
            marginal_budget = (
                cfg.communication_amortized_q_time_ratio
                * cfg.lyapunov_rhythm_target
            )
            structural_deadline_safe = True
            if selected.mode == "join":
                existing_window = (
                    self._reason_aware_structural_group_windows.get(
                        selected.group_id
                    )
                )
                structural_latest = max(
                    existing_window[1] if existing_window else -math.inf,
                    self._virtual_now + structural.safe_duration,
                )
                structural_deadline_safe = (
                    self._virtual_now + structural_qmax.safe_duration
                    <= structural_latest
                )
            if (
                structural_qmax.predicted_duration
                - structural.predicted_duration
                <= marginal_budget
                and structural_qmax.safe_duration
                - structural.safe_duration
                <= marginal_budget
                and structural_deadline_safe
            ):
                structural_amortized_q = True
                amortized_q_eligible = True
        route = recommend_reason_aware_route(
            cause=cause,
            v23_action=selected,
            scored_actions=scored,
            service_age_periods=service_age_periods,
            minimum_anchor_age_periods=cfg.reason_aware_min_anchor_age_periods,
            system_healthy=system_healthy,
            background_sojourn_periods=cfg.reason_aware_background_sojourn_periods,
            rhythm_target=cfg.lyapunov_rhythm_target,
        )
        if (
            structural is not None
            and structural.eligible
            and selected is not None
            and selected.mode == "create"
        ):
            # The real V2.3 group is created immediately after this Shadow
            # record.  Preserve a counterfactual window keyed by that real id
            # so later dispatches can observe the corrected cadence cost
            # without mutating the real group.
            self._reason_aware_structural_group_windows[applied_group] = (
                self._virtual_now + structural.predicted_duration,
                self._virtual_now + structural.safe_duration,
            )
        elif (
            structural is not None
            and structural.eligible
            and selected is not None
            and selected.mode == "join"
        ):
            current = self._reason_aware_structural_group_windows.get(
                selected.group_id
            )
            if current is not None:
                self._reason_aware_structural_group_windows[selected.group_id] = (
                    max(
                        current[0],
                        self._virtual_now + structural.predicted_duration,
                    ),
                    max(
                        current[1],
                        self._virtual_now + structural.safe_duration,
                    ),
                )
        corrected_window = (
            self._reason_aware_structural_group_windows.get(selected.group_id)
            if selected is not None and selected.mode == "join" else None
        )
        corrected_sojourn = (
            max(0.0, corrected_window[0] - self._virtual_now)
            if corrected_window else 0.0
        )
        corrected_cadence_excess = max(
            0.0, corrected_sojourn - cfg.lyapunov_rhythm_target
        )
        selected_join_sojourn = (
            float(selected.predicted_sojourn)
            if selected is not None and selected.mode == "join" else 0.0
        )
        observed_join_sojourn = max(
            corrected_sojourn, selected_join_sojourn,
        )
        mature_join_limit = (
            cfg.reason_aware_cadence_max_ratio
            * cfg.lyapunov_rhythm_target
        )
        mature_long_join = bool(
            selected is not None
            and selected.mode == "join"
            and cause.mature
            and cause.label != "extreme_communication_bound"
            and observed_join_sojourn > mature_join_limit
        )
        structural_long_join = bool(
            corrected_window is not None
            and corrected_cadence_excess > 0.0
            and cause.label != "extreme_communication_bound"
        )
        elastic_join_avoidance = False
        same_q_create = None
        coordinated_batch_role = ""
        if (
            (structural_long_join or mature_long_join)
            and selected is not None
        ):
            create_actions, *_ = self._effective_service_v2_create_actions(
                client_id, curve, required_q=applied_q,
            )
            same_q_create = next(
                (action for action in create_actions if action.q == applied_q),
                None,
            )
            if same_q_create is not None:
                same_q_create = self._lyapunov_policy.score(
                    [same_q_create],
                    rhythm_debt=self._lyapunov_rhythm_debt,
                    workload_debt=self._lyapunov_workload_debt.get(
                        client_id, 0.0
                    ),
                    qmax=self.max_local_steps,
                    qmin=self.min_local_steps,
                    fedcompass_join_q=self._lyapunov_q_references.get(
                        client_id, -1
                    ),
                )[0]
            if same_q_create is not None and same_q_create.legal:
                route = ReasonAwareRoute(
                    lane="elastic_unified",
                    mode="create",
                    group_id=-1,
                    q=selected.q,
                    reason=(
                        "mature_join_cadence_avoidance"
                        if mature_long_join
                        else "corrected_anchor_cadence_avoidance"
                    ),
                    compatible_background_group=-1,
                    anchor_eligible=False,
                    changed=True,
                )
                elastic_join_avoidance = True
        shadow_action = (
            same_q_create
            if elastic_join_avoidance
            else next((
                action for action in scored
                if (
                    action.legal
                    and action.mode == route.mode
                    and action.group_id == route.group_id
                )
            ), selected)
        )
        self._reason_aware_routing_trace_buffer.append({
            "decision_id": self._decision_id(client_id),
            "virtual_time": self._virtual_now,
            "client_id": client_id,
            "slow_cause": cause.label,
            "classification_confidence": cause.confidence,
            "predictor_mature": int(cause.mature),
            "compute_ratio": cause.compute_ratio,
            "communication_ratio": cause.communication_ratio,
            "availability_ratio": cause.availability_ratio,
            "spike_ratio": cause.spike_ratio,
            "service_age": service_age,
            "service_age_periods": service_age_periods,
            "recent_cadence_median": recent_median if recent else "",
            "recent_cadence_max": recent_max if recent else "",
            "rhythm_debt": self._lyapunov_rhythm_debt,
            "active_groups": len(self.arrival_group),
            "at_risk_groups": at_risk_groups,
            "system_healthy": int(system_healthy),
            "recommended_lane": route.lane,
            "route_reason": route.reason,
            "compatible_background_group": route.compatible_background_group,
            "anchor_eligible": int(route.anchor_eligible),
            "v23_mode": applied_mode,
            "v23_group_id": applied_group,
            "v23_q": applied_q,
            "shadow_mode": route.mode,
            "shadow_group_id": route.group_id,
            # Structural invariant: Reason-Aware Shadow never changes Q.
            "shadow_q": route.q,
            "q_unchanged": int(route.q == applied_q),
            "recommendation_changed": int(
                route.mode != applied_mode
                or (
                    route.mode == "join"
                    and route.group_id != applied_group
                )
            ),
            "shadow_predicted_sojourn": (
                shadow_action.predicted_sojourn if shadow_action else ""
            ),
            "shadow_holding_wait": (
                shadow_action.holding_wait if shadow_action else ""
            ),
            "shadow_external_wait": (
                shadow_action.external_wait if shadow_action else ""
            ),
            "shadow_cadence_excess": (
                max(
                    0.0,
                    shadow_action.predicted_sojourn - cfg.lyapunov_rhythm_target,
                ) if shadow_action and shadow_action.mode == "join" else 0.0
            ),
            "shadow_deadline_safe": (
                int(shadow_action.deadline_safe) if shadow_action else ""
            ),
            "shadow_safety_slack": (
                shadow_action.latest_arrival_time
                - shadow_action.safe_finish_time
                if shadow_action else ""
            ),
            "one_report_structural_enabled": int(
                cfg.reason_aware_one_report_structural_shadow
            ),
            "one_report_structural_eligible": int(
                structural is not None and structural.eligible
            ),
            "one_report_structural_reason": (
                structural.reason if structural else "not_evaluated"
            ),
            "one_report_structural_q": (
                structural.q if structural else ""
            ),
            "one_report_structural_predicted_duration": (
                structural.predicted_duration if structural else ""
            ),
            "one_report_structural_safe_duration": (
                structural.safe_duration if structural else ""
            ),
            "one_report_structural_predicted_finish": (
                self._virtual_now + structural.predicted_duration
                if structural else ""
            ),
            "one_report_structural_safe_finish": (
                self._virtual_now + structural.safe_duration
                if structural else ""
            ),
            "one_report_structural_fixed_duration": (
                structural.fixed_duration if structural else ""
            ),
            "one_report_structural_compute_duration": (
                structural.compute_duration if structural else ""
            ),
            "one_report_structural_communication_ratio": (
                structural.communication_ratio if structural else ""
            ),
            "one_report_structural_q_unchanged": int(
                structural is None
                or structural.q == applied_q
            ),
            "structural_group_window_active": int(
                corrected_window is not None
            ),
            "structural_group_expected_time": (
                corrected_window[0] if corrected_window else ""
            ),
            "structural_group_latest_time": (
                corrected_window[1] if corrected_window else ""
            ),
            "structural_group_sojourn": (
                corrected_sojourn if corrected_window else ""
            ),
            "structural_group_cadence_excess": (
                corrected_cadence_excess if corrected_window else ""
            ),
            "join_observed_sojourn": observed_join_sojourn,
            "join_cadence_limit": mature_join_limit,
            "mature_long_join_avoidance": int(mature_long_join),
            "elastic_join_avoidance": int(elastic_join_avoidance),
            "same_q_create_candidate": int(same_q_create is not None),
            "same_q_create_legal": int(
                same_q_create is not None and same_q_create.legal
            ),
            "same_q_create_score": (
                same_q_create.score if same_q_create is not None else ""
            ),
            "same_q_create_expected_time": (
                same_q_create.group_frontier_time
                if same_q_create is not None else ""
            ),
            "same_q_create_latest_time": (
                same_q_create.latest_arrival_time
                if same_q_create is not None else ""
            ),
            "coordinated_batch_role": coordinated_batch_role,
            "coordinated_batch_group_id": (
                route.group_id if coordinated_batch_role else ""
            ),
            "fair_debt_score": fair_debt_score,
            "communication_amortized_q_enabled": int(
                cfg.communication_amortized_q_shadow
            ),
            "communication_amortized_q_eligible": int(
                amortized_q_eligible
            ),
            "communication_amortized_q": (
                structural_qmax.q
                if structural_amortized_q and structural_qmax is not None
                else (
                    amortized_q_point.q
                    if amortized_q_point is not None else ""
                )
            ),
            "communication_amortized_q_added": (
                structural_qmax.q - applied_q
                if structural_amortized_q and structural_qmax is not None
                else (
                    amortized_q_point.q - applied_q
                    if amortized_q_point is not None else 0
                )
            ),
            "communication_amortized_added_predicted_duration": (
                structural_qmax.predicted_duration
                - structural.predicted_duration
                if (
                    structural_amortized_q
                    and structural_qmax is not None
                    and structural is not None
                )
                else (
                    amortized_q_point.predicted_duration
                    - point.predicted_duration
                    if amortized_q_point is not None and point is not None
                    else 0.0
                )
            ),
            "communication_amortized_added_safe_duration": (
                structural_qmax.safe_duration - structural.safe_duration
                if (
                    structural_amortized_q
                    and structural_qmax is not None
                    and structural is not None
                )
                else (
                    amortized_q_point.safe_duration - point.safe_duration
                    if amortized_q_point is not None and point is not None
                    else 0.0
                )
            ),
            "communication_amortized_deadline_safe": int(
                structural_amortized_q
                or amortized_q_point is None
                or selected is None
                or selected.mode != "join"
                or amortized_q_point.safe_finish_time
                <= selected.latest_arrival_time
            ),
        })

    def _finalize_reason_aware_batch(
        self, next_virtual_time: float | None = None,
    ) -> None:
        rows = self._reason_aware_routing_trace_buffer
        if not rows:
            return
        batch_time = float(rows[-1]["virtual_time"])
        if (
            next_virtual_time is not None
            and math.isclose(
                batch_time, next_virtual_time,
                rel_tol=0.0, abs_tol=1e-9,
            )
        ):
            return
        batch = []
        for row in reversed(rows):
            if not math.isclose(
                float(row["virtual_time"]), batch_time,
                rel_tol=0.0, abs_tol=1e-9,
            ):
                break
            if int(row.get("elastic_join_avoidance", 0)) == 1:
                batch.append(row)
        if len(batch) <= 1:
            return
        # The widest legal same-Q create window is the only anchor that can
        # safely admit every narrower same-time candidate.
        anchor = max(
            batch,
            key=lambda row: float(row["same_q_create_latest_time"]),
        )
        group_id = -1_000_000 - len(rows)
        anchor_latest = float(anchor["same_q_create_latest_time"])
        for row in batch:
            candidate_latest = float(row["same_q_create_latest_time"])
            row["shadow_group_id"] = group_id
            row["coordinated_batch_group_id"] = group_id
            row["recommended_lane"] = "elastic_unified"
            if row is anchor:
                row["shadow_mode"] = "create"
                row["route_reason"] = "coordinated_same_batch_anchor"
                row["anchor_eligible"] = 1
                row["coordinated_batch_role"] = "anchor"
            elif candidate_latest <= anchor_latest:
                row["shadow_mode"] = "join"
                row["route_reason"] = "coordinated_same_batch_join"
                row["anchor_eligible"] = 0
                row["coordinated_batch_role"] = "join"
                row["shadow_deadline_safe"] = 1
            else:
                # Defensive fallback; max(latest) should make this unreachable.
                row["shadow_mode"] = "create"
                row["route_reason"] = "coordinated_incompatible_create"
                row["coordinated_batch_role"] = "extra_anchor"
            row["recommendation_changed"] = 1

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
            "predicted_finish_times": {client_id: point.predicted_finish_time},
            "safe_finish_times": {client_id: point.safe_finish_time},
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

    def pop_effective_service_q_traces(self):
        rows = self._effective_service_q_trace_buffer
        self._effective_service_q_trace_buffer = []
        return rows

    def pop_effective_service_region_traces(self):
        rows = self._effective_service_region_trace_buffer
        self._effective_service_region_trace_buffer = []
        return rows

    def pop_effective_service_shadow_outcome_traces(self):
        self._settle_effective_service_shadow_groups()
        rows = self._effective_service_shadow_outcome_trace_buffer
        self._effective_service_shadow_outcome_trace_buffer = []
        return rows

    def pop_reason_aware_routing_traces(self):
        self._finalize_reason_aware_batch()
        rows = self._reason_aware_routing_trace_buffer
        self._reason_aware_routing_trace_buffer = []
        return rows

    def pop_fair_contribution_traces(self):
        rows = self._fair_contribution_trace_buffer
        self._fair_contribution_trace_buffer = []
        return rows

    def pop_contribution_restoration_traces(self):
        rows = self._contribution_restoration_trace_buffer
        self._contribution_restoration_trace_buffer = []
        return rows

    def pop_micro_hold_traces(self):
        rows = self._micro_hold_trace_buffer
        self._micro_hold_trace_buffer = []
        return rows

    def pop_unified_batch_dispatch_traces(self):
        rows = self._unified_batch_trace_buffer
        self._unified_batch_trace_buffer = []
        return rows
