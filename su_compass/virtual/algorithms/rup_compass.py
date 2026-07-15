"""RUP-Compass controller with independently switchable scheduling layers.

Existing-group workload assignment and an optional risk-constrained admission
gate are overridden.  Rejected mismatches still use FedCompass' original group
creation; deadline handling, general buffer semantics and aggregation remain
unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from su_compass.scheduling.policies import RUPConfig, RUPWorkloadPolicy
from su_compass.scheduling.predictors import AdaptiveLatencyPredictor
from su_compass.virtual.algorithms.fedcompass import VirtualFedCompassController
from su_compass.virtual.event import VirtualEvent


class VirtualRUPCompassController(VirtualFedCompassController):
    def __init__(self, *args, rup_config: RUPConfig, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rup_config = rup_config
        self._rup_policy = RUPWorkloadPolicy(AdaptiveLatencyPredictor(), rup_config)
        self._client_runtime_states: Dict[str, Any] = {}
        self._rup_trace_buffer: List[Dict[str, Any]] = []
        self._group_admission_trace_buffer: List[Dict[str, Any]] = []
        # Apply rejection happens before any group/controller mutation.  Keep the
        # trace row by reference until FedCompass _create_group() supplies the
        # actual group and Q.
        self._pending_group_admission_by_client: Dict[str, Dict[str, Any]] = {}

    def update_global_accuracy(self, accuracy: float) -> None:
        """Expose the latest validation accuracy to stage-aware workload policy."""
        self._rup_policy.update_global_accuracy(accuracy)

    @property
    def algorithm_name(self) -> str:
        return "rup_compass"

    def on_client_upload(self, event: VirtualEvent, virtual_now: float):
        state = event.payload.get("runtime_state")
        if state is not None:
            self._client_runtime_states[event.client_id] = state
        report = event.payload.get("report")
        observation = event.payload.get("rup_training_observation")
        if report is not None:
            self._rup_policy.observe_upload(
                event.client_id, report.round_time, observation,
                upload_time=virtual_now, late=bool(report.late),
            )
        return super().on_client_upload(event, virtual_now)

    def pop_rup_outcomes(self) -> List[Dict[str, Any]]:
        return self._rup_policy.pop_outcomes()

    def get_rup_terminal_state(self) -> Dict[str, Any]:
        return self._rup_policy.terminal_state()

    def _assign_group(self, client_id: str):
        """Trace first/new-group passthroughs; existing-group decisions trace themselves."""
        before = len(self._rup_trace_buffer)
        events = super()._assign_group(client_id)
        if len(self._rup_trace_buffer) == before:
            group_id = self.client_info[client_id].get("goa", -1)
            if group_id >= 0 and group_id in self.arrival_group:
                group = self.arrival_group[group_id]
                q = int(self.client_info[client_id]["local_steps"])
                decision = self._rup_policy.passthrough(
                    client_id=client_id, q=q, dispatch_time=self._virtual_now,
                    expected_arrival_time=group["expected_arrival_time"],
                    latest_arrival_time=group["latest_arrival_time"],
                    reason="fedcompass_group_creation_passthrough",
                )
                trace = decision.to_trace(self._virtual_now, group_id)
                trace["decision"] = "create_or_first_group"
                trace["profile_type"] = ""
                self._rup_trace_buffer.append(trace)
        return events

    def _join_group(self, client_id: str) -> bool:
        """Reproduce FedCompass group selection, then replace only its Q."""
        curr_time = self._virtual_now
        speed = self.client_info[client_id]["speed"]
        assigned_group = -1
        baseline_q = -1
        for group_id in self.arrival_group:
            remaining = self.arrival_group[group_id]["expected_arrival_time"] - curr_time
            if remaining <= 0:
                continue
            candidate_q = math.floor(remaining / speed)
            if (
                candidate_q < self.min_local_steps
                or candidate_q < baseline_q
                or candidate_q > self.max_local_steps
            ):
                continue
            assigned_group = group_id
            baseline_q = candidate_q
        if assigned_group == -1:
            return False

        group = self.arrival_group[assigned_group]
        decision = self._rup_policy.decide(
            client_id=client_id,
            dispatch_time=curr_time,
            speed_smoothed=speed,
            runtime_state=self._client_runtime_states.get(client_id),
            expected_arrival_time=group["expected_arrival_time"],
            latest_arrival_time=group["latest_arrival_time"],
            qmin=self.min_local_steps,
            qmax=self.max_local_steps,
            baseline_q=baseline_q,
        )
        applied_q = decision.applied_q

        group_mismatch = not decision.state_safe_feasible
        requested_admission_mode = self.rup_config.group_admission_mode
        admission_apply_mode = requested_admission_mode in {"apply", "conservative"}
        current_group_size = len(group["clients"])
        expected_window = max(
            group["latest_arrival_time"] - group["expected_arrival_time"], 0.0
        )
        late_slack = expected_window * self.rup_config.group_admission_late_slack_ratio
        lateness_margin = decision.predicted_finish_time - group["latest_arrival_time"]
        severe_late_risk = lateness_margin > late_slack
        preserves_group_size = current_group_size >= self.rup_config.group_admission_min_group_size
        conservative_reject = (
            requested_admission_mode == "conservative"
            and group_mismatch
            and severe_late_risk
            and preserves_group_size
        )
        apply_reject = requested_admission_mode == "apply" and group_mismatch
        admission_can_apply = self.rup_config.mode == "apply" and admission_apply_mode
        should_reject = admission_can_apply and (apply_reject or conservative_reject)
        admitted = not should_reject
        if requested_admission_mode == "off":
            shadow_action = "join_existing_group"
            applied_action = "join_existing_group"
            reason = "rup_group_admission_off"
        elif requested_admission_mode == "conservative" and group_mismatch:
            shadow_action = "create_group"
            applied_action = "create_group" if not admitted else "join_existing_group"
            if admitted and not severe_late_risk:
                reason = "rup_conservative_keep_join_low_late_risk"
            elif admitted and not preserves_group_size:
                reason = "rup_conservative_keep_join_small_group"
            elif not admitted:
                reason = "rup_conservative_reject_to_fedcompass_create_group"
            else:
                reason = "rup_conservative_shadow_keep_join"
        elif group_mismatch:
            shadow_action = "create_group"
            applied_action = "create_group" if not admitted else "join_existing_group"
            reason = (
                "rup_group_mismatch_reject_to_fedcompass_create_group"
                if not admitted else "rup_group_mismatch_shadow_keep_join"
            )
        else:
            shadow_action = "join_existing_group"
            applied_action = "join_existing_group"
            reason = "rup_safe_q_admit"

        admission_trace = {
            "virtual_time": curr_time,
            "client_id": client_id,
            "mode": requested_admission_mode,
            "candidate_group_id": assigned_group,
            "fedcompass_q": baseline_q,
            "candidate_trust_q": applied_q,
            "trust_q_safe_feasible": int(decision.state_safe_feasible),
            "group_mismatch": int(group_mismatch),
            "current_group_size": current_group_size,
            "conservative_min_group_size": self.rup_config.group_admission_min_group_size,
            "predicted_finish_time": decision.predicted_finish_time,
            "safe_finish_time": decision.safe_finish_time,
            "lateness_margin": lateness_margin,
            "late_slack": late_slack,
            "severe_late_risk": int(severe_late_risk),
            "preserves_group_size": int(preserves_group_size),
            "shadow_action": shadow_action,
            "applied_action": applied_action,
            "admitted": int(admitted),
            "fedcompass_create_group_id": self.group_counter if not admitted else -1,
            "actual_group_id": assigned_group if admitted else -1,
            "actual_dispatched_q": applied_q if admitted else -1,
            "reason": reason,
        }
        self._group_admission_trace_buffer.append(admission_trace)

        trace = decision.to_trace(curr_time, assigned_group)
        trace["decision"] = "join_group" if admitted else "reject_group_create"
        trace["profile_type"] = ""
        self._rup_trace_buffer.append(trace)
        if not admitted:
            self._pending_group_admission_by_client[client_id] = admission_trace
            return False

        group["clients"].append(client_id)
        self.client_info[client_id]["goa"] = assigned_group
        self.client_info[client_id]["local_steps"] = applied_q
        self.client_info[client_id]["start_time"] = curr_time

        remaining = group["expected_arrival_time"] - curr_time
        self._record_dispatch_decision(
            client_id=client_id,
            decision="join_group",
            assigned_group=assigned_group,
            assigned_steps=applied_q,
            speed_raw=speed,
            remaining_time=remaining,
            target_arrival=group["expected_arrival_time"],
            latest_arrival=group["latest_arrival_time"],
        )
        return True

    def _create_group(self, client_id: str):
        """Use unmodified FedCompass creation and enrich a rejected admission."""
        new_events = super()._create_group(client_id)
        pending = self._pending_group_admission_by_client.pop(client_id, None)
        if pending is not None:
            pending["actual_group_id"] = int(self.client_info[client_id]["goa"])
            pending["actual_dispatched_q"] = int(
                self.client_info[client_id]["local_steps"]
            )
        return new_events

    def pop_rup_traces(self) -> List[Dict[str, Any]]:
        rows = list(self._rup_trace_buffer)
        self._rup_trace_buffer.clear()
        return rows

    def pop_group_admission_traces(self) -> List[Dict[str, Any]]:
        """Return RUP group-gate facts through the shared admission trace."""
        rows = list(self._group_admission_trace_buffer)
        self._group_admission_trace_buffer.clear()
        return rows
