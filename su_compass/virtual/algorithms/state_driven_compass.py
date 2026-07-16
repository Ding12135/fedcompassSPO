"""State-native FedCompass scheduler with modular ablation modes."""

from __future__ import annotations

from typing import Any, Dict, List

from su_compass.scheduling.fedcompass_reference import (
    existing_group_reference,
    new_group_reference_q,
    new_group_reference_window,
)
from su_compass.scheduling.policies.joint_group_q import JointGroupQPolicy
from su_compass.scheduling.state_time_model import StateTimeModel, state_group_window
from su_compass.scheduling.state_driven_config import StateDrivenConfig
from su_compass.virtual.algorithms.fedcompass import VirtualFedCompassController
from su_compass.virtual.event import EventType, VirtualEvent


class VirtualStateDrivenCompassController(VirtualFedCompassController):
    def __init__(self, *args, state_driven_config: StateDrivenConfig, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state_driven_config = state_driven_config
        self._state_time_model = StateTimeModel()
        self._joint_policy = JointGroupQPolicy(state_driven_config.target_band_ratio)
        self._runtime_states: Dict[str, Any] = {}
        self._state_time_trace_buffer: List[dict] = []
        self._joint_trace_buffer: List[dict] = []
        self._candidate_trace_buffer: List[dict] = []
        self._state_group_creation_trace_buffer: List[dict] = []

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
        state = event.payload.get("runtime_state")
        if state is not None:
            self._runtime_states[event.client_id] = state
        return super().on_client_upload(event, virtual_now)

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
        return self._create_state_group(client_id, q, fed_reference_q=fed_q)

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
