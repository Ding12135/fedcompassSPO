import copy
import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from su_compass.scheduling.state_driven_config import StateDrivenConfig
from su_compass.scheduling.state_time_model import QTimeCandidate
from su_compass.virtual.algorithms.fedcompass import VirtualFedCompassController
from su_compass.virtual.algorithms.state_driven_compass import VirtualStateDrivenCompassController
from su_compass.virtual.event import EventType, VirtualEvent
from su_compass.virtual.trace import TraceWriter


class _FakeTimeModel:
    name = "fake"
    version = "test"

    def predict_q(self, *, dispatch_time, q, **kwargs):
        duration = q * 0.1
        return QTimeCandidate(
            q=q, predictor_name="fake", predictor_source="mature_state",
            num_reports=10, used_fallback=False, fallback_reason="",
            predicted_duration=duration, safe_duration=duration + 2.0,
            predicted_finish_time=dispatch_time + duration,
            safe_finish_time=dispatch_time + duration + 2.0,
            uncertainty=2.0, compute_duration=duration,
            communication_duration=0, spike_duration=0,
            availability_duration=0, availability_risk_duration=0,
        )

    def predict_curve(self, *, qs, **kwargs):
        return [self.predict_q(q=q, **kwargs) for q in qs], True


class _FallbackTimeModel(_FakeTimeModel):
    def predict_q(self, *, dispatch_time, q, **kwargs):
        base = super().predict_q(dispatch_time=dispatch_time, q=q, **kwargs)
        return QTimeCandidate(**{
            **base.__dict__, "predictor_source": "fedcompass_fallback",
            "used_fallback": True, "fallback_reason": "predictor_fallback",
            "safe_duration": base.predicted_duration,
            "safe_finish_time": base.predicted_finish_time,
        })


def _base(controller):
    controller._client_dispatch_index = {"client_0": 0}
    controller.client_info = {
        "client_0": {"speed": 0.1, "timestamp": 0, "local_steps": 100, "goa": -1}
    }
    controller._virtual_now = 5.0
    controller._prepare_next_decision("client_0")
    return controller


class StateDrivenControllerTest(unittest.TestCase):
    @staticmethod
    def _warm_pooled_safety(controller):
        for index in range(8):
            client_id = f"warm_{index}"
            controller._calibrated_shadow.predict(
                client_id=client_id, local_steps=40,
                baseline_duration=4.0, baseline_safe_duration=5.0,
            )
            controller._calibrated_shadow.observe(
                client_id=client_id, local_steps=40, actual_duration=5.0,
                compute_duration=4.0, communication_duration=1.0,
            )

    def test_dynamic_frontier_uses_only_current_pending_predictions(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=2, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(),
        ))
        group = {
            "clients": ["client_0"],
            "expected_arrival_time": 80.0,
            "latest_arrival_time": 100.0,
            "predicted_finish_times": {"client_0": 30.0, "finished": 90.0},
        }
        self.assertEqual(controller._dynamic_group_frontier(group), 30.0)

    def test_lyapunov_queue_updates_only_from_real_aggregation(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(
                lyapunov_mode="shadow", lyapunov_rhythm_target=10,
                lyapunov_client_target_rates="client_0=0.1",
            ),
        ))
        controller._client_ids = ["client_0"]
        controller._virtual_now = 15
        controller._record_aggregation_trace(
            "single", {"client_0": 0}, {"client_0": 100},
            {"client_0": 0}, {"client_0": "d"}, -1, 1, 0,
        )
        self.assertEqual(controller._lyapunov_rhythm_debt, 5)
        self.assertEqual(controller._lyapunov_workload_debt["client_0"], 1.0)
        row = controller.pop_lyapunov_queue_traces()[0]
        self.assertEqual(row["delta_t"], 15)

    def test_lyapunov_shadow_keeps_safe_reuse_action(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(
                new_group_window_mode="state_apply_fixed_q",
                new_group_q_mode="fedcompass", lyapunov_mode="shadow",
            ),
        ))
        controller._state_time_model = _FakeTimeModel()
        controller.arrival_group = {
            0: {"clients": [], "arrived_clients": [],
                "expected_arrival_time": 15.0, "latest_arrival_time": 28.0,
                "created_time": 0.0}
        }
        controller._assign_group("client_0")
        self.assertEqual(controller.client_info["client_0"]["goa"], 0)
        trace = controller.pop_lyapunov_decision_traces()[0]
        self.assertEqual(trace["mode"], "shadow")
        self.assertEqual(trace["applied_mode"], "join")

    def test_effective_service_v2_region_three_recommends_controlled_create(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(
                new_group_window_mode="state_apply_fixed_q",
                new_group_q_mode="fedcompass", lyapunov_mode="shadow",
                lyapunov_action_scope="effective_service_v2",
                lyapunov_enable_workload_queue=False,
                lyapunov_q_reference_spec="client_0=80",
                finite_sample_safety_calibration=True,
            ),
        ))
        controller._state_time_model = _FakeTimeModel()
        controller._assign_group("client_0")
        region = controller.pop_effective_service_region_traces()[0]
        self.assertEqual(region["region"], "region_3")
        self.assertEqual(region["recommended_mode"], "create")
        self.assertIn(region["recommended_q"], {40, 64, 80, 88})

    def test_effective_service_v2_obvious_join_blocks_create_competition(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(
                lyapunov_mode="shadow", lyapunov_action_scope="effective_service_v2",
            ),
        ))
        join = SimpleNamespace(
            legal=True, mode="join", external_wait=0.1, holding_wait=1.0, score=1.0,
            q=80, group_id=3,
        )
        create = SimpleNamespace(
            legal=True, mode="create", external_wait=10.0, holding_wait=10.0,
            score=-10.0,
            q=80, group_id=-1,
        )
        decision = controller._choose_effective_service_v2(
            "client_0", [join, create], recruit_expected=10.0,
            recruit_safe=12.0, recruitment_source="test",
            predicted_group_size=2,
        )
        self.assertEqual(decision.action.mode, "join")
        self.assertEqual(
            controller.pop_effective_service_region_traces()[0]["region"],
            "region_1",
        )

    def test_effective_service_v2_1_cold_start_defers_to_safe_reuse(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(
                new_group_window_mode="state_apply_fixed_q",
                new_group_q_mode="fedcompass", lyapunov_mode="shadow",
                lyapunov_action_scope="effective_service_v2_1",
                lyapunov_enable_workload_queue=False,
                lyapunov_q_reference_spec="client_0=80",
                finite_sample_safety_calibration=True,
            ),
        ))
        controller._state_time_model = _FakeTimeModel()
        controller._assign_group("client_0")
        region = controller.pop_effective_service_region_traces()[0]
        self.assertEqual(region["region"], "cold_start")
        self.assertEqual(region["recommended_mode"], "defer")
        self.assertIsNone(controller._effective_service_shadow_groups)

    def test_effective_service_v2_1_create_becomes_joinable_shadow_group(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=2, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(
                new_group_window_mode="state_apply_fixed_q",
                new_group_q_mode="fedcompass", lyapunov_mode="shadow",
                lyapunov_action_scope="effective_service_v2_1",
                lyapunov_enable_workload_queue=False,
                lyapunov_q_reference_spec="client_0=80,client_1=80",
                finite_sample_safety_calibration=True,
            ),
        ))
        controller._state_time_model = _FakeTimeModel()
        self._warm_pooled_safety(controller)
        controller._assign_group("client_0")
        first = controller.pop_effective_service_region_traces()[0]
        self.assertEqual(first["recommended_mode"], "create")
        controller.client_info["client_1"] = {
            "speed": 0.1, "timestamp": 0, "local_steps": 100, "goa": -1,
        }
        controller._client_dispatch_index["client_1"] = 0
        controller._virtual_now = 6.0
        controller._assign_group("client_1")
        second = controller.pop_effective_service_region_traces()[0]
        self.assertEqual(second["recommended_mode"], "join")
        groups = controller._effective_service_shadow_groups
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(next(iter(groups.values()))["clients"]), 2)

    def test_effective_service_v2_1_seed_ignores_expired_and_empty_groups(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=2, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(
                lyapunov_mode="shadow",
                lyapunov_action_scope="effective_service_v2_1",
            ),
        ))
        controller.arrival_group = {
            1: {"clients": ["client_0"], "latest_arrival_time": 5.0},
            2: {"clients": [], "latest_arrival_time": 20.0},
            3: {"clients": ["client_1"], "latest_arrival_time": 20.0},
        }
        groups = controller._ensure_effective_service_shadow_groups()
        self.assertEqual(set(groups), {3})
        self.assertEqual(groups[3]["clients"], ["client_1"])
        self.assertEqual(controller.pop_effective_service_shadow_outcome_traces(), [])

    def test_effective_service_v2_1_blocks_duplicate_counterfactual_dispatch(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(
                new_group_window_mode="state_apply_fixed_q",
                new_group_q_mode="fedcompass", lyapunov_mode="shadow",
                lyapunov_action_scope="effective_service_v2_1",
                lyapunov_enable_workload_queue=False,
                lyapunov_q_reference_spec="client_0=80",
                finite_sample_safety_calibration=True,
            ),
        ))
        controller._state_time_model = _FakeTimeModel()
        self._warm_pooled_safety(controller)
        controller._assign_group("client_0")
        controller.pop_effective_service_region_traces()
        controller._virtual_now = 6.0
        controller._prepare_next_decision("client_0")
        *_, decision = controller._lyapunov_actions("client_0")
        self.assertFalse(decision.feasible)
        region = controller.pop_effective_service_region_traces()[0]
        self.assertEqual(region["reason"], "shadow_dispatch_blocked")

    def test_effective_service_v2_1_apply_uses_selected_create_window(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(
                new_group_window_mode="state_apply_fixed_q",
                new_group_q_mode="fedcompass", lyapunov_mode="apply",
                lyapunov_action_scope="effective_service_v2_1",
                lyapunov_enable_workload_queue=False,
                lyapunov_q_reference_spec="client_0=80",
                finite_sample_safety_calibration=True,
            ),
        ))
        controller._state_time_model = _FakeTimeModel()
        self._warm_pooled_safety(controller)
        events = controller._assign_group("client_0")
        group = next(iter(controller.arrival_group.values()))
        self.assertEqual(group["time_source"], "effective_service_v2_1_apply")
        self.assertEqual(events[0].time, group["latest_arrival_time"])
        self.assertIsNone(controller._effective_service_shadow_groups)

    def test_effective_service_v2_2_clips_and_prices_recruitment_safe_time(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(
                lyapunov_mode="apply",
                lyapunov_action_scope="effective_service_v2_1",
                lyapunov_q_reference_spec="client_0=80",
                finite_sample_safety_calibration=True,
                lyapunov_recruit_safe_cap_ratio=2.0,
                lyapunov_create_safe_cost=True,
            ),
        ))
        controller._state_time_model = _FakeTimeModel()
        self._warm_pooled_safety(controller)
        controller._lyapunov_recent_intervals = [10.0] * 9 + [20.0, 99.0, 120.0]
        curve, _ = controller._curve("client_0")
        actions, expected, safe, source, _, expected_raw, raw, cap = (
            controller._effective_service_v2_create_actions("client_0", curve)
        )
        self.assertEqual(expected_raw, 10.0)
        self.assertEqual(raw, 120.0)
        self.assertAlmostEqual(cap, 32.8)
        self.assertAlmostEqual(safe, 32.8)
        self.assertIn("rhythm_trust_clipped", source)
        self.assertTrue(all(
            action.predicted_sojourn == action.predicted_duration + safe
            for action in actions
        ))

    def test_lyapunov_apply_can_create_instead_of_long_holding_join(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=2, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(
                new_group_window_mode="state_apply_fixed_q",
                new_group_q_mode="fedcompass", lyapunov_mode="apply",
                lyapunov_max_holding_wait=100,
            ),
        ))
        controller._state_time_model = _FakeTimeModel()
        controller._lyapunov_rhythm_debt = 100
        controller.client_info["client_1"] = {
            "speed": 0.1, "timestamp": 0, "local_steps": 200, "goa": 0,
        }
        controller.arrival_group = {
            0: {"clients": ["client_1"], "arrived_clients": [],
                "expected_arrival_time": 85.0, "latest_arrival_time": 100.0,
                "created_time": 0.0}
        }
        controller.group_counter = 1
        events = controller._assign_group("client_0")
        self.assertTrue(events)
        self.assertNotEqual(controller.client_info["client_0"]["goa"], 0)
        trace = controller.pop_lyapunov_decision_traces()[0]
        self.assertEqual(trace["recommended_mode"], "create")
        self.assertEqual(trace["recommendation_applied"], 1)

    def test_lyapunov_traces_flush_to_independent_files(self):
        with TemporaryDirectory() as directory:
            writer = TraceWriter(directory, "state_driven_compass")
            writer.record_lyapunov_decision({
                key: 0 for key in (
                    "decision_id", "virtual_time", "client_id", "mode",
                    "rhythm_debt", "workload_debt", "num_actions",
                    "num_legal_actions", "rejected_holding_actions",
                    "rejected_q_actions", "recommended_mode",
                    "recommended_group_id", "recommended_q",
                    "recommended_score", "recommended_sojourn",
                    "recommended_holding_wait", "recommended_external_wait",
                    "applied_mode", "applied_group_id", "applied_q",
                    "recommendation_applied",
                )
            })
            writer.record_lyapunov_queue({
                key: 0 for key in (
                    "aggregation_id", "virtual_time", "delta_t", "client_id",
                    "rhythm_debt_before", "rhythm_debt_after",
                    "workload_debt_before", "target_workload_arrival",
                    "effective_work_service", "workload_debt_after",
                    "participated",
                )
            })
            writer.flush()
            from pathlib import Path
            self.assertTrue((Path(directory) / "lyapunov_decision_trace.csv").exists())
            self.assertTrue((Path(directory) / "lyapunov_queue_trace.csv").exists())

    def test_effective_service_region_trace_flushes_cold_and_guarded_rows(self):
        with TemporaryDirectory() as directory:
            writer = TraceWriter(directory, "state_driven_compass")
            writer.record_effective_service_region({
                "decision_id": "cold", "region": "cold_start",
                "recruitment_expected": "", "recruitment_safe": "",
            })
            writer.record_effective_service_region({
                "decision_id": "guarded", "region": "region_3",
                "recruitment_expected": 16.4,
                "recruitment_expected_raw": 60.0,
                "recruitment_safe": 32.8,
                "recruitment_safe_raw": 99.0,
                "recruitment_safe_cap": 32.8,
                "create_safe_cost_enabled": 1,
            })
            writer.flush()
            from pathlib import Path
            path = Path(directory) / "effective_service_region_shadow_trace.csv"
            self.assertTrue(path.exists())
            self.assertIn("recruitment_safe_cap", path.read_text())

    def test_state_group_uses_predicted_and_safe_finish(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(),
        ))
        controller._state_time_model = _FakeTimeModel()
        events = controller._create_state_group("client_0", 200)
        group = controller.arrival_group[0]
        self.assertEqual(group["expected_arrival_time"], 25.0)
        self.assertEqual(group["latest_arrival_time"], 27.0)
        self.assertGreaterEqual(group["latest_arrival_time"], 27.0)
        self.assertEqual(events[0].time, group["latest_arrival_time"])

    def test_shadow_existing_group_matches_fedcompass_mutation(self):
        state = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(
                existing_group_mode="state_shadow",
                new_group_window_mode="state_shadow_fixed_q",
                new_group_q_mode="fedcompass",
            ),
        ))
        state._state_time_model = _FakeTimeModel()
        fed = _base(VirtualFedCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
        ))
        group = {
            0: {"clients": [], "arrived_clients": [],
                "expected_arrival_time": 15.0, "latest_arrival_time": 18.0,
                "created_time": 0.0}
        }
        state.arrival_group = copy.deepcopy(group)
        fed.arrival_group = copy.deepcopy(group)
        self.assertEqual(state._join_group("client_0"), fed._join_group("client_0"))
        self.assertEqual(state.arrival_group, fed.arrival_group)
        for key in ("goa", "local_steps", "start_time"):
            self.assertEqual(state.client_info["client_0"].get(key), fed.client_info["client_0"].get(key))

    def test_cold_start_fallback_preserves_fedcompass_window(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(),
        ))
        controller._state_time_model = _FallbackTimeModel()
        controller._create_state_group("client_0", 200)
        group = controller.arrival_group[0]
        self.assertEqual(group["expected_arrival_time"], 25.0)
        self.assertEqual(group["latest_arrival_time"], 29.0)
        self.assertEqual(group["time_source"], "fedcompass_speed")

    def test_reason_shadow_closes_against_real_fallback_action(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(
                new_group_window_mode="state_apply_fixed_q",
                new_group_q_mode="fedcompass",
                lyapunov_mode="apply",
                lyapunov_action_scope="effective_service_v2_1",
                lyapunov_enable_workload_queue=False,
                reason_aware_routing_shadow=True,
                reason_aware_one_report_structural_shadow=True,
            ),
        ))
        controller._state_time_model = _FallbackTimeModel()
        events = controller._assign_group("client_0")
        self.assertTrue(events)
        row = controller.pop_reason_aware_routing_traces()[0]
        self.assertEqual(row["v23_mode"], "create")
        self.assertEqual(row["v23_group_id"], 0)
        self.assertEqual(row["v23_q"], 200)
        self.assertEqual(row["shadow_q"], 200)
        self.assertEqual(row["q_unchanged"], 1)

    def test_fallback_join_executes_parent_policy(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(),
        ))
        controller._state_time_model = _FallbackTimeModel()
        controller.arrival_group = {
            0: {"clients": [], "arrived_clients": [],
                "expected_arrival_time": 15.0, "latest_arrival_time": 18.0,
                "created_time": 0.0}
        }
        self.assertTrue(controller._join_group("client_0"))
        self.assertEqual(controller.client_info["client_0"]["local_steps"], 100)
        trace = controller.pop_joint_group_q_traces()[0]
        self.assertEqual(trace["fallback_to_fedcompass"], 1)
        self.assertEqual(trace["state_control_active"], 0)

    def test_calibrated_predictor_shadow_closes_on_real_outcome(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(),
        ))
        controller._state_time_model = _FakeTimeModel()
        controller._append_dispatch("client_0", 5.0, 100, 15.0, 17.0)
        decision_id = controller.client_info["client_0"]["decision_id"]
        report = SimpleNamespace(
            round_time=12.0, local_steps=100, train_time=10.0,
            communication_time=2.0, spike_delay=0.0, availability_wait=0.0,
        )
        controller._close_shadow_outcomes(VirtualEvent(
            time=17.0, event_type=EventType.CLIENT_UPLOAD, client_id="client_0",
            payload={"decision_id": decision_id, "report": report},
        ))
        row = controller.pop_calibrated_predictor_shadow_traces()[0]
        self.assertEqual(row["actual_duration"], 12.0)
        self.assertIn("safe_prediction_better", row)
        with TemporaryDirectory() as directory:
            writer = TraceWriter(directory, "state_driven_compass")
            writer.record_calibrated_predictor_shadow(row)
            writer.flush()

    def test_predictor_native_new_group_shadow_has_outcome(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=1, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(),
        ))
        controller._state_time_model = _FakeTimeModel()
        controller.arrival_group = {
            3: {"clients": ["client_0"], "arrived_clients": [],
                "expected_arrival_time": 15.0, "latest_arrival_time": 18.0}
        }
        controller._record_predictor_native_group_shadow(
            client_id="client_0", applied_q=100, fed_reference_q=100,
        )
        decision_id = controller.client_info["client_0"]["decision_id"]
        report = SimpleNamespace(
            round_time=12.0, local_steps=100, train_time=10.0,
            communication_time=2.0, spike_delay=0.0, availability_wait=0.0,
        )
        controller._close_shadow_outcomes(VirtualEvent(
            time=17.0, event_type=EventType.CLIENT_UPLOAD, client_id="client_0",
            payload={"decision_id": decision_id, "report": report},
        ))
        row = controller.pop_predictor_native_group_shadow_traces()[0]
        self.assertEqual(row["applied_q"], 100)
        self.assertIn("counterfactual_actual_duration", row)
        self.assertIn("native_prediction_better", row)
        with TemporaryDirectory() as directory:
            writer = TraceWriter(directory, "state_driven_compass")
            writer.record_predictor_native_group_shadow(row)
            writer.flush()

    def test_fixed_q_state_window_applies_fedcompass_reference_q(self):
        controller = _base(VirtualStateDrivenCompassController(
            aggregator=object(), num_clients=2, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            state_driven_config=StateDrivenConfig(
                existing_group_mode="state_apply",
                new_group_window_mode="state_apply_fixed_q",
                new_group_q_mode="fedcompass",
            ),
        ))
        controller._state_time_model = _FakeTimeModel()
        controller.client_info["client_1"] = {
            "speed": 0.02, "timestamp": 0, "local_steps": 100, "goa": 7,
        }
        controller.arrival_group = {
            7: {"clients": ["client_1"], "arrived_clients": [],
                "expected_arrival_time": 9.0, "latest_arrival_time": 10.0}
        }
        controller._create_group("client_0")
        self.assertEqual(controller.client_info["client_0"]["local_steps"], 90)
        row = controller.pop_state_group_creation_traces()[0]
        self.assertEqual(row["fedcompass_reference_q"], 90)
        self.assertEqual(row["state_assigned_q"], 90)


if __name__ == "__main__":
    unittest.main()
