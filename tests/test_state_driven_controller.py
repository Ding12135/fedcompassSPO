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
