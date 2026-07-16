import copy
import unittest

from su_compass.scheduling.state_driven_config import StateDrivenConfig
from su_compass.scheduling.state_time_model import QTimeCandidate
from su_compass.virtual.algorithms.fedcompass import VirtualFedCompassController
from su_compass.virtual.algorithms.state_driven_compass import VirtualStateDrivenCompassController


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


if __name__ == "__main__":
    unittest.main()
