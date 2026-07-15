"""Controller-level invariants for RUP risk-constrained group admission."""

import unittest

from su_compass.scheduling.policies import RUPConfig

try:
    from su_compass.virtual.algorithms.rup_compass import VirtualRUPCompassController
except ModuleNotFoundError as exc:
    if exc.name not in {"torch", "numpy"}:
        raise
    VirtualRUPCompassController = None


class _Decision:
    def __init__(self, safe, applied_q=80):
        self.state_safe_feasible = safe
        self.applied_q = applied_q

    def to_trace(self, virtual_time, assigned_group):
        return {
            "virtual_time": virtual_time,
            "assigned_group": assigned_group,
        }


class _Policy:
    def __init__(self, decision):
        self.decision = decision

    def decide(self, **kwargs):
        return self.decision


@unittest.skipIf(
    VirtualRUPCompassController is None,
    "controller integration test requires the experiment torch/numpy runtime",
)
class RUPGroupAdmissionTest(unittest.TestCase):
    def _controller(self, admission_mode, safe):
        controller = VirtualRUPCompassController(
            aggregator=object(), num_clients=2, min_local_steps=40,
            max_local_steps=200, speed_momentum=0.9,
            latest_time_factor=1.2, num_global_epochs=10,
            rup_config=RUPConfig(
                mode="apply", group_admission_mode=admission_mode,
            ),
        )
        controller._rup_policy = _Policy(_Decision(safe=safe))
        controller._virtual_now = 0.0
        controller.group_counter = 1
        controller.client_info = {
            "client_0": {"speed": 0.1, "goa": -1},
            "client_1": {"speed": 0.1, "goa": 0},
        }
        controller.arrival_group = {
            0: {
                "clients": ["client_1"], "arrived_clients": [],
                "expected_arrival_time": 10.0, "latest_arrival_time": 12.0,
                "created_time": 0.0,
            }
        }
        return controller

    def test_apply_rejects_mismatch_then_uses_original_create_group(self):
        controller = self._controller("apply", safe=False)
        self.assertFalse(controller._join_group("client_0"))
        self.assertNotIn("client_0", controller.arrival_group[0]["clients"])

        controller._create_group("client_0")
        self.assertEqual(controller.client_info["client_0"]["goa"], 1)
        row = controller.pop_group_admission_traces()[0]
        self.assertEqual(row["admitted"], 0)
        self.assertEqual(row["actual_group_id"], 1)
        self.assertEqual(
            row["actual_dispatched_q"],
            controller.client_info["client_0"]["local_steps"],
        )

    def test_shadow_keeps_original_existing_group(self):
        controller = self._controller("shadow", safe=False)
        self.assertTrue(controller._join_group("client_0"))
        self.assertEqual(controller.client_info["client_0"]["goa"], 0)
        row = controller.pop_group_admission_traces()[0]
        self.assertEqual(row["shadow_action"], "create_group")
        self.assertEqual(row["applied_action"], "join_existing_group")


if __name__ == "__main__":
    unittest.main()
