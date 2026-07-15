"""FedCompass原始新建组反事实Shadow测试。"""

import copy
import unittest

from su_compass.scheduling.policies import (
    GroupCreationCounterfactualShadowPolicy,
    calculate_fedcompass_new_group_plan,
)
from su_compass.scheduling.types import LatencyPrediction


class _Predictor:
    @property
    def name(self):
        return "fake"

    def __init__(self, safe_duration):
        self.safe_duration = safe_duration

    def predict(self, context):
        return LatencyPrediction(
            predicted_duration=self.safe_duration,
            predicted_finish_time=context.dispatch_time + self.safe_duration,
            safe_duration=self.safe_duration,
            compute_duration=self.safe_duration,
            communication_duration=0.0,
            spike_duration=0.0,
            availability_duration=0.0,
            uncertainty=0.0,
            used_fallback=False,
            num_reports=1,
            predictor_name="fake",
        )


class GroupCreationCounterfactualTest(unittest.TestCase):
    def setUp(self):
        self.groups = {
            3: {
                "clients": ["fast"], "arrived_clients": [],
                "expected_arrival_time": 20.0, "latest_arrival_time": 22.0,
            }
        }
        self.client_info = {
            "fast": {"speed": 0.05}, "target": {"speed": 0.1},
        }

    def test_original_formula_and_no_mutation(self):
        groups_before = copy.deepcopy(self.groups)
        clients_before = copy.deepcopy(self.client_info)
        plan = calculate_fedcompass_new_group_plan(
            dispatch_time=10.0, client_speed=0.1, groups=self.groups,
            client_info=self.client_info, qmin=40, qmax=200,
            latest_time_factor=1.2,
        )
        # est=22+0.05*200=32；floor((32-10)/0.1)=220超过Qmax，
        # 原公式没有采纳该候选，最终回退Qmax。
        self.assertEqual(plan.local_steps, 200)
        self.assertEqual(plan.expected_arrival_time, 30.0)
        self.assertEqual(plan.latest_arrival_time, 34.0)
        self.assertEqual(self.groups, groups_before)
        self.assertEqual(self.client_info, clients_before)

    def _evaluate(self, safe_duration):
        return GroupCreationCounterfactualShadowPolicy(_Predictor(safe_duration)).evaluate(
            client_id="target", dispatch_time=10.0, speed_smoothed=0.1,
            runtime_state=None, groups=self.groups, client_info=self.client_info,
            current_group_id=3, current_q=40,
            current_predicted_finish_time=30.0, current_safe_finish_time=30.0,
            qmin=40, qmax=200, latest_time_factor=1.2,
        )

    def test_counterfactual_safe(self):
        result = self._evaluate(20.0)
        self.assertTrue(result.counterfactual_safe_feasible)
        self.assertEqual(result.shadow_action, "create_group_candidate")
        self.assertGreaterEqual(result.counterfactual_q, 40)
        self.assertLessEqual(result.counterfactual_q, 200)

    def test_safer_but_still_unsafe(self):
        result = self._evaluate(25.0)
        self.assertFalse(result.counterfactual_safe_feasible)
        self.assertEqual(result.shadow_action, "observe_safer_but_unsafe")

    def test_not_safer(self):
        result = self._evaluate(40.0)
        self.assertEqual(result.shadow_action, "keep_current_candidate")


if __name__ == "__main__":
    unittest.main()
