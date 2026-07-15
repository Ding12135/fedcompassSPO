"""状态感知新组Q可行性Shadow测试。"""

import unittest

from su_compass.scheduling.policies import StateGroupCreationQShadowPolicy
from su_compass.scheduling.types import LatencyPrediction


class _LinearPredictor:
    @property
    def name(self):
        return "linear_fake"

    def __init__(self, fixed_delay):
        self.fixed_delay = fixed_delay

    def predict(self, context):
        duration = context.local_steps * 0.05 + self.fixed_delay
        return LatencyPrediction(
            predictor_name=self.name, predicted_duration=duration,
            predicted_finish_time=context.dispatch_time + duration,
            safe_duration=duration,
        )


class StateGroupCreationQShadowTest(unittest.TestCase):
    def test_alternative_q_can_make_window_safe(self):
        # latest增量斜率=0.1*1.2，大于预测计算斜率0.05；较大的Q能够摊薄固定延迟。
        result = StateGroupCreationQShadowPolicy(_LinearPredictor(8.0)).recommend(
            client_id="client_0", dispatch_time=10.0, speed_smoothed=0.1,
            runtime_state=None, original_q=40,
            original_expected_arrival_time=14.0, original_safe_feasible=False,
            qmin=40, qmax=200, latest_time_factor=1.2,
        )
        self.assertGreater(result.num_safe_q_candidates, 0)
        self.assertGreater(result.recommended_q, 40)
        self.assertGreaterEqual(result.recommended_safe_slack, 0.0)
        self.assertEqual(result.reason, "alternative_q_makes_original_window_safe")

    def test_no_safe_q_is_reported_without_fake_repair(self):
        result = StateGroupCreationQShadowPolicy(_LinearPredictor(100.0)).recommend(
            client_id="client_0", dispatch_time=10.0, speed_smoothed=0.1,
            runtime_state=None, original_q=80,
            original_expected_arrival_time=18.0, original_safe_feasible=False,
            qmin=40, qmax=200, latest_time_factor=1.2,
        )
        self.assertEqual(result.num_safe_q_candidates, 0)
        self.assertEqual(result.recommended_q, 80)
        self.assertEqual(result.shadow_action, "no_safe_q_candidate")

    def test_original_safe_q_is_preserved(self):
        result = StateGroupCreationQShadowPolicy(_LinearPredictor(0.0)).recommend(
            client_id="client_0", dispatch_time=10.0, speed_smoothed=0.1,
            runtime_state=None, original_q=100,
            original_expected_arrival_time=20.0, original_safe_feasible=True,
            qmin=40, qmax=200, latest_time_factor=1.2,
        )
        self.assertEqual(result.recommended_q, 100)
        self.assertEqual(result.reason, "original_q_already_safe")


if __name__ == "__main__":
    unittest.main()
