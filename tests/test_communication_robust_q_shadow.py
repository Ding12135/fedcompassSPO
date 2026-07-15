"""固定状态组时间窗的通信稳健Q Shadow测试。"""

import unittest
from types import SimpleNamespace

from su_compass.scheduling.policies import CommunicationRobustQShadowPolicy
from su_compass.scheduling.types import LatencyPrediction


class _LinearPredictor:
    @property
    def name(self):
        return "linear"

    def predict(self, context):
        duration = context.local_steps * 0.1 + 5.0
        return LatencyPrediction(
            predictor_name=self.name,
            predicted_duration=duration,
            predicted_finish_time=context.dispatch_time + duration,
            uncertainty=1.0,
            safe_duration=duration + 1.0,
        )


class CommunicationRobustQShadowTest(unittest.TestCase):
    def test_reduces_to_largest_robust_q(self):
        state = SimpleNamespace(communication_time_std=2.0)
        result = CommunicationRobustQShadowPolicy(
            _LinearPredictor(), risk_beta=2.0
        ).recommend(
            client_id="client_0", dispatch_time=0.0, speed_smoothed=0.1,
            runtime_state=state, original_q=100, group_latest_time=17.0,
            qmin=40, qmax=200,
        )
        # 额外风险为3；safe=0.1Q+6，因此最大可行Q为80。
        self.assertEqual(result.recommended_q, 80)
        self.assertEqual(result.q_reduction, 20)
        self.assertTrue(result.robust_safe_feasible)

    def test_keeps_original_when_already_safe(self):
        state = SimpleNamespace(communication_time_std=0.5)
        result = CommunicationRobustQShadowPolicy(
            _LinearPredictor(), risk_beta=1.0
        ).recommend(
            client_id="client_0", dispatch_time=0.0, speed_smoothed=0.1,
            runtime_state=state, original_q=60, group_latest_time=13.0,
            qmin=40, qmax=200,
        )
        self.assertEqual(result.recommended_q, 60)
        self.assertEqual(result.shadow_action, "keep_original_q")

    def test_q_never_below_original_bounds(self):
        state = SimpleNamespace(communication_time_std=100.0)
        result = CommunicationRobustQShadowPolicy(
            _LinearPredictor(), risk_beta=2.0
        ).recommend(
            client_id="client_0", dispatch_time=0.0, speed_smoothed=0.1,
            runtime_state=state, original_q=50, group_latest_time=1.0,
            qmin=40, qmax=200,
        )
        self.assertEqual(result.recommended_q, 50)
        self.assertFalse(result.robust_safe_feasible)


if __name__ == "__main__":
    unittest.main()
