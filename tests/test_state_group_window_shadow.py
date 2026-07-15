"""固定Q状态预测新组时间窗Shadow测试。"""

import unittest

from su_compass.scheduling.policies import StateGroupWindowShadowPolicy
from su_compass.scheduling.types import LatencyPrediction


class _Predictor:
    @property
    def name(self):
        return "window_fake"

    def __init__(self, mean_duration, safe_duration):
        self.mean_duration = mean_duration
        self.safe_duration = safe_duration

    def predict(self, context):
        return LatencyPrediction(
            predictor_name=self.name,
            predicted_duration=self.mean_duration,
            predicted_finish_time=context.dispatch_time + self.mean_duration,
            safe_duration=self.safe_duration,
        )


class StateGroupWindowShadowTest(unittest.TestCase):
    def _evaluate(self, mean_duration, safe_duration):
        return StateGroupWindowShadowPolicy(
            _Predictor(mean_duration, safe_duration)
        ).evaluate(
            client_id="client_0", dispatch_time=10.0, speed_smoothed=0.1,
            runtime_state=None, fixed_q=100,
            speed_expected_arrival_time=20.0,
            speed_latest_arrival_time=22.0,
            qmin=40, qmax=200, latest_time_factor=1.2,
        )

    def test_fixed_q_and_lambda_are_preserved(self):
        result = self._evaluate(mean_duration=15.0, safe_duration=17.0)
        self.assertEqual(result.fixed_q, 100)
        self.assertEqual(result.state_expected_arrival_time, 25.0)
        self.assertEqual(result.state_latest_arrival_time, 28.0)
        self.assertEqual(result.state_safe_slack, 1.0)
        self.assertEqual(result.shadow_action, "state_window_candidate")

    def test_safer_but_still_unsafe_is_not_marked_safe(self):
        result = self._evaluate(mean_duration=15.0, safe_duration=19.0)
        self.assertFalse(result.state_window_safe_feasible)
        self.assertEqual(result.shadow_action, "observe_safer_but_unsafe")

    def test_invalid_fixed_q_fails(self):
        with self.assertRaises(ValueError):
            StateGroupWindowShadowPolicy(_Predictor(1.0, 1.0)).evaluate(
                client_id="client_0", dispatch_time=0.0, speed_smoothed=0.1,
                runtime_state=None, fixed_q=201,
                speed_expected_arrival_time=20.0,
                speed_latest_arrival_time=22.0,
                qmin=40, qmax=200, latest_time_factor=1.2,
            )


if __name__ == "__main__":
    unittest.main()
