"""通信尾部风险Shadow测试。"""

import unittest
from types import SimpleNamespace

from su_compass.scheduling.policies import CommunicationTailRiskShadowPolicy


class CommunicationTailRiskShadowTest(unittest.TestCase):
    def test_p90_margin_only_adds_uncovered_tail(self):
        state = SimpleNamespace(
            num_reports=5,
            communication_time_mean=10.0,
            communication_time_std=2.0,
            communication_time_p90=15.0,
            communication_time_recent_max=18.0,
        )
        result = CommunicationTailRiskShadowPolicy().evaluate(
            runtime_state=state, original_safe_slack=2.5,
            existing_uncertainty=3.0,
        )
        self.assertEqual(result.incremental_p90_margin, 2.0)
        self.assertEqual(result.p90_calibrated_safe_slack, 0.5)
        self.assertTrue(result.p90_safe_feasible)
        self.assertFalse(result.max_safe_feasible)

    def test_insufficient_history_is_explicit(self):
        state = SimpleNamespace(
            num_reports=2,
            communication_time_mean=10.0,
            communication_time_std=1.0,
            communication_time_p90=12.0,
            communication_time_recent_max=12.0,
        )
        result = CommunicationTailRiskShadowPolicy(min_reports=3).evaluate(
            runtime_state=state, original_safe_slack=5.0,
            existing_uncertainty=1.0,
        )
        self.assertEqual(result.shadow_action, "insufficient_history")
        self.assertFalse(result.p90_safe_feasible)

    def test_policy_does_not_mutate_runtime_state(self):
        state = SimpleNamespace(
            num_reports=4,
            communication_time_mean=10.0,
            communication_time_std=1.0,
            communication_time_p90=13.0,
            communication_time_recent_max=14.0,
        )
        before = dict(vars(state))
        CommunicationTailRiskShadowPolicy().evaluate(
            runtime_state=state, original_safe_slack=1.0,
            existing_uncertainty=1.0,
        )
        self.assertEqual(vars(state), before)


if __name__ == "__main__":
    unittest.main()
