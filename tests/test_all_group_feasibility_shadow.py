"""全量已有组可行性Shadow策略测试。"""

import copy
import unittest

from su_compass.scheduling.policies import AllGroupFeasibilityShadowPolicy
from su_compass.scheduling.policies.q_shadow import QRecommendation


class _FakeQPolicy:
    def recommend(self, **kwargs):
        expected = kwargs["expected_arrival_time"]
        safe = expected >= 20.0
        q = 80 if safe else kwargs["qmin"]
        return QRecommendation(
            recommended_q=q, predicted_duration=1.0,
            predicted_finish_time=expected, safe_duration=1.0,
            safe_finish_time=expected + (0.0 if safe else 100.0),
            expected_deviation=0.0, safe_feasible=safe,
            hit_qmin=q == kwargs["qmin"], hit_qmax=False,
            reason="fake", num_safe_candidates=int(safe),
        )


class AllGroupFeasibilityShadowTest(unittest.TestCase):
    def setUp(self):
        self.policy = AllGroupFeasibilityShadowPolicy(_FakeQPolicy())
        self.groups = {
            1: {"expected_arrival_time": 15.0, "latest_arrival_time": 16.0},
            2: {"expected_arrival_time": 20.0, "latest_arrival_time": 21.0},
        }

    def test_state_only_repairs_current_mismatch_without_mutation(self):
        before = copy.deepcopy(self.groups)
        result = self.policy.recommend(
            client_id="client_0", dispatch_time=10.0, speed_smoothed=0.02,
            runtime_state=None, groups=self.groups, current_group_id=1,
            qmin=40, qmax=200,
        )
        state_only = next(c for c in result.candidates if c.group_id == 2)
        self.assertEqual(state_only.feasibility_class, "state_only")
        self.assertEqual(result.shadow_action, "switch_existing_group")
        self.assertTrue(result.mismatch_repaired)
        self.assertEqual(self.groups, before)

    def test_current_safe_is_kept(self):
        groups = {2: self.groups[2]}
        result = self.policy.recommend(
            client_id="client_0", dispatch_time=10.0, speed_smoothed=0.1,
            runtime_state=None, groups=groups, current_group_id=2,
            qmin=40, qmax=200,
        )
        self.assertEqual(result.shadow_action, "keep_current_group")

    def test_no_safe_group_considers_create(self):
        groups = {1: self.groups[1]}
        result = self.policy.recommend(
            client_id="client_0", dispatch_time=10.0, speed_smoothed=0.1,
            runtime_state=None, groups=groups, current_group_id=1,
            qmin=40, qmax=200,
        )
        self.assertEqual(result.shadow_action, "consider_create_group")
        self.assertEqual(result.recommended_group_id, -1)


if __name__ == "__main__":
    unittest.main()
