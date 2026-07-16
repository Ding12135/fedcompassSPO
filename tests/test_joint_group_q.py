import unittest

from su_compass.scheduling.policies.joint_group_q import JointGroupQPolicy
from su_compass.scheduling.state_time_model import QTimeCandidate, state_group_window


def point(q, finish, safe=None):
    safe = finish if safe is None else safe
    return QTimeCandidate(
        q=q, predictor_name="test", predictor_source="mature_state",
        num_reports=10, used_fallback=False, fallback_reason="",
        predicted_duration=finish, safe_duration=safe,
        predicted_finish_time=finish, safe_finish_time=safe,
        uncertainty=0, compute_duration=finish,
        communication_duration=0, spike_duration=0,
        availability_duration=0, availability_risk_duration=0,
    )


class JointGroupQPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = JointGroupQPolicy(target_band_ratio=0.10)

    def test_earliest_aligned_safe_group_precedes_later_group(self):
        groups = {
            1: {"expected_arrival_time": 10.0, "latest_arrival_time": 20.0},
            2: {"expected_arrival_time": 15.0, "latest_arrival_time": 25.0},
        }
        rows = self.policy.enumerate_candidates(
            now=0, groups=groups, curve=[point(50, 10.5), point(100, 15.0)],
        )
        decision = self.policy.choose(rows)
        self.assertTrue(decision.feasible)
        self.assertEqual(decision.group_id, 1)
        self.assertEqual(decision.q, 50)

    def test_alignment_tie_chooses_larger_q(self):
        groups = {1: {"expected_arrival_time": 10.0, "latest_arrival_time": 20.0}}
        rows = self.policy.enumerate_candidates(
            now=0, groups=groups, curve=[point(50, 9.5), point(60, 10.5)],
        )
        self.assertEqual(self.policy.choose(rows).q, 60)

    def test_deadline_safe_without_target_alignment_is_infeasible(self):
        groups = {1: {"expected_arrival_time": 10.0, "latest_arrival_time": 20.0}}
        rows = self.policy.enumerate_candidates(
            now=0, groups=groups, curve=[point(50, 15.0, 16.0)],
        )
        self.assertTrue(rows[0].deadline_safe)
        self.assertFalse(rows[0].target_aligned)
        self.assertFalse(self.policy.choose(rows).feasible)

    def test_unsafe_aligned_candidate_is_infeasible(self):
        groups = {1: {"expected_arrival_time": 10.0, "latest_arrival_time": 11.0}}
        rows = self.policy.enumerate_candidates(
            now=0, groups=groups, curve=[point(50, 10.0, 12.0)],
        )
        self.assertTrue(rows[0].target_aligned)
        self.assertFalse(rows[0].deadline_safe)
        self.assertFalse(self.policy.choose(rows).feasible)

    def test_state_group_window_never_truncates_safe_finish(self):
        expected, latest = state_group_window(point(50, 10.0, 18.0), 0.5)
        self.assertEqual(expected, 10.0)
        self.assertEqual(latest, 18.0)

    def test_state_group_window_applies_minimum_slack(self):
        expected, latest = state_group_window(point(50, 10.0, 10.1), 0.5)
        self.assertEqual(expected, 10.0)
        self.assertEqual(latest, 10.5)


if __name__ == "__main__":
    unittest.main()
