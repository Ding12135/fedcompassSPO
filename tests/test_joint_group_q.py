import unittest
from tempfile import TemporaryDirectory

from su_compass.scheduling.policies.joint_group_q import JointGroupQPolicy
from su_compass.scheduling.state_time_model import QTimeCandidate, state_group_window
from su_compass.virtual.trace import TraceWriter


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

    def test_each_group_uses_largest_deadline_safe_q(self):
        groups = {
            1: {"expected_arrival_time": 10.0, "latest_arrival_time": 20.0},
        }
        rows = self.policy.enumerate_candidates(
            now=0, groups=groups,
            curve=[point(50, 10.0), point(100, 15.0), point(120, 18.0, 21.0)],
        )
        decision = self.policy.choose(rows)
        self.assertTrue(decision.feasible)
        self.assertEqual(decision.group_id, 1)
        self.assertEqual(decision.q, 100)

    def test_alignment_band_consolidates_into_larger_group(self):
        groups = {
            1: {"clients": ["a"], "expected_arrival_time": 10.0,
                "latest_arrival_time": 20.0},
            2: {"clients": ["b", "c", "d"], "expected_arrival_time": 10.3,
                "latest_arrival_time": 21.0},
        }
        rows = self.policy.enumerate_candidates(
            now=0, groups=groups, curve=[point(60, 10.0)],
        )
        decision = self.policy.choose(rows)
        self.assertEqual(decision.group_id, 2)
        self.assertEqual(decision.group_size, 3)

    def test_deadline_safe_without_target_alignment_is_feasible(self):
        groups = {1: {"expected_arrival_time": 10.0, "latest_arrival_time": 20.0}}
        rows = self.policy.enumerate_candidates(
            now=0, groups=groups, curve=[point(50, 15.0, 16.0)],
        )
        self.assertTrue(rows[0].deadline_safe)
        self.assertFalse(rows[0].target_aligned)
        self.assertTrue(self.policy.choose(rows).feasible)

    def test_expected_time_passed_but_deadline_safe_is_feasible(self):
        groups = {1: {"expected_arrival_time": 10.0, "latest_arrival_time": 20.0}}
        rows = self.policy.enumerate_candidates(
            now=11.0, groups=groups, curve=[point(50, 16.0, 18.0)],
        )
        self.assertTrue(rows[0].expected_already_passed)
        self.assertTrue(self.policy.choose(rows).feasible)

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

    def test_new_candidate_and_decision_fields_flush(self):
        groups = {
            1: {"clients": ["a"], "arrived_clients": ["b"],
                "expected_arrival_time": 10.0, "latest_arrival_time": 20.0}
        }
        candidate = self.policy.enumerate_candidates(
            now=0, groups=groups, curve=[point(50, 10.0)],
        )[0]
        with TemporaryDirectory() as directory:
            writer = TraceWriter(directory, "state_driven_compass")
            row = candidate.to_trace()
            row.update({"decision_id": "d0", "client_id": "c0", "selected_by_state": 1})
            writer.record_joint_group_q_candidate(row)
            writer.record_joint_group_q({
                "decision_id": "d0", "state_group_size_before": 2,
            })
            writer.flush()


if __name__ == "__main__":
    unittest.main()
