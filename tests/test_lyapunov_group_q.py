import math
import unittest

from su_compass.scheduling.policies.lyapunov_group_q import (
    LyapunovAction,
    LyapunovGroupQPolicy,
)


def _action(mode, q, sojourn, *, group=0, holding=0, safe=True):
    return LyapunovAction(
        mode=mode, group_id=group, q=q,
        predicted_finish_time=sojourn, safe_finish_time=sojourn,
        group_frontier_time=sojourn, latest_arrival_time=sojourn + 1,
        deadline_safe=safe, holding_wait=holding, external_wait=0,
        predicted_sojourn=sojourn, effective_work=q / 200,
        utility=math.log1p(q / 40),
    )


class LyapunovGroupQPolicyTest(unittest.TestCase):
    def _policy(self, **kwargs):
        defaults = dict(
            rhythm_target=10, tradeoff_v=1, max_holding_wait=20,
            q_trust_eta=1.1, create_penalty=0.25,
        )
        defaults.update(kwargs)
        return LyapunovGroupQPolicy(**defaults)

    def test_holding_and_q_constraints_are_hard_gates(self):
        policy = self._policy()
        scored = policy.score([
            _action("join", 200, 10, holding=21),
            _action("join", 150, 10),
            _action("create", 100, 12, group=-1),
        ], rhythm_debt=0, workload_debt=0, qmax=200, qmin=40,
            fedcompass_join_q=100)
        self.assertEqual(scored[0].rejection_reason, "holding_wait_exceeds_cap")
        self.assertEqual(scored[1].rejection_reason, "q_exceeds_trust_region")
        self.assertTrue(scored[2].legal)

    def test_rhythm_debt_prefers_shorter_create_action(self):
        policy = self._policy(max_holding_wait=100)
        scored = policy.score([
            _action("join", 200, 80, holding=70),
            _action("create", 100, 10, group=-1),
        ], rhythm_debt=100, workload_debt=0, qmax=200, qmin=40)
        decision = policy.choose(scored)
        self.assertTrue(decision.feasible)
        self.assertEqual(decision.action.mode, "create")

    def test_workload_debt_can_repay_with_more_effective_work(self):
        policy = self._policy(create_penalty=0)
        scored = policy.score([
            _action("join", 40, 10),
            _action("join", 200, 12, group=1),
        ], rhythm_debt=0, workload_debt=10, qmax=200, qmin=40)
        self.assertEqual(policy.choose(scored).action.q, 200)


if __name__ == "__main__":
    unittest.main()
