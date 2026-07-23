import math
import unittest
from dataclasses import replace

from su_compass.scheduling.policies.lyapunov_group_q import (
    LyapunovAction,
    LyapunovGroupQPolicy,
    choose_effective_service_v2,
)


def _action(
    mode, q, sojourn, *, group=0, holding=0, extension=0,
    duration=None, affected=1, safe=True,
):
    duration = sojourn - holding if duration is None else duration
    return LyapunovAction(
        mode=mode, group_id=group, q=q,
        predicted_finish_time=duration, predicted_duration=duration,
        safe_finish_time=duration,
        group_frontier_time=sojourn, latest_arrival_time=sojourn + 1,
        deadline_safe=safe, holding_wait=holding, external_wait=extension,
        affected_pending_clients=affected,
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
        self.assertEqual(scored[0].rejection_reason, "extreme_holding_wait")
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

    def test_join_v2_rhythm_debt_prices_frontier_extension_not_holding(self):
        policy = self._policy(
            action_scope="join_only_v2", max_holding_wait=100,
            max_holding_ratio=100, holding_weight=0,
        )
        scored = policy.score([
            _action("join", 100, 60, group=0, holding=50, extension=0),
            _action("join", 100, 15, group=1, holding=0, extension=5),
        ], rhythm_debt=100, workload_debt=0, qmax=200, qmin=40)
        self.assertEqual(policy.choose(scored).action.group_id, 0)

    def test_join_v2_soft_holding_cost_can_prefer_better_alignment(self):
        policy = self._policy(
            action_scope="join_only_v2", max_holding_wait=100,
            max_holding_ratio=100, holding_weight=1,
        )
        scored = policy.score([
            _action("join", 100, 60, group=0, holding=50),
            _action("join", 100, 15, group=1, holding=5),
        ], rhythm_debt=0, workload_debt=0, qmax=200, qmin=40)
        self.assertEqual(policy.choose(scored).action.group_id, 1)

    def test_join_v2_extreme_holding_requires_absolute_and_ratio_excess(self):
        policy = self._policy(
            action_scope="join_only_v2", max_holding_wait=80,
            max_holding_ratio=4, holding_weight=0,
        )
        scored = policy.score([
            _action("join", 100, 190, group=0, holding=90, duration=100),
            _action("join", 100, 100, group=1, holding=90, duration=10),
        ], rhythm_debt=0, workload_debt=0, qmax=200, qmin=40)
        self.assertTrue(scored[0].legal)
        self.assertEqual(scored[1].rejection_reason, "extreme_holding_wait")

    def test_effective_service_selector_uses_shared_region_hysteresis(self):
        scored = [
            replace(
                _action("join", 80, 20, group=3, holding=5, extension=3),
                legal=True, score=1.0,
            ),
            replace(_action("create", 80, 10, group=-1), legal=True, score=0.8),
        ]
        selection = choose_effective_service_v2(
            scored, obvious_extension_limit=1,
            obvious_holding_limit=10, create_hysteresis=0.1,
        )
        self.assertEqual(selection.region, "region_2")
        self.assertEqual(selection.decision.action.mode, "create")

    def test_effective_service_selector_keeps_obvious_join(self):
        scored = [
            replace(
                _action("join", 80, 10, group=2, holding=1, extension=0.5),
                legal=True, score=10.0,
            ),
            replace(_action("create", 80, 5, group=-1), legal=True, score=0.0),
        ]
        selection = choose_effective_service_v2(
            scored, obvious_extension_limit=1,
            obvious_holding_limit=10, create_hysteresis=0.1,
        )
        self.assertEqual(selection.region, "region_1")
        self.assertEqual(selection.decision.action.group_id, 2)

    def test_join_cadence_cost_avoids_already_long_group_under_high_h(self):
        actions = [
            _action("join", 80, 60, group=2, holding=50, extension=0),
            _action("create", 80, 15, group=-1),
        ]
        common = dict(
            action_scope="effective_service_v2_1", max_holding_wait=100,
            max_holding_ratio=100, holding_weight=0,
        )
        without = self._policy(**common).score(
            actions, rhythm_debt=100, workload_debt=0, qmax=200, qmin=40,
        )
        with_cost = self._policy(**common, join_cadence_weight=1.0).score(
            actions, rhythm_debt=100, workload_debt=0, qmax=200, qmin=40,
        )
        self.assertEqual(self._policy(**common).choose(without).action.mode, "join")
        self.assertEqual(
            self._policy(**common, join_cadence_weight=1.0).choose(with_cost).action.mode,
            "create",
        )


if __name__ == "__main__":
    unittest.main()
