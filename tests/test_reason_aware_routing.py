import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from su_compass.scheduling.policies.lyapunov_group_q import LyapunovAction
from su_compass.scheduling.policies.reason_aware_routing import (
    classify_slow_cause,
    recommend_reason_aware_route,
)
from su_compass.scheduling.state_driven_config import StateDrivenConfig
from su_compass.virtual.trace import TraceWriter


def _point(
    *, duration=100.0, compute=1.0, communication=99.0,
    availability=0.0, risk=0.0, spike=0.0, reports=2,
    fallback=False, source="mature_state",
):
    return SimpleNamespace(
        predicted_duration=duration, compute_duration=compute,
        communication_duration=communication,
        availability_duration=availability,
        availability_risk_duration=risk, spike_duration=spike,
        num_reports=reports, used_fallback=fallback,
        predictor_source=source,
    )


def _action(mode, group, q, sojourn, score, *, legal=True):
    return LyapunovAction(
        mode=mode, group_id=group, q=q,
        predicted_finish_time=sojourn, predicted_duration=sojourn,
        safe_finish_time=sojourn, group_frontier_time=sojourn,
        latest_arrival_time=sojourn + 10, deadline_safe=True,
        holding_wait=0.0, external_wait=0.0, affected_pending_clients=1,
        predicted_sojourn=sojourn, effective_work=q / 200,
        utility=1.0, score=score, legal=legal,
    )


class ReasonAwareRoutingTest(unittest.TestCase):
    def test_feature_is_default_off(self):
        self.assertFalse(StateDrivenConfig().reason_aware_routing_shadow)

    def test_extreme_communication_is_state_based(self):
        cause = classify_slow_cause(_point())
        self.assertEqual(cause.label, "extreme_communication_bound")
        self.assertTrue(cause.mature)
        self.assertGreater(cause.confidence, 0)

    def test_client7_like_profile_is_not_extreme(self):
        cause = classify_slow_cause(_point(
            duration=100, compute=5, communication=94.5, spike=0.5,
        ))
        self.assertEqual(cause.label, "communication_bound")

    def test_cold_profile_never_claims_a_slow_cause(self):
        cause = classify_slow_cause(_point(reports=1))
        self.assertEqual(cause.label, "cold_start")
        self.assertFalse(cause.mature)

    def test_extreme_client_reuses_compatible_background_group(self):
        cause = classify_slow_cause(_point())
        v23 = _action("create", -1, 52, 120, 5)
        background = _action("join", 16, 52, 100, 3)
        route = recommend_reason_aware_route(
            cause=cause, v23_action=v23,
            scored_actions=[v23, background],
            service_age_periods=8, minimum_anchor_age_periods=4,
            system_healthy=True, background_sojourn_periods=2,
            rhythm_target=16.4,
        )
        self.assertEqual(route.lane, "background")
        self.assertEqual(route.mode, "join")
        self.assertEqual(route.group_id, 16)
        self.assertEqual(route.q, v23.q)
        self.assertTrue(route.changed)

    def test_shadow_anchor_preserves_v23_q(self):
        cause = classify_slow_cause(_point())
        v23 = _action("join", 3, 52, 10, 4)
        create = _action("create", -1, 52, 20, 2)
        route = recommend_reason_aware_route(
            cause=cause, v23_action=v23,
            scored_actions=[v23, create],
            service_age_periods=8, minimum_anchor_age_periods=4,
            system_healthy=True, background_sojourn_periods=2,
            rhythm_target=16.4,
        )
        self.assertTrue(route.anchor_eligible)
        self.assertEqual(route.mode, "create")
        self.assertEqual(route.q, 52)

    def test_unhealthy_system_preserves_v23_action(self):
        cause = classify_slow_cause(_point())
        v23 = _action("create", -1, 52, 20, 1)
        route = recommend_reason_aware_route(
            cause=cause, v23_action=v23, scored_actions=[v23],
            service_age_periods=8, minimum_anchor_age_periods=4,
            system_healthy=False, background_sojourn_periods=2,
            rhythm_target=16.4,
        )
        self.assertFalse(route.anchor_eligible)
        self.assertFalse(route.changed)
        self.assertEqual(route.q, 52)

    def test_independent_shadow_trace_flushes(self):
        with TemporaryDirectory() as directory:
            writer = TraceWriter(directory, "state_driven_compass")
            writer.record_reason_aware_routing({
                "decision_id": "d1",
                "client_id": "client_5",
                "slow_cause": "extreme_communication_bound",
                "v23_q": 52,
                "shadow_q": 52,
                "q_unchanged": 1,
            })
            writer.flush()
            path = Path(directory) / "reason_aware_routing_shadow_trace.csv"
            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
            self.assertIn("slow_cause", text)
            self.assertIn("q_unchanged", text)
            self.assertIn("one_report_structural_eligible", text)
            self.assertIn("elastic_join_avoidance", text)


if __name__ == "__main__":
    unittest.main()
