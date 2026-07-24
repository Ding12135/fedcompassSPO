import unittest
from pathlib import Path

from su_compass.scheduling.policies.controlled_q_candidates import (
    controlled_create_qs,
    controlled_join_qs,
)
from su_compass.scheduling.state_time_model import QTimeCandidate
from su_compass.scheduling.state_driven_config import StateDrivenConfig
from su_compass.experiments.run_cifar10_state_driven import command_for


def _point(q, *, now=0.0, fixed=10.0, per_step=0.5, safe_extra=2.0, fallback=False):
    duration = fixed + per_step * q
    return QTimeCandidate(
        q=q, predictor_name="test", predictor_source=(
            "fedcompass_fallback" if fallback else "mature_state"
        ), num_reports=10, used_fallback=fallback, fallback_reason="",
        predicted_duration=duration, safe_duration=duration + safe_extra,
        predicted_finish_time=now + duration,
        safe_finish_time=now + duration + safe_extra, uncertainty=safe_extra,
        compute_duration=per_step * q, communication_duration=fixed,
        spike_duration=0.0, availability_duration=0.0,
        availability_risk_duration=0.0,
    )


class ControlledQCandidatesTest(unittest.TestCase):
    def test_effective_service_v2_preset_routes_shadow_safety_and_no_z(self):
        command = command_for(
            "state_driven_effective_service_v2_shadow", 120, 2026, Path("out"),
        )
        value = lambda flag: command[command.index(flag) + 1]
        self.assertEqual(value("--sd_lyapunov_mode"), "shadow")
        self.assertEqual(value("--sd_lyapunov_action_scope"), "effective_service_v2")
        self.assertEqual(value("--sd_finite_sample_safety_calibration"), "on")
        self.assertEqual(value("--sd_lyapunov_workload_queue"), "off")

    def test_effective_service_v2_1_preset_routes_stateful_shadow(self):
        command = command_for(
            "state_driven_effective_service_v2_1_shadow", 120, 2026, Path("out"),
        )
        value = lambda flag: command[command.index(flag) + 1]
        self.assertEqual(value("--sd_lyapunov_mode"), "shadow")
        self.assertEqual(value("--sd_lyapunov_action_scope"), "effective_service_v2_1")
        self.assertEqual(value("--sd_finite_sample_safety_calibration"), "on")
        self.assertEqual(value("--sd_lyapunov_workload_queue"), "off")

    def test_effective_service_v2_2_preset_enables_recruitment_guards(self):
        command = command_for(
            "state_driven_effective_service_v2_2_apply_smoke",
            40, 2026, Path("out"),
        )
        value = lambda flag: command[command.index(flag) + 1]
        self.assertEqual(value("--sd_lyapunov_mode"), "apply")
        self.assertEqual(value("--sd_lyapunov_recruit_safe_cap_ratio"), "2.0")
        self.assertEqual(value("--sd_lyapunov_create_safe_cost"), "on")

    def test_effective_service_v2_3_preset_prices_join_cadence(self):
        command = command_for(
            "state_driven_effective_service_v2_3_apply_smoke",
            40, 2026, Path("out"),
        )
        value = lambda flag: command[command.index(flag) + 1]
        self.assertEqual(value("--sd_lyapunov_recruit_safe_cap_ratio"), "2.0")
        self.assertEqual(value("--sd_lyapunov_create_safe_cost"), "on")
        self.assertEqual(value("--sd_lyapunov_join_cadence_weight"), "1.0")

    def test_effective_service_v2_4_is_shadow_over_v23(self):
        command = command_for(
            "state_driven_effective_service_v2_4_routing_shadow",
            40, 2026, Path("out"),
        )
        value = lambda flag: command[command.index(flag) + 1]
        self.assertEqual(value("--sd_lyapunov_mode"), "apply")
        self.assertEqual(value("--sd_lyapunov_join_cadence_weight"), "1.0")
        self.assertEqual(value("--sd_reason_aware_routing_shadow"), "on")
        self.assertEqual(
            value("--sd_reason_aware_one_report_structural_shadow"), "on",
        )
        self.assertEqual(value("--sd_lyapunov_workload_queue"), "off")

    def test_quality_gated_v2_5_keeps_v23_apply_and_new_actions_shadow(self):
        command = command_for(
            "state_driven_quality_gated_v2_5_shadow",
            120, 2026, Path("out"),
        )
        value = lambda flag: command[command.index(flag) + 1]
        self.assertEqual(value("--sd_lyapunov_mode"), "apply")
        self.assertEqual(value("--sd_lyapunov_join_cadence_weight"), "1.0")
        self.assertEqual(value("--sd_fair_contribution_shadow"), "on")
        self.assertEqual(value("--sd_communication_amortized_q_shadow"), "on")
        self.assertEqual(value("--sd_contribution_restoration_shadow"), "on")
        self.assertEqual(value("--sd_micro_hold_shadow"), "on")
        self.assertEqual(value("--sd_unified_batch_dispatch_mode"), "off")

    def test_mature_cadence_v2_6_keeps_v23_apply(self):
        command = command_for(
            "state_driven_mature_cadence_v2_6_shadow",
            240, 2026, Path("out"),
        )
        value = lambda flag: command[command.index(flag) + 1]
        self.assertEqual(value("--sd_lyapunov_mode"), "apply")
        self.assertEqual(value("--sd_reason_aware_routing_shadow"), "on")
        self.assertEqual(value("--sd_contribution_restoration_shadow"), "on")
        self.assertEqual(value("--sd_micro_hold_shadow"), "on")
        self.assertEqual(value("--sd_unified_batch_dispatch_mode"), "off")

    def test_effective_service_v1_cannot_be_enabled_in_apply(self):
        with self.assertRaisesRegex(ValueError, "Shadow-only"):
            StateDrivenConfig(
                lyapunov_mode="apply", lyapunov_action_scope="effective_service_v1",
            )

    def test_effective_service_v2_cannot_be_enabled_in_apply(self):
        with self.assertRaisesRegex(ValueError, "Shadow-only"):
            StateDrivenConfig(
                lyapunov_mode="apply", lyapunov_action_scope="effective_service_v2",
            )

    def test_effective_service_v2_1_apply_is_gated_to_stateful_scope(self):
        config = StateDrivenConfig(
            lyapunov_mode="apply", lyapunov_action_scope="effective_service_v2_1",
        )
        self.assertEqual(config.lyapunov_mode, "apply")
        self.assertEqual(config.lyapunov_action_scope, "effective_service_v2_1")

    def test_create_candidates_use_only_group_independent_anchors(self):
        curve = [_point(q) for q in range(40, 201)]
        self.assertEqual(
            controlled_create_qs(
                curve=curve, reference_q=80, qmin=40, qmax=200,
                trust_eta=1.1,
            ),
            (40, 64, 80, 88),
        )

    def test_raw_alignment_cannot_escape_group_independent_trust_cap(self):
        curve = [_point(q) for q in range(40, 201)]
        result = controlled_join_qs(
            curve=curve, target_time=110.0, deadline=120.0,
            group_safe_frontier=100.0, reference_q=80,
            qmin=40, qmax=200, trust_eta=1.1,
        )
        self.assertEqual(result.raw_align_q, 200)
        self.assertEqual(result.trust_upper_q, 88)
        self.assertEqual(result.controlled_align_q, 88)
        self.assertNotIn(200, result.candidate_qs)

    def test_group_already_at_risk_has_no_candidates(self):
        result = controlled_join_qs(
            curve=[_point(40)], target_time=30.0, deadline=20.0,
            group_safe_frontier=21.0, reference_q=80,
            qmin=40, qmax=200, trust_eta=1.1,
        )
        self.assertEqual(result.candidate_qs, ())
        self.assertEqual(result.reason, "group_already_at_risk")

    def test_fallback_curve_does_not_add_state_alignment_anchor(self):
        curve = [_point(q, fallback=True) for q in range(40, 101)]
        result = controlled_join_qs(
            curve=curve, target_time=60.0, deadline=100.0,
            group_safe_frontier=50.0, reference_q=80,
            qmin=40, qmax=200, trust_eta=1.1,
        )
        self.assertFalse(result.reliable)
        self.assertNotIn(result.controlled_align_q, result.candidate_qs)


if __name__ == "__main__":
    unittest.main()
