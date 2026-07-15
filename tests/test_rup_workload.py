"""Ablation and fallback invariants for the pluggable RUP workload policy."""

import unittest
from types import SimpleNamespace

from su_compass.scheduling.policies import RUPConfig, RUPWorkloadPolicy
from su_compass.scheduling.types import LatencyPrediction


class _LinearPredictor:
    @property
    def name(self):
        return "linear"

    def predict(self, context):
        duration = 5.0 + context.local_steps * 0.1
        return LatencyPrediction(
            predictor_name=self.name,
            predicted_duration=duration,
            predicted_finish_time=context.dispatch_time + duration,
            uncertainty=1.0,
            safe_duration=duration + 1.0,
            num_reports=getattr(context.runtime_state, "num_reports", 0),
        )


def _state(reports=10):
    return SimpleNamespace(num_reports=reports)


class RUPWorkloadPolicyTest(unittest.TestCase):
    def _decide(self, policy, client="client_0", baseline=100, latest=20.0):
        return policy.decide(
            client_id=client, dispatch_time=0.0, speed_smoothed=0.1,
            runtime_state=_state(), expected_arrival_time=15.0,
            latest_arrival_time=latest, qmin=40, qmax=200,
            baseline_q=baseline,
        )

    def test_off_is_strict_noop(self):
        result = self._decide(RUPWorkloadPolicy(
            _LinearPredictor(), RUPConfig(mode="off")
        ))
        self.assertEqual(result.recommended_q, 100)
        self.assertEqual(result.applied_q, 100)

    def test_group_admission_mode_is_independently_configurable(self):
        self.assertEqual(RUPConfig().group_admission_mode, "apply")
        self.assertEqual(
            RUPConfig(group_admission_mode="off").group_admission_mode, "off"
        )
        with self.assertRaises(ValueError):
            RUPConfig(group_admission_mode="invalid")

    def test_shadow_never_changes_dispatched_q(self):
        config = RUPConfig(
            mode="shadow", utility_enabled=False, budget_enabled=False,
            soft_boundary_enabled=False,
        )
        result = self._decide(RUPWorkloadPolicy(_LinearPredictor(), config), baseline=80)
        self.assertNotEqual(result.recommended_q, result.baseline_q)
        self.assertEqual(result.applied_q, result.baseline_q)

    def test_every_layer_can_be_disabled(self):
        config = RUPConfig(
            mode="apply", state_enabled=False, residual_risk_enabled=False,
            trust_region_enabled=False, soft_boundary_enabled=False,
            utility_enabled=False, budget_enabled=False,
        )
        result = self._decide(RUPWorkloadPolicy(_LinearPredictor(), config))
        self.assertEqual(result.applied_q, 100)
        self.assertEqual(result.enabled_layers, "")

    def test_no_safe_q_keeps_fedcompass(self):
        config = RUPConfig(
            utility_enabled=False, budget_enabled=False,
            soft_boundary_enabled=False,
        )
        result = self._decide(
            RUPWorkloadPolicy(_LinearPredictor(), config), latest=5.0,
        )
        self.assertFalse(result.state_safe_feasible)
        self.assertEqual(result.applied_q, 100)
        self.assertEqual(result.fallback_reason, "no_safe_q_keep_fedcompass")

    def test_utility_is_bounded_and_requires_positive_progress(self):
        policy = RUPWorkloadPolicy(
            _LinearPredictor(),
            RUPConfig(soft_boundary_enabled=False, budget_enabled=False),
        )
        for _ in range(5):
            policy.observe_upload("client_0", 15.0, SimpleNamespace(
                finite=True, loss_before=4.0, loss_delta_per_step=0.01,
                num_train_samples=100,
            ))
            policy.observe_upload("client_1", 15.0, SimpleNamespace(
                finite=True, loss_before=1.0, loss_delta_per_step=0.01,
                num_train_samples=100,
            ))
        result = self._decide(policy)
        self.assertEqual(result.utility_normalized, 1.1)
        self.assertGreaterEqual(result.utility_q, result.soft_q)

        no_progress = RUPWorkloadPolicy(_LinearPredictor(), policy.config)
        for _ in range(5):
            no_progress.observe_upload("client_0", 15.0, SimpleNamespace(
                finite=True, loss_before=4.0, loss_delta_per_step=-0.01,
                num_train_samples=100,
            ))
        result = self._decide(no_progress)
        self.assertEqual(result.utility_confidence, 0.0)
        self.assertEqual(result.utility_normalized, 1.0)


if __name__ == "__main__":
    unittest.main()
