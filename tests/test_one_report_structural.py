import unittest

from su_compass.scheduling.state_driven_config import StateDrivenConfig
from su_compass.scheduling.policies.one_report_structural import (
    predict_one_report_structural,
)


class OneReportStructuralTest(unittest.TestCase):
    def test_online_shadow_is_default_off(self):
        config = StateDrivenConfig()
        self.assertFalse(config.reason_aware_one_report_structural_shadow)
        self.assertEqual(config.reason_aware_one_report_communication_gate, 0.95)

    def test_preserves_fixed_communication_when_q_decreases(self):
        result = predict_one_report_structural(
            q=50,
            observed_q=200,
            observed_round_duration=150.0,
            observed_compute_duration=4.0,
            observed_communication_duration=146.0,
        )
        self.assertTrue(result.eligible)
        self.assertAlmostEqual(result.predicted_duration, 147.0)
        self.assertAlmostEqual(result.safe_duration, 165.0)

    def test_gate_rejects_non_communication_dominated_client(self):
        result = predict_one_report_structural(
            q=50,
            observed_q=200,
            observed_round_duration=10.0,
            observed_compute_duration=3.0,
            observed_communication_duration=7.0,
        )
        self.assertFalse(result.eligible)

    def test_requires_exactly_one_report(self):
        result = predict_one_report_structural(
            q=50,
            observed_q=200,
            observed_round_duration=150.0,
            observed_compute_duration=4.0,
            observed_communication_duration=146.0,
            num_reports=2,
        )
        self.assertFalse(result.eligible)

    def test_invalid_q_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_one_report_structural(
                q=0,
                observed_q=200,
                observed_round_duration=150.0,
                observed_compute_duration=4.0,
                observed_communication_duration=146.0,
            )

    def test_config_rejects_invalid_gate(self):
        with self.assertRaises(ValueError):
            StateDrivenConfig(
                reason_aware_one_report_communication_gate=1.1,
            )


if __name__ == "__main__":
    unittest.main()
