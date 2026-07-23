import unittest

from su_compass.scheduling.predictors.regime_calibrated import (
    RegimeCalibratedPredictor,
)


class RegimeCalibratedPredictorTest(unittest.TestCase):
    def test_cold_start_uses_protected_baseline(self):
        predictor = RegimeCalibratedPredictor(min_observations=2)
        result = predictor.predict(
            client_id="c0", local_steps=100,
            baseline_duration=12.0, baseline_safe_duration=15.0,
        )
        self.assertEqual(result.predicted_duration, 12.0)
        self.assertEqual(result.safe_duration, 15.0)
        self.assertFalse(result.used_candidate)

    def test_observation_updates_without_future_leakage(self):
        predictor = RegimeCalibratedPredictor(min_observations=1)
        first = predictor.predict(
            client_id="c0", local_steps=100,
            baseline_duration=20.0, baseline_safe_duration=22.0,
        )
        self.assertEqual(first.raw_duration, 20.0)
        predictor.observe(
            client_id="c0", local_steps=100, actual_duration=12.0,
            compute_duration=10.0, communication_duration=2.0,
        )
        second = predictor.predict(
            client_id="c0", local_steps=50,
            baseline_duration=20.0, baseline_safe_duration=22.0,
        )
        # A single residual is not enough to estimate temporal dependence;
        # the protected structural baseline remains unchanged.
        self.assertAlmostEqual(second.raw_duration, 20.0)

    def test_positive_residual_calibrates_safe_margin(self):
        predictor = RegimeCalibratedPredictor(min_observations=1)
        predictor.predict(
            client_id="c0", local_steps=10,
            baseline_duration=10.0, baseline_safe_duration=10.0,
        )
        predictor.observe(
            client_id="c0", local_steps=10, actual_duration=15.0,
            compute_duration=10.0, communication_duration=5.0,
        )
        result = predictor.predict(
            client_id="c0", local_steps=10,
            baseline_duration=15.0, baseline_safe_duration=15.0,
        )
        self.assertGreaterEqual(result.raw_safe_duration - result.raw_duration, 5.0)

    def test_finite_sample_calibration_pools_short_client_history(self):
        predictor = RegimeCalibratedPredictor(
            target_coverage=0.85, finite_sample_pooling=True,
            client_calibration_min=2,
        )
        for client, error in (("a", 1.0), ("b", 4.0)):
            predictor.predict(
                client_id=client, local_steps=10,
                baseline_duration=10.0, baseline_safe_duration=10.0,
            )
            predictor.observe(
                client_id=client, local_steps=10, actual_duration=10.0 + error,
                compute_duration=10.0, communication_duration=error,
            )
        margin, source, n, rank, _, pooled = predictor.preview_safety_margin(
            client_id="c", analytical_margin=0.5,
        )
        self.assertEqual(source, "pooled_finite")
        self.assertEqual((n, rank), (2, 2))
        self.assertEqual(margin, pooled)
        self.assertEqual(margin, 4.0)


if __name__ == "__main__":
    unittest.main()
