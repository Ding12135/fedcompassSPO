import unittest

from su_compass.scheduling.fedcompass_reference import (
    existing_group_reference, new_group_reference_window,
)


class FedCompassReferenceTest(unittest.TestCase):
    def test_existing_group_reference_matches_largest_q_rule(self):
        groups = {
            1: {"expected_arrival_time": 10.0, "latest_arrival_time": 12.0},
            2: {"expected_arrival_time": 15.0, "latest_arrival_time": 18.0},
        }
        result = existing_group_reference(
            now=0.0, speed=0.1, groups=groups, qmin=40, qmax=200,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.group_id, 2)
        self.assertEqual(result.q, 150)

    def test_new_group_window_matches_original_formula(self):
        expected, latest = new_group_reference_window(
            now=5.0, q=100, speed=0.2, latest_time_factor=1.2,
        )
        self.assertEqual(expected, 25.0)
        self.assertEqual(latest, 29.0)


if __name__ == "__main__":
    unittest.main()
