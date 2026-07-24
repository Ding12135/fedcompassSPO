import unittest

from su_compass.scheduling.policies.unified_batch_dispatch import (
    rank_unified_batch,
)


class UnifiedBatchDispatchTests(unittest.TestCase):
    def test_contribution_deficit_can_outweigh_duration_without_speed_class(self):
        ranked = rank_unified_batch(
            ["fast", "currently_slow"],
            fair_debt={"fast": 0.1, "currently_slow": 1.5},
            safe_duration={"fast": 10.0, "currently_slow": 40.0},
            rhythm_target=16.0,
        )
        self.assertEqual(ranked[0].client_id, "currently_slow")

    def test_stale_or_unserviceable_client_does_not_win_on_debt_alone(self):
        ranked = rank_unified_batch(
            ["fresh", "stale"],
            fair_debt={"fresh": 0.4, "stale": 2.0},
            safe_duration={"fresh": 16.0, "stale": 16.0},
            rhythm_target=16.0,
            freshness={"fresh": 1.0, "stale": 0.05},
            service_probability={"fresh": 1.0, "stale": 0.5},
        )
        self.assertEqual(ranked[0].client_id, "fresh")

    def test_ties_are_deterministic_and_not_speed_ordered(self):
        ranked = rank_unified_batch(
            ["client_7", "client_1", "client_3"],
            fair_debt={},
            safe_duration={},
            rhythm_target=16.0,
        )
        self.assertEqual(
            [row.client_id for row in ranked],
            ["client_1", "client_3", "client_7"],
        )


if __name__ == "__main__":
    unittest.main()
