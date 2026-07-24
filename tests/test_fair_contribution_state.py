import unittest

from su_compass.scheduling.policies.fair_contribution_state import (
    FairContributionState,
)


class FairContributionStateTests(unittest.TestCase):
    def test_uses_normalized_real_aggregator_weights(self):
        state = FairContributionState(["client_0", "client_1", "client_2"])
        records = state.update(
            {"client_0": 0, "client_1": 1},
            alpha=0.9,
            staleness_fn=lambda stale: 1.0 / (stale + 1),
        )
        by_client = {row.client_id: row for row in records}
        self.assertAlmostEqual(by_client["client_0"].raw_weight, 0.3)
        self.assertAlmostEqual(by_client["client_1"].raw_weight, 0.15)
        self.assertAlmostEqual(by_client["client_0"].normalized_weight, 2.0 / 3.0)
        self.assertAlmostEqual(by_client["client_1"].normalized_weight, 1.0 / 3.0)
        self.assertEqual(by_client["client_2"].normalized_weight, 0.0)
        self.assertAlmostEqual(by_client["client_2"].fair_debt_raw, 0.15)
        self.assertAlmostEqual(
            by_client["client_0"].target_effective_contribution, 0.15,
        )
        self.assertAlmostEqual(by_client["client_0"].effective_contribution, 0.3)

    def test_raw_debt_is_not_discarded_when_score_is_capped(self):
        state = FairContributionState(
            ["client_0", "client_1"], score_cap=0.2,
        )
        for _ in range(4):
            records = state.update(
                {"client_0": 0}, alpha=0.9, staleness_fn=lambda _: 1.0,
            )
        row = {item.client_id: item for item in records}["client_1"]
        self.assertGreater(row.fair_debt_raw, row.fair_debt_score)
        self.assertEqual(row.fair_debt_score, 0.2)
        self.assertGreater(row.fair_debt_overflow, 0.0)

    def test_jain_uses_cumulative_effective_share(self):
        state = FairContributionState(["client_0", "client_1"])
        state.update(
            {"client_0": 0}, alpha=0.9, staleness_fn=lambda _: 1.0,
        )
        self.assertAlmostEqual(state.jain_index(), 0.5)
        state.update(
            {"client_1": 0}, alpha=0.9, staleness_fn=lambda _: 1.0,
        )
        self.assertAlmostEqual(state.jain_index(), 1.0)


if __name__ == "__main__":
    unittest.main()
