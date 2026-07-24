import unittest

from su_compass.scheduling.policies.fair_contribution_state import (
    FairContributionState,
)
from su_compass.scheduling.policies.quality_gated_contribution import (
    recommend_contribution_restoration,
)


class QualityGatedContributionTests(unittest.TestCase):
    def _records(self):
        state = FairContributionState(["a", "b"], score_cap=2.0)
        state.raw_debt["b"] = 1.0
        return state.update(
            {"a": 0, "b": 3},
            alpha=0.9,
            staleness_fn=lambda stale: (stale + 1) ** -0.5,
        )

    def test_bonus_is_bounded_and_client_id_agnostic(self):
        rows = recommend_contribution_restoration(
            self._records(),
            local_steps={"a": 100, "b": 100},
            qmax=100,
            rhythm_debt=0.0,
            rhythm_stop=10.0,
            debt_score_cap=2.0,
            bonus_mass_cap=0.05,
            staleness_hard_cap=8,
        )
        by_id = {row.client_id: row for row in rows}
        self.assertTrue(by_id["b"].eligible)
        self.assertGreater(by_id["b"].proposed_share, by_id["b"].base_share)
        self.assertLessEqual(sum(row.allocated_bonus for row in rows), 0.05)
        self.assertAlmostEqual(sum(row.proposed_share for row in rows), 1.0)

    def test_rhythm_stop_disables_restoration(self):
        rows = recommend_contribution_restoration(
            self._records(),
            local_steps={"a": 100, "b": 100},
            qmax=100,
            rhythm_debt=10.0,
            rhythm_stop=10.0,
            debt_score_cap=2.0,
            bonus_mass_cap=0.05,
            staleness_hard_cap=8,
        )
        self.assertFalse(any(row.eligible for row in rows))
        self.assertTrue(all(row.allocated_bonus == 0 for row in rows))

    def test_staleness_cap_remains_hard(self):
        state = FairContributionState(["x", "y"], score_cap=2.0)
        state.raw_debt["y"] = 2.0
        records = state.update(
            {"x": 0, "y": 9},
            alpha=0.9,
            staleness_fn=lambda stale: (stale + 1) ** -0.5,
        )
        rows = recommend_contribution_restoration(
            records,
            local_steps={"x": 100, "y": 100},
            qmax=100,
            rhythm_debt=0,
            rhythm_stop=10,
            debt_score_cap=2,
            bonus_mass_cap=0.05,
            staleness_hard_cap=8,
        )
        y = next(row for row in rows if row.client_id == "y")
        self.assertFalse(y.eligible)
        self.assertEqual(y.reason, "staleness_hard_cap")


if __name__ == "__main__":
    unittest.main()
