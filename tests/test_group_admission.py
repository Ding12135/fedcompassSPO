"""Risk-Constrained Group Admission的模式与不变性单元测试。"""

import unittest

from su_compass.scheduling.policies.group_admission import (
    RiskConstrainedGroupAdmissionPolicy,
)


class GroupAdmissionPolicyTest(unittest.TestCase):
    def test_safe_group_is_always_admitted(self):
        for mode in ("shadow", "apply"):
            decision = RiskConstrainedGroupAdmissionPolicy(mode).decide(True)
            self.assertTrue(decision.admitted)
            self.assertEqual(decision.applied_action, "join_existing_group")

    def test_shadow_keeps_mismatch_join_unchanged(self):
        decision = RiskConstrainedGroupAdmissionPolicy("shadow").decide(False)
        self.assertTrue(decision.admitted)
        self.assertEqual(decision.shadow_action, "create_group")
        self.assertEqual(decision.applied_action, "join_existing_group")

    def test_apply_rejects_only_mismatch(self):
        decision = RiskConstrainedGroupAdmissionPolicy("apply").decide(False)
        self.assertFalse(decision.admitted)
        self.assertEqual(decision.applied_action, "create_group")

    def test_invalid_mode_fails_fast(self):
        with self.assertRaises(ValueError):
            RiskConstrainedGroupAdmissionPolicy("off")


if __name__ == "__main__":
    unittest.main()
