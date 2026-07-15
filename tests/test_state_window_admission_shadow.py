"""状态时间窗组合准入Shadow Gate测试。"""

import unittest

from su_compass.scheduling.policies import StateWindowAdmissionShadowPolicy


class StateWindowAdmissionShadowTest(unittest.TestCase):
    def setUp(self):
        self.policy = StateWindowAdmissionShadowPolicy()

    def test_current_safe_is_kept(self):
        result = self.policy.decide(
            current_group_safe=True, other_existing_group_safe=True,
            state_new_group_safe=True,
        )
        self.assertEqual(result.action, "keep_current_group")

    def test_other_safe_group_precedes_creation(self):
        result = self.policy.decide(
            current_group_safe=False, other_existing_group_safe=True,
            state_new_group_safe=True,
        )
        self.assertEqual(result.action, "switch_existing_group")

    def test_state_window_group_only_when_existing_groups_fail(self):
        result = self.policy.decide(
            current_group_safe=False, other_existing_group_safe=False,
            state_new_group_safe=True,
        )
        self.assertEqual(result.action, "create_state_window_group")

    def test_unresolved_is_explicit(self):
        result = self.policy.decide(
            current_group_safe=False, other_existing_group_safe=False,
            state_new_group_safe=False,
        )
        self.assertEqual(result.action, "unresolved_mismatch")


if __name__ == "__main__":
    unittest.main()
