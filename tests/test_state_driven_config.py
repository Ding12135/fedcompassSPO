import unittest

from su_compass.scheduling.state_driven_config import StateDrivenConfig


class StateDrivenConfigTest(unittest.TestCase):
    def test_all_five_legal_combinations(self):
        combinations = [
            ("fedcompass", "fedcompass", "fedcompass"),
            ("state_shadow", "state_shadow_fixed_q", "fedcompass"),
            ("state_apply", "fedcompass", "fedcompass"),
            ("state_apply", "state_apply_fixed_q", "fedcompass"),
            ("state_apply", "state_apply", "qmax_anchor"),
        ]
        for existing, window, q_mode in combinations:
            StateDrivenConfig(
                existing_group_mode=existing,
                new_group_window_mode=window,
                new_group_q_mode=q_mode,
            )

    def test_qmax_with_fedcompass_window_is_rejected(self):
        with self.assertRaises(ValueError):
            StateDrivenConfig(
                existing_group_mode="state_apply",
                new_group_window_mode="fedcompass",
                new_group_q_mode="qmax_anchor",
            )

    def test_lyapunov_is_default_off_and_modes_validate(self):
        self.assertEqual(StateDrivenConfig().lyapunov_mode, "off")
        StateDrivenConfig(lyapunov_mode="shadow")
        StateDrivenConfig(lyapunov_mode="apply")
        with self.assertRaises(ValueError):
            StateDrivenConfig(lyapunov_mode="invalid")


if __name__ == "__main__":
    unittest.main()
