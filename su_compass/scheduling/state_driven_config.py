"""Validated paper-level configuration for State-Driven FedCompass."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StateDrivenConfig:
    existing_group_mode: str = "state_apply"
    new_group_window_mode: str = "state_apply"
    new_group_q_mode: str = "qmax_anchor"
    candidate_trace_level: str = "summary"
    target_band_ratio: float = 0.05
    alignment_equivalence_band: float = 0.5
    min_group_slack: float = 0.5
    max_group_slack: float = 120.0
    cold_start_mode: str = "original_fedcompass"
    initial_dispatch_mode: str = "original_qmax"
    state_control_start_condition: str = "predictor_non_fallback_after_min_reports"
    safe_window_overflow_policy: str = "keep_safe_window_and_record_anomaly"
    calibrated_predictor_shadow: bool = True
    predictor_native_new_group_shadow: bool = True
    calibrated_shadow_target_coverage: float = 0.85

    def __post_init__(self) -> None:
        legal = {
            ("fedcompass", "fedcompass", "fedcompass"),
            ("state_shadow", "state_shadow_fixed_q", "fedcompass"),
            ("state_apply", "fedcompass", "fedcompass"),
            ("state_apply", "state_apply_fixed_q", "fedcompass"),
            ("state_apply", "state_apply", "qmax_anchor"),
        }
        combo = (self.existing_group_mode, self.new_group_window_mode, self.new_group_q_mode)
        if combo not in legal:
            raise ValueError(f"illegal State-Driven mode combination: {combo}")
        if self.candidate_trace_level not in {"none", "summary", "full"}:
            raise ValueError("candidate_trace_level must be none, summary or full")
        if (
            self.target_band_ratio < 0
            or self.alignment_equivalence_band < 0
            or self.min_group_slack < 0
        ):
            raise ValueError("target band and group slack must be non-negative")
        if self.max_group_slack < self.min_group_slack:
            raise ValueError("max_group_slack must be >= min_group_slack")
        if not 0.5 < self.calibrated_shadow_target_coverage < 1.0:
            raise ValueError("calibrated shadow target coverage must be in (0.5, 1)")
