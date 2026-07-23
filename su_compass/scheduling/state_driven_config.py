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
    finite_sample_safety_calibration: bool = False
    lyapunov_mode: str = "off"
    lyapunov_rhythm_target: float = 16.4
    lyapunov_v: float = 1.0
    lyapunov_max_holding_wait: float = 80.0
    lyapunov_q_trust_eta: float = 1.1
    lyapunov_create_penalty: float = 0.25
    lyapunov_enable_rhythm_queue: bool = True
    lyapunov_enable_workload_queue: bool = True
    lyapunov_client_target_rates: str = ""
    lyapunov_action_scope: str = "joint_v1"
    lyapunov_holding_weight: float = 0.0
    lyapunov_max_holding_ratio: float = 1_000_000.0
    lyapunov_q_reference_spec: str = ""
    lyapunov_region_extension_ratio: float = 0.10
    lyapunov_create_hysteresis: float = 0.10
    lyapunov_recruit_safe_cap_ratio: float = 1_000_000.0
    lyapunov_create_safe_cost: bool = False
    lyapunov_join_cadence_weight: float = 0.0
    reason_aware_routing_shadow: bool = False
    reason_aware_min_anchor_age_periods: float = 4.0
    reason_aware_background_sojourn_periods: float = 2.0
    reason_aware_cadence_median_ratio: float = 1.25
    reason_aware_cadence_max_ratio: float = 2.0
    reason_aware_one_report_structural_shadow: bool = False
    reason_aware_one_report_communication_gate: float = 0.95
    reason_aware_one_report_safety_fraction: float = 0.10

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
        if self.lyapunov_mode not in {"off", "shadow", "apply"}:
            raise ValueError("lyapunov_mode must be off, shadow or apply")
        if self.lyapunov_rhythm_target <= 0 or self.lyapunov_v < 0:
            raise ValueError("Lyapunov rhythm target must be positive and V non-negative")
        if (
            self.lyapunov_max_holding_wait < 0
            or self.lyapunov_q_trust_eta < 1.0
            or self.lyapunov_create_penalty < 0
        ):
            raise ValueError("invalid Lyapunov constraint parameter")
        if self.lyapunov_action_scope not in {
            "joint_v1", "join_only_v2", "effective_service_v1", "effective_service_v2",
            "effective_service_v2_1",
        }:
            raise ValueError("invalid Lyapunov action scope")
        if self.lyapunov_action_scope in {
            "effective_service_v1", "effective_service_v2",
        } and self.lyapunov_mode == "apply":
            raise ValueError("stateless effective_service modes are Shadow-only")
        if self.lyapunov_holding_weight < 0 or self.lyapunov_max_holding_ratio < 0:
            raise ValueError("invalid Lyapunov holding configuration")
        if self.lyapunov_region_extension_ratio < 0 or self.lyapunov_create_hysteresis < 0:
            raise ValueError("invalid regional decision configuration")
        if self.lyapunov_recruit_safe_cap_ratio < 1.0:
            raise ValueError("recruit safe cap ratio must be at least one")
        if self.lyapunov_join_cadence_weight < 0:
            raise ValueError("join cadence weight must be non-negative")
        if (
            self.reason_aware_min_anchor_age_periods < 0
            or self.reason_aware_background_sojourn_periods < 0
            or self.reason_aware_cadence_median_ratio <= 0
            or self.reason_aware_cadence_max_ratio <= 0
        ):
            raise ValueError("invalid reason-aware routing configuration")
        if not 0 <= self.reason_aware_one_report_communication_gate <= 1:
            raise ValueError("one-report communication gate must be in [0, 1]")
        if self.reason_aware_one_report_safety_fraction < 0:
            raise ValueError("one-report safety fraction must be non-negative")
