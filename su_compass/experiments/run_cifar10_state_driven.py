"""One-command CIFAR-10 State-Driven FedCompass preset runner."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from su_compass.experiments.run_cifar10_main import (
    RTX4090_BASE_STEP_TIME, RTX4090_MODEL_SIZE_MB, RTX4090_UPDATE_SIZE_MB,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESETS = {
    "state_driven_fc": ("fedcompass", "fedcompass", "fedcompass", "summary"),
    "state_driven_shadow": ("state_shadow", "state_shadow_fixed_q", "fedcompass", "full"),
    "state_driven_join": ("state_apply", "fedcompass", "fedcompass", "summary"),
    "state_driven_fixed_q_window": ("state_apply", "state_apply_fixed_q", "fedcompass", "summary"),
    "state_driven_safe_reuse_v1": ("state_apply", "state_apply_fixed_q", "fedcompass", "summary"),
    "state_driven_full": ("state_apply", "state_apply", "qmax_anchor", "summary"),
}

# 0.8 x the frozen seed-2026 FedCompass staleness-adjusted workload rates.
# These are experiment calibration inputs, not values learned from the
# candidate run; the manifest records them verbatim for reproducibility.
BASELINE_TARGET_RATES_08 = (
    "client_0=0.041279,client_1=0.043813,client_2=0.007019,"
    "client_3=0.025474,client_4=0.021321,client_5=0.000186,"
    "client_6=0.000346,client_7=0.002284"
)

# Frozen seed-2026 FedCompass per-client median Q.  This is a replay/Shadow
# calibration input, not a value learned from the candidate trajectory.  It
# must be replaced by a multi-seed frozen statistic before Apply.
BASELINE_MEDIAN_Q_SEED2026 = (
    "client_0=174,client_1=188,client_2=84,client_3=156,"
    "client_4=111,client_5=47,client_6=47,client_7=64"
)

LYAPUNOV_PRESETS = {
    "state_driven_lyapunov_shadow": ["--sd_lyapunov_mode", "shadow"],
    "state_driven_lyapunov_v1": ["--sd_lyapunov_mode", "apply"],
    "state_driven_lyapunov_no_h": [
        "--sd_lyapunov_mode", "apply", "--sd_lyapunov_rhythm_queue", "off",
    ],
    "state_driven_lyapunov_no_z": [
        "--sd_lyapunov_mode", "apply", "--sd_lyapunov_workload_queue", "off",
    ],
    "state_driven_lyapunov_no_holding_cap": [
        "--sd_lyapunov_mode", "apply", "--sd_lyapunov_max_holding_wait", "1000000",
    ],
    "state_driven_lyapunov_no_qcap": [
        "--sd_lyapunov_mode", "apply", "--sd_lyapunov_q_trust_eta", "1000000",
    ],
    "state_driven_lyapunov_join_v2_shadow": [
        "--sd_lyapunov_mode", "shadow",
        "--sd_lyapunov_action_scope", "join_only_v2",
        "--sd_lyapunov_workload_queue", "off",
        "--sd_lyapunov_holding_weight", "0.25",
        "--sd_lyapunov_max_holding_wait", "100.0",
        "--sd_lyapunov_max_holding_ratio", "4.0",
    ],
    "state_driven_lyapunov_join_v2": [
        "--sd_lyapunov_mode", "apply",
        "--sd_lyapunov_action_scope", "join_only_v2",
        "--sd_lyapunov_workload_queue", "off",
        "--sd_lyapunov_holding_weight", "0.25",
        "--sd_lyapunov_max_holding_wait", "100.0",
        "--sd_lyapunov_max_holding_ratio", "4.0",
    ],
    "state_driven_effective_service_v1_shadow": [
        "--sd_lyapunov_mode", "shadow",
        "--sd_lyapunov_action_scope", "effective_service_v1",
        "--sd_lyapunov_workload_queue", "off",
        "--sd_lyapunov_holding_weight", "0.25",
        "--sd_lyapunov_max_holding_wait", "85.0",
        "--sd_lyapunov_max_holding_ratio", "6.5",
        "--sd_lyapunov_q_reference_spec", BASELINE_MEDIAN_Q_SEED2026,
    ],
    "state_driven_effective_service_v2_shadow": [
        "--sd_lyapunov_mode", "shadow",
        "--sd_lyapunov_action_scope", "effective_service_v2",
        "--sd_lyapunov_workload_queue", "off",
        "--sd_lyapunov_holding_weight", "0.25",
        "--sd_lyapunov_max_holding_wait", "85.0",
        "--sd_lyapunov_max_holding_ratio", "6.5",
        "--sd_lyapunov_q_reference_spec", BASELINE_MEDIAN_Q_SEED2026,
        "--sd_finite_sample_safety_calibration", "on",
        "--sd_lyapunov_region_extension_ratio", "0.10",
        "--sd_lyapunov_create_hysteresis", "0.10",
    ],
    "state_driven_effective_service_v2_1_shadow": [
        "--sd_lyapunov_mode", "shadow",
        "--sd_lyapunov_action_scope", "effective_service_v2_1",
        "--sd_lyapunov_workload_queue", "off",
        "--sd_lyapunov_holding_weight", "0.25",
        "--sd_lyapunov_max_holding_wait", "85.0",
        "--sd_lyapunov_max_holding_ratio", "6.5",
        "--sd_lyapunov_q_reference_spec", BASELINE_MEDIAN_Q_SEED2026,
        "--sd_finite_sample_safety_calibration", "on",
        "--sd_lyapunov_region_extension_ratio", "0.10",
        "--sd_lyapunov_create_hysteresis", "0.10",
    ],
    "state_driven_effective_service_v2_1_apply_smoke": [
        "--sd_lyapunov_mode", "apply",
        "--sd_lyapunov_action_scope", "effective_service_v2_1",
        "--sd_lyapunov_workload_queue", "off",
        "--sd_lyapunov_holding_weight", "0.25",
        "--sd_lyapunov_max_holding_wait", "85.0",
        "--sd_lyapunov_max_holding_ratio", "6.5",
        "--sd_lyapunov_q_reference_spec", BASELINE_MEDIAN_Q_SEED2026,
        "--sd_finite_sample_safety_calibration", "on",
        "--sd_lyapunov_region_extension_ratio", "0.10",
        "--sd_lyapunov_create_hysteresis", "0.10",
    ],
    "state_driven_effective_service_v2_2_apply_smoke": [
        "--sd_lyapunov_mode", "apply",
        "--sd_lyapunov_action_scope", "effective_service_v2_1",
        "--sd_lyapunov_workload_queue", "off",
        "--sd_lyapunov_holding_weight", "0.25",
        "--sd_lyapunov_max_holding_wait", "85.0",
        "--sd_lyapunov_max_holding_ratio", "6.5",
        "--sd_lyapunov_q_reference_spec", BASELINE_MEDIAN_Q_SEED2026,
        "--sd_finite_sample_safety_calibration", "on",
        "--sd_lyapunov_region_extension_ratio", "0.10",
        "--sd_lyapunov_create_hysteresis", "0.10",
        "--sd_lyapunov_recruit_safe_cap_ratio", "2.0",
        "--sd_lyapunov_create_safe_cost", "on",
    ],
    "state_driven_effective_service_v2_3_apply_smoke": [
        "--sd_lyapunov_mode", "apply",
        "--sd_lyapunov_action_scope", "effective_service_v2_1",
        "--sd_lyapunov_workload_queue", "off",
        "--sd_lyapunov_holding_weight", "0.25",
        "--sd_lyapunov_max_holding_wait", "85.0",
        "--sd_lyapunov_max_holding_ratio", "6.5",
        "--sd_lyapunov_q_reference_spec", BASELINE_MEDIAN_Q_SEED2026,
        "--sd_finite_sample_safety_calibration", "on",
        "--sd_lyapunov_region_extension_ratio", "0.10",
        "--sd_lyapunov_create_hysteresis", "0.10",
        "--sd_lyapunov_recruit_safe_cap_ratio", "2.0",
        "--sd_lyapunov_create_safe_cost", "on",
        "--sd_lyapunov_join_cadence_weight", "1.0",
    ],
    "state_driven_effective_service_v2_4_routing_shadow": [
        "--sd_lyapunov_mode", "apply",
        "--sd_lyapunov_action_scope", "effective_service_v2_1",
        "--sd_lyapunov_workload_queue", "off",
        "--sd_lyapunov_holding_weight", "0.25",
        "--sd_lyapunov_max_holding_wait", "85.0",
        "--sd_lyapunov_max_holding_ratio", "6.5",
        "--sd_lyapunov_q_reference_spec", BASELINE_MEDIAN_Q_SEED2026,
        "--sd_finite_sample_safety_calibration", "on",
        "--sd_lyapunov_region_extension_ratio", "0.10",
        "--sd_lyapunov_create_hysteresis", "0.10",
        "--sd_lyapunov_recruit_safe_cap_ratio", "2.0",
        "--sd_lyapunov_create_safe_cost", "on",
        "--sd_lyapunov_join_cadence_weight", "1.0",
        "--sd_reason_aware_routing_shadow", "on",
        "--sd_reason_aware_one_report_structural_shadow", "on",
    ],
    "state_driven_unified_fair_batch_v2_4_shadow": [
        "--sd_lyapunov_mode", "apply",
        "--sd_lyapunov_action_scope", "effective_service_v2_1",
        "--sd_lyapunov_workload_queue", "off",
        "--sd_lyapunov_holding_weight", "0.25",
        "--sd_lyapunov_max_holding_wait", "85.0",
        "--sd_lyapunov_max_holding_ratio", "6.5",
        "--sd_lyapunov_q_reference_spec", BASELINE_MEDIAN_Q_SEED2026,
        "--sd_finite_sample_safety_calibration", "on",
        "--sd_lyapunov_region_extension_ratio", "0.10",
        "--sd_lyapunov_create_hysteresis", "0.10",
        "--sd_lyapunov_recruit_safe_cap_ratio", "2.0",
        "--sd_lyapunov_create_safe_cost", "on",
        "--sd_lyapunov_join_cadence_weight", "1.0",
        "--sd_reason_aware_one_report_structural_shadow", "on",
        "--sd_reason_aware_routing_shadow", "on",
        "--sd_fair_contribution_shadow", "on",
        "--sd_communication_amortized_q_shadow", "on",
        "--sd_unified_batch_dispatch_mode", "off",
    ],
    "state_driven_quality_gated_v2_5_shadow": [
        "--sd_lyapunov_mode", "apply",
        "--sd_lyapunov_action_scope", "effective_service_v2_1",
        "--sd_lyapunov_workload_queue", "off",
        "--sd_lyapunov_holding_weight", "0.25",
        "--sd_lyapunov_max_holding_wait", "85.0",
        "--sd_lyapunov_max_holding_ratio", "6.5",
        "--sd_lyapunov_q_reference_spec", BASELINE_MEDIAN_Q_SEED2026,
        "--sd_finite_sample_safety_calibration", "on",
        "--sd_lyapunov_region_extension_ratio", "0.10",
        "--sd_lyapunov_create_hysteresis", "0.10",
        "--sd_lyapunov_recruit_safe_cap_ratio", "2.0",
        "--sd_lyapunov_create_safe_cost", "on",
        "--sd_lyapunov_join_cadence_weight", "1.0",
        "--sd_reason_aware_one_report_structural_shadow", "on",
        "--sd_reason_aware_routing_shadow", "on",
        "--sd_fair_contribution_shadow", "on",
        "--sd_communication_amortized_q_shadow", "on",
        "--sd_contribution_restoration_shadow", "on",
        "--sd_micro_hold_shadow", "on",
        "--sd_unified_batch_dispatch_mode", "off",
    ],
    "state_driven_mature_cadence_v2_6_shadow": [
        "--sd_lyapunov_mode", "apply",
        "--sd_lyapunov_action_scope", "effective_service_v2_1",
        "--sd_lyapunov_workload_queue", "off",
        "--sd_lyapunov_holding_weight", "0.25",
        "--sd_lyapunov_max_holding_wait", "85.0",
        "--sd_lyapunov_max_holding_ratio", "6.5",
        "--sd_lyapunov_q_reference_spec", BASELINE_MEDIAN_Q_SEED2026,
        "--sd_finite_sample_safety_calibration", "on",
        "--sd_lyapunov_region_extension_ratio", "0.10",
        "--sd_lyapunov_create_hysteresis", "0.10",
        "--sd_lyapunov_recruit_safe_cap_ratio", "2.0",
        "--sd_lyapunov_create_safe_cost", "on",
        "--sd_lyapunov_join_cadence_weight", "1.0",
        "--sd_reason_aware_one_report_structural_shadow", "on",
        "--sd_reason_aware_routing_shadow", "on",
        "--sd_fair_contribution_shadow", "on",
        "--sd_communication_amortized_q_shadow", "on",
        "--sd_contribution_restoration_shadow", "on",
        "--sd_micro_hold_shadow", "on",
        "--sd_unified_batch_dispatch_mode", "off",
    ],
}
LYAPUNOV_PRESETS["state_driven_bounded_fairness_v2_7_shadow"] = [
    *LYAPUNOV_PRESETS["state_driven_mature_cadence_v2_6_shadow"],
    "--sd_communication_amortized_q_max_ratio", "3.0",
]
for _preset in LYAPUNOV_PRESETS:
    PRESETS[_preset] = ("state_apply", "state_apply_fixed_q", "fedcompass", "summary")


def command_for(preset: str, budget: int, seed: int, output_dir: Path) -> list[str]:
    existing, window, new_q, trace_level = PRESETS[preset]
    command = [
        sys.executable, "-u", "-m", "su_compass.experiments.run_virtual_fl",
        "--algorithm", "state_driven_compass",
        "--server_config", "su_compass/config/virtual_fedcompass_cifar10_bn_partition_fix.yaml",
        "--client_config", "examples/config/client_cifar10_dual_moderate.yaml",
        "--num_clients", "8", "--num_global_epochs", str(budget),
        "--min_local_steps", "40", "--max_local_steps", "200",
        "--seed", str(seed), "--base_step_time", str(RTX4090_BASE_STEP_TIME),
        "--model_size_mb", str(RTX4090_MODEL_SIZE_MB),
        "--update_size_mb", str(RTX4090_UPDATE_SIZE_MB),
        "--output_dir", str(output_dir), "--algorithm_variant", preset,
        "--sd_existing_group_mode", existing,
        "--sd_new_group_window_mode", window,
        "--sd_new_group_q_mode", new_q,
        "--sd_candidate_trace_level", trace_level,
    ]
    if preset in LYAPUNOV_PRESETS:
        command.extend([
            "--sd_lyapunov_rhythm_target", "16.4",
            "--sd_lyapunov_v", "1.0",
            "--sd_lyapunov_max_holding_wait", "80.0",
            "--sd_lyapunov_q_trust_eta", "1.1",
            "--sd_lyapunov_create_penalty", "0.25",
            "--sd_lyapunov_client_target_rates", BASELINE_TARGET_RATES_08,
        ])
        command.extend(LYAPUNOV_PRESETS[preset])
    return command


def run_with_live_log(command: list[str], *, cwd: Path, log_path: Path) -> int:
    """Mirror child output to the terminal and run.log without extra tools."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_root", default="su_compass/output/state_driven_ablation")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output_root / args.preset / f"seed{args.seed}"
    marker = output / "run_complete.json"
    if marker.exists() and not args.force:
        print(f"[skip] completed: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    command = command_for(args.preset, args.budget, args.seed, output)
    if args.no_progress:
        command.append("--no_progress")
    (output / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    returncode = run_with_live_log(command, cwd=PROJECT_ROOT, log_path=output / "run.log")
    if returncode != 0 or not marker.exists():
        raise SystemExit(f"[fail] {output / 'run.log'}")
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
