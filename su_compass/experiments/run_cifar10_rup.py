"""One-command CIFAR-10 RUP-Compass full/shadow/ablation runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from su_compass.experiments.run_cifar10_main import (
    RTX4090_BASE_STEP_TIME, RTX4090_MODEL_SIZE_MB, RTX4090_UPDATE_SIZE_MB,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESETS = {
    "full": {
        "rup_group_admission": "conservative",
        "rup_accuracy_priority": "on",
        "rup_accuracy_q_floor_ratio": "1.0",
        "rup_accuracy_q_boost_ratio": "0.05",
        "rup_accuracy_q_boost_min_confidence": "0.5",
        "rup_accuracy_q_boost_start_accuracy": "50.0",
    },
    "stage_boost50": {
        "rup_group_admission": "conservative",
        "rup_accuracy_priority": "on",
        "rup_accuracy_q_floor_ratio": "1.0",
        "rup_accuracy_q_boost_ratio": "0.10",
        "rup_accuracy_q_boost_min_confidence": "0.3",
        "rup_accuracy_q_boost_start_accuracy": "50.0",
    },
    "q_only_stable": {
        "rup_group_admission": "off",
        "rup_accuracy_priority": "on",
        "rup_accuracy_q_floor_ratio": "1.0",
        "rup_accuracy_q_boost_ratio": "0.0",
        "rup_risk_gated_floor": "on",
        "rup_risk_gated_floor_min_safe_candidates": "10",
        "rup_risk_gated_floor_slack_ratio": "0.05",
        "rup_q_smoothness": "on",
        "rup_q_smooth_max_increase_ratio": "0.10",
        "rup_q_smooth_max_decrease_ratio": "0.20",
    },
    "aggressive_group_admission": {"rup_group_admission": "apply"},
    "shadow": {"rup_mode": "shadow", "rup_prox": "off"},
    "off": {"rup_mode": "off", "rup_group_admission": "off", "rup_prox": "off"},
    "state_only": {"rup_utility": "off", "rup_budget": "off", "rup_group_admission": "off", "rup_prox": "off"},
    "state_prox": {"rup_utility": "off", "rup_budget": "off", "rup_group_admission": "off"},
    "state_utility": {"rup_group_admission": "off", "rup_prox": "off"},
    "no_residual": {"rup_residual_risk": "off"},
    "no_trust": {"rup_trust": "off"},
    "no_soft_boundary": {"rup_soft_boundary": "off"},
    "no_utility": {"rup_utility": "off"},
    "no_budget": {"rup_budget": "off"},
    "no_group_admission": {"rup_group_admission": "off"},
    "no_accuracy_priority": {"rup_accuracy_priority": "off"},
    "no_prox": {"rup_prox": "off"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="CIFAR-10 RUP-Compass experiment")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_root", default="su_compass/output/cifar10_rup")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    output = PROJECT_ROOT / args.output_root / args.preset / f"seed{args.seed}"
    marker = output / "experiment_config.json"
    if marker.exists() and not args.force:
        print(f"[skip] {args.preset} seed={args.seed} already complete: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "su_compass.experiments.run_virtual_fl",
        "--algorithm", "rup_compass",
        "--server_config", "su_compass/config/virtual_fedcompass_cifar10_bn_partition_fix.yaml",
        "--client_config", "examples/config/client_cifar10_dual_moderate.yaml",
        "--num_clients", "8", "--num_global_epochs", str(args.budget),
        "--min_local_steps", "40", "--max_local_steps", "200",
        "--seed", str(args.seed),
        "--base_step_time", str(RTX4090_BASE_STEP_TIME),
        "--model_size_mb", str(RTX4090_MODEL_SIZE_MB),
        "--update_size_mb", str(RTX4090_UPDATE_SIZE_MB),
        "--output_dir", str(output),
        "--algorithm_variant", f"rup_{args.preset}",
    ]
    for key, value in PRESETS[args.preset].items():
        command.extend([f"--{key}", value])
    if args.no_progress:
        command.append("--no_progress")
    log_path = output.parent / f"seed{args.seed}.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not marker.exists():
        raise SystemExit(f"[fail] see {log_path}")
    print(f"[done] {args.preset} seed={args.seed}: {output}")


if __name__ == "__main__":
    main()
