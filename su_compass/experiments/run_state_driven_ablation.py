"""Run all fixed State-Driven presets from one unchanged code version."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from su_compass.experiments.run_cifar10_main import (
    RTX4090_BASE_STEP_TIME, RTX4090_MODEL_SIZE_MB, RTX4090_UPDATE_SIZE_MB,
)
from su_compass.experiments.run_cifar10_state_driven import (
    PROJECT_ROOT, run_with_live_log,
)


PRESETS = [
    "state_driven_fc", "state_q_only", "state_driven_shadow", "state_driven_join",
    "state_driven_fixed_q_window", "state_driven_full",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_root", default="su_compass/output/state_driven_ablation")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()
    failed = []
    for preset in PRESETS:
        if preset == "state_q_only":
            output = PROJECT_ROOT / args.output_root / preset / f"seed{args.seed}"
            marker = output / "run_complete.json"
            if marker.exists() and not args.force:
                print(f"[skip] completed: {output}")
                continue
            output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-u", "-m", "su_compass.experiments.run_virtual_fl",
                "--algorithm", "state_compass",
                "--server_config", "su_compass/config/virtual_fedcompass_cifar10_bn_partition_fix.yaml",
                "--client_config", "examples/config/client_cifar10_dual_moderate.yaml",
                "--num_clients", "8", "--num_global_epochs", str(args.budget),
                "--min_local_steps", "40", "--max_local_steps", "200",
                "--seed", str(args.seed), "--base_step_time", str(RTX4090_BASE_STEP_TIME),
                "--model_size_mb", str(RTX4090_MODEL_SIZE_MB),
                "--update_size_mb", str(RTX4090_UPDATE_SIZE_MB),
                "--group_admission_mode", "shadow", "--output_dir", str(output),
                "--algorithm_variant", preset,
            ]
            (output / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        else:
            command = [
                sys.executable, "-m", "su_compass.experiments.run_cifar10_state_driven",
                "--preset", preset, "--budget", str(args.budget), "--seed", str(args.seed),
                "--output_root", args.output_root,
            ]
        if args.force and preset != "state_q_only":
            command.append("--force")
        if args.no_progress:
            command.append("--no_progress")
        if preset == "state_q_only":
            returncode = run_with_live_log(
                command, cwd=PROJECT_ROOT, log_path=output / "run.log",
            )
        else:
            returncode = subprocess.run(command, cwd=PROJECT_ROOT).returncode
        if returncode:
            failed.append(preset)
        elif preset == "state_q_only":
            from su_compass.experiments.analyze_state_driven import write_summary
            write_summary(output)
    if failed:
        raise SystemExit(f"failed presets: {', '.join(failed)}")
    from su_compass.experiments.analyze_state_driven import write_ablation_comparison
    comparison = write_ablation_comparison(
        PROJECT_ROOT / args.output_root, PRESETS, args.seed,
    )
    print(f"[done] comparison -> {comparison}")


if __name__ == "__main__":
    main()
