"""Run the controlled CIFAR-10 FedCompass convergence candidate.

This entry point deliberately runs only the baseline.  StateCompass should be
re-evaluated after the shared training base reaches the accuracy gate.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from su_compass.experiments.run_cifar10_main import (
    RTX4090_BASE_STEP_TIME,
    RTX4090_MODEL_SIZE_MB,
    RTX4090_UPDATE_SIZE_MB,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--qmax", type=int, default=80)
    parser.add_argument("--qmin", type=int, default=40)
    parser.add_argument(
        "--output_dir",
        default="su_compass/output/cifar10_baseline_convergence/seed2026",
    )
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    command = [
        sys.executable,
        "-m",
        "su_compass.experiments.run_virtual_fl",
        "--algorithm", "fedcompass",
        "--server_config", "su_compass/config/virtual_fedcompass_cifar10_convergence.yaml",
        "--client_config", "examples/config/client_cifar10_standard_dirichlet.yaml",
        "--num_clients", "8",
        "--num_global_epochs", str(args.budget),
        "--min_local_steps", str(args.qmin),
        "--max_local_steps", str(args.qmax),
        "--seed", str(args.seed),
        "--base_step_time", str(RTX4090_BASE_STEP_TIME),
        "--model_size_mb", str(RTX4090_MODEL_SIZE_MB),
        "--update_size_mb", str(RTX4090_UPDATE_SIZE_MB),
        "--output_dir", args.output_dir,
        "--algorithm_variant", "fedcompass_convergence_v1",
    ]
    if args.no_progress:
        command.append("--no_progress")
    raise SystemExit(subprocess.run(command, cwd=PROJECT_ROOT).returncode)


if __name__ == "__main__":
    main()
