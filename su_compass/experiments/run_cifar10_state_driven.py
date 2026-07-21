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


def command_for(preset: str, budget: int, seed: int, output_dir: Path) -> list[str]:
    existing, window, new_q, trace_level = PRESETS[preset]
    return [
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
