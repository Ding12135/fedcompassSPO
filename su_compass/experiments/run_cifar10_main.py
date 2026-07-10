"""批量运行 CIFAR-10 FedCompass / Oort-Compass 主实验。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHODS = ("fedcompass", "q_only", "q_and_group")
DEFAULT_SEEDS = (2026, 2027, 2028)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 Oort-Compass 主实验")
    parser.add_argument("--budget", type=int, default=1500,
                        help="client update budget；CIFAR Dirichlet 论文级收敛固定标准 1500")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--methods", nargs="+", choices=DEFAULT_METHODS, default=list(DEFAULT_METHODS)
    )
    parser.add_argument(
        "--output_root", default="su_compass/output/cifar10_main",
    )
    parser.add_argument(
        "--server_config",
        default="su_compass/config/virtual_fedcompass_cifar10_8.yaml",
    )
    parser.add_argument("--client_config", default="examples/config/client_cifar10.yaml")
    parser.add_argument("--base_step_time", type=float, default=0.0703)
    parser.add_argument("--model_size_mb", type=float, default=42.66)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, method: str, seed: int, output: Path) -> list[str]:
    algorithm = "fedcompass" if method == "fedcompass" else "oort_compass"
    command = [
        sys.executable,
        "-m",
        "su_compass.experiments.run_virtual_fl",
        "--algorithm",
        algorithm,
        "--server_config",
        args.server_config,
        "--client_config",
        args.client_config,
        "--num_clients",
        "8",
        "--num_global_epochs",
        str(args.budget),
        "--seed",
        str(seed),
        "--base_step_time",
        str(args.base_step_time),
        "--model_size_mb",
        str(args.model_size_mb),
        "--update_size_mb",
        str(args.model_size_mb),
        "--output_dir",
        str(output),
        "--algorithm_variant",
        method,
    ]
    if algorithm == "oort_compass":
        command.extend(["--oort_mode", method])
    if args.no_progress:
        command.append("--no_progress")
    return command


def main() -> None:
    args = parse_args()
    output_root = PROJECT_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for method in args.methods:
            output = output_root / method / f"seed{seed}"
            marker = output / "experiment_config.json"
            if marker.exists() and not args.force:
                print(f"[skip] {method} seed={seed} 已完成")
                continue

            output.mkdir(parents=True, exist_ok=True)
            command = build_command(args, method, seed, output)
            log_path = output_root / f"{method}_seed{seed}.log"
            print(f"[run] {method} seed={seed} budget={args.budget}")
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            if result.returncode != 0 or not marker.exists():
                raise SystemExit(
                    f"[fail] {method} seed={seed}，详情见 {log_path}"
                )
            print(f"[done] {method} seed={seed} -> {output}")

    print("[done] CIFAR-10 主实验全部完成")


if __name__ == "__main__":
    main()
