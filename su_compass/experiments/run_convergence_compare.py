"""
su_compass.experiments.run_convergence_compare — FedCompass vs Oort-Compass 收敛对比。

仅对比 fedcompass 与 oort_compass(q_only)，用于在补全 FedAvg/FedAsync 之前
先判断 Oort 引入是否有收益、是否值得继续投入。

用法：
    python -m su_compass.experiments.run_convergence_compare
    python -m su_compass.experiments.run_convergence_compare --budget 300 --seeds 2026 2027 2028
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from su_compass.experiments.console import (
    BatchExperimentProgress,
    gather_hardware_info,
    print_experiment_banner,
)

OUTPUT_ROOT = Path("su_compass/output/convergence_compare")
DEFAULT_BUDGET = 250  # client update budget；budget=30 时 final~85%，250 目标冲到 95%+
DEFAULT_SEEDS: Tuple[int, ...] = (2026, 2027, 2028)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_one(
    algorithm: str,
    seed: int,
    budget: int,
    oort_mode: str,
    force: bool,
    output_root: Path,
    batch: Optional[BatchExperimentProgress] = None,
    run_index: int = 0,
) -> bool:
    name = "fedcompass" if algorithm == "fedcompass" else "q_only"
    out = _project_root() / output_root / name / f"seed{seed}"
    marker = out / "experiment_config.json"
    if marker.exists() and not force:
        if batch is not None:
            batch.begin_run(name, seed, run_index)
            batch.log(f"  ⊘ 跳过（已存在） {name} seed={seed}")
            batch.end_run(True, 0.0)
        else:
            print(f"[skip] {name} seed={seed}")
        return True

    run_args = [
        sys.executable, "-m", "su_compass.experiments.run_virtual_fl",
        "--algorithm", algorithm,
        "--seed", str(seed),
        "--num_global_epochs", str(budget),
        "--num_clients", "8",
        "--output_dir", str(out),
        "--algorithm_variant", name,
    ]
    if algorithm == "oort_compass":
        run_args += ["--oort_mode", oort_mode]

    if batch is not None:
        batch.begin_run(name, seed, run_index)
    else:
        print(f"\n[run] {' '.join(run_args)}")
    t0 = time.time()
    proc = subprocess.run(run_args, cwd=_project_root())
    elapsed = time.time() - t0
    ok = proc.returncode == 0 and marker.exists()
    if batch is not None:
        batch.end_run(ok, elapsed)
    else:
        print(f"[{'OK' if ok else 'FAIL'}] {name} seed={seed} ({elapsed:.0f}s)")
    return ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FedCompass vs Oort-Compass 收敛对比")
    p.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                   help="client update budget（FedCompass 终止条件）")
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--oort_mode", default="q_only", choices=["q_only", "q_and_group"])
    p.add_argument("--output_root", type=str, default=str(OUTPUT_ROOT),
                   help="相对项目根的输出目录")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no_progress", action="store_true", help="关闭硬件横幅与批量进度条")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    show_ui = not args.no_progress
    out_root = Path(args.output_root)
    (_project_root() / out_root).mkdir(parents=True, exist_ok=True)

    hardware = gather_hardware_info()
    print_experiment_banner(
        title="SU-Compass 收敛对比实验 (FedCompass vs Oort)",
        run_lines=[
            f"训练预算  {args.budget} client updates",
            f"Seeds     {args.seeds}",
            f"Oort模式  {args.oort_mode}",
            f"输出目录  {out_root}",
        ],
        hardware=hardware,
        enabled=show_ui,
    )

    runs: List[Tuple[str, int]] = []
    for seed in args.seeds:
        runs.append(("fedcompass", seed))
        runs.append(("oort_compass", seed))

    all_ok = True
    with BatchExperimentProgress(len(runs), "收敛对比", enabled=show_ui) as batch:
        for i, (algo, seed) in enumerate(runs, start=1):
            all_ok &= _run_one(
                algo, seed, args.budget, args.oort_mode, args.force, out_root, batch, i,
            )

    if not all_ok:
        sys.exit(1)
    print("\n[done] 收敛对比实验完成。运行 analyze_convergence_compare 查看结果。")


if __name__ == "__main__":
    main()
