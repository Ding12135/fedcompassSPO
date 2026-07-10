"""
su_compass.experiments.console — 实验终端输出：硬件环境横幅与进度条。

参考 PyTorch Lightning / HuggingFace Accelerate 的常见做法：
    - 启动时打印可复现的硬件与运行环境摘要
    - 训练/调度过程中用 tqdm 显示预算进度与关键指标
    - 非 TTY 环境（日志重定向）自动降级为纯文本
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore


# ──────────────────────────── 硬件信息 ────────────────────────────

@dataclass
class HardwareInfo:
    """单次实验启动时可记录的硬件/软件环境。"""

    hostname: str
    platform: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: str
    cudnn_version: str
    gpu_devices: List[str] = field(default_factory=list)
    gpu_memory_gb: List[str] = field(default_factory=list)
    cpu_name: str = ""
    cpu_count: int = 0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    train_device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hostname": self.hostname,
            "platform": self.platform,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "gpu_devices": self.gpu_devices,
            "gpu_memory_gb": self.gpu_memory_gb,
            "cpu_name": self.cpu_name,
            "cpu_count": self.cpu_count,
            "ram_total_gb": self.ram_total_gb,
            "ram_available_gb": self.ram_available_gb,
            "train_device": self.train_device,
        }


def _read_linux_meminfo() -> tuple[float, float]:
    """读取 Linux 内存总量与可用量 (GB)。"""
    total_kb = avail_kb = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
    except OSError:
        return 0.0, 0.0
    return total_kb / (1024 ** 2), avail_kb / (1024 ** 2)


def _read_cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def gather_hardware_info(train_device: str = "cuda") -> HardwareInfo:
    """采集当前机器硬件与 PyTorch 运行环境。"""
    import torch

    gpu_names: List[str] = []
    gpu_mem: List[str] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_names.append(props.name)
            total_gb = props.total_memory / (1024 ** 3)
            try:
                free_b, total_b = torch.cuda.mem_get_info(i)
                gpu_mem.append(f"{free_b / (1024**3):.1f}/{total_gb:.1f} GB free")
            except Exception:
                gpu_mem.append(f"{total_gb:.1f} GB total")

    ram_total, ram_avail = _read_linux_meminfo()
    if ram_total <= 0 and platform.system() != "Linux":
        ram_total = ram_avail = 0.0

    _cudnn = getattr(torch.backends, "cudnn", None)
    cudnn_ver = "N/A"
    if _cudnn is not None:
        v = getattr(_cudnn, "version", None)
        if v is not None:
            cudnn_ver = str(v() if callable(v) else v)

    return HardwareInfo(
        hostname=platform.node(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=getattr(torch.version, "cuda", None) or "N/A",
        cudnn_version=cudnn_ver,
        gpu_devices=gpu_names,
        gpu_memory_gb=gpu_mem,
        cpu_name=_read_cpu_name(),
        cpu_count=os.cpu_count() or 0,
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        train_device=train_device,
    )


def _box_line(text: str, width: int = 68) -> str:
    inner = f" {text} "
    if len(inner) > width:
        inner = inner[: width - 1] + " "
    return f"║{inner:<{width}}║"


def print_experiment_banner(
    *,
    title: str,
    run_lines: List[str],
    hardware: Optional[HardwareInfo] = None,
    enabled: bool = True,
) -> None:
    """打印实验启动横幅（硬件 + 本次 run 配置）。"""
    if not enabled:
        return

    width = 68
    top = f"╔{'═' * width}╗"
    mid = f"╠{'═' * width}╣"
    bot = f"╚{'═' * width}╝"

    lines = [top, _box_line(title, width), mid]
    lines.append(_box_line(f"启动时间  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", width))
    lines.append(mid)
    lines.append(_box_line("实验配置", width))
    for item in run_lines:
        lines.append(_box_line(item, width))

    if hardware is not None:
        lines.append(mid)
        lines.append(_box_line("硬件 / 运行环境", width))
        lines.append(_box_line(f"主机      {hardware.hostname}", width))
        lines.append(_box_line(f"系统      {hardware.platform}", width))
        lines.append(_box_line(f"Python    {hardware.python_version}", width))
        lines.append(_box_line(f"PyTorch   {hardware.torch_version}", width))
        lines.append(_box_line(f"训练设备  {hardware.train_device}", width))
        if hardware.cuda_available and hardware.gpu_devices:
            for i, (name, mem) in enumerate(zip(hardware.gpu_devices, hardware.gpu_memory_gb)):
                lines.append(_box_line(f"GPU [{i}]  {name}  ({mem})", width))
            lines.append(_box_line(f"CUDA      {hardware.cuda_version}  |  cuDNN {hardware.cudnn_version}", width))
        else:
            lines.append(_box_line("GPU       未检测到 CUDA 设备，将使用 CPU", width))
        if hardware.cpu_name:
            lines.append(_box_line(f"CPU       {hardware.cpu_name[:48]}", width))
            lines.append(_box_line(f"CPU 核心  {hardware.cpu_count}", width))
        if hardware.ram_total_gb > 0:
            lines.append(_box_line(
                f"内存      {hardware.ram_available_gb:.1f} / {hardware.ram_total_gb:.1f} GB 可用",
                width,
            ))

    lines.append(bot)
    print("\n".join(lines), flush=True)


def print_run_footer(
    *,
    success: bool,
    elapsed_s: float,
    summary_lines: List[str],
    enabled: bool = True,
) -> None:
    """实验结束摘要。"""
    if not enabled:
        return
    status = "完成" if success else "异常结束"
    print()
    print(f"{'─' * 70}")
    print(f"  SU-Compass 实验{status}  |  耗时 {elapsed_s:.1f}s ({elapsed_s / 60:.1f} min)")
    for line in summary_lines:
        print(f"  · {line}")
    print(f"{'─' * 70}")
    print(flush=True)


# ──────────────────────────── 单次实验进度条 ────────────────────────────

class ExperimentProgress:
    """虚拟 FL 主循环进度条：以 client-update budget 为主刻度。"""

    def __init__(self, total_budget: int, enabled: bool = True) -> None:
        self.total_budget = max(1, total_budget)
        self.enabled = bool(enabled and tqdm is not None and sys.stderr.isatty())
        self._pbar: Optional[Any] = None
        self._last_budget = 0

    def start(self, desc: str = "实验进度") -> None:
        if not self.enabled:
            return
        self._pbar = tqdm(
            total=self.total_budget,
            desc=desc,
            unit="upd",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
            file=sys.stderr,
        )

    def log(self, message: str) -> None:
        """在进度条上方输出一条状态（不破坏条形显示）。"""
        if self._pbar is not None:
            self._pbar.write(f"  ▸ {message}")
        else:
            print(f"  ▸ {message}", flush=True)

    def set_phase(self, phase: str) -> None:
        if self._pbar is not None:
            self._pbar.set_description_str(phase[:28])

    def update_budget(
        self,
        budget_used: int,
        *,
        global_version: Optional[int] = None,
        accuracy: Optional[float] = None,
        virtual_time: Optional[float] = None,
        phase: Optional[str] = None,
    ) -> None:
        delta = max(0, budget_used - self._last_budget)
        self._last_budget = budget_used

        parts: List[str] = []
        if global_version is not None:
            parts.append(f"v={global_version}")
        if accuracy is not None:
            parts.append(f"acc={accuracy:.2f}%")
        if virtual_time is not None:
            parts.append(f"t={virtual_time:.1f}s")
        postfix = " | ".join(parts) if parts else ""

        if self._pbar is None:
            if delta > 0 or phase:
                msg = f"[{budget_used}/{self.total_budget}]"
                if postfix:
                    msg += f" {postfix}"
                if phase:
                    msg = f"{phase} {msg}"
                print(msg, flush=True)
            return

        if phase:
            self._pbar.set_description_str(phase[:28])
        if delta > 0:
            self._pbar.update(delta)
        if postfix:
            self._pbar.set_postfix_str(postfix, refresh=True)

    def on_train_start(self, client_id: str, local_steps: int, client_round: int) -> None:
        self.set_phase(f"GPU训练 {client_id}")
        if self._pbar is not None:
            self._pbar.set_postfix_str(f"Q={local_steps} round={client_round}", refresh=True)

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None


# ──────────────────────────── 批量实验进度条 ────────────────────────────

class BatchExperimentProgress:
    """批量运行器（Stage A–D / convergence compare）的外层进度。"""

    def __init__(self, total: int, stage_name: str, enabled: bool = True) -> None:
        self.total = total
        self.stage_name = stage_name
        self.enabled = bool(enabled and tqdm is not None and sys.stderr.isatty())
        self._pbar: Optional[Any] = None
        self._ok = 0
        self._fail = 0
        self._t0 = time.time()

    def __enter__(self) -> "BatchExperimentProgress":
        if self.enabled:
            self._pbar = tqdm(
                total=self.total,
                desc=f"批量实验 {self.stage_name}",
                unit="run",
                dynamic_ncols=True,
                file=sys.stderr,
            )
        else:
            print(f"\n{'=' * 70}\n  批量实验: {self.stage_name}  (共 {self.total} 组)\n{'=' * 70}")
        return self

    def __exit__(self, *args: object) -> None:
        elapsed = time.time() - self._t0
        if self._pbar is not None:
            self._pbar.close()
        print(
            f"\n  批量 [{self.stage_name}] 结束: "
            f"成功 {self._ok} / 失败 {self._fail} / 共 {self.total}  |  耗时 {elapsed / 60:.1f} min",
            flush=True,
        )

    def log(self, message: str) -> None:
        if self._pbar is not None:
            self._pbar.write(message)
        else:
            print(message, flush=True)

    def begin_run(self, name: str, seed: int, index: int) -> None:
        label = f"[{index}/{self.total}] {name} seed={seed}"
        if self._pbar is not None:
            self._pbar.set_postfix_str(label, refresh=True)
            self._pbar.write(f"  ▶ 开始 {label}")
        else:
            print(f"\n  ▶ 开始 {label}", flush=True)

    def end_run(self, ok: bool, elapsed_s: float) -> None:
        if ok:
            self._ok += 1
        else:
            self._fail += 1
        status = "OK" if ok else "FAIL"
        msg = f"  ◀ {status} ({elapsed_s:.0f}s)"
        if self._pbar is not None:
            self._pbar.write(msg)
            self._pbar.update(1)
        else:
            print(msg, flush=True)
