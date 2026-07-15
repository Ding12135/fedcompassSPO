# 云端 CIFAR-10 收敛实验说明（FedCompass vs q_only）

> **口径提示（2026-07-14）：** 本文中的历史运行流程继续保留；正式数据划分、论文对齐和三方法公平比较要求以
> [`../cifar说明/数据划分与公平对比规范.md`](../cifar说明/数据划分与公平对比规范.md) 为准。

本文档面向**云端 Agent / 远程执行环境**，说明如何在云 GPU 上跑通 CIFAR-10 主实验、判断是否收敛、如何分析结果，以及后续优化调优方向。

---

## 1. 实验目标

**当前阶段分两步，必须按顺序执行：**

1. **第一阶段（必做）**：在云 GPU 上校准 `base_step_time` / `model_size_mb`；
2. **第二阶段（正式实验）**：固定 **budget=1500**，跑 **FedCompass + q_only × seeds 2026/2027/2028** 到论文级收敛平台。

> **禁止跳过第一阶段直接跑正式实验。** 未校准虚拟端侧参数时，TTA / late / deadline 等系统指标不可比。

**暂不跑：**

- `q_and_group`（本地 seed2026 已证实 CIFAR 上 staleness 过高、精度更差）
- loss-based statistical utility（下一阶段）
- MNIST（本地已有结果）

**论文叙事定位：**

> 验证 reason-aware Oort-Compass（q_only）在难数据集 CIFAR-10 上，是否能在**不系统性降低最终精度**的前提下，改善 **late / deadline / 早期 TTA**。

---

## 2. 前置条件检查清单

Agent 开跑前必须确认：

| 检查项 | 命令 / 路径 | 通过标准 |
|--------|------------|----------|
| GPU 可用 | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` | `True` + 有 GPU 名称 |
| 项目可导入 | `pip install -e .` 后 `python -c "import su_compass"` | 无报错 |
| CIFAR-10 数据 | `ls examples/datasets/RawData/cifar-10-batches-py` | 目录存在 |
| 稳定性修复已包含 | 检查以下文件 | 见 §2.1 |
| 磁盘空间 | `df -h .` | 建议 ≥ 10GB 空闲 |

### 2.1 必须包含的代码修复（无则先同步最新代码）

以下修复是 CIFAR 能稳定跑通的前提，**baseline 与 q_only 共用**：

1. **staleness 多项式方向**：`(u+1)^(-a)`，不是 `(u+1)^a`
   - `src/appfl/aggregator/fedcompass_aggregator.py`
   - `src/appfl/aggregator/fedasync_aggregator.py`
2. **梯度裁剪在 `optimizer.step()` 之前**
   - `src/appfl/trainer/naive_trainer.py`
3. **CIFAR 专用配置**
   - `su_compass/config/virtual_fedcompass_cifar10_8.yaml`
4. **NaN 保护**：本地更新含 NaN/Inf 时中止
   - `su_compass/virtual/client_runtime.py`
5. **批量 runner / analyzer**
   - `su_compass/experiments/run_cifar10_main.py`
   - `su_compass/experiments/analyze_cifar10_main.py`

---

## 3. 环境初始化

```bash
cd <项目根目录>    # 例如 /root/gpufree-data/fedcompass

conda create -n fedcompass python=3.10 -y
conda activate fedcompass
pip install -e .

# 下载 CIFAR-10（国内镜像）
python scripts/prepare_datasets.py --datasets cifar10 --source cn
```

---

## 4. 第一阶段：GPU 训练参数校准（正式实验前必做）

虚拟时间实验需要把**真实 GPU 训练耗时**映射为虚拟端侧参数。不同 GPU（3060 / 4090 / A100）的单步耗时差异很大，**第一次实验必须先做校准，不得跳过**。

需校准的两个量：

| 参数 | 含义 | 用途 |
|------|------|------|
| `base_step_time` | 计算能力 1.0 时每个 local step 的真实耗时（秒） | 虚拟调度时间轴 |
| `model_size_mb` | ResNet-18 模型 state_dict 字节数（MB） | 虚拟下载/上传耗时 |

```bash
python - <<'PY'
import copy, os, sys, time
from pathlib import Path
from omegaconf import OmegaConf

sys.path.insert(0, str(Path("examples").resolve()))
os.chdir("examples")
from appfl.agent import APPFLClientAgent, APPFLServerAgent

server = APPFLServerAgent(
    server_agent_config=OmegaConf.load("../su_compass/config/virtual_fedcompass_cifar10_8.yaml")
)
client = APPFLClientAgent(
    client_agent_config=OmegaConf.load("config/client_cifar10.yaml")
)
client.load_config(server.get_client_configs())
client.load_parameters(copy.deepcopy(server.model.state_dict()))
client.trainer.train_configs.num_local_steps = 40
client.train()  # warmup
t0 = time.perf_counter()
for _ in range(3):
    client.train()
dt = (time.perf_counter() - t0) / 3
base_step = dt / 40
n = sum(p.numel() for p in server.model.parameters())
size_mb = sum(p.numel() * p.element_size() for p in server.model.state_dict().values()) / (1024**2)
print(f"base_step_time={base_step:.6f}")
print(f"model_size_mb={size_mb:.4f}")
PY
```

记下输出，后续命令中替换 `BASE_STEP` 和 `MODEL_MB`。

校准结果写入环境变量，后续所有实验共用：

```bash
export BASE_STEP=<云上测得的 base_step_time>
export MODEL_MB=<云上测得的 model_size_mb>
```

当前正式云端环境已于 2026-07-13 在 **NVIDIA GeForce RTX 4090** 上完成校准：

```text
base_step_time = 0.015435538
model_size_mb  = 42.662056
update_size_mb = 42.662056
40 steps 中位耗时 = 0.617422 秒
```

上述数值已固化为 `run_cifar10_main.py` 的 CIFAR 默认值。在同一台
RTX 4090 上运行主实验时可以不再显式传入这三个参数；更换 GPU
或修改模型后必须重新校准。

**第一阶段完成标准：** 已输出 `BASE_STEP` 和 `MODEL_MB`，并确认单次 round-40 训练可稳定完成、无 NaN。

---

## 5. 论文收敛目标（FedCompass ICLR 2024）

正式实验必须跑到**接近论文报告的 top validation accuracy**，不能停在 budget=600。

### 5.1 论文中的 CIFAR-10 精度（Appendix F, Table 16/17）

论文使用 ResNet-18 + SGD lr=0.1 + CIFAR-10，报告的是 **top validation accuracy**（非单次 final）：

**Dual Dirichlet 划分（与当前实验最接近）**

当前 `client_cifar10.yaml` 使用 `partition_strategy: dirichlet_noniid, alpha2=0.5`，对应论文 **Dual Dirichlet** 划分。

| 客户端数 m | 异构设置 | FedCompass top acc（论文） |
|-----------|---------|---------------------------|
| 5 | homogeneous | **59.29 ± 3.49%** |
| 5 | normal (σ=0.3μ) | **59.98 ± 3.65%** |
| 5 | exponential | **61.23 ± 2.67%** |
| 10 | homogeneous | **54.51 ± 3.50%** |
| 10 | normal | **54.95 ± 3.68%** |
| 10 | exponential | **57.29 ± 1.98%** |

**Class 划分（更高，但不是当前设置）**

| 客户端数 m | FedCompass top acc（论文） |
|-----------|---------------------------|
| 5 | **72.66 – 73.96%** |
| 10 | **66.27 – 68.73%** |

### 5.2 本项目的收敛参照（8 客户端 + Dirichlet α2=0.5）

我们 8 客户端介于论文 m=5 与 m=10 之间，**不应拿 Class 划分的 66–74% 做目标**。

| 指标 | 论文参照 | 本项目收敛目标 |
|------|---------|---------------|
| **top accuracy（FedCompass）** | Dirichlet m=10: ~55–57%；m=5: ~59–61% | **max acc ≥ 55%**，理想 **57–60%** |
| **TTA 对比阈值** | Table 1: Dirichlet 用 **50%** 作目标精度 | 报告 TTA@50；辅助看 TTA@55 |
| **末段稳定性** | 论文报 top acc 均值 ± 标准差 | last-10 与 max 差距 **< 2pp** |

论文 Table 1 中 CIFAR-10 的 TTA 基准精度：

- **Class partition → 目标 60%**
- **Dirichlet partition → 目标 50%**（我们应对齐这个）

### 5.3 固定 budget 标准（云实验统一执行）

本地 budget=600 时 fedcompass max 仅 **43.88%**，末段精度还在波动下降，远未达到论文水平。云实验**不再做递增探测**，统一使用固定 budget：

```text
BUDGET = 1500   （client update budget，FedCompass 终止条件）
```

| 依据 | 说明 |
|------|------|
| 本地 600 → max 44% | 约为论文目标（55%）的 80%，需显著更多训练量 |
| 安全系数 2.5× | 600 × 2.5 = 1500，覆盖异步尾部波动与未收敛余量 |
| 论文无固定轮数 | 论文按 wall-clock 对齐跑到收敛；1500 是当前框架下的统一工程标准 |
| 两方法必须相同 | fedcompass 与 q_only 都用 **1500**，保证公平对比 |

**验收标准（跑满 1500 后检查）：**

| 指标 | 通过线 |
|------|--------|
| FedCompass max acc | **≥ 55%**（对齐论文 Dirichlet m≈8–10） |
| TTA@50 | 应能达到（论文 Dirichlet 对比基准） |
| last-10 与 max 差距 | < 3pp |
| 无 NaN | `finite=true` |

若云上 fedcompass seed2026 跑满 1500 后 max 仍 < 50%，说明是实现差距（如 BN 聚合、虚拟框架口径），不是再加 budget 能解决的——记录现象并回报，不要擅自改 budget。

### 5.4 论文为什么不追求「高精度」

论文**不是以 ImageNet 级或中心化训练的高精度为目标**，原因如下：

1. **对比基准是「各算法在同等条件下能达到的 top acc」**  
   论文原文：*"The dataset target accuracy is carefully selected based on each algorithm's top achievable accuracy during training."*  
   即：目标精度来自算法自身可达上限，而非 centralized SOTA。

2. **Dirichlet Non-IID 下精度天然较低**  
   - CIFAR Dirichlet 的 TTA 对比目标仅 **50%**（Table 1）  
   - Class 划分才用 **60%**  
   - 中心化 ResNet-18 在 CIFAR-10 可达 90%+，但联邦 + Non-IID + 异步下 **50–60% 已是正常平台**

3. **论文核心贡献是「异构下的收敛效率」，不是精度 SOTA**  
   Table 1 比的是**达到目标精度的 wall-clock 时间**（FedCompass = 1.00× baseline），关注的是：
   - 谁更快到达 50%（Dirichlet）或 60%（Class）
   - 不是在 90% 上比高低

4. **部分异步算法在部分设置下根本达不到目标**  
   Table 1 中 FedAsync / FedBuff 在多个 CIFAR 格子上为 **"—"**，表示超过半数 run 达不到目标精度——说明论文承认异步 FL 在难设置下的精度天花板。

5. **研究问题是 cross-silo 异构调度，不是刷榜**  
   客户端设备异构、数据 Non-IID、半异步 staleness 都会压低上限；论文要证明的是 FedCompass 在**这个上限附近**比 FedAsync/FedBuff 更快、更稳。

**对我们实验的含义：**

- 应对齐论文的 **Dirichlet top acc ~55–60%** 和 **TTA@50%**，而不是追求 70%+  
- 与 q_only 的对比也应在「接近论文收敛平台」时进行，才有可比性  
- 若我们 1500 budget 仍只有 40%+，需查实现问题，而非盲目加 budget 或换高精度目标

---

## 6. 实验矩阵

### 6.1 本轮要跑的实验

| 阶段 | 实验 | algorithm | seeds | budget | 说明 |
|------|------|-----------|-------|--------|------|
| 第一阶段 | GPU 校准 | — | — | — | §4，必做 |
| 第二阶段 | 正式对比 | fedcompass + q_only | 2026/2027/2028 | **1500（固定）** | 主实验 |

**公平性要求：** baseline 与 q_only 必须使用完全相同的：

- `server_config`: `su_compass/config/virtual_fedcompass_cifar10_8.yaml`
- `client_config`: `examples/config/client_cifar10.yaml`
- `base_step_time`、`model_size_mb`（第一阶段校准值）
- `num_clients=8`，`Qmin=40`，`Qmax=200`，**`budget=1500`**
- Oort 默认 λ：`comm=1.0, late=1.0, var=0.5, avail=0.5`

### 6.2 正式批量实验（fedcompass + q_only，budget=1500）

```bash
export BASE_STEP=<第一阶段校准值>
export MODEL_MB=<第一阶段校准值>
export BUDGET=1500

nohup python -m su_compass.experiments.run_cifar10_main \
  --budget "$BUDGET" \
  --seeds 2026 2027 2028 \
  --methods fedcompass q_only \
  --base_step_time "$BASE_STEP" \
  --model_size_mb "$MODEL_MB" \
  --output_root su_compass/output/cifar10_main \
  --no_progress \
  > su_compass/output/cifar10_main/run.log 2>&1 &
```

跑满后验收：

```bash
python3 - <<'PY'
import csv, statistics, sys
from pathlib import Path
p = Path("su_compass/output/cifar10_main/fedcompass/seed2026/global_eval_trace.csv")
rows = list(csv.DictReader(p.open()))
accs = [float(r["test_accuracy"]) for r in rows]
max_acc, last10 = max(accs), statistics.mean(accs[-10:])
print(f"max={max_acc:.2f}%  last10={last10:.2f}%  gap={max_acc-last10:.2f}pp")
for t in [50, 55]:
    hit = next((float(r["virtual_time"]) for r in rows if float(r["test_accuracy"]) >= t), None)
    print(f"TTA@{t}: {hit}")
print("PASS" if max_acc >= 55.0 else "BELOW_PAPER_TARGET")
PY
```

监控进度：

```bash
tail -f su_compass/output/cifar10_main/run.log
ls su_compass/output/cifar10_main/*.log
```

**断点续跑：** 已有 `experiment_config.json` 的 run 会自动跳过；强制重跑加 `--force`。

---

## 7. 输出目录结构

```
su_compass/output/cifar10_main/
├── fedcompass/
│   ├── seed2026/
│   ├── seed2027/
│   └── seed2028/
├── q_only/
│   ├── seed2026/
│   ├── seed2027/
│   └── seed2028/
├── fedcompass_seed2026.log
├── q_only_seed2027.log
├── ...
└── analysis/
    ├── report.md
    └── report.json
```

每个 seed 目录内关键文件：

| 文件 | 用途 |
|------|------|
| `experiment_config.json` | 完整实验配置（含硬件信息） |
| `global_eval_trace.csv` | 精度-时间曲线（主图数据源） |
| `aggregation_trace.csv` | staleness 分析 |
| `scheduler_trace.csv` | late 分析 |
| `group_trace.csv` | deadline 分析 |
| `oort_trace.csv` | q_only 专有：Q 方向、penalty |

---

## 8. 结果分析

### 8.1 一键汇总

```bash
python -m su_compass.experiments.analyze_cifar10_main \
  --output_root su_compass/output/cifar10_main

cat su_compass/output/cifar10_main/analysis/report.md
```

### 8.2 必看指标

**精度侧：**

| 指标 | 来源 | 判断标准 |
|------|------|----------|
| final accuracy | `global_eval_trace` 末次 | q_only 不低于 baseline 超过 1pp |
| max accuracy | 全程最高 | 参考上限 |
| last-10 mean | 末 10 次平均 | 减少异步尾部波动 |
| TTA@40/50/55 | 首次达到阈值的 virtual_time | **主看 TTA@50**（论文 Dirichlet 基准） |

**系统侧：**

| 指标 | 来源 | q_only 期望 |
|------|------|-------------|
| mean staleness | `aggregation_trace` | 可略升，但不应 > 2.5 |
| late uploads | `scheduler_trace` | 比 baseline 降 ≥ 25% |
| deadline triggers | `group_trace` | 比 baseline 降 ≥ 25% |
| mean Q | `round_reports` | 应低于 baseline（更保守） |

**稳定性：**

| 指标 | 判断 |
|------|------|
| `finite=true` | 无 NaN/Inf |
| 精度大跌次数 | 连续评估跌幅 >5pp 的次数应少 |

### 8.3 本地 seed2026 参考结果（budget=600，未收敛，仅供对照）

| 方法 | final | max | last-10 | TTA@30 | late | stale |
|------|-------|-----|---------|--------|------|-------|
| fedcompass | 34.6% | 43.9% | 35.5% | 789 | 85 | 1.05 |
| q_only | 37.3% | 43.1% | 37.3% | 357 | 31 | 1.61 |

初步结论：q_only 系统指标显著改善，但 budget=600 时 max 仅 ~44%，**距论文 Dirichlet 收敛目标（~55–60%）差约 12–16pp**。

---

## 9. 异常排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| loss/acc 变 NaN | staleness 指数方向错误 / 无梯度裁剪 | 确认 §2.1 修复已同步 |
| acc 锁死 10% | 全局模型权重爆炸 | 同上；检查 `gradient_based=True` + `send_gradient=True` |
| max acc < 50%（budget≥1200） | budget 不足或训练配置问题 | 继续加大 budget；检查 BN 聚合、学习率 |
| q_only 与 baseline 调度完全相同 | Oort mode 未传入 | 确认 `--oort_mode q_only` |
| 实验极慢 | 未用 GPU | 检查 `device: cuda` |
| 虚拟时间尺度异常 | `base_step_time` 未校准 | 重跑 §4 第一阶段校准 |

---

## 10. 后续优化与调优路线

根据本轮 CIFAR 收敛实验结果，按以下优先级推进。**每步只改一个因素，单 seed 验证后再扩 3 seeds。**

### P0：确认收敛与统计可信度

```text
1. 第一阶段 GPU 校准 → 第二阶段 fedcompass + q_only（budget=1500）
2. 确认 FedCompass max acc 达到论文级 ~55%+（Dirichlet, 8 clients）
3. 用 3 seed 均值判断 q_only 是否在收敛状态下优于 baseline
```

**最低可发表标准（CIFAR, Dirichlet）：**

- FedCompass baseline max acc 接近论文：**≥ 55%**（3 seed 均值）
- TTA@50：q_only 在 3/3 seeds 优于 FedCompass
- last-10 accuracy：q_only 均值不低于 baseline 超过 1pp
- late 或 deadline：至少下降 25%

### P1：q_only 超参调优（若精度 OK 但 TTA@40+ 慢）

本地观察到 q_only 早期快、冲高慢，可能因 Q 过保守导致 staleness 上升。可扫描：

| 参数 | 当前默认 | 扫描范围 | 目的 |
|------|----------|----------|------|
| `λ_late` | 1.0 | 0.5 / 1.0 / 2.0 | 平衡 late vs staleness |
| `λ_comm` | 1.0 | 0.5 / 1.0 / 2.0 | 网络差客户端惩罚力度 |
| `penalty_clip` | 3.0 | 2.0 / 3.0 / 5.0 | 防止 Q 被压得过低 |
| `q_floor_ratio` | 未启用 | 0.3 / 0.5 | 保精度下限（若已实现） |

单 seed 快速验证命令：

```bash
python -m su_compass.experiments.run_virtual_fl \
  --algorithm oort_compass --oort_mode q_only \
  --oort_lambda_late 0.5 \
  --seed 2026 --num_global_epochs 600 \
  --server_config su_compass/config/virtual_fedcompass_cifar10_8.yaml \
  --client_config examples/config/client_cifar10.yaml \
  --base_step_time "$BASE_STEP" --model_size_mb "$MODEL_MB" \
  --output_dir su_compass/output/cifar10_tune/lambda_late_0.5/seed2026 \
  --no_progress
```

### P2：训练侧收敛优化（若 max acc 上不去）

| 方向 | 说明 |
|------|------|
| 增大 budget | 已固定 1500；若仍不达标查 BN 聚合等非 budget 因素 |
| BN buffer 聚合 | 当前组聚合对 BN 做平均，可能伤害 ResNet eval；考虑只聚合 parameters |
| 学习率 | 论文 0.1；若震荡大可试 0.05 |
| batch size | 已对齐论文 128 |

### P3：q_and_group 修复（当前不建议主实验）

本地 CIFAR 上 `q_and_group` 因 `join_failed` 过高导致组碎片化。若后续要重试：

- 降低 `risk_threshold`（0.5 → 0.7）
- 降低 `slack_min_ratio`（0.25 → 0.15）
- 或改为仅对极高风险客户端过滤

**必须满足 Gate：** `dispatch_decision_trace` 与 q_only 至少有 1 处不同，且精度不系统性低于 q_only。

### P4：接入 loss-based statistical utility

在 q_only 系统收益确认后，将 `statistical_utility()` 从常数 1 改为基于 local loss 的归一化值，观察强 Non-IID 下是否带来额外精度收益。见 `后续优化方向.md` §3.2。

---

## 11. 结果回传

实验完成后，将以下目录打包回传本地：

```bash
# 云上打包
tar czf cifar10_main_results.tar.gz su_compass/output/cifar10_main/

# 本地解压合并
tar xzf cifar10_main_results.tar.gz -C /path/to/fedcompass/
```

本地合并分析：

```bash
python -m su_compass.experiments.analyze_cifar10_main \
  --output_root su_compass/output/cifar10_main
```

若本地已有 seed2026、云上有 2027/2028，确保目录结构一致后直接合并进同一 `cifar10_main/` 再分析。

---

## 12. Agent 执行摘要（可直接复制）

```bash
# === 第一阶段：GPU 校准（必做） ===
cd <项目根目录>
conda activate fedcompass
pip install -e .
python scripts/prepare_datasets.py --datasets cifar10 --source cn
# 运行 §4 校准脚本：
export BASE_STEP=<校准值>
export MODEL_MB=<校准值>

# === 第二阶段：正式实验（fedcompass + q_only, budget=1500 固定） ===
export BUDGET=1500

nohup python -m su_compass.experiments.run_cifar10_main \
  --budget "$BUDGET" \
  --seeds 2026 2027 2028 \
  --methods fedcompass q_only \
  --base_step_time "$BASE_STEP" \
  --model_size_mb "$MODEL_MB" \
  --output_root su_compass/output/cifar10_main \
  --no_progress \
  > su_compass/output/cifar10_main/run.log 2>&1 &

# === 分析 ===
python -m su_compass.experiments.analyze_cifar10_main \
  --output_root su_compass/output/cifar10_main
cat su_compass/output/cifar10_main/analysis/report.md
```

**固定标准：** `BUDGET=1500`，不做递增探测。  
**不要跑：** `q_and_group`、MNIST、loss utility。

---

## 13. 相关文档

- 实验阶段划分：`实验方案.md`
- 方法优化路线：`后续优化方向.md`
- 代码级修复说明：`主实验代码优化与安全实现.md`
- 本地同步脚本：`scripts/sync_to_gpufree.sh`
