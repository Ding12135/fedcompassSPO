# RUP-Compass 可插拔实验说明

当前实验与历史实验的口径差异、完整解耦矩阵和结果归因规则，见
[`RUP-Compass实验差异与解耦插拔说明.md`](./RUP-Compass实验差异与解耦插拔说明.md)。

RUP-Compass 不修改 FedCompass 的建组公式、deadline、buffer 和聚合；仅新增一个
风险约束分组准入 Gate。Gate 拒绝无安全 Q 的已有组后，仍调用 FedCompass 原始
建组路径。全部新增层均由参数控制，关闭后无需改代码。

## 推荐首次运行

```bash
python -m su_compass.experiments.run_cifar10_rup \
  --preset full --budget 50 --seed 2026
```

冒烟通过后：

```bash
python -m su_compass.experiments.run_cifar10_rup \
  --preset full --budget 600 --seed 2026 --force
```

## 预设消融

| preset | 实际启用内容 |
|---|---|
| `full` | 状态、残差风险、Trust、软边界、Utility、预算守恒、风险分组准入、FedProx |
| `shadow` | 计算完整调度建议但执行 FedCompass Q；关闭 FedProx |
| `off` | 调度和训练均退化为 FedCompass 路径 |
| `state_only` | 状态风险 + Trust + 软边界 |
| `state_prox` | state_only + FedProx |
| `state_utility` | 状态 + Utility + 预算守恒，不使用 FedProx |
| `no_residual` | Full 去掉残差风险校准 |
| `no_trust` | Full 去掉双 Trust Region |
| `no_soft_boundary` | Full 去掉 Q 软边界 |
| `no_utility` | Full 去掉统计效用 |
| `no_budget` | Full 去掉滚动预算守恒 |
| `no_group_admission` | Full 去掉风险约束分组准入，恢复 FedCompass 原分组 |
| `no_prox` | Full 去掉 FedProx |

示例：

```bash
python -m su_compass.experiments.run_cifar10_rup --preset no_utility --budget 600
```

## 任意组合

底层入口支持逐层开关：

```bash
python -m su_compass.experiments.run_virtual_fl \
  --algorithm rup_compass \
  --server_config su_compass/config/virtual_fedcompass_cifar10_bn_partition_fix.yaml \
  --client_config examples/config/client_cifar10_dual_moderate.yaml \
  --num_clients 8 --num_global_epochs 600 --seed 2026 \
  --rup_state on \
  --rup_residual_risk on \
  --rup_trust on \
  --rup_soft_boundary on \
  --rup_utility off \
  --rup_budget on \
  --rup_group_admission apply \
  --rup_prox on \
  --rup_prox_mu 1e-4 \
  --output_dir su_compass/output/custom_rup
```

`--rup_group_admission off|shadow|apply` 可独立关闭、旁路观察或应用准入。
`--rup_mode shadow` 不应用 Q 或准入动作；同时设置 `--rup_mode off`、
`--rup_group_admission off`、`--rup_prox off` 可完全关闭。

## 新增观测表

`rup_decision_trace.csv` 每次派发记录：

- FedCompass、状态、Trust、软边界、Utility、预算层各自的 Q；
- 推荐 Q 与实际 Q；
- 预测/安全完成时间、残差风险、候选数；
- Utility 原值、归一化值、置信度、EMA loss 与进步率；
- 滚动预算比例、债务和本次补偿；
- 所有 fallback 原因和启用层。

`rup_training_trace.csv` 每轮客户端训练记录：

- 固定小批次上的训练前后 loss；
- 每步 loss 改善；
- FedProx 系数与平均近端惩罚；
- 本地样本数和数值有限性。

`group_admission_trace.csv` 每次已有组候选记录：

- 该组是否存在安全 Q，以及是否判定为 group mismatch；
- Shadow 建议与真实动作；
- Apply 拒绝后的 FedCompass 新组 ID 和实际派发 Q。

原有 `scheduler_trace.csv`、`aggregation_trace.csv`、`group_trace.csv`、
`global_eval_trace.csv` 保持不变，便于与 FedCompass 直接对比。

统一生成对比和机制报告：

```bash
python -m su_compass.experiments.analyze_rup_compass \
  --baseline_dir su_compass/output/cifar10_baseline_dual_moderate_budget600/seed2026 \
  --rup_dir su_compass/output/cifar10_rup/full/seed2026 \
  --output_dir su_compass/output/cifar10_rup/analysis
```

## 无收益时的无代码回退

- 精度下降但系统收益存在：先运行 `no_utility`，再运行 `no_prox`。
- 总训练量偏差超过 5%：保留 `rup_budget on`，收紧预算区间参数。
- Q 仍集中在 200：保留 `rup_soft_boundary on`，调低 `--rup_soft_qmax`。
- late 没改善：保留状态层，调整 `--rup_residual_quantile`。
- 分组数量或 deadline 增加：运行 `no_group_admission`，恢复原分组作为回退。
- 任一层疑似有害：对应 `--rup_* off`，不修改实现。
