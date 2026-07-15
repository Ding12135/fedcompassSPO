# RUP-Compass 实验差异、解耦与模块插拔说明

本文说明当前 RUP-Compass 实验与此前 CIFAR 实验的区别，以及如何在不修改代码的情况下完成模块关闭、Shadow 验证、消融实验和失败回退。

## 1. 实验阶段划分

项目中的 CIFAR 结果来自三个不同阶段，不能直接混入同一张主结果表。

| 阶段 | 数据划分 | 预算 | 方法 | 主要用途 |
|---|---|---:|---|---|
| 历史 Oort-Compass | P0 Dual `(8,0.5)` | 600/1500 | FedCompass、q_only、q_and_group | 早期系统代价与分组探索 |
| StateCompass 长程实验 | P0 Dual `(8,0.5)` | 1200 | FedCompass、Trust-Q、Q+Group | 强 Non-IID 压力测试和长期稳定性诊断 |
| 当前正式实验 | P1 Dual `(8,2)` | 600 | FedCompass、RUP-Compass | 论文主设置和完整方法验证 |

### 1.1 P0 与 P1 的区别

P0 使用：

```yaml
alpha1: 8.0
alpha2: 0.5
dirichlet_mode: legacy_two_level
```

P0 同时具有很强的数据量异构与标签异构，适合压力测试，但 FedCompass 在该设置下无法达到 TTA50。

P1 使用：

```yaml
alpha1: 8.0
alpha2: 2.0
dirichlet_mode: legacy_two_level
```

对应配置：

```text
examples/config/client_cifar10_dual_moderate.yaml
```

P1 保留明显 Non-IID，同时避免统计异构完全掩盖动态端侧状态的调度影响。当前正确的 P1 FedCompass baseline 为：

```text
max accuracy：65.67%
last-10：60.83%
TTA50：239.88
late：83
deadline group：54/135
```

因此，后续主实验统一使用 P1；P0 1200-budget 结果只作为强异构和长程稳定性证据。

## 2. 旧方法与当前方法的结构差异

### 2.1 FedCompass baseline

FedCompass 使用历史平均速度分配 Q：

```text
历史 round_time / Q
→ 选择 arrival group
→ 根据剩余时间计算 Q
→ 本地 SGD
→ 原始 FedCompass 聚合
```

它没有显式区分计算、通信、突发延迟和 availability，也不使用客户端统计效用和近端约束。

### 2.2 历史 Oort-Compass

历史 q_only 主要用系统惩罚修正 Q 的计算分母：

```text
通信比例 + late率 + 波动 + availability
→ system penalty
→ effective step time
→ 修改Q
```

当时的 `statistical_utility()` 实际仍为常数 1，因此不是完整的统计效用方法。q_and_group 还修改了分组准入，后续 CIFAR 结果显示分组改动容易增加模型波动。

### 2.3 StateCompass Trust-Q

StateCompass 将客户端耗时拆分为：

```text
计算时间
下载时间
上传时间
突发延迟
availability等待
```

然后在已有 FedCompass group 内重新推荐 Q。P0 1200-budget 中，它显著降低 late 和 deadline，但带来：

```text
Q极端比例：20.5% → 48.7%
总本地步数：增加约7%
完成虚拟时间：增加约9.7%
```

这说明原始 Trust-Q 有收益，但还缺少工作量稳定和公平预算控制。

### 2.4 当前 RUP-Compass

RUP-Compass 在不修改聚合的前提下增加：

```text
多维状态预测
+ 历史预测残差风险校准
+ 双Trust Region
+ Q软边界
+ 有界统计Utility
+ 滚动训练预算守恒
+ FedProx本地稳定
```

完整流程：

```text
FedCompass选择原始已有group和参考Q
→ 状态预测构造安全Q集合
→ 残差分位数校准安全余量
→ Trust Region限制变化
→ 软边界减少不必要的40/200
→ Utility在±10%内微调
→ 预算控制器补偿或削减Q
→ FedProx执行本地训练
→ FedCompass原始分组和聚合
```

## 3. RUP-Compass 保持不变的部分

当前方法明确不修改：

```text
FedCompass已有group选择
FedCompass新建group公式
group expected/latest time
deadline timer
general buffer
group buffer
staleness函数
服务器聚合权重
BN buffer聚合策略
客户端更新上传格式
```

RUP调度只覆盖“加入已有 group”后的本地 Q。first group 和新建 group 使用 FedCompass 原始 Q，并在 `rup_decision_trace.csv` 中标记：

```text
fedcompass_group_creation_passthrough
```

这样可以把收益归因于工作量调度，而不是分组或聚合变化。

## 4. 模块边界

| 模块 | 控制内容 | 是否改变Q | 是否改变训练目标 | 是否改变分组/聚合 |
|---|---|:---:|:---:|:---:|
| State | 多维完成时间预测和安全Q集合 | 是 | 否 | 否 |
| Residual Risk | 历史预测残差分位数安全余量 | 是 | 否 | 否 |
| Trust Region | 相对FedCompass和上轮Q的变化范围 | 是 | 否 | 否 |
| Soft Boundary | 避免不必要的Q=40/200 | 是 | 否 | 否 |
| Utility | 根据可信统计价值微调Q | 是 | 否 | 否 |
| Budget | 控制滚动总训练量偏差 | 是 | 否 | 否 |
| Group Admission | 无安全Q时拒绝已有组，并调用FedCompass原始建组 | 否 | 否 | 是（仅准入选择） |
| FedProx | 增加本地近端目标 | 否 | 是 | 否 |

Group Admission 不重写建组公式，不改变组 deadline、general buffer 或聚合器。
它只修复“所有合法 Q 都无法安全赶上当前组”这一明确失配；安全候选保持原组。

模块实现位置：

```text
调度策略：su_compass/scheduling/policies/rup_workload.py
控制器：su_compass/virtual/algorithms/rup_compass.py
训练适配器：su_compass/virtual/training/rup_adapter.py
统一入口：su_compass/experiments/run_virtual_fl.py
CIFAR入口：su_compass/experiments/run_cifar10_rup.py
分析入口：su_compass/experiments/analyze_rup_compass.py
```

## 5. 三种总模式

### 5.1 Apply

```bash
--rup_mode apply
```

计算并真实执行 RUP Q；若 `--rup_group_admission apply`，对无安全 Q 的已有组
拒绝准入并沿用 FedCompass 原始建组；若 `--rup_prox on`，同时使用 FedProx。

### 5.2 Shadow

```bash
--rup_mode shadow
```

计算完整的状态、Trust、Utility和预算建议，但真实执行 FedCompass Q。Shadow 模式自动不应用 FedProx，确保训练轨迹仍是 baseline 训练目标。

用途：

- 检查推荐Q分布；
- 检查安全候选和风险余量；
- 检查Utility是否集中到少数客户端；
- 在不改变模型结果的情况下校准策略。

### 5.3 Off

```bash
--rup_mode off
```

调度退化为 FedCompass Q，FedProx也不会应用。仍可保留少量 RUP trace，用于验证关闭策略后的等价性。

## 6. 每层独立开关

```bash
--rup_state on|off
--rup_residual_risk on|off
--rup_trust on|off
--rup_soft_boundary on|off
--rup_utility on|off
--rup_budget on|off
--rup_group_admission off|shadow|apply
--rup_prox on|off
```

示例：只使用状态安全 Q：

```bash
python -m su_compass.experiments.run_virtual_fl \
  --algorithm rup_compass \
  --rup_mode apply \
  --rup_state on \
  --rup_residual_risk on \
  --rup_trust on \
  --rup_soft_boundary on \
  --rup_utility off \
  --rup_budget off \
  --rup_group_admission off \
  --rup_prox off \
  ...
```

只检查 FedProx：

```bash
--rup_state off \
--rup_residual_risk off \
--rup_trust off \
--rup_soft_boundary off \
--rup_utility off \
--rup_budget off \
--rup_prox on
```

此时已有 group 的 Q 保持 FedCompass 值，仅本地训练目标改变。

## 7. 预设消融实验

推荐使用：

```bash
python -m su_compass.experiments.run_cifar10_rup \
  --preset <名称> --budget 600 --seed 2026
```

| preset | State/Risk | Trust/Soft | Utility | Budget | Group | FedProx | 用途 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| `full` | ✓ | ✓ | ✓ | ✓ | Apply | ✓ | 最终完整方法 |
| `shadow` | 建议 | 建议 | 建议 | 建议 | Shadow |  | 不干预校准 |
| `off` |  |  |  |  |  |  | 退化等价检查 |
| `state_only` | ✓ | ✓ |  |  |  |  | 状态调度贡献 |
| `state_prox` | ✓ | ✓ |  |  |  | ✓ | 状态+本地稳定 |
| `state_utility` | ✓ | ✓ | ✓ | ✓ |  |  | 状态+资源价值 |
| `no_residual` | ✓但无残差校准 | ✓ | ✓ | ✓ | Apply | ✓ | 风险校准贡献 |
| `no_trust` | ✓ | 仅软边界 | ✓ | ✓ | Apply | ✓ | Trust贡献 |
| `no_soft_boundary` | ✓ | 仅Trust | ✓ | ✓ | Apply | ✓ | 极端Q控制贡献 |
| `no_utility` | ✓ | ✓ |  | ✓ | Apply | ✓ | Utility贡献 |
| `no_budget` | ✓ | ✓ | ✓ |  | Apply | ✓ | 预算公平贡献 |
| `no_group_admission` | ✓ | ✓ | ✓ | ✓ |  | ✓ | 分组准入贡献 |
| `no_prox` | ✓ | ✓ | ✓ | ✓ | Apply |  | FedProx贡献 |

## 8. 推荐实验顺序

### Stage 0：关闭等价性

```bash
python -m su_compass.experiments.run_cifar10_rup \
  --preset off --budget 50 --seed 2026
```

检查：

- dispatch Q与FedCompass一致；
- 分组和聚合路径一致；
- 无FedProx惩罚；
- 无NaN/Inf。

### Stage 1：Shadow校准

```bash
python -m su_compass.experiments.run_cifar10_rup \
  --preset shadow --budget 50 --seed 2026
```

检查：

- `recommended_q`与`applied_q`分开；
- `applied_q == baseline_q`；
- 分组 mismatch 只写建议，不改变真实 group；
- Utility范围在 `[0.9,1.1]`；
- no-safe-Q和fallback原因合理；
- 滚动预算比例没有持续偏离。

### Stage 2：Full冒烟

```bash
python -m su_compass.experiments.run_cifar10_rup \
  --preset full --budget 50 --seed 2026
```

检查：

- loss和prox penalty均为有限数；
- Q没有全部集中到40或200；
- 无安全Q拒绝事件均由FedCompass原始建组路径接管；
- 训练量没有快速偏离baseline；
- global accuracy能够正常上升。

### Stage 3：Full正式单种子

```bash
python -m su_compass.experiments.run_cifar10_rup \
  --preset full --budget 600 --seed 2026 --force
```

只有 Full 通过精度安全和系统收益门禁后，才进入消融。

### Stage 4：最小必要消融

优先运行：

```text
state_only
state_prox
state_utility
full
```

如果某项结果无法解释，再运行 `no_residual/no_trust/no_soft_boundary/no_budget` 细分机制。

## 9. 实验解耦原则

### 9.1 数据与训练底座冻结

所有正式对比必须保持：

```text
P1 Dual Dirichlet (8,2)
partition seed=42
runtime seed相同
ResNet-18
SGD lr=0.1
batch size=128
Q硬范围40–200
weighted BN
equal aggregation
相同client update budget
```

不能只给某个方法改变数据划分、学习率、BN或预算。

### 9.2 一次只解释一个差异

| 对比 | 能够归因的结论 |
|---|---|
| FedCompass vs state_only | 状态风险调度整体贡献 |
| state_only vs state_prox | FedProx边际贡献 |
| state_only vs state_utility | Utility+预算分配贡献 |
| state_utility vs Full | FedProx在完整调度上的贡献 |
| no_utility vs Full | Utility边际贡献 |
| no_budget vs Full | 预算守恒贡献及公平性代价 |
| no_trust vs Full | Q变化约束贡献 |
| no_soft_boundary vs Full | Q极端控制贡献 |

不能使用 `FedCompass vs Full` 单独声称每个子模块分别有效；该对比只能证明完整方法有效。

### 9.3 Shadow结果不能作为真实收益

Shadow只能用于证明：

- 推荐形态合理；
- 策略不会越过约束；
- 可能存在改进机会。

Shadow不能证明：

- late真实下降；
- deadline真实下降；
- TTA真实改善；
- accuracy真实提高。

这些必须由 Apply 实验给出。

## 10. Trace解读

### 10.1 `rup_decision_trace.csv`

一行对应一次派发决策。建议按以下顺序分析：

```text
baseline_q
→ raw_state_q
→ trust_q
→ soft_q
→ utility_q
→ budget_q
→ recommended_q
→ applied_q
```

如果某层前后 Q 相同，说明该层本次没有改变决策，不代表模块未启用。

重要字段：

| 字段 | 含义 |
|---|---|
| `num_safe_candidates` | 满足风险deadline的Q数量 |
| `residual_margin` | 历史预测残差产生的额外风险余量 |
| `trust_lower/upper` | 本次Trust Region边界 |
| `utility_normalized` | 最终Utility倍率，限制在0.9–1.1 |
| `utility_confidence` | Utility证据可信度 |
| `budget_ratio_before` | 调整前滚动训练量比例 |
| `budget_debt_before` | 相对FedCompass欠缺或超出的本地步数 |
| `budget_adjustment` | 本次预算控制增加或减少的Q |
| `fallback_reason` | 冷启动、无安全Q或其他回退原因 |
| `enabled_layers` | 当前实验实际启用的模块 |

### 10.2 `rup_training_trace.csv`

| 字段 | 用途 |
|---|---|
| `loss_before/after` | Utility和本地训练收益 |
| `loss_delta_per_step` | 判断高loss客户端是否仍能有效学习 |
| `prox_mu` | 本轮实际近端系数 |
| `mean_prox_penalty` | FedProx约束强度 |
| `mean_base_loss` | 基础分类loss |
| `finite` | 数值稳定性 |

### 10.3 `group_admission_trace.csv`

`group_admission_trace.csv` 是分组模块的因果事实表：

| 字段 | 含义 |
|---|---|
| `candidate_group_id` | FedCompass原本选择的已有组 |
| `group_mismatch` | 该组是否不存在任何安全Q |
| `shadow_action` | 风险Gate建议加入还是建组 |
| `applied_action` | 本次实际执行动作 |
| `actual_group_id` | 最终真实组，Apply拒绝后由原始建组回填 |
| `actual_dispatched_q` | 最终真实训练Q，不能与被拒绝候选Q混用 |

分组消融只比较 `full` 与 `no_group_admission`。其余参数、seed 和预算必须相同，
收益可归因于风险准入；不要用 Shadow 结果宣称真实 late、deadline 或精度收益。

### 10.4 原有Trace

```text
scheduler_trace.csv：late、完成时间、端侧状态
group_trace.csv：deadline、all_arrived、group规模
aggregation_trace.csv：staleness、参与客户端、实际Q
global_eval_trace.csv：accuracy、loss、TTA和AUC
```

## 11. 自动对比报告

```bash
python -m su_compass.experiments.analyze_rup_compass \
  --baseline_dir su_compass/output/cifar10_baseline_dual_moderate_budget600/seed2026 \
  --rup_dir su_compass/output/cifar10_rup/full/seed2026 \
  --output_dir su_compass/output/cifar10_rup/analysis
```

输出：

```text
report.json
report.md
```

报告同时包含：

- accuracy、last-10、波动和TTA；
- AUC、late、deadline和staleness；
- 总训练量和Q边界命中；
- 各层Q改变率；
- fallback分布；
- Utility、风险余量与FedProx观测。

## 12. 无好结果时的模块回退

不修改代码，按症状选择预设或开关。

| 症状 | 首先关闭/调整 | 解释 |
|---|---|---|
| 精度下降，late改善 | `no_utility` | 排查统计资源倾斜 |
| 精度仍下降 | `no_prox` | 排查近端系数过强 |
| 总训练量低于baseline超过5% | 保留Budget，收紧目标区间 | 排除少训练伪收益 |
| 总训练量高于baseline超过5% | `--rup_budget on` | 抑制额外训练量 |
| Q=200仍过多 | 保留soft boundary，降低soft_qmax | 减少高Q漂移 |
| Q=40过多、TTA恶化 | 降低风险分位数或`no_residual` | 排查风险过度保守 |
| late没有改善 | 提高残差分位数 | 增加完成时间保护 |
| Q变化过于剧烈 | 保留Trust并收紧上下界 | 稳定连续决策 |
| Utility长期只倾向一个客户端 | 收窄Utility到`[0.95,1.05]` | 防止资源集中 |
| 新组数或deadline比例上升 | `no_group_admission` | 回退FedCompass原分组并隔离准入影响 |
| Prox penalty远大于base loss | 降低`rup_prox_mu` | 防止本地训练受限过强 |

建议回退顺序：

```text
Full
→ no_utility
→ state_prox
→ state_only
```

只要 `state_only` 仍保持已有 late/deadline收益，论文的多维状态调度主结论就不会丢失；Utility、Budget和FedProx的收益可以通过消融决定是否进入最终主表。

## 13. 论文结果使用原则

主实验采用：

```text
FedCompass vs RUP-Compass Full
P1 Dual (8,2)
seeds 2026/2027/2028
budget 600
```

P0 1200-budget 结果作为：

- 强 Non-IID 压力测试；
- 原始 Trust-Q 长程收益证据；
- Q两极化和训练量偏移的动机证据。

论文中应分别报告：

1. 模型质量：max、last-10、std、最大回退；
2. 收敛效率：TTA50/55/60、Accuracy-Time AUC；
3. 系统可靠性：late、deadline、staleness；
4. 资源公平：总本地步数、客户端训练份额、Q分布；
5. 方法机制：安全候选、风险余量、Utility和Prox penalty。

完整方法有收益后再运行消融，避免在最终方法尚未通过前消耗大量实验时间。
