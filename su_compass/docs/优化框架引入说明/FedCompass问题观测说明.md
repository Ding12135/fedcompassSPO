# FedCompass 问题观测说明

## 1. 模块目的

本次新增的观测代码用于在**不改变 FedCompass 调度结果**的前提下，记录：

1. FedCompass 下发时使用历史平滑 `speed` 预测的下一轮完成时间；
2. 客户端下一轮的实际完成时间；
3. 预测误差与客户端迟到、端侧画像和不同慢因之间的关系；
4. arrival group 最终由全部客户端到达还是 deadline 触发。

该模块只用于发现和证明问题，不是新的 Q 分配或分组策略。

## 2. 代码位置

- `su_compass/diagnostics/fedcompass_problem.py`
  - 保存下发时的预测信息；
  - 与客户端下一轮实际返回配对；
  - 计算预测误差；
  - 生成按画像类型汇总的问题报告。
- `su_compass/experiments/run_virtual_fl.py`
  - 在 FedCompass 实验中创建观测器；
  - 把真实 dispatch decision 和 round report 交给观测器；
  - 不把观测结果传回 controller。
- `su_compass/virtual/trace.py`
  - 统一写出诊断 CSV 和 JSON。

## 3. 新增输出

### 3.1 `fedcompass_prediction_trace.csv`

每行表示一次“下发预测—下一轮实际完成”的配对。首轮客户端没有历史
`speed`，因此不纳入预测误差。

关键字段：

| 字段 | 含义 |
|---|---|
| `speed_smoothed_at_dispatch` | FedCompass 下发时真实使用的历史平滑单步耗时 |
| `predicted_duration` | `Q × speed_smoothed_at_dispatch` |
| `actual_duration` | 客户端下一轮实际总耗时 |
| `signed_prediction_error` | `actual - predicted`，正数表示 FedCompass 低估耗时 |
| `absolute_prediction_error` | 预测绝对误差 |
| `relative_prediction_error` | 绝对误差占实际耗时的比例 |
| `target_arrival_error` | 实际完成时间相对 group expected time 的偏差 |
| `deadline_margin` | `latest - actual_finish`，负数表示迟到 |
| `train_time` | 纯训练耗时 |
| `communication_time` | 下载与上传耗时之和 |
| `spike_delay` | 突发卡顿耗时 |
| `availability_wait` | 不可用等待耗时 |

### 3.2 `fedcompass_problem_report.json`

实验结束自动汇总：

- 总体 MAE、MAPE 和平均有符号误差；
- 低估超过实际耗时 10% 的次数；
- 迟到与按时记录的平均预测误差；
- 预测误差与迟到的 Pearson 相关系数；
- 每种 runtime profile 的独立统计；
- group deadline 触发比例。

## 4. 如何运行

继续使用原有 `fedcompass` 实验命令即可，不需要增加参数。只要：

```text
--algorithm fedcompass
```

实验输出目录中就会自动增加上述两个文件。`oort_compass` 不生成该基线问题
报告，避免把已受 Oort 干预的结果混入 FedCompass 原方法证据。

## 4.1 第一阶段状态预测Shadow输出

新增完成时间预测器后，同一目录还会生成：

```text
latency_prediction_shadow_trace.csv
latency_prediction_shadow_report.json
```

该实验固定使用FedCompass已经分配的真实Q，同时比较：

```text
FedCompass预测 = speed_smoothed × Q
状态分解预测 = compute_step_mean × Q
             + communication_mean
             + spike_mean
             + availability_wait_mean
```

状态在dispatch时冻结。本轮完成后的新状态只允许预测下一轮，禁止用于回头预测
本轮。`state_used_fallback=1`表示冷启动样本不足，状态预测器退化为FedCompass
预测；这类记录中两种预测值应完全一致。

### Availability事件自适应V1

第一轮针对效果较差的availability客户端增加事件型处理，其他耗时分量保持
上一版公式不变。RuntimeState新增：

```text
unavailable_event_rate
availability_wait_when_unavailable_mean
availability_wait_when_unavailable_std
unavailable_event_count
last_available
rounds_since_last_unavailable
```

当窗口内不可用事件不足2次时，不启用专门策略，完全沿用原分解预测。当事件
样本充分且不可用率低于50%时：

```text
普通点预测中的availability等待 = 0
安全预测中的availability风险
  = 条件等待均值 + 条件等待标准差
```

这样普通轮次不会因为少量不可用事件而每轮承担平均等待，但`safe_duration`仍会
保留不可用尾部风险。策略根据状态字段触发，不按`availability_limited`标签或
客户端ID硬编码。

新增Shadow字段：

```text
state_availability_risk_duration
state_availability_event_rate
state_availability_event_count
state_availability_strategy_active
```

V1实验重点检查：availability客户端MAE是否改善，以及未激活该策略的其他客户端
预测是否与上一版保持一致。

### Availability安全风险校准V2

V1普通点预测已经降低availability客户端MAE，但其安全风险直接使用：

```text
条件等待均值 + 条件等待标准差
```

这相当于假设下一轮一定不可用，造成safe duration误报偏多。V2仅校准风险：

```text
availability_risk
= unavailable_event_rate
  × (conditional_wait_mean + conditional_wait_std)
```

普通点预测、事件激活条件和其他耗时分量全部不变。新增公式同时表达不可用事件
发生的概率与发生后的等待代价，目标是在保持迟到召回的同时降低安全预测误报。

## 4.2 Shadow Q推荐

预测器通过后，实验会额外生成：

```text
q_recommendation_shadow_trace.csv
q_recommendation_shadow_report.json
```

Shadow Q策略枚举`Qmin`到`Qmax`的整数Q，优先选择：

```text
safe_finish(Q) <= latest_arrival_time
```

并在安全候选中选择`mean_finish(Q)`最接近`expected_arrival_time`的Q；偏差
相同时选择更大的Q。如果没有任何安全Q，则推荐Qmin并记录：

```text
no_safe_q_fallback_qmin
```

该阶段FedCompass仍执行原Q，Shadow推荐不会写入Dispatch或client_info。报告只
用于检查Q分布、边界命中、安全可行性和预测到达偏差，不能作为真实late、
deadline、TTA或accuracy收益。

## 4.3 StateCompass Q-only真实接入

`--algorithm state_compass`启用Q-only控制器。它严格先按FedCompass原公式选择
group，再只修改该group下的Q，因此不包含新分组策略：

```text
FedCompass选择group
        ↓
状态预测器枚举该group下合法Q
        ↓
存在安全Q：实际采用状态Q
不存在安全Q：保持FedCompass原Q，并记录group_mismatch
```

首组和新建group仍完整沿用FedCompass。实验额外输出`state_q_trace.csv`，记录：

```text
fedcompass_q / state_q / q_difference
state_safe_feasible / group_mismatch
expected/latest及推荐原因
```

无安全Q时不再强制Qmin，因为Qmin仍会迟到且损失本地训练量；该事件保留给后续
group策略处理。Q-only实验用于验证真实late、deadline、TTA、staleness与精度，
不能声称已经完成arrival group优化。

### Trust-Q V2稳定化

Q-only V1降低了late、deadline和staleness，但平均Q上升并导致TTA变慢。V2在
不修改预测器和分组的情况下增加两层保护：

```text
普通客户端：状态Q最多比FedCompass原Q增加20%
已有至少2次不可用事件：状态Q不得高于FedCompass原Q
无安全Q：保持FedCompass原Q并记录group_mismatch
```

V2采用非对称信任域：安全预测要求的减Q不做20%限制，因为将Q强行拉高可能重新
超过deadline；只限制导致聚合间隔变长的增Q。`state_q_trace.csv`新增：

```text
raw_state_q / state_q / max_increase_q
q_increase_clipped
availability_increase_guard
unavailable_event_count
applied_reason
```

V2目标是在保持late、deadline、staleness改善的同时，抑制平均Q上升并恢复TTA。

## 4.4 RCP-GS分组Shadow验证

RCP-GS（Risk-Constrained Pareto Group Scheduler）用于处理Q-only记录的
`group_mismatch`：当前group中即使采用Qmin，状态安全完成时间仍超过deadline。
第一阶段仅做旁路推荐，不实际修改客户端所属group。

候选筛选顺序如下：

```text
FedCompass原规则认可的已有group
        ↓
状态safe finish不超过原latest
        ↓
mean finish处于原group到达余量内
        ↓
候选Q不低于当前Trust-Q基线Q
        ↓
到达偏差不劣于基线，并至少一项严格改善
```

多个候选按“Q最大、到达偏差最小、expected更早”选择。无Pareto候选且基线安全时
保持原组；基线不安全且无安全已有组时，只建议沿用FedCompass原`create_group`
路径，不自定义新group时间。这样保留Q边界、arrival group生命周期、deadline、
general buffer和聚合机制。

新增两张表：

- `group_candidate_shadow_trace.csv`：逐候选约束结果和拒绝原因；
- `group_recommendation_shadow_trace.csv`：逐dispatch的保持/换组/create建议。

Shadow结果只能证明“存在可行且不减训练量的换组机会”。只有将相同策略真实接入
并进行多种子对比后，才能声称late、staleness、TTA或精度收益。

`latency_prediction_shadow_report.json`会按总体和画像类型给出两种方法的MAE、
MAPE、有符号误差、状态方法改善/持平/退化次数及冷启动次数。该报告只能用于
判断预测器是否更准确，不能直接证明尚未真正执行的状态Q策略有效。

## 5. 如何形成论文证据

如果多种子结果稳定出现以下现象，可以支持“历史 speed 对动态端侧状态适应
不足”的问题定义：

1. 动态画像的 MAE 明显高于稳定画像；
2. 迟到样本的 `signed_prediction_error` 明显为正；
3. `signed_error_late_correlation` 稳定为正；
4. `network_poor`、`compute_volatile` 或 `availability_limited` 的低估误差显著；
5. group deadline 触发比例较高。

论文表述必须保留边界：相关性和 deadline 比例可以证明问题现象存在，但不能
单独证明某个新 Q 或分组策略有效。新方法的有效性仍需后续 shadow 反事实实验和
实际接入实验验证。

## 6. 不变性要求

启用观测后，以下原有文件应与相同 seed、相同配置的观测前实验一致：

- `aggregation_trace.csv`
- `dispatch_decision_trace.csv`
- `group_trace.csv`
- `scheduler_trace.csv` 中的调度字段
- `round_reports.csv` 中的 `local_steps`

如果这些内容发生变化，说明观测代码意外影响了调度，实验结果不能用于论文。
