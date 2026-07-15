# GP-BC-RUP 最终实验方案

## 冻结方法

最终方法保留 FedCompass 原始分组，只优化已有组内的本地步数 Q。状态预测、残差风险、
trust region 和统计效用共同产生无约束 Q；最后一层累计预算 guard 保证相对 FedCompass
的累计训练量债务不超过 1%。高效用且有安全余量的客户端优先偿还债务。

固定 preset：`gp_bc_rup`。禁止对 seed 2026 的结果做参数回调后再宣称主结果。

## 单次主实验

```bash
python -m su_compass.experiments.run_cifar10_rup \
  --preset gp_bc_rup \
  --budget 600 \
  --seed 2026 \
  --output_root su_compass/output/cifar10_gp_bc_budget600
```

分析：

```bash
python -m su_compass.experiments.analyze_rup_compass \
  --baseline_dir su_compass/output/cifar10_baseline_dual_moderate_budget600/seed2026 \
  --rup_dir su_compass/output/cifar10_gp_bc_budget600/gp_bc_rup/seed2026 \
  --output_dir su_compass/output/rup_analysis_gp_bc/seed2026
```

## 预注册成功条件

- final accuracy >= 64.5，max accuracy >= 65.0；
- last-10 标准差不高于 FedCompass；
- total local steps 比值位于 [0.99, 1.02]；
- late rate 与 deadline rate 均低于 FedCompass；
- TTA@62 或 TTA@64 至少一项优于 FedCompass。

## 已记录的诊断数据

`rup_decision_trace.csv` 记录每次决策的 baseline、无约束和最终 Q，安全候选上下界，
trust region，预测 deadline slack，统计效用及置信度，残差风险边界，滑动窗口预算，
累计 baseline/applied workload、work ratio、work debt、debt limit、guard 和偿债触发，
以及 baseline/applied Q 是否属于风险安全集合。

`rup_training_trace.csv` 记录 loss 前后值、每步 loss 改善、样本数、prox penalty 和数值
稳定性。`scheduler_trace.csv`、`group_trace.csv` 与 `aggregation_trace.csv` 分别用于分析
迟到、deadline、group size、staleness 和实际聚合训练量。

`rup_outcome_trace.csv` 使用 dispatch ID 将预测时延、安全时延与实际 round duration、
预测误差、安全边界越界和 deadline miss 逐任务闭环。`rup_terminal_state.json` 区分
dispatched、completed、aggregated workload，并记录结束时在途任务和已完成未聚合任务。
分析报告还输出逐客户端 workload、late rate、staleness、utility、loss progress、CV 和
Gini，检查偿债是否集中于少数 non-IID 客户端。

## 若未通过，按证据只允许一次定向修正

- work ratio < 0.99：检查 guard 是否覆盖 passthrough，并把 debt ratio 从 0.01 收紧到 0；
- work ratio 合格但 final accuracy 低：检查 Q 增量是否集中在低 progress 客户端，将偿债阈值从
  utility >= 1.0 提高到 1.05；
- late/deadline 变差且 guard 频繁：训练量约束与安全候选冲突，保持 1% debt，不得增加 boost；
- TTA 不占优但系统指标与精度通过：将论文主张定为系统稳定性，不再针对单 seed 调参；
- last-10 波动大：检查 staleness 和每客户端 Q 增量，优先限制偿债单次增幅，不启用旧 smoothness。

开发 seed 通过后，仅用完全相同配置补 seed 2027、2028 做统计验证。
