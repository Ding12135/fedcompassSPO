# Risk-Constrained Group Admission 说明

该模块只处理Trust-Q已经确认的 `group_mismatch`：FedCompass先按原规则选择已有
arrival group，Trust-Q再枚举该组内的安全Q。存在安全Q时，无论模式为何都正常
加入并使用Trust-Q；不存在安全Q时，Shadow模式保持原行为但记录建议，Apply模式
拒绝加入，由父类 `_assign_group()` 调用FedCompass原始 `_create_group()`。

StateCompass 的通用策略实现位于 `su_compass/scheduling/policies/group_admission.py`，
接入点位于 `su_compass/virtual/algorithms/state_compass.py`。RUP-Compass 使用同一准入
语义，直接消费 RUP 已产生的 `state_safe_feasible`，接入点位于
`su_compass/virtual/algorithms/rup_compass.py`。两者均不修改Q上下界、arrival group
生命周期、deadline、general buffer或聚合逻辑。

运行开关：

```bash
--algorithm state_compass --group_admission_mode shadow
--algorithm state_compass --group_admission_mode apply
--algorithm rup_compass --rup_group_admission off
--algorithm rup_compass --rup_group_admission shadow
--algorithm rup_compass --rup_group_admission apply
```

RUP 的总模式具有最高优先级：`--rup_mode shadow` 即使搭配分组 `apply` 也只观察，
`--rup_mode off` 不产生分组干预。正式 `full` 默认使用分组 `apply`，可通过
`--preset no_group_admission` 无代码回退。

独立输出 `group_admission_trace.csv`。其中 `candidate_trust_q` 是已有组候选Q，
`shadow_action` 表示风险策略建议，`applied_action` 表示本次真实动作。Apply拒绝时，
父类原始建组完成后会回填 `actual_group_id` 和 `actual_dispatched_q`；二者可与
`dispatch_decision_trace.csv` 的 `create_group` 行核对，避免把被拒绝候选Q误当成
真实训练Q。

Shadow不变性应通过同配置、同种子分别运行改动前Trust-Q与当前Shadow，比较
`group_trace.csv`、`dispatch_decision_trace.csv`。Apply阶段则检查所有
`group_mismatch=1` 行均满足 `admitted=0`，且对应客户端在同一虚拟时刻存在
`decision=create_group` 的派发记录。预测改善仅说明调度信号更准确，不能直接描述
为真实训练收益；TTA、精度等必须由真实接入实验单独报告。
