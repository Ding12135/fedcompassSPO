# 端侧模型接入 FedCompass 说明

本文档说明如何把当前已经验证完成的端侧运行模型接入 FedCompass 半异步联邦学习框架，用于在单张 RTX 4090 上开展 8 个异构客户端的虚拟时间实验。

核心目标：

- 使用一张 RTX 4090 完成 8 个逻辑客户端的真实本地训练。
- 使用端侧模型生成每个客户端的虚拟下载、训练、上传、卡顿和不可用等待时间。
- 使用虚拟时间代替真实 wall-clock 等待，让 FedCompass server 按虚拟完成时间调度。
- 保留 FedCompass 半异步思想：客户端不全同步等待，也不是单个客户端到达就立即无条件更新，而是按 arrival group 进行分组聚合。
- 在不真实等待慢客户端的情况下，快速开展异构客户端实验。

## 1. 当前已有基础

当前已经完成并验证的端侧模块包括：

```text
su_compass/runtime/
├── profile.py          # 五类端侧画像参数
├── virtual_runtime.py  # 虚拟运行时间模型
├── state.py            # ClientRoundReport 与 RuntimeStateTracker
└── validate_profiles.py # 8 客户端画像验证脚本
```

已有验证输出：

```text
su_compass/output/runtime_profile_8clients/
├── stable_fast/
├── stable_slow/
├── compute_volatile/
├── network_poor/
├── availability_limited/
└── summary/
```

当前五类客户端已经形成清晰状态签名：

| 画像类型 | 客户端 | 主要特征 |
|---------|--------|----------|
| `stable_fast` | `client_0`, `client_1` | 快且稳定 |
| `stable_slow` | `client_2` | 计算慢但稳定 |
| `compute_volatile` | `client_3`, `client_4` | 平均速度中等但波动大 |
| `network_poor` | `client_5`, `client_6` | 通信占比高，尤其上传慢 |
| `availability_limited` | `client_7` | 可用性等待明显 |

接入前验证结论见：

- [`端侧模型验证.md`](../端侧模型验证/端侧模型验证.md)
- [`客户端画像参数.md`](../端侧模型/客户端画像参数.md)
- [`端侧模型.md`](../端侧模型/端侧模型.md)

## 2. 接入后的实验语义

接入后的实验不是 8 张真实设备并行训练，而是：

```text
一张 RTX 4090
  -> 顺序/分批完成 8 个逻辑客户端的真实训练
  -> 每个客户端训练得到真实 local_model
  -> 端侧模型生成该客户端的虚拟 round_time
  -> server 按 virtual_finish_time 处理客户端上传事件
  -> FedCompass 用虚拟时间估计速度、分组和聚合
```

也就是说：

- **模型更新是真的**：每个客户端都会用真实数据、本地模型和指定 local steps 训练。
- **端侧耗时是虚拟的**：下载、上传、卡顿、不可用等待由 `VirtualRuntimeModel` 生成。
- **客户端并行是逻辑并行**：虽然训练计算在一张 GPU 上执行，但 server 看到的是 8 个客户端按虚拟完成时间异步到达。
- **半异步调度是真的按 FedCompass 语义执行**：server 按虚拟 arrival group 处理聚合，不是简单同步，也不是简单串行。

推荐论文表述：

```text
We run model training on a single RTX 4090 GPU and emulate heterogeneous client arrivals with an event-driven virtual-time runtime model. Each logical client performs real local training, while its download, upload, runtime variability, and availability delays are generated from predefined runtime profiles. The FedCompass scheduler observes virtual dispatch and finish times to perform semi-asynchronous grouping and local-step assignment.
```

## 3. 为什么使用虚拟时间

当前只有一张 RTX 4090，如果用真实等待或真实多进程并行，会有三个问题：

1. 8 个客户端同时抢一张 GPU，真实训练时间会被 CUDA 上下文切换和显存竞争污染。
2. 如果用 `sleep` 模拟慢客户端，`network_poor` 或 `availability_limited` 会显著拉长实验 wall-clock。
3. 真实等待不利于快速调参和重复实验。

因此当前更适合采用虚拟时间：

| 方案 | 优点 | 缺点 | 当前是否推荐 |
|------|------|------|-------------|
| 真实 sleep | 最贴近真实 wall-clock 半异步 | 实验很慢 | 暂不推荐 |
| MPI/gRPC 多进程真实并行 | 接近真实部署 | 单 GPU 抢占严重，异构不可控 | 暂不推荐 |
| 事件驱动虚拟时间 | 快、可复现、异构可控 | 需要单独实现虚拟调度 runner | 推荐 |

## 4. 总体接入路线

接入应分三阶段进行。

### 4.1 阶段一：虚拟时间 FedCompass Runner

目标：先不改原始 `CompassScheduler` 的真实 `time.time()` 和 `threading.Timer` 逻辑，而是新增一个实验 runner，复现 FedCompass 的核心半异步调度逻辑。

推荐新增位置：

```text
su_compass/experiments/run_virtual_fedcompass.py
```

或：

```text
examples/virtual/run_virtual_fedcompass.py
```

该 runner 负责：

1. 创建 8 个逻辑客户端。
2. 给每个客户端绑定一个端侧画像。
3. 调用真实 trainer 完成本地训练。
4. 调用 `VirtualRuntimeModel.simulate_round()` 生成虚拟完成时间。
5. 将客户端上传事件放入虚拟事件队列。
6. server 按 `virtual_finish_time` 取出事件。
7. 使用 FedCompass 风格逻辑更新速度、分配 local steps、创建或加入 arrival group。
8. 触发单客户端或 group aggregation。
9. 记录 runtime state、scheduler trace 和 group trace。

### 4.2 阶段二：观测对齐原始 FedCompass

目标：把 `ClientRoundReport -> RuntimeStateTracker` 接到原始 `CompassScheduler` 中，但只做观测，不改变调度。

建议接入位置：

```text
src/appfl/scheduler/compass_scheduler.py
```

关键位置：

```text
_record_info()
```

这一阶段只记录：

```text
client_update_time
client_steps
client_speed
state_snapshot
```

不改：

```text
_assign_group()
_join_group()
_create_group()
_group_update()
FedCompassAggregator
```

### 4.3 阶段三：reason-aware 调度扩展

目标：在原始 FedCompass speed-aware 调度基础上，引入慢因状态。

可探索策略：

| 慢因 | 可能策略 |
|------|----------|
| 计算慢 | 继续使用 FedCompass 的 Q 调整 |
| 通信慢 | 避免简单增加 Q，防止上传瓶颈进一步拖慢 |
| 波动大 | 降低进入紧 deadline group 的概率 |
| 可用性差 | 单独分组或延后调度 |

第三阶段不应立即开始。应先完成阶段一和阶段二，确保基线实验稳定。

## 5. 虚拟时间 Runner 的核心设计

### 5.1 核心对象

虚拟时间实验 runner 建议维护以下状态：

```text
VirtualFedCompassRunner
├── clients
│   ├── client_id
│   ├── profile
│   ├── local_steps
│   ├── virtual_available_time
│   ├── last_global_timestamp
│   └── local_model
├── event_queue
│   └── 按 virtual_finish_time 排序
├── runtime_trackers
│   └── 每个客户端一个 RuntimeStateTracker
├── scheduler_state
│   ├── client_info
│   ├── arrival_group
│   ├── group_buffer
│   └── global_timestamp
└── aggregator
```

### 5.2 单轮客户端执行流程

每个逻辑客户端的一轮参与流程：

```text
1. server 给 client_i 下发 global_model 和 local_steps
2. client_i 在 RTX 4090 上真实训练 local_steps 步
3. client_i 生成 local_model/update
4. VirtualRuntimeModel 根据 client_i 的 profile 生成虚拟 round result
5. 得到 dispatch_time、finish_time、train_time、download_time、upload_time、spike_delay、availability_wait
6. 构造 ClientRoundReport
7. 将 (finish_time, client_id, local_model, report) 放入 event_queue
```

### 5.3 Server 处理事件流程

server 侧不按真实训练完成顺序处理，而按虚拟时间处理：

```text
while not training_finished:
    event = event_queue.pop_min_finish_time()
    virtual_now = event.finish_time
    client_id = event.client_id
    local_model = event.local_model
    report = event.report

    update RuntimeStateTracker(report)
    update client speed = report.round_time / report.local_steps
    run FedCompass-style schedule logic
    assign next local_steps
    dispatch next round for this client if needed
```

关键点：

- 真实训练顺序可以是串行的。
- server 处理顺序必须由 `virtual_finish_time` 决定。
- `virtual_now` 替代原始 FedCompass 里的 `time.time() - self.start_time`。

## 6. FedCompass 半异步逻辑如何保留

FedCompass 的核心不是物理并行，而是调度语义：

```text
快客户端先到达
server 记录速度
server 为客户端创建或加入 arrival group
同组客户端接近同一目标时间到达
到齐或超过 latest time 后触发 group aggregation
```

虚拟时间 runner 应保留以下逻辑：

| FedCompass 逻辑 | 虚拟时间实现 |
|----------------|--------------|
| `client_speed = update_time / local_steps` | `client_speed = virtual_round_time / local_steps` |
| 当前时间 `curr_time` | 当前事件的 `virtual_finish_time` |
| `expected_arrival_time` | 虚拟目标到达时间 |
| `latest_arrival_time` | 虚拟最晚到达时间 |
| `threading.Timer` | 事件队列中检查 group deadline |
| 到达 group | 根据虚拟时间判断 |
| late update | `finish_time > latest_arrival_time` |

不建议在虚拟时间 runner 中使用 `threading.Timer`，因为它依赖真实 wall-clock，会破坏虚拟时间语义。

## 7. 8 客户端画像映射

默认使用当前已经验证过的 8 客户端映射：

| 客户端 | 画像 |
|--------|------|
| `client_0` | `stable_fast` |
| `client_1` | `stable_fast` |
| `client_2` | `stable_slow` |
| `client_3` | `compute_volatile` |
| `client_4` | `compute_volatile` |
| `client_5` | `network_poor` |
| `client_6` | `network_poor` |
| `client_7` | `availability_limited` |

如果 FedCompass 主代码中客户端 ID 是数字，需要统一映射：

```text
0 -> client_0
1 -> client_1
...
7 -> client_7
```

## 8. 输出文件设计

虚拟时间 FedCompass 实验建议输出：

```text
su_compass/output/virtual_fedcompass_mnist8/
├── client_states/
│   ├── client_0/
│   │   ├── round_reports.csv
│   │   ├── state_trace.csv
│   │   └── client_summary.json
│   ├── ...
│   └── client_7/
├── summary/
│   ├── summary_table.csv
│   ├── all_round_reports.csv
│   ├── all_state_traces.csv
│   ├── runtime_overview_bar.png
│   ├── speed_stability_scatter.png
│   └── bottleneck_scatter.png
├── scheduler_trace.csv
├── group_trace.csv
├── training_metrics.csv
└── experiment_config.json
```

### 8.1 `scheduler_trace.csv`

用于分析 FedCompass 调度行为。

建议字段：

```text
virtual_time
round_idx
client_id
profile_type
assigned_group
local_steps
step_time
speed_estimate
round_time
communication_ratio
availability_rate
late
staleness
global_timestamp
```

### 8.2 `group_trace.csv`

用于分析 arrival group 是否按预期工作。

建议字段：

```text
group_id
created_time
expected_arrival_time
latest_arrival_time
client_ids
arrived_client_ids
aggregation_time
group_size
late_clients
max_arrival_gap
```

### 8.3 `training_metrics.csv`

用于分析模型训练效果。

建议字段：

```text
global_timestamp
virtual_time
num_global_updates
num_client_updates
train_loss
validation_loss
validation_accuracy
```

## 9. 分阶段验证方法

### 9.1 验证一：端侧画像保持一致

目标：确认接入 FedCompass runner 后，端侧画像仍与独立验证一致。

检查：

- `stable_fast` 仍然最快且稳定。
- `network_poor` 仍然通信占比最高。
- `availability_limited` 仍然不可用等待最高。
- `compute_volatile` 仍然 `step_time_cv` 较高。

通过标准：

- 输出的 `summary_table.csv` 与 `runtime_profile_8clients` 中的相对关系一致。
- 最大快慢倍数仍约为 `stable_fast` 到 `network_poor` 的 8-10 倍范围。

### 9.2 验证二：虚拟时间顺序正确

目标：确认 server 按虚拟完成时间处理上传事件。

检查：

- `scheduler_trace.csv` 中 `virtual_time` 单调递增。
- 每个事件的处理顺序与 `finish_time` 排序一致。
- 真实训练完成顺序不影响 server 处理顺序。

通过标准：

- 没有出现后完成的虚拟事件先被 server 处理。
- 同一客户端下一轮 `dispatch_time` 不早于上一轮 `finish_time`。

### 9.3 验证三：FedCompass 分组逻辑正确

目标：确认半异步 arrival group 正常工作。

检查：

- 快客户端可以更早创建 group。
- 慢客户端根据速度加入合适 group 或创建新 group。
- 同一 group 内客户端的虚拟到达时间接近。
- 超过 `latest_arrival_time` 的客户端被标记为 late。

通过标准：

- `group_trace.csv` 中 group 的 `expected_arrival_time`、`latest_arrival_time` 和实际 arrival 能对应。
- `local_steps` 在 `[Qmin, Qmax]` 内。
- 快客户端平均 `local_steps` 高于慢客户端。

### 9.4 验证四：训练指标正常

目标：确认虚拟时间接入没有破坏联邦训练。

检查：

- loss 能下降。
- accuracy 能上升。
- global update 数量合理。
- staleness 没有异常爆炸。

通过标准：

- 同样配置下，FedCompass baseline 和虚拟时间版本的模型训练趋势合理。
- 不要求每次曲线完全相同，但不能出现明显训练崩溃。

## 10. 推荐实验顺序

为了又稳又快地开始实验，建议按以下顺序：

### 第一步：小规模 smoke

配置：

```text
dataset = MNIST
num_clients = 8
global_updates = 20 或 50
Qmin = 40
Qmax = 200
profiles = 8 客户端默认映射
```

目标：

- runner 能跑通。
- 输出文件完整。
- 虚拟时间、画像状态和 local_steps 正常。

### 第二步：短程对比实验

运行：

```text
FedAvg + virtual profiles
FedAsync + virtual profiles
FedCompass + virtual profiles
```

目标：

- FedAvg 被 `network_poor` 和 `availability_limited` 拖慢。
- FedAsync 更新快但 staleness 更明显。
- FedCompass 能通过 group 和 Q 分配减轻 straggler 影响。

### 第三步：完整 MNIST 8 客户端实验

配置尽量对齐论文：

```text
Qmin = 40
Qmax = 200
latest_time_factor = 1.2
local_lr = 0.003
num_clients = 8
```

目标：

- 得到主实验曲线。
- 输出 wall-clock equivalent virtual time。
- 生成 runtime state 和 scheduler/group trace。

### 第四步：扩展到 CIFAR10

在 MNIST 跑通后再扩展到 CIFAR10，避免一开始被训练成本和模型复杂度拖慢。

## 11. 与原始 FedCompass 的关系

当前接入方案不是推翻 FedCompass，而是把端侧画像作为可控异构来源接入其半异步思想。

| 原始 FedCompass | 当前接入方案 |
|----------------|--------------|
| 依赖真实 `time.time()` 记录客户端到达 | 使用虚拟 `dispatch_time/finish_time` |
| 真实客户端快慢由环境决定 | 五类端侧画像控制快慢和慢因 |
| 主要记录 `speed = update_time / steps` | 额外记录训练、通信、卡顿、可用性状态 |
| 使用真实 `threading.Timer` 等待 group deadline | 使用事件队列模拟 group deadline |
| 只知道客户端慢 | 能解释为什么慢 |

需要在论文中明确：

- 这是 FedCompass 的虚拟时间实验实现。
- 模型训练真实执行。
- 异构到达由端侧画像模拟。
- 调度器观察虚拟时间完成事件。

## 12. 风险与注意事项

### 12.1 不要混用真实时间和虚拟时间

同一个实验中，调度时间应统一来自虚拟时间。不要一部分逻辑用 `time.time()`，另一部分用 `virtual_finish_time`。

尤其注意：

```text
client_update_time
client_start_time
expected_arrival_time
latest_arrival_time
late 判断
staleness 统计
```

这些都应使用虚拟时间。

### 12.2 不要让真实训练顺序决定 server 到达顺序

即使真实训练是串行的，server 也必须按事件队列中的 `virtual_finish_time` 处理。

错误做法：

```text
for client in clients:
    train client
    server immediately handles update
```

正确做法：

```text
for client in ready_clients:
    train client
    push event to queue

server pop event by minimum virtual_finish_time
server handles update
```

### 12.3 第一版不直接改 reason-aware 调度

第一版只需要跑出：

- 原始 FedCompass 风格调度
- 端侧状态记录
- 五类异构画像影响

不要马上让通信占比、可用性等状态影响 Q 分配。否则很难判断问题来自端侧接入还是新调度策略。

## 13. 最小可行版本

最小可行版本只需要做到：

1. 8 个逻辑客户端能真实训练。
2. 每个客户端绑定一个 `ClientRuntimeProfile`。
3. 每轮训练后调用 `VirtualRuntimeModel.simulate_round()`。
4. 事件队列按 `finish_time` 排序。
5. server 用 `finish_time` 作为当前时间。
6. `speed = round_time / local_steps`。
7. local steps 始终在 `[Qmin, Qmax]` 内。
8. 输出 `round_reports.csv`、`state_trace.csv`、`scheduler_trace.csv`、`group_trace.csv`。

完成这 8 点，就可以开始做第一组 FedCompass 异构半异步实验。

## 14. 最终实验目标

接入完成后，实验应能回答以下问题：

1. 在单 GPU 资源下，是否能模拟 8 个异构客户端的半异步联邦学习？
2. FedCompass 是否能根据虚拟速度为不同客户端分配不同 local steps？
3. 五类画像是否在真实联邦训练中仍表现出稳定差异？
4. 网络慢、计算慢、波动大和可用性差是否会导致不同的调度行为？
5. 原始 FedCompass speed-aware 调度是否会把通信慢误认为计算慢？
6. 后续 reason-aware 调度是否有改进空间？

当前推荐结论：

```text
先实现单 GPU 真实训练 + 虚拟时间事件队列 + FedCompass 半异步分组。
先记录端侧状态，不立即改变调度策略。
先跑 MNIST 8 客户端 smoke，再跑完整 MNIST，最后扩展到 CIFAR10。
```
