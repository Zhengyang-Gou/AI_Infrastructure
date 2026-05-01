# PipeDream
## 基本做法
1. 把 DNN 模型划分成若干连续层组
2. 每个连续层组称为一个 stage
3. 每个 stage 分配给一个或多个 worker
4. 多个 minibatch 同时在 pipeline 中流动
5. 不同 worker 同时处理不同 minibatch 的 forward 或 backward

### PipeDream 为什么能减少通信
PipeDream 可以减少 inter-worker communication，因为它只需要在相邻 stage 之间传输 layer inputs / outputs

PipeDream 的通信有两个特点：
1. 只在相邻 stage 之间通信
2. 是 peer-to-peer communication，而不是 all-to-all communication

### 难点
pipelining 看起来简单，但 DNN training 有一个传统流水线没有的特殊问题——双向性

DNN 训练是：

Forward:
stage 1 -> stage 2 -> stage 3 -> stage 4

Backward:
stage 4 -> stage 3 -> stage 2 -> stage 1

backward 依赖 forward 产生的状态和中间结果，例如 activation，以及 forward 时使用的权重版本，所以 DNN pipeline 的难点不只是调度，还包括同一个 minibatch 的 backward 必须使用以及它 forward 时对应的中间状态和权重版本

## Pipeline Parallelism
PipeDream 的 pipeline parallelism，简称 PP，是把模型按连续层切成多个 stage，每个 stage 放到一个 GPU / worker 上。每个 stage 负责自己那一段层的 forward 和 backward。PipeDream 结合的是 intra-batch parallelism 和 inter-batch parallelism

### 为什么 Pipeline Parallelism 比数据并行更快
**Pipeline 通信更少**

数据并行：所有 worker 同步大梯度，通常 all-reduce / all-to-all 风格

PipeDream：相邻 stage 点对点传 activation / gradient

**Pipeline 能重叠计算和通信**
PipeDream 的每个 stage 在完成一个 minibatch 的 forward 后，可以异步把 activation 发给下一个 stage，同时自己开始处理另一个 minibatch

类似地，backward 完成后，它把 gradient 发给前一个 stage，同时开始下一个任务

### Work Partitioning
解决的问题：

- 模型应该怎么切成 stage？
- 每个 stage 应该分配几个 worker？
- 哪些 stage 需要复制？

#### partition 目标
1. balance work 让每个 stage 的处理时间尽量接近

2. minimize communication 尽量减少跨 stage 通信，并考虑硬件拓扑
PipeDream 的 partition 是 topology-aware 的：大的输出应该尽量走高带宽链路

#### Step1 - Profiler：先测每层的三个量
PipeDream 先做一个短 profiling run，记录每层的三个量：
- T_l: layer 的 forward + backward 总计算时间

- a_l: layer 的输出 activation 大小，backward 中对应 input gradient 大小也按这个估计

- w_l: layer 的参数大小
这三个量分别估计：
- 计算瓶颈
- 跨 stage activation / gradient 通信
- stage 复制后数据并行同步参数的成本

#### Step2 - stage replication
PipeDream 不要求每个 stage 只能放在一个 GPU 上，如果某个 stage 太慢，可以复制它，让多个 GPU 处理不同 minibatch

例如：

Worker 1 + Worker 2: Stage 1 的两个 replica

Worker 3:            Stage 2

这等价于对某个 stage 局部使用数据并行，这样做的目的是让 stage 吞吐率接近

#### Step3 - 动态规划 partition

