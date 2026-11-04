# 00 · vLLM 源码纵览

本文只串联模块与交接，不展开源码、算法和逐轮推演。固定配置见 [README](README.md#1-固定运行配置)。

## 1. 启动：先准备可复用的引擎

```text
01 入口：接收配置，创建引擎
    ↓
02 核心：组织组件初始化
    ├─ 调用 05：创建模型、加载权重、准备执行侧缓存并预热
    └─ 创建 03：让调度器取得缓存配置，准备接收请求
```

05 负责实际准备资源；04 负责缓存布局、容量规则和运行期间的块管理。02 只解释初始化调用的顺序。

## 2. 运行：一条请求怎样得到结果

```text
用户 Prompt
    ↓
01 入口：分词、校验、登记输出状态
    │ EngineCoreRequest
    ↓
02 核心：接收请求，组织每轮执行
    ├─ 调用 03 调度：决定谁执行、算多少 token
    │      └─ 调用 04 KV 管理：查询前缀、申请或释放块
    │ SchedulerOutput
    ↓
06 执行：准备张量，运行模型
    ├─ 调用 07 Attention：执行注意力并读写 KV
    └─ 调用 08 Sampling：把 logits 变成输出 token
    │ ModelRunnerOutput
    ↓
02 核心：把执行结果交给 03 更新请求，取得核心增量
    │ EngineCoreOutputs
    ↓
01 入口：解码、组织 RequestOutput，交给用户
```

未结束的请求继续下一轮。02 组织调用，03 决定请求是否继续。

## 3. 三个扩展作用在哪里

| 模块 | 在基础主线上增加什么 |
| --- | --- |
| [09 · Runtime Optimizations](09-Runtime-Optimizations/README.md) | 编译与图执行、异步调度、批次重叠、推测解码 |
| [10 · Distributed Execution](10-Distributed-Execution/README.md) | 将模型与执行任务分布到多个设备和进程 |
| [11 · Online Serving](11-Online-Serving/README.md) | HTTP 请求、异步调用、流式交付与连接生命周期 |

## 4. 去哪里读

| 模块                                                                     | 唯一的主问题                        |
| ---------------------------------------------------------------------- | ----------------------------- |
| [01 · Inference Entry](01-Inference-Entry/README.md)                   | 用户输入怎样成为请求，核心增量怎样成为用户结果？      |
| [02 · Engine Core](02-Engine-Core/README.md)                           | 请求与结果怎样跨过核心边界，核心怎样组织调用？       |
| [03 · Scheduler](03-Scheduler/README.md)                               | 谁执行、算多少、何时等待、抢占或结束？           |
| [04 · KV Cache Management](04-KV-Cache-Management/README.md)           | KV 块怎样分配、复用、淘汰与回收？            |
| [05 · Model and Weight Loading](05-Model-and-Weight-Loading/README.md) | 模型、权重和执行资源怎样就绪？               |
| [06 · Model Execution](06-Model-Execution/README.md)                   | 执行计划怎样变成张量与一次模型执行？            |
| [07 · Attention Backend](07-Attention-Backend/README.md)               | 注意力后端怎样选择，怎样读写 KV 并调用 kernel？ |
| [08 · Sampling](08-Sampling/README.md)                                 | logits 怎样变成采样 token 和概率信息？    |

各章 README 定义具体边界。正文遇到其他模块，只说明接口的输入、输出与本模块的调用位置；算法和状态演变留在唯一负责的章节。
