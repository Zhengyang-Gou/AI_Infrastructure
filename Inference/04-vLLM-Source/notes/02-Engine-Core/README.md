# 02 · Engine Core

**本章只回答：核心怎样接收请求、组织调用并返回结果？**

## 1. 职责边界

| 项目 | 契约 |
| --- | --- |
| 输入 | EngineCoreRequest、取消 ID 与 VllmConfig |
| 输出 | EngineCoreOutputs；交给 Scheduler 的 Request 或结束指令 |
| 本章负责 | 核心组件初始化顺序、请求接收与转交、同步 step 的调用顺序、客户端选择与进程驱动、取消消息处理 |
| 到此为止 | 调用 Scheduler、Executor 和初始化接口；不进入它们的算法与状态实现 |

| 内容 | 唯一归属 |
| --- | --- |
| Request 字段、入队、预算、核心判停、请求退场 | 03 Scheduler |
| 缓存容量计算、块池与实际回收 | 04 KV Cache Management |
| 模型、权重、KV 张量和基础预热 | 05 Model and Weight Loading |
| Executor/Worker 内部、输入张量与前向 | 06 Model Execution |
| batch queue、多批在途、异步占位与推测路径 | 09 Runtime Optimizations |
| 多设备 Worker 拓扑与跨 rank 协调 | 10 Distributed Execution |

核心的单进程/后台进程驱动属于本章；模型怎样分布到多个设备属于 10。

## 2. 篇目边界

| 篇目 | 负责到哪里 |
| --- | --- |
| [01 · EngineCore 与初始化](Core-Modules/01-EngineCore-and-Initialization.md) | 组件创建和依赖传入的顺序；不推导资源计算 |
| [02 · Request 与请求接收](Core-Modules/02-Request-and-Admission.md) | 收到 EngineCoreRequest，调用转换工厂，再交给 Scheduler；到 add_request 接口为止 |
| [03 · Step 与执行闭环](Core-Modules/03-Step-and-Execution-Loop.md) | schedule → execute/sample → update 的数据与调用顺序 |
| [04 · 核心客户端与进程循环](Core-Modules/04-Core-Client-and-Process-Loop.md) | 客户端选择、消息队列和谁调用 step_fn |
| [05 · 完成与取消的核心交接](Core-Modules/05-Completion-and-Abort.md) | 返回完成信息、转交取消 ID、处理取消队列；不展开判停与资源清理 |

## 3. 阅读方式与验收

先读 [Overview](Overview.md)，再按 01 → 05 阅读。固定配置见 [总目录 §1](../README.md#1-固定运行配置)。

- 能解释核心为什么先初始化执行资源，再创建 Scheduler。
- 能区分核心接收、Scheduler 接纳与实际执行。
- 能将一份 SchedulerOutput 与对应 ModelRunnerOutput 交回同一次更新。
- 能解释同进程取输出与后台循环驱动的差别。
- 能指出取消何时交给 Scheduler，完成信息何时返回客户端。

[源码纵览](../00-Overview.md) · [下一章：Scheduler](../03-Scheduler/README.md)
