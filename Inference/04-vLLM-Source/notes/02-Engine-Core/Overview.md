# Engine Core Overview

**EngineCore 负责调用顺序和数据交接。** 本章把 Scheduler 与 Executor 视为已经给出接口契约的组件。

## 1. 输入、输出与持有的对象

```text
EngineCoreClient
    │ 请求与取消 ID
    ↓
EngineCore
    ├─ scheduler：接收 Request、产生计划、消费模型结果
    ├─ model_executor：接收计划、返回模型结果
    └─ structured_output_manager：提供结构化输出依赖
    │ EngineCoreOutputs
    ↑ 返回客户端
```

| 对象 | 在本章的角色 |
| --- | --- |
| EngineCoreRequest | 外部交来的提交数据 |
| Request | 核心调用转换工厂得到、交给 Scheduler 管理的对象 |
| SchedulerOutput | Scheduler 返回、交给执行器的计划 |
| ModelRunnerOutput | 执行器返回、交给 Scheduler 的结果 |
| EngineCoreOutputs | 更新接口返回、核心交给客户端的输出 |

本章不展开这些对象在其他组件内部的字段变化。

## 2. 初始化与请求接收

初始化依次组织：创建 Executor → 调用缓存初始化接口 → 将缓存配置交给 Scheduler → 准备核心驱动。资源怎样算、怎样分配由对应实现负责。

接收请求时，核心调用 Request.from_engine_core_request，将得到的 Request 传给 Scheduler.add_request。到这个接口为止；请求表、队列和后续生命周期属于调度器。

## 3. 同步一轮的固定交接

```text
检查 Scheduler 是否有工作
    → schedule()：取得计划 P
    → execute_model(P)：取得 Future
    → Future.result()；需要时 sample_tokens()：取得结果 R
    → 处理执行期间到达的取消
    → update_from_output(P, R)：取得核心输出 O
    → 返回 O 与本轮是否执行模型的标记
```

核心必须把同一轮的 P 与 R 配对交给更新接口。至于谁被选中、分到多少 token、怎样判停和生成核心输出，均由 Scheduler 负责。

## 4. 谁驱动这个调用序列

| 路径 | 请求怎样送达 | 谁推进核心 | 输出怎样返回 |
| --- | --- | --- | --- |
| 同进程 InprocClient | 直接调用预处理与接收接口 | get_output 调用 step_fn，再调用 post_step | 直接返回客户端索引 0 的输出 |
| 后台进程 SyncMPClient | 发送 ADD，输入线程预处理后交给 input_queue | EngineCoreProc.run_busy_loop | output_queue → 发送线程 → 客户端 outputs_queue |

step_fn 表示配置选中的核心步骤。固定主线使用同步 step；多批在途的实现归 09。

## 5. 完成与取消到哪里为止

- 正常完成：Scheduler 在更新结果中提供结束信息，核心将输出返回客户端。
- 外部取消：核心接收内部 ID，将结束指令交给 Scheduler.finish_requests。
- 执行期间收到取消：先处理 aborts_queue，再将模型结果交给更新接口。

核心不执行文本解码，不决定结束条件，也不直接操作块池。

[章节目录](README.md) · [开始阅读 Core 01](Core-Modules/01-EngineCore-and-Initialization.md)
