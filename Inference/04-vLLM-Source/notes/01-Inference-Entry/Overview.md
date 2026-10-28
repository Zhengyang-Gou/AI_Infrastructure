# Inference Entry Overview

本文说明入口模块的组件框架、执行路线与数据变化。先建立整体认识，再沿 [Core-Modules](Core-Modules/) 中的六篇笔记阅读具体函数与状态实现。

## 1. 整体职责与阅读场景

入口连接用户调用与引擎核心：将 prompts 和生成参数准备成可提交的请求，持续处理核心输出，再返回用户结果。

本章以本地源码的 decoder-only 普通文本、离线 `LLM.generate`、`n=1` 为主线。引擎初始化先建立组件与执行资源；随后一次次生成调用复用引擎，创建各自的请求状态。调度、KV Cache 分配与模型前向由核心及下游模块负责。

## 2. 组件框架：对象怎样组织

```text
LLM：用户持有的入口对象
├─ 继承 OfflineInferenceMixin：复用离线流程编排方法
└─ 持有 LLMEngine：组织请求与结果的核心交接
   ├─ renderer：具体 Renderer 实例，公共基类是 BaseRenderer
   ├─ InputProcessor：请求校验与构造
   ├─ OutputProcessor：请求输出状态、解码与结果构造
   └─ EngineCoreClient：提交请求、取输出、取消
          ↓↑
      Engine Core：调度、缓存与模型执行
```

`LLMEngine` 创建 renderer 等组件，`LLM` 复用其中的 renderer 与输入处理器引用；同一组件可以被多处引用。Mixin 方法运行在同一个 `LLM` 实例上，组件则是被引用和调用的对象。

| 组件 | 在入口主线中的职责 | 对应 Core |
| --- | --- | --- |
| LLM | 汇集初始化配置，提供公共接口 | 01 |
| OfflineInferenceMixin | 对齐批量参数，提交并收集请求 | 02 |
| Renderer / BaseRenderer | 解析后的输入渲染、分词与 EngineInput 构造 | 03 |
| InputProcessor | 校验并补全执行请求 | 04 |
| LLMEngine | 衔接 processor、输出状态和核心客户端 | 05 |
| EngineCoreClient | 隔离核心执行与通信方式 | 05；内部继续到下一章 |
| OutputProcessor / RequestState | 跨轮累积输出、解码、判停、组织结果 | 06 |

## 3. 执行路线：先初始化，再提交和收集

### 3.1 初始化路线

```text
用户创建 LLM
    → 汇集 EngineArgs
    → 引擎工厂创建 VllmConfig、选择 Executor
    → 创建 LLMEngine
        → 创建 renderer、InputProcessor、OutputProcessor
        → 创建 EngineCoreClient，准备核心执行体系
    → LLM 保存配置与组件引用，初始化计数器并预热 renderer
```

入口配置的处理见 [01 · LLM 与初始化](Core-Modules/01-LLM-and-Initialization.md)。模型与 GPU 资源的准备会继续进入核心和执行器；renderer 预热是输入处理侧的工作。

### 3.2 请求提交路线

```text
LLM.generate
    → OfflineInferenceMixin._run_completion
        → _add_completion_requests
            → 对齐 prompts 与逐请求参数
            → 逐条预处理，调用 renderer 得到 EngineInput
            → 分配外部 ID，设置 FINAL_ONLY
            → LLMEngine.add_request
                → InputProcessor.process_inputs → EngineCoreRequest
                → 分配内部 ID，保留外部 ID
                → OutputProcessor.add_request → 登记 RequestState
                → EngineCoreClient.add_request → 核心
```

提交阶段通过生成器逐条渲染、逐条提交；后台核心可能在入口提交整批请求期间就开始执行。输出状态在提交核心之前建立，为后续增量结果准备好上下文。

### 3.3 结果收集路线

```text
OfflineInferenceMixin._run_engine
    → 仍有未完成请求时，重复 LLMEngine.step
        → EngineCoreClient.get_output → EngineCoreOutputs
        → OutputProcessor.process_outputs
            → 用内部 ID 找 RequestState
            → 更新 Token、文本与结束信息
            → 按输出模式构造 RequestOutput
            → 完成清理，返回需要核心取消的请求（如有）
        → EngineCoreClient.abort_requests（如有）
        → 将用户结果交给收集循环
    → 收集 finished 结果，按外部 ID 恢复输入顺序
```

同步 `generate` 最终返回完整结果，内部仍持续处理增量。`FINAL_ONLY` 限制用户对象的交付时机，不会跳过中间 Token 的处理。

`LLMEngine.step` 是入口的一轮交互。在同进程模式下，客户端取输出会推进核心；多进程模式下则取得后台核心输出。一轮交互不与单个请求的一次前向或一个输出 Token 严格对应。

## 4. 数据路线：每次转换增加什么

```text
Prompt
  → 解析、渲染与分词 → EngineInput
  → 结合参数与请求信息 → EngineCoreRequest
  → 核心调度和执行 → EngineCoreOutput（多轮增量）
  → 结合 RequestState 解码与组织 → RequestOutput
```

| 数据表示 | 核心含义 | 负责转换的组件 |
| --- | --- | --- |
| Prompt | 用户提供的文本、Token IDs 等输入形式 | 离线编排与输入解析 |
| EngineInput | 统一的模型输入内容，普通文本路径为 TokensInput | Renderer |
| EngineCoreRequest | 加入经过验证的生成参数、ID、到达时间等执行信息 | InputProcessor，经 LLMEngine 继续提交 |
| EngineCoreOutput | 一条请求本轮新增的 Token 与状态 | Engine Core 及下游 |
| RequestOutput | 面向用户的输入信息、候选结果、完成状态等 | OutputProcessor 与 RequestState |

`EngineCoreOutputs` 是多条 `EngineCoreOutput` 的容器。`RequestOutput.outputs` 中的 `CompletionOutput` 是一个候选；它与外层请求结果属于不同层次。

入口分配的数字字符串是外部 ID。内部编号后，核心和输出状态表用内部 ID 关联请求；输出处理再将外部 ID 写回 `RequestOutput`。离线结果因此能够按输入顺序排序。

## 5. 进入具体实现的阅读路线

| 顺序 | 本站关注的问题 | 具体实现 |
| --- | --- | --- |
| 01 | 配置什么时候创建？调用怎样委托？ | [LLM 与初始化](Core-Modules/01-LLM-and-Initialization.md) |
| 02 | 一批输入怎样对齐、逐条提交、统一收集？ | [离线推理流程编排](Core-Modules/02-Offline-Inference-Flow.md) |
| 03 | 输入在哪里解析、分词，哪些字段被保留？ | [Renderer 与输入表示](Core-Modules/03-Renderer-and-EngineInput.md) |
| 04 | 参数和长度在哪里校验？请求 ID 怎样变化？ | [InputProcessor 与请求构造](Core-Modules/04-InputProcessor-and-EngineCoreRequest.md) |
| 05 | 提交与状态注册怎样衔接？客户端怎样隔离执行模式？ | [LLMEngine 与核心边界](Core-Modules/05-LLMEngine-and-Core-Boundary.md) |
| 06 | 增量怎样还原成文本与用户结果？何时清理和反馈取消？ | [输出处理与文本还原](Core-Modules/06-OutputProcessor-and-Detokenization.md) |

每站先识别输入、输出和持续保存的状态，再阅读对应函数。函数开头的文件、类、名称及行范围可以用于回到本地源码继续追踪。

## 6. 用同一条请求回看全过程

假设离线入口收到一条普通文本 Prompt，生成一个候选：

| 阶段 | 发生什么 | 此时主要保存什么 |
| --- | --- | --- |
| 输入准备 | 文本转换为 Token IDs，建立 EngineInput | 输入内容与到达时间 |
| 请求构造 | 校验、克隆和补全参数，建立核心请求 | 外部与内部 ID、输入与执行配置 |
| 提交核心 | 先登记 RequestState，再提交 | 输出模式、解码器、ID 对应关系 |
| 多轮返回 | 核心增量按内部 ID 找到请求状态 | 已生成 Token、累计文本与停止检查状态 |
| 请求完成 | 构造最终对象，恢复外部 ID，清理状态 | 完成的 RequestOutput |

如果输出侧先检测到停止字符串，还会通知核心取消；如果提交中途失败，编排层会取消已成功添加的请求。入口状态与核心执行资源通过这些反馈共同收尾。

到这里，入口侧主线已经闭合。下一章从 [Engine Core](../02-Engine-Core/) 继续追踪请求如何进入调度器、执行计划怎样下发，以及输出如何产生。

[返回章节目录](README.md) · [开始阅读 Core 01](Core-Modules/01-LLM-and-Initialization.md)
