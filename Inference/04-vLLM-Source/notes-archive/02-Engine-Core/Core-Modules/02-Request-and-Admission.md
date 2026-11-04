# 02 · Request 与请求接收

**这一篇看入口提交的 EngineCoreRequest，如何变成核心持续保存的 Request，并进入等待队列。**

## 1. 职责与边界

请求跨入核心时，需要从一份提交数据，变成可以反复更新的执行状态：

```text
EngineCoreRequest
    → EngineCore.preprocess_add_request
    → Request.from_engine_core_request：创建 Request
    → EngineCore.add_request
    → Scheduler.add_request：记录请求并入队
```

入队之后，请求等待 Scheduler 安排。它何时被选中、分到多少 token 预算，由 [Scheduler 章节](../../03-Scheduler/) 分析。

## 2. 源码定位与对象区别

| 文件 | 类 | 关注内容 |
| --- | --- | --- |
| [vllm/v1/engine/core.py](../../../vllm/vllm/v1/engine/core.py) | EngineCore | preprocess_add_request、add_request |
| [vllm/v1/request.py](../../../vllm/vllm/v1/request.py) | Request、RequestStatus | from_engine_core_request、初始化状态与 token 记录 |
| [vllm/v1/core/sched/scheduler.py](../../../vllm/vllm/v1/core/sched/scheduler.py) | Scheduler | add_request、_enqueue_waiting_request |

| 对象 | 主要用途 | 在哪里保存 |
| --- | --- | --- |
| EngineCoreRequest | 携带入口提交的输入、参数与请求信息 | 在客户端与核心之间交接 |
| Request | 记录核心执行进度、已生成 token 与生命周期状态 | Scheduler 的请求表及相应队列 |
| RequestState | 记录解码进度、输出方式等用户结果上下文 | 入口 OutputProcessor |

核心的 Request 和入口的 RequestState 通过同一个内部请求 ID 关联。核心继续使用入口分配好的内部 ID，这里没有重新生成另一套编号。

## 3. 实现分析

### 3.1 preprocess_add_request 创建核心 Request

普通文本路径到达的关键代码是：

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：preprocess_add_request
# 源码范围：L983–L991（连续节选）
req = Request.from_engine_core_request(request, self.request_block_hasher)
if req.use_structured_output:
    # Note on thread safety: no race condition.
    # `grammar_init` is only invoked in input processing thread. For
    # `structured_output_manager`, each request is independent and
    # grammar compilation is async. Scheduler always checks grammar
    # compilation status before scheduling request.
    self.structured_output_manager.grammar_init(req)
return req, request.current_wave
```

`Request.from_engine_core_request(...)` 会调用 Request 构造函数，将提交数据中的 ID、输入、参数等传过去，再初始化执行状态。这个步骤复用已经准备好的输入，普通文本路径在这里不再分词。

方法返回 `(req, request_wave)`。`req` 是要加入调度器的 Request；`request_wave` 用于数据并行的请求轮次协调，单核心主线只需知道它会一起传给 `add_request()`。

结构化输出请求还会初始化 grammar，多模态输入会在该方法前面的分支处理接收侧缓存。它们属于输入接收时的附加处理。

### 3.2 Request 为跨轮执行准备状态

新的普通请求先进入等待状态：

```python
# 文件：vllm/v1/request.py
# 类：Request
# 函数：__init__
# 源码范围：L97–L99（连续节选）
self.status = RequestStatus.WAITING
self.events: list[EngineCoreEvent] = []
self.stop_reason: int | str | None = None
```

输入长度、已生成部分和完整 token 历史也在构造时准备：

```python
# 文件：vllm/v1/request.py
# 类：Request
# 函数：__init__
# 源码范围：L140–L148（连续节选）
self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
    prompt_token_ids, prompt_embeds
)
self._output_token_ids: list[int] = []
self._all_token_ids: list[int] = (
    self.prompt_token_ids.copy()
    if self.prompt_token_ids is not None
    else [0] * self.num_prompt_tokens
)
```

以 4 个输入 token 的请求为例，初始状态可以这样理解：

| 字段 | 初始内容与作用 |
| --- | --- |
| prompt_token_ids | 4 个输入 token 的 ID |
| num_prompt_tokens | 4，输入长度 |
| output_token_ids | 空序列，尚未生成输出 |
| num_computed_tokens | 0，调度侧执行进度的起点 |
| status | WAITING，等待安排 |
| max_tokens | 本请求允许生成的最大输出 token 数 |

后续生成的新 token 会保存在 Request 中，下一轮可以继续使用。`num_computed_tokens` 表示调度侧推进的进度，调度时就可能增加；异步执行时它可能包含在途计算，不能直接当成已经全部完成的 GPU 计算量。

### 3.3 EngineCore 将 Request 交给 Scheduler

`EngineCore.add_request()` 检查请求 ID 类型、任务支持情况及传输配置等信息后，进入下面这一行：

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：add_request
# 源码范围：L479–L479（连续节选）
self.scheduler.add_request(request)
```

注意这两个方法同名，但对象不同：外层 `EngineCore.add_request()` 组织核心接收，里面的 `Scheduler.add_request()` 才负责登记到调度器。

新的非流式输入请求，会进入 Scheduler 的新请求分支：

```python
# 文件：vllm/v1/core/sched/scheduler.py
# 类：Scheduler
# 函数：add_request
# 源码范围：L2230–L2235（连续节选）
self._enqueue_waiting_request(request)
self.requests[request.request_id] = request
if self.connector is not None:
    self.connector.on_new_request(request)
if self.log_stats:
    request.record_event(EngineCoreEventType.QUEUED)
```

其中两个动作最重要：

- `_enqueue_waiting_request(request)`：放入等待队列；普通 `WAITING` 请求进入 `waiting`，需要额外依赖的请求有相应等待队列。
- `self.requests[request.request_id] = request`：按内部 ID 保存 Request，执行结果返回时可以找到同一个对象。

队列与请求表承担不同作用：队列组织调度顺序，请求表支持按 ID 查找。两处引用的是同一个 Request 对象。

### 3.4 入队后怎样继续

```text
WAITING：等待调度
    ↓ Scheduler 接纳并安排
RUNNING：进入运行集合，经历一轮或多轮执行
    ├─ 继续：更新进度，参加后续调度
    ├─ 被抢占：进入 PREEMPTED 等待恢复
    └─ 结束：进入相应 FINISHED 状态
```

`RUNNING` 请求可能仍在处理 Prompt 分块，也可能正在 Decode。生命周期状态和 Prefill / Decode 计算阶段是两个不同维度。

## 4. 用一条请求走到等待队列

假设入口提交内部 ID 为 `0-ab12cd34` 的请求，输入有 4 个 token，最多生成 2 个 token：

```text
EngineCoreRequest：携带输入、生成参数、内部 ID
    ↓ 创建 Request
保存输入，输出序列为空，status=WAITING
    ↓ 加入 Scheduler
waiting 中有这条请求
requests["0-ab12cd34"] 指向它
    ↓
等待下一次 schedule() 决定执行内容
```

此时核心已经接收请求；模型计算由执行循环安排。

## 5. 阅读检查

- 为什么核心需要 Request，而入口还需要 RequestState？
- 请求表与等待队列各自解决什么问题？
- 请求进入 RUNNING，是否意味着已经进入 Decode？

[章节目录](../README.md) · [上一篇：EngineCore 与初始化](01-EngineCore-and-Initialization.md) · [下一篇：Step 与执行闭环](03-Step-and-Execution-Loop.md)
