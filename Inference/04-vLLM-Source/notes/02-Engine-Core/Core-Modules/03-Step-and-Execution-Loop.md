# 03 · Step 与执行闭环

**这一篇是本章中心：EngineCore 怎样组织一轮调度、执行和结果更新。** 本篇只展开普通同步 `step()`，到调度与执行接口为止。

## 1. 职责与边界

```text
Scheduler 保存的请求
    ↓ schedule
SchedulerOutput：本轮执行计划
    ↓ Executor 执行与采样
ModelRunnerOutput：模型返回的结果
    ↓ update_from_output
更新 Request，构造 EngineCoreOutputs
    ↓
未完成请求留待下一轮，增量结果返回入口
```

EngineCore 组织这几步的顺序与数据传递。请求选择策略、缓存块分配算法和模型输入张量准备，分别在对应组件章节深入。

## 2. 源码定位与数据交接

| 文件 | 类 | 关注内容 |
| --- | --- | --- |
| [vllm/v1/engine/core.py](../../../vllm/vllm/v1/engine/core.py) | EngineCore | step、post_step；多批在途实现归 09 |
| [vllm/v1/core/sched/scheduler.py](../../../vllm/vllm/v1/core/sched/scheduler.py) | Scheduler | schedule、update_from_output |
| [vllm/v1/executor/abstract.py](../../../vllm/vllm/v1/executor/abstract.py) | Executor | execute_model、sample_tokens |
| [vllm/v1/core/sched/output.py](../../../vllm/vllm/v1/core/sched/output.py) | SchedulerOutput | 选中的请求、逐请求计算量、缓存块信息、结束请求通知 |
| [vllm/v1/outputs.py](../../../vllm/vllm/v1/outputs.py) | ModelRunnerOutput | 请求 ID 映射、sampled_token_ids 等 |
| [vllm/v1/engine/__init__.py](../../../vllm/vllm/v1/engine/__init__.py) | EngineCoreOutput、EngineCoreOutputs | 单请求增量与返回入口的一批结果 |

| 数据 | 应怎样理解 |
| --- | --- |
| SchedulerOutput | 执行计划，说明谁参与、每条请求算多少以及相关缓存信息 |
| ModelRunnerOutput | 执行侧结果，包含各请求本轮采样出的 token 等信息 |
| EngineCoreOutput | 一条请求准备交回入口的增量与结束信息 |
| EngineCoreOutputs | 多条核心输出的容器，还可携带统计等信息 |

## 3. 实现分析

### 3.1 检查核心是否还有工作，取得本轮计划

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：step
# 源码范围：L591–L595（连续节选）
# Check for any requests remaining in the scheduler - unfinished,
# or finished and not yet removed from the batch.
if not self.scheduler.has_requests():
    return {}, False
scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
```

`has_requests()` 判断调度器是否还有工作，包括未完成请求和结束后的必要收尾。没有工作时，返回空字典和 `False`。

`schedule()` 返回 `SchedulerOutput`，其中的 `num_scheduled_tokens` 记录各请求本轮参与计算的 token 数，`total_num_scheduled_tokens` 是总和。计划中还带有缓存块及新旧请求信息，供执行侧组织输入。

基础 EngineCore 的 `_should_throttle_prefills()` 返回 `False`；数据并行实现可以覆盖它。本篇沿普通单核心路径阅读。

### 3.2 执行模型，取得可用于更新的结果

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：step
# 源码范围：L596–L604（连续节选）
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
with (
    self.capture_iteration_details(scheduler_output) as iteration_details,
    self.log_error_detail(scheduler_output),
):
    model_output = future.result()
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)
```

这几行的执行顺序是：

1. `execute_model(..., non_block=True)` 下发执行，返回 Future，即用于等待并取得执行结果的对象。
2. 准备结构化输出可能需要的 grammar 信息，普通文本路径先掌握参数传递的位置。
3. `future.result()` 等待并取回这次执行的返回值，也会传播执行异常。
4. 返回值为 `None` 时，继续调用 `sample_tokens()`，得到 `ModelRunnerOutput`。

这里的 `None` 是继续进入采样调用的分支条件，不能据此认为请求已经没有后续工作。具体执行器如何拆分模型前向和采样，由 [Model Execution](../../06-Model-Execution/) 分析。

`non_block=True` 让下发接口先返回 Future；普通 `step()` 随后仍会调用 `.result()`。这个参数本身不能说明整轮已经实现多批次重叠。

### 3.3 先处理取消，再更新请求与输出

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：step
# 源码范围：L606–L614（连续节选）
# Before processing the model output, process any aborts that happened
# during the model execution.
self._process_aborts_queue()
engine_core_outputs = self.scheduler.update_from_output(
    scheduler_output, model_output
)
self._attach_iteration_details(engine_core_outputs, iteration_details)

return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

在模型执行期间，输入线程可能已经收到取消。核心先处理这些取消，再把本轮模型结果交给 Scheduler，避免继续将被取消请求当成正常活跃请求。

`update_from_output()` 结合计划和结果：计划说明本轮安排了哪些请求、多少计算，模型结果提供每条请求产生的 token 等信息。调用返回按客户端组织的核心输出。Request 如何更新、判停和退场，均留在 Scheduler 内部。

返回值分为两部分：

| 返回内容 | 用途 |
| --- | --- |
| 按 client_index 组织的 EngineCoreOutputs 字典 | 将结果交回对应客户端；单客户端主线通常取索引 0 |
| total_num_scheduled_tokens > 0 | 标记本轮是否安排了 token 计算，交给驱动层及 post_step 使用 |

因此，`EngineCore.step()` 的返回类型与入口 `LLMEngine.step()` 返回的用户结果列表不同。入口还需要解码和组织 `RequestOutput`。

### 3.4 谁开始下一轮，post_step 做什么

`step()` 组织一轮工作。驱动它反复运行的是客户端调用或后台核心循环，见 [核心客户端与进程循环](04-Core-Client-and-Process-Loop.md)。

调用方在取得结果后调用 `post_step()`。固定的无推测生成主线不进入其扩展处理；优化路径的具体工作归 09。本篇只确认驱动方会在 step 之后调用这个钩子。

## 4. 只检查交接，不复演调度

| 顺序 | 核心持有的数据 | 核心动作 |
| --- | --- | --- |
| 1 | 计划 P | 传给 execute_model |
| 2 | 对应结果 R | 等待并取得；需要时继续 sample_tokens |
| 3 | 同一轮 P 与 R | 传给 update_from_output |
| 4 | 核心输出 O | 交还驱动方 |

本篇不计算 P 内的 token 数，不修改 Request 进度，不解释采样算法。多批次在途的计划/Future 配对归 09。

## 5. 阅读检查

- 能否沿 schedule → execute_model / sample_tokens → update_from_output 解释每次数据交接？
- 为什么 Executor 需要执行计划，Scheduler 更新时还需要同一份计划？
- 更新接口返回的字典与入口返回的用户结果列表有什么区别？
- 谁调用 step，谁取得并转发它的返回值？

[章节目录](../README.md) · [上一篇：Request 与请求接收](02-Request-and-Admission.md) · [下一篇：核心客户端与进程循环](04-Core-Client-and-Process-Loop.md)
