# 05 · 完成、取消与清理

**这一篇看请求如何退出核心执行：正常生成结束，或者由外部发来取消。** 核心需要更新状态、移出调度队列、处理资源引用，并与入口输出清理衔接。

## 1. 职责与边界

```text
正常完成：模型结果 → 更新 Request → 命中结束条件 → 清理 → 返回结束信息

取消：入口或输出侧发起 → 核心客户端 → EngineCore.abort_requests
                                     → Scheduler.finish_requests → 清理
```

核心与入口分别保存 Request 和 RequestState。结束一条请求，需要让核心停止安排它，并让入口停止维护它的输出状态。

## 2. 源码定位与结束信息

| 文件 | 类或作用域 | 关注内容 |
| --- | --- | --- |
| [vllm/v1/core/sched/scheduler.py](../../../vllm/vllm/v1/core/sched/scheduler.py) | Scheduler | _update_request_with_output、finish_requests、_free_request、_free_blocks |
| [vllm/v1/core/sched/utils.py](../../../vllm/vllm/v1/core/sched/utils.py) | 模块级函数 | check_stop |
| [vllm/v1/engine/core.py](../../../vllm/vllm/v1/engine/core.py) | EngineCore、EngineCoreProc | abort_requests、_process_aborts_queue、ABORT 消息接收 |
| [vllm/v1/engine/llm_engine.py](../../../vllm/vllm/v1/engine/llm_engine.py) | LLMEngine | abort_request、step |
| [vllm/v1/engine/output_processor.py](../../../vllm/vllm/v1/engine/output_processor.py) | OutputProcessor | 停止字符串反馈与入口状态清理 |

RequestStatus 描述核心生命周期，EngineCoreOutput 的 `finish_reason` 则把结束原因带回入口。常见对应关系如下：

| 核心状态 | 含义 | 返回原因 |
| --- | --- | --- |
| FINISHED_STOPPED | 命中 EOS 或停止 token | STOP |
| FINISHED_LENGTH_CAPPED | 达到输出或总长度限制 | LENGTH |
| FINISHED_ABORTED | 收到核心取消 | ABORT |
| FINISHED_ERROR | 执行相关错误导致结束 | ERROR |

状态枚举也包含其他结束分支。本篇沿普通生成和取消阅读，重点关注状态怎样进入清理路线。

## 3. 实现分析

### 3.1 正常完成：追加新 token 后检查结束条件

`Scheduler.update_from_output()` 会将新 token 交给下面的方法处理：

```python
# 文件：vllm/v1/core/sched/scheduler.py
# 类：Scheduler
# 函数：_update_request_with_output
# 源码范围：L2101–L2111（连续节选）
stopped = False
for num_new, output_token_id in enumerate(new_token_ids, 1):
    request.append_output_token_ids(output_token_id)

    # Check for stop and update request state.
    # This must be called before we make the EngineCoreOutput.
    stopped = check_stop(request, self.max_model_len)
    if stopped:
        del new_token_ids[num_new:]  # Trim new tokens if needed.
        break
return new_token_ids, stopped
```

每追加一个输出 token，便检查它是否触发结束。若一轮有多个新 token，命中结束条件后会截去后面多余的 token，再构造核心输出。

`check_stop()` 中普通生成相关的核心判断是：

```python
# 文件：vllm/v1/core/sched/utils.py
# 类：无（模块级函数）
# 函数：check_stop
# 源码范围：L100–L117（连续节选）
if request.num_output_tokens < sampling_params.min_tokens:
    return False

last_token_id = request.output_token_ids[-1]
if last_token_id == sampling_params.eos_token_id:
    request.status = RequestStatus.FINISHED_STOPPED
    return True

if last_token_id in (sampling_params.stop_token_ids or ()):
    request.status = RequestStatus.FINISHED_STOPPED
    request.stop_reason = last_token_id
    return True
if (
    request.num_tokens >= max_model_len
    or request.num_output_tokens >= request.max_tokens
):
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    return True
```

它按顺序检查最短生成长度、EOS、停止 token 和长度上限；源码还有按配置启用的重复模式检测。`num_tokens` 包含输入与已生成部分，`num_output_tokens` 只计算输出。

停止 token 是整数 ID，文本停止字符串由入口 detokenizer 检查。两种结束判断分别发生在核心和输出侧。

### 3.2 完成后移出队列，安排请求清理

正常生成判停后，`update_from_output()` 进入完成处理，调用 `_free_request()` 安排资源清理，并将结束请求移出相应队列。普通无流式输入、无传输延迟的路径会走到：

```python
# 文件：vllm/v1/core/sched/scheduler.py
# 类：Scheduler
# 函数：_free_request
# 源码范围：L2317–L2327（连续节选）
self.encoder_cache_manager.free(request)
request_id = request.request_id
self.finished_req_ids.add(request_id)
if self.finished_req_ids_dict is not None:
    self.finished_req_ids_dict[request.client_index].add(request_id)

delay_free_blocks |= connector_delay_free_blocks
if not delay_free_blocks:
    self._free_blocks(request)

return kv_xfer_params, ec_xfer_params
```

这里记录结束 ID，供执行侧和相关客户端处理收尾，再按条件释放资源。实际删除核心请求记录的位置是：

```python
# 文件：vllm/v1/core/sched/scheduler.py
# 类：Scheduler
# 函数：_free_blocks
# 源码范围：L2329–L2332（连续节选）
def _free_blocks(self, request: Request):
    assert request.is_finished()
    self._free_request_blocks(request)
    del self.requests[request.request_id]
```

`_free_request_blocks()` 负责把请求占用的 KV 块引用交回缓存管理器；存在在途 GPU 写入时会安排延迟释放。释放引用后，缓存内容是否作为前缀保留，由 [KV Cache Management](../../04-KV-Cache-Management/) 分析。

核心还会通过后续 `SchedulerOutput.finished_req_ids` 通知 Worker 清理缓存的请求状态。因此，用户请求已经完成后，核心仍可能需要执行收尾工作。

### 3.3 用户取消：先清理入口，再通知核心

```python
# 文件：vllm/v1/engine/llm_engine.py
# 类：LLMEngine
# 函数：abort_request
# 源码范围：L212–L216（连续节选）
def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
    """Remove request_ids from EngineCore and Detokenizer."""

    request_ids = self.output_processor.abort_requests(request_ids, internal)
    self.engine_core.abort_requests(request_ids)
```

输出处理器先处理外部与内部 ID 的对应关系，并清理入口状态；返回内部 ID 后，由客户端通知核心。调用方已有内部 ID 时，可以使用 `internal=True`。

核心取消方法的关键调用是：

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：abort_requests
# 源码范围：L491–L491（连续节选）
self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
```

`Scheduler.finish_requests()` 找到尚未结束的 Request，移出相应队列，然后设置结束状态并进入清理：

```python
# 文件：vllm/v1/core/sched/scheduler.py
# 类：Scheduler
# 函数：finish_requests
# 源码范围：L2295–L2296（连续节选）
request.status = finished_status
self._free_request(request, delay_free_blocks=delay_free_blocks)
```

这两行对每条有效请求执行，方法最后返回实际结束的请求列表。

不存在或已经结束的请求会被跳过，因此重复收到相同取消可以安全处理。普通取消并不要求先取回一个最终生成回答。

### 3.4 输出侧命中停止字符串，也会通知核心取消

例如配置 `stop=["<END>"]`，解码文本累计为“答案<END>”，而核心尚未报告结束：

```text
OutputProcessor 检测到停止字符串
    → 按输出配置截取文本，构造完成结果
    → 清理入口 RequestState
    → 将内部 ID 放入 reqs_to_abort
    → LLMEngine.step 通过客户端通知核心
    → 核心执行取消清理
```

用户看到的结果可以是正常的 `stop`，而核心资源通过取消路线收尾。输出结果的结束原因与核心接到的取消动作承担不同职责。

### 3.5 执行期间到达的取消，在哪里处理

多进程输入线程接收 ABORT 时，会同时写入核心 `input_queue` 和 `aborts_queue`。普通输入队列维持消息处理顺序，单独的取消队列让核心能在模型执行返回后尽快处理取消。

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：_process_aborts_queue
# 源码范围：L741–L749（连续节选）
def _process_aborts_queue(self):
    if not self.aborts_queue.empty():
        request_ids = []
        while not self.aborts_queue.empty():
            ids = self.aborts_queue.get_nowait()
            # Should be a list here, but also handle string just in case.
            request_ids.extend((ids,) if isinstance(ids, str) else ids)
        # More efficient to abort all as a single batch.
        self.abort_requests(request_ids)
```

`step()` 和 batch queue 路径都在 `update_from_output()` 之前调用它。已经下发的 GPU 工作可能完成，但核心可以先终止请求，再处理返回结果；有在途计算时资源回收还需满足安全条件。

如果某条取消消息先到取消队列，而对应 ADD 仍在等待处理，普通输入队列中的那份 ABORT 仍能按消息顺序再次生效。两个队列与重复取消处理共同避免遗漏。

## 4. 对照两条结束路线

假设 A、B 都在运行，A 最多生成 2 个 token：

| 事件 | 核心动作 | 入口动作 |
| --- | --- | --- |
| A 生成第 2 个 token，达到上限 | 设置长度结束状态，移出队列，安排资源清理，返回结束信息 | 累积到最终增量，构造 RequestOutput，清理输出状态 |
| 用户取消 B | 接收内部 ID，设置取消状态，移出队列并安排清理 | 取消调用先清理输出状态，再通知核心 |
| B 已下发的结果稍后到达 | 根据请求状态跳过或处理失效结果，遵守在途资源回收规则 | 已无对应状态时忽略迟到输出 |

核心状态、Worker 保存的请求信息、KV 引用和入口状态分别由对应组件收尾。请求生命周期的结束，需要这些动作协调完成。

## 5. 阅读检查

- EOS、停止 token、停止字符串分别在哪里检查？
- 为什么输出侧已经完成，仍可能需要通知核心取消？
- 用户完成与核心不再需要任何收尾，是否是同一个时刻？
- 取消队列为什么与普通输入队列配合使用？

[章节目录](../README.md) · [上一篇：核心客户端与进程循环](04-Core-Client-and-Process-Loop.md) · [回到整体框架](../Overview.md) · [下一章：Scheduler](../../03-Scheduler/)
