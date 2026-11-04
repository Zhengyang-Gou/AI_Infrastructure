# 05 · 结果更新、判停与请求退场

## 1. 职责与边界

输入是同一轮的 SchedulerOutput 与 ModelRunnerOutput，或带内部 ID 的外部结束指令。输出是更新后的 Request、核心增量和结束请求通知。

本篇负责追加正式 token、EOS/停止 token/长度等核心判停、队列退出与调用资源释放接口。停止字符串属于 01；KV 引用如何实际回收属于 04；执行侧如何删除请求状态属于 06。

源码：`vllm/v1/core/sched/scheduler.py`、`vllm/v1/core/sched/utils.py`、`vllm/v1/request.py`。路径相对于源码根目录。

## 2. 追加正式输出与判停

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

## 3. 结束状态与输出原因

RequestStatus 表示核心生命周期，finish_reason 用于将结果原因带回入口。

| 核心状态 | 含义 | 返回原因 |
| --- | --- | --- |
| FINISHED_STOPPED | 命中 EOS 或停止 token | STOP |
| FINISHED_LENGTH_CAPPED | 达到输出或总长度限制 | LENGTH |
| FINISHED_ABORTED | 收到核心取消 | ABORT |
| FINISHED_ERROR | 执行相关错误导致结束 | ERROR |

## 4. 退出调度与交还资源

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

`_free_request_blocks()` 是资源交还的调用边界。本篇只确认请求何时退出、何时发出释放指令；块引用、延迟回收与缓存保留的实现由 04 负责。

核心还会通过后续 `SchedulerOutput.finished_req_ids` 通知 Worker 清理缓存的请求状态。因此，用户请求已经完成后，核心仍可能需要执行收尾工作。

## 5. 外部指定结束

`finish_requests(request_ids, finished_status)` 接收内部 ID 和结束状态。它查找有效请求，移出相应队列，再设置状态并进入共用清理路线：

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

## 6. 阅读检查

- 达到 max_tokens 与命中 EOS，分别对应什么状态？
- 一轮返回多个 token 时，为什么逐个追加并检查结束？
- 请求退出调度与 KV 物理块重新可用为什么是两个动作？

[章节目录](../README.md) · [本章概览](../Overview.md)
