# 06 · 输出处理与文本还原

本篇跟踪最后一道边界：核心返回携带内部请求 ID 的增量 Token，入口如何将其累积、解码、判停并组织成用户结果。

## 1. 职责与边界

```text
提交时：EngineCoreRequest → RequestState，记录输入与输出配置

返回时：EngineCoreOutput
    → 按内部 ID 找 RequestState
    → 增量解码与停止字符串检查
    → 按输出模式构造 CompletionOutput / RequestOutput
    → 清理完成状态，必要时请求核心取消
```

`OutputProcessor` 管理入口输出状态；`RequestState` 保存单请求的上下文；detokenizer 负责 Token 到文本的增量还原。采样已在执行侧完成，本篇消费的是采样结果。

## 2. 源码定位与核心状态

| 文件 | 类或作用域 | 重点内容 |
| --- | --- | --- |
| [v1/engine/output_processor.py](../../../vllm/vllm/v1/engine/output_processor.py) | `OutputProcessor` | `add_request`、`process_outputs`、`_finish_request`、`abort_requests` |
| 同上 | `RequestState` | `from_new_request`、`make_request_output`、`_new_completion_output`、`_new_request_output` |
| [v1/engine/detokenizer.py](../../../vllm/vllm/v1/engine/detokenizer.py) | `IncrementalDetokenizer`、`BaseIncrementalDetokenizer` | 解码器选择、`update`、`get_next_output_text` |
| 同上 | 模块级函数 | `check_stop_strings` |
| [outputs.py](../../../vllm/vllm/outputs.py) | `RequestOutput`、`CompletionOutput` | 用户侧请求结果与单个候选结果 |

| 状态 | 作用 |
| --- | --- |
| `request_states` | 内部请求 ID 到 `RequestState` 的映射 |
| `external_req_ids` | 外部 ID 到内部 ID 列表的映射，支持取消与标识还原 |
| `RequestState.output_kind` | 决定累计、增量或只在结束时输出 |
| `RequestState.detokenizer` | 保存已生成 Token IDs、解码文本与停止检查状态 |
| `RequestState.logprobs_processor` | 按配置处理生成与 Prompt 的 logprobs |
| `RequestState.queue` | 在线异步路径可使用队列；离线主线直接收集返回对象 |

`EngineCoreOutput` 描述执行增量，`RequestState` 保存跨轮状态，`RequestOutput` 是交付给用户的结果。三者不能相互替代。

## 3. 实现分析

### 3.1 提交时建立 RequestState

```python
# 文件：vllm/v1/engine/output_processor.py
# 类：OutputProcessor
# 函数：add_request
# 源码范围：L525–L554（连续节选）
def add_request(
    self,
    request: EngineCoreRequest,
    prompt: str | None,
    parent_req: ParentRequest | None = None,
    request_index: int = 0,
    queue: RequestOutputCollector | None = None,
) -> None:
    request_id = request.request_id
    req_state = self.request_states.get(request_id)
    if req_state is not None:
        self._update_streaming_request_state(req_state, request, prompt)
        return

    req_state = RequestState.from_new_request(
        tokenizer=self.tokenizer,
        request=request,
        prompt=prompt,
        parent_req=parent_req,
        request_index=request_index,
        queue=queue,
        log_stats=self.log_stats,
        stream_interval=self.stream_interval,
    )
    self.request_states[request_id] = req_state
    if parent_req:
        self.parent_requests[parent_req.request_id] = parent_req

    # Track the external_req_id -> [internal_req_id, ...] mapping
    self.external_req_ids[req_state.external_req_id].append(request_id)
```

新的普通请求通过 `RequestState.from_new_request` 创建状态，登记到内部 ID 的索引，并维护外部到内部的对应关系。已有状态的分支用于流式输入更新，本篇先沿新请求路径阅读。

工厂从请求参数中取出输出模式，并创建 logprobs 处理器与解码器：

```python
# 文件：vllm/v1/engine/output_processor.py
# 类：RequestState
# 函数：from_new_request
# 源码范围：L223–L238（连续节选）
if sampling_params := request.sampling_params:
    if not sampling_params.detokenize:
        tokenizer = None
    output_kind = sampling_params.output_kind
    if sampling_params.stream_interval is not None:
        # clamp to the engine-level stream interval.
        stream_interval = max(sampling_params.stream_interval, stream_interval)
    logprobs_processor = LogprobsProcessor.from_new_request(
        tokenizer=tokenizer,
        request=request,
    )
    detokenizer = IncrementalDetokenizer.from_new_request(
        tokenizer=tokenizer,
        request=request,
    )
    max_tokens_param = sampling_params.max_tokens
```

状态随后还记录输入内容、内外 ID、到达时间等信息。因此，核心每轮只需返回增量，入口就能依靠状态恢复上下文。

### 3.2 取得增量后，先找到对应状态

```python
# 文件：vllm/v1/engine/output_processor.py
# 类：OutputProcessor
# 函数：process_outputs
# 源码范围：L619–L624（连续节选）
for engine_core_output in engine_core_outputs:
    req_id = engine_core_output.request_id
    req_state = self.request_states.get(req_id)
    if req_state is None:
        # Ignore output for already-aborted request.
        continue
```

如果请求已经在入口被取消，对应状态可能已经不存在。此时直接忽略迟到输出，避免将它当成新请求或继续生成用户结果。

### 3.3 增量解码也承担文本停止判断

普通生成分支将新 Token 交给请求自己的解码器：

```python
# 文件：vllm/v1/engine/output_processor.py
# 类：OutputProcessor
# 函数：process_outputs
# 源码范围：L652–L665（连续节选）
if pooling_output is None:
    assert req_state.detokenizer is not None
    assert req_state.logprobs_processor is not None
    # 2) Detokenize the token ids into text and perform stop checks.
    stop_string = req_state.detokenizer.update(
        new_token_ids, finish_reason == FinishReason.STOP
    )
    if stop_string:
        finish_reason = FinishReason.STOP
        stop_reason = stop_string

    # 3) Compute sample and prompt logprobs for request,
    # if required.
    req_state.logprobs_processor.update_from_output(engine_core_output)
```

`finish_reason` 最初来自核心；如果文本解码发现停止字符串，输出侧会将结束原因更新为 `STOP`，并记录命中的字符串。

解码器工厂会根据是否启用文本解码及 tokenizer 能力，选择只记录 Token 的基础实现、快速实现或慢速实现。快速、慢速实现共用 `BaseIncrementalDetokenizer` 的停止检查框架，具体逐 Token 解码分别由子类实现。

其 `update` 中的文本累积与停止检查如下；函数前面还处理空增量及核心 STOP Token 的显示规则：

```python
# 文件：vllm/v1/engine/detokenizer.py
# 类：BaseIncrementalDetokenizer
# 函数：update
# 源码范围：L116–L143（连续节选）
# 1) Detokenize the new token ids incrementally.
stop_check_offset = len(self.output_text)
for new_token_id in new_token_ids:
    self.token_ids.append(new_token_id)
    self.output_text += self.decode_next(new_token_id)
    # Support min_tokens, see https://github.com/vllm-project/vllm/pull/22014
    if self.min_tokens and self.num_output_tokens() <= self.min_tokens:
        stop_check_offset = len(self.output_text)

if skipped_stop_token_id is not None:
    # Cleanup after skipping detokenization.
    self.token_ids.append(skipped_stop_token_id)

# 2) Evaluate stop strings.
stop_string = None
if self.stop and self.num_output_tokens() > self.min_tokens:
    stop = check_stop_strings(
        output_text=self.output_text,
        new_char_count=len(self.output_text) - stop_check_offset,
        stop=self.stop,
        include_in_output=self.include_stop_str_in_output,
    )
    if stop is not None:
        stop_string, truncate_to = stop
        if truncate_to != -1:
            self.output_text = self.output_text[:truncate_to]

return stop_string
```

停止字符串可能跨 Token、跨轮次形成，不能仅对当前单个 Token 独立解码后的文本进行判断。`check_stop_strings` 会覆盖与新文本相邻的历史后缀；有多个命中时，选择文本中最早完成的停止字符串，相同结束位置按停止列表顺序选择。

为避免将未来可能属于停止字符串的片段提前流式输出，初始化时会准备一个缓冲长度：

```python
# 文件：vllm/v1/engine/detokenizer.py
# 类：BaseIncrementalDetokenizer
# 函数：__init__
# 源码范围：L87–L91（连续节选）
if self.stop and not self.include_stop_str_in_output:
    self.stop_buffer_length = max(len(s) for s in self.stop) - 1
else:
    self.stop_buffer_length = 0
self._last_output_text_offset: int = 0
```

`include_stop_str_in_output=False` 时，显示文本可以截去停止字符串，但已生成 Token ID 仍可能记录在状态中。文本截断与删除执行历史的 Token IDs 是不同操作。

### 3.4 输出模式决定何时交付结果

```python
# 文件：vllm/v1/engine/output_processor.py
# 类：RequestState
# 函数：make_request_output
# 源码范围：L285–L290（连续节选）
finished = finish_reason is not None
final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

if not finished and final_only:
    # Only the final output is required in FINAL_ONLY mode.
    return None
```

| 输出模式 | 用户对象包含什么 | 何时产生 |
| --- | --- | --- |
| `CUMULATIVE` | 截至当前的累计输出 | 满足输出间隔规则时 |
| `DELTA` | 自上次交付以来的增量 | 满足输出间隔规则时 |
| `FINAL_ONLY` | 完整最终输出 | 请求完成时 |

`FINAL_ONLY` 只限制交付时机。`process_outputs` 仍先处理增量 Token 和状态，再决定是否构造用户对象。输出间隔 `stream_interval` 还可以进一步控制交付频率。

生成候选时，普通离线模式会使用全部已记录的输出 Token：

```python
# 文件：vllm/v1/engine/output_processor.py
# 类：RequestState
# 函数：_new_completion_output
# 源码范围：L394–L402（连续节选）
assert self.detokenizer is not None
assert self.logprobs_processor is not None
finished = finish_reason is not None
delta = self.output_kind == RequestOutputKind.DELTA

# Prepare text and token_ids, based on delta mode
text = self.detokenizer.get_next_output_text(finished, delta)
if not delta:
    token_ids = self.detokenizer.output_token_ids
```

函数后面再构造 `CompletionOutput`，填写候选索引、文本、Token IDs、logprobs 和结束原因。`n=1` 时只有一个候选；`n>1` 时由父请求协调多个候选的归属与完成状态。

用户请求对象使用外部 ID：

```python
# 文件：vllm/v1/engine/output_processor.py
# 类：RequestState
# 函数：_new_request_output
# 源码范围：L373–L386（连续节选）
return RequestOutput(
    request_id=external_req_id,  # request_id is what was provided externally
    lora_request=self.lora_request,
    prompt=self.prompt,
    prompt_token_ids=prompt_token_ids,
    prompt_logprobs=prompt_logprobs,
    outputs=cast(list[CompletionOutput], outputs),
    finished=finished,
    kv_transfer_params=kv_transfer_params,
    ec_transfer_params=ec_transfer_params,
    num_cached_tokens=self.num_cached_tokens,
    num_cache_creation_tokens=self.num_cache_creation_tokens,
    metrics=self.stats,
)
```

这一步让上游离线循环能继续使用入口的递增数字 ID 排序，而不需要解释核心内部的随机后缀。

### 3.5 完成清理与核心取消反馈

输出侧发现请求完成后，普通非流式输入路径会清理状态；若核心尚未报告完成，还会记录需要取消的内部 ID：

```python
# 文件：vllm/v1/engine/output_processor.py
# 类：OutputProcessor
# 函数：process_outputs
# 源码范围：L686–L699（连续节选）
# Free completed requests.
if finish_reason is not None:
    if req_state.streaming_input:
        if req_state.input_chunk_queue:
            update = req_state.input_chunk_queue.popleft()
            req_state.apply_streaming_update(update)
        else:
            req_state.input_chunk_queue = None
    else:
        self._finish_request(req_state)
        if not engine_core_output.finished:
            # If req not finished in EngineCore, but Detokenizer
            # detected stop string, abort needed in EngineCore.
            reqs_to_abort.append(req_id)
```

处理器返回的 `OutputProcessorOutput` 同时带有用户结果与 `reqs_to_abort`。上一篇的 `LLMEngine.step` 将后者交给客户端通知核心。

状态清理本身移除请求索引，并在没有剩余对应请求时移除外部 ID 映射：

```python
# 文件：vllm/v1/engine/output_processor.py
# 类：OutputProcessor
# 函数：_finish_request
# 源码范围：L713–L725（连续节选）
def _finish_request(self, req_state: RequestState) -> None:
    req_id = req_state.request_id
    self.request_states.pop(req_id)

    internal_ids = self.external_req_ids[req_state.external_req_id]
    internal_ids.remove(req_id)
    if not internal_ids:
        del self.external_req_ids[req_state.external_req_id]

    # Remove parent request if applicable.
    parent_req = req_state.parent_req
    if parent_req and not parent_req.child_requests:
        self.parent_requests.pop(parent_req.request_id, None)
```

这属于入口状态清理；GPU KV 的引用回收由执行核心继续负责。

## 4. 手工推演：跨轮次的停止字符串

设 `stop=["<END>"]`、`include_stop_str_in_output=False`、`min_tokens=0`，采用 `FINAL_ONLY`：

```text
第 1 次增量：累计文本成为“答案<EN”
    → 停止字符串尚未完整出现，继续保留状态
    → 请求未结束，不产生最终 RequestOutput

第 2 次增量：追加“D>”，累计文本出现“<END>”
    → 命中停止字符串，将显示文本截为“答案”
    → 输出侧标记 STOP，构造最终 RequestOutput
    → 如果核心原本尚未结束，将内部 ID 加入 reqs_to_abort
    → LLMEngine 通知核心取消并完成资源清理
```

这里的文本片段用于说明解码行为，实际 Token 切分取决于 tokenizer。若设置 `min_tokens`，停止检查还需遵守最短生成长度要求。

## 5. 设计原因与自测

按请求维护状态，才能将不同请求交错返回的增量正确归属。解码与停止字符串处理需要文本上下文；核心侧的 EOS、长度等判停与输出侧的文本判停通过取消反馈连接。结果对象使用外部 ID，让执行标识与用户接口各自稳定。

读完后应能解释：为什么状态必须在提交时创建？`FINAL_ONLY` 为什么仍需每轮处理？为什么停止字符串会跨轮次？入口状态清理与 KV 回收分别由谁负责？

六篇回读时，沿「入口接口 → 流程编排 → 输入内容 → 执行请求 → 核心交接 → 输出还原」追踪同一条请求，再进入下一章的核心执行闭环。

[章节目录](../README.md) · [上一篇：LLMEngine 与核心边界](05-LLMEngine-and-Core-Boundary.md) · [回到整体框架](../Overview.md) · [下一章：Engine Core](../../02-Engine-Core/)
