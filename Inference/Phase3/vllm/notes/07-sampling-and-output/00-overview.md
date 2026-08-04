# Sampling and Output

## 1. 学习目标

本阶段从模型 hidden states 产生 logits 开始，追踪每个请求的 SamplingParams 如何批量作用于 logits、采样 token 如何回写 Scheduler、token IDs 如何增量解码，最终如何组装成用户看到的 `RequestOutput`。

完成本阶段后，应能够说明以下链路：

```text
hidden states
→ model.compute_logits()
→ Sampler
→ sampled token IDs
→ ModelRunnerOutput
→ Scheduler.update_from_output()
→ EngineCoreOutput
→ OutputProcessor.process_outputs()
→ IncrementalDetokenizer
→ CompletionOutput / RequestOutput
```

## 2. 阅读文件与顺序

| 顺序 | 笔记 | 源码文件 | 阅读目的 |
| --- | --- | --- | --- |
| 1 | `01-sampler.md` | `vllm/v1/worker/gpu/sample/sampler.py` | 理解 logits 处理、top-k/top-p 与 token 采样 |
| 2 | `02-scheduler.md` | `vllm/v1/core/sched/scheduler.py` | 理解 sampled tokens 如何写回 Request、判断停止并形成核心输出 |
| 3 | `03-output-processor.md` | `vllm/v1/engine/output_processor.py` | 理解核心输出如何变成前端请求输出 |
| 4 | `04-detokenizer.md` | `vllm/v1/engine/detokenizer.py` | 理解 token IDs 如何增量变成文本并检查 stop string |

`scheduler.py` 在调度阶段已经阅读过。本阶段只重新聚焦 `update_from_output()` 及其结果回写和结束请求路径，不重复展开 `schedule()`。

## 3. 核心对象

| 对象 | 作用 |
| --- | --- |
| `Sampler` | 按请求状态处理 logits，执行采样并计算所需 logprobs |
| `SamplerOutput` | 保存 GPU 上的 sampled token IDs、logprobs 与采样统计 |
| `ModelRunnerOutput` | 将 Model Runner 的采样结果传回 Scheduler |
| `Request` | 保存已生成 token、已计算 token 数、状态与停止原因 |
| `EngineCoreOutput` | 将单个请求的新 token、finish reason 与 logprobs 交给前端 |
| `OutputProcessor` | 维护前端请求状态并组装 `RequestOutput` |
| `IncrementalDetokenizer` | 将新增 token IDs 增量解码为文本并检查 stop string |
| `CompletionOutput` / `RequestOutput` | 分别表示单条候选序列与最终用户请求结果 |

## 4. 主执行流程

### GPU 采样

```text
GPUModelRunner.sample_tokens()
→ 选择需要计算 logits 的 hidden states
→ model.compute_logits()
→ Sampler.__call__(logits, input_batch)
→ logit bias / penalties / bad words
→ temperature / min-p / top-k / top-p
→ greedy 或随机采样路径
→ SamplerOutput
→ ModelRunnerOutput
```

### Scheduler 结果回写

```text
Scheduler.update_from_output()
→ 按 request_id 取得 sampled token IDs
→ Request.append_output_token_ids()
→ check_stop()
→ 更新 RequestStatus 与 stop_reason
→ 必要时释放请求资源
→ EngineCoreOutput
→ EngineCoreOutputs
```

### 前端输出处理

```text
EngineCoreOutput
→ OutputProcessor.process_outputs()
→ IncrementalDetokenizer.update()
→ stop string 检查
→ LogprobsProcessor 更新
→ RequestState.make_request_output()
→ CompletionOutput
→ RequestOutput
→ LLM.generate() 或异步请求队列
```

## 5. 关键数据变化

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| Logits 计算 | 选定位置的 hidden states | 每个待采样位置的 logits |
| Sampler | logits、`InputBatch`、逐请求采样状态 | `SamplerOutput` |
| Model Runner | `SamplerOutput` 与 batch 映射 | `ModelRunnerOutput` |
| Scheduler 回写 | `SchedulerOutput`、`ModelRunnerOutput` | `EngineCoreOutputs` |
| 前端处理 | `EngineCoreOutput`、前端 `RequestState` | `RequestOutput` 或队列事件 |
| 增量解码 | 新 token IDs、tokenizer 状态 | 新文本、累计文本或匹配到的 stop string |

## 6. 停止条件的两层处理

- Scheduler 侧根据 token ID、EOS、最大生成长度和最大模型长度等条件更新 `RequestStatus`。
- Detokenizer 侧检查字符串级 stop 条件，因为 stop string 只有解码成文本后才能可靠识别。
- 如果 Detokenizer 先发现 stop string，而 Engine Core 仍认为请求未结束，`OutputProcessor` 会把该请求加入 `reqs_to_abort`，通知核心停止继续执行。

## 7. 完成标准

- 能说明不同请求的 SamplingParams 如何在同一个 batch 中生效。
- 能说明 temperature、min-p、top-k、top-p 与最终采样的先后关系。
- 能说明 sampled token 如何写入 `Request` 并形成 `EngineCoreOutput`。
- 能区分 token 级停止条件与字符串级停止条件。
- 能说明 DELTA、CUMULATIVE 和 FINAL_ONLY 输出模式如何影响前端结果。
- 能追踪最终文本进入 `CompletionOutput.text` 和 `RequestOutput.outputs`。

## 8. 当前结论

采样与输出不是单个函数完成的：GPU Sampler 负责从 logits 选 token，Scheduler 负责把 token 变成请求状态和核心输出，OutputProcessor 与 Detokenizer 再负责文本、流式语义、logprobs 和最终用户对象。

## 补充专题

- `05-logits-logprobs-and-structured-output.md`：补齐 logits 变换、概率信息和 grammar mask。
