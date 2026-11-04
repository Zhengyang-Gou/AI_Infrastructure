# 主线走查：一条请求穿过全部模块

这条走查用 [README 案例 A](README.md)（预算 4、`max_num_seqs=2`、B 先入队 2 token、A 后入队 6 token、各最多生成 2 个 token）。每一步只回答三件事：**谁在做、状态变了什么、去哪里读细节**。

读完全表应该能自己讲出：一条请求在哪个对象里被改了哪个字段，以及它在模块之间被交给谁。模块正文负责展开细节，本页只负责把线索串起来。

## 1. 逐步骤走查

| 步 | 谁在做 | 状态变化（案例 A） | 读哪里 |
| --- | --- | --- | --- |
| 1 | `llm = LLM(...)` | 组件、模型、KV 存储、调度器就绪；尚无任何请求 | [05 · 模型与权重加载](05-Model-and-Weight-Loading/)（待写）、[01 · LLM 与初始化](01-Inference-Entry/Core-Modules/01-LLM-and-Initialization.md) |
| 2 | `llm.generate([B, A])` | 一批输入与生成参数对齐，逐条提交，最后统一收集 | [01 · 离线推理流程编排](01-Inference-Entry/Core-Modules/02-Offline-Inference-Flow.md) |
| 3 | `BaseRenderer.render_cmpl` | `Prompt → EngineInput`：得到 token ids（B=2 个，A=6 个） | [01 · Renderer 与输入表示](01-Inference-Entry/Core-Modules/03-Renderer-and-EngineInput.md) |
| 4 | `InputProcessor.process_inputs` | `EngineInput → EngineCoreRequest`：校验长度与参数，写入 `request_id`、`prompt_token_ids`、采样参数 | [01 · InputProcessor 与请求构造](01-Inference-Entry/Core-Modules/04-InputProcessor-and-EngineCoreRequest.md) |
| 5 | `OutputProcessor.add_request` | 入口侧登记 `RequestState`，建立**外部 ID ↔ 内部 ID** 的对应 | [01 · LLMEngine 与核心边界](01-Inference-Entry/Core-Modules/05-LLMEngine-and-Core-Boundary.md) |
| 6 | `InprocClient.add_request` → `EngineCore.add_request` | `EngineCoreRequest → Request`，状态为 `WAITING`，进入等待队列（B 在前，A 在后） | [02 · Request 与请求接收](02-Engine-Core/Core-Modules/02-Request-and-Admission.md) |
| 7 | `Scheduler.schedule(budget=4)` | 预算 4 分给 B=2、A=2；B 的 Prefill 完成，A 只算了 6 个中的 2 个（Chunked Prefill）。两条的 `num_computed_tokens` 都从 0 推进到 2 | [03 · Scheduler §4–§5](03-Scheduler/Overview.md) |
| 8 | 执行侧 `Executor.execute_model` → `GPUModelRunner` | 为两条请求建立执行侧状态与 block table，把本轮 token、位置、`slot_mapping` 组装成输入张量，前向后采样 | [06 · Model Execution](06-Model-Execution/)（待写）、[04 · KV Cache Management](04-KV-Cache-Management/)（待写） |
| 9 | `Scheduler.update_from_output` | B 追加第 1 个输出 token（`num_tokens` 2→3）；A 本轮没有输出 token，只是 Prefill 进度增加 | [03 · Scheduler §8](03-Scheduler/Overview.md) |
| 10 | `EngineCoreOutputs` → `LLMEngine.step` → `OutputProcessor` | 增量 token 解码成文本；离线用 `FINAL_ONLY`，此时**还不会**返回给用户，但判停与解码状态每轮都要维护 | [01 · 输出处理与文本还原](01-Inference-Entry/Core-Modules/06-OutputProcessor-and-Detokenization.md) |
| 11 | 下一轮 `step()` | B 走 Decode（1 个 token），A 继续 Prefill（预算剩 3，推进到 5/6）；第 3 轮 C 加入，第 4 轮全部结束 | [03 · Scheduler §6](03-Scheduler/Overview.md) |
| 12 | `check_stop` → `Scheduler.finish_requests` | 达到 `max_tokens` 的两条请求标记为 `FINISHED_STOPPED`，退出运行集合，释放 KV 块引用 | [02 · 完成、取消与清理](02-Engine-Core/Core-Modules/05-Completion-and-Abort.md) |
| 13 | `_run_engine` 收齐结果 | 按提交顺序返回 `[RequestOutput(B), RequestOutput(A)]`，各含完整文本与结束原因 | [01 · 离线推理流程编排](01-Inference-Entry/Core-Modules/02-Offline-Inference-Flow.md) |

第 9、10 步要分清：**调度侧进度**（`num_computed_tokens`）在 `schedule()` 里就推进了，早于结果返回；**请求新增 token**（`num_tokens`）要等 `update_from_output()`。两者不同步，这也是“本轮算了多少 token”和“本轮生成多少 token”容易混淆的原因。

## 2. 换成块账本看（案例 B）

把案例 A 换成真实预算 256、`block_size=16`，只看一条 600-token 请求，就能看清第 8 步的 KV 部分：

| 轮次 | 本轮计算量 | `num_computed_tokens` | 累计块数 | `slot_mapping` 覆盖的位置 |
| --- | ---: | ---: | ---: | --- |
| 1 | 256 | 256 | 16 | 0–255，块 0–15 |
| 2 | 256 | 512 | 32 | 256–511，块 16–31 |
| 3 | 88 | 600 | 38 | 512–599，块 32–37 |

第 3 轮结束时 Prefill 完成并采样出第 1 个 token，此后每轮只加 1 个 token。块数是 `ceil(num_computed_tokens / 16)`，但每轮只申请**新增**的那几块，旧块继续持有引用——这条“只增量申请”是第 7、8 步能对上账的关键。

## 3. 这张表怎么用

- **当索引用**：读某个模块前，先看它在表里占了哪几步，读完回到那几步确认能讲出来。
- **当进度表用**：每写完一个模块，就把对应格子里的“（待写）”换成链接，这张表随之变成完成度清单。目前 04、05、06 三格还是空的。
- **当排查表用**：调不动的时候先定位卡在第几步，再去那一格对应的对象里看字段，比通读源码快。

维护约定：模块内的函数改名或拆分后，回来同步这张表；表里不写行号，行号只出现在模块正文。
