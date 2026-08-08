# OfflineInferenceMixin

## 1. 本篇学习目标

这一篇只回答一个核心问题：

> `LLM.generate()` 进入 `_run_completion()` 之后，vLLM 如何把“一批请求提交到引擎”和“同步驱动引擎直到全部完成”组织起来？

完成本篇后，应该能够回答：

1. `_run_completion()` 为什么是 `LLM.generate()` 的直接下游？
2. “提交请求”和“驱动执行”为什么是两个独立阶段？
3. 一批 `prompts`、`SamplingParams`、LoRA 和 priority 如何对齐？
4. prompt 在哪里从公开 API 输入变成 `EngineInput`？
5. 单个请求最终在哪里进入 `LLMEngine.add_request()`？
6. 谁循环调用 `LLMEngine.step()`？
7. 为什么最终结果需要按 `request_id` 排序？
8. `OfflineInferenceMixin` 做什么，又明确不做什么？

本篇暂时不深入：

- `LLMEngine.add_request()` 内部如何构造 `EngineCoreRequest`；
- `LLMEngine.step()` 如何与 `EngineCoreClient` 交互；
- Scheduler 如何选择 request / token；
- KV Cache 如何分配；
- ModelExecutor 如何执行模型；
- Sampling 如何产生 token。

这些分别留到后续章节。

---

## 2. 文件定位

- 文件路径：`vllm/entrypoints/offline_utils.py`
- 所属层次：离线 API 编排层
- 核心职责：将批量输入规范化、渲染并逐个提交给 `LLMEngine`，随后同步循环调用 `LLMEngine.step()`，直到当前请求全部完成。
- 在调用链中的位置：

```text
LLM.generate()
    ↓
OfflineInferenceMixin
    ↓
LLMEngine.add_request() / LLMEngine.step()
```

`LLM` 提供公开 API，`LLMEngine` 负责前端请求状态和输出处理，而 `OfflineInferenceMixin` 位于两者之间，负责把一次同步的离线 API 调用编排成若干次 `LLMEngine` 操作。

一句话定位：

> `OfflineInferenceMixin` 不负责“怎么调度、怎么算模型”，而负责“先把请求送进去，再不断推进引擎，最后把结果收回来”。

---

## 3. 先建立最重要的结构

整个文件当前最值得掌握的是三个函数：

```text
_run_completion()              ★★★★★
├── _add_completion_requests() ★★★★★
└── _run_engine()              ★★★★★
```

它们对应两个阶段：

```text
                _run_completion()
                /               \
               /                 \
              ▼                   ▼
_add_completion_requests()    _run_engine()
          │                       │
          ▼                       ▼
       提交请求                驱动执行
```

必须建立的核心认知：

> **提交 request ≠ 执行 request。**

`generate()` 之所以对用户表现为一个同步阻塞 API，是因为 `_run_completion()` 把这两个阶段连续执行了。

---

## 4. 主调用链

### 4.1 从 `LLM.generate()` 进入

上一层：

```text
LLM.generate()
    ↓
self._run_completion(...)
```

`_run_completion()` 由 `OfflineInferenceMixin` 提供。

它本身几乎没有模型逻辑，只规定执行顺序：

```python
self._add_completion_requests(
    prompts=prompts,
    params=params,
    use_tqdm=use_tqdm,
    lora_request=lora_request,
    priority=priority,
    tokenization_kwargs=tokenization_kwargs,
    mm_processor_kwargs=mm_processor_kwargs,
)

return self._run_engine(
    use_tqdm=use_tqdm,
    output_type=output_type,
)
```

因此可以直接抽象为：

```text
LLM.generate()
    ↓
_run_completion()
    │
    ├── 1. _add_completion_requests()
    │
    └── 2. _run_engine()
```

这就是本篇最重要的结构。

---

## 5. 第一阶段：提交请求

### 5.1 `_add_completion_requests()`

这一阶段的目标不是执行模型，而是把公开 API 输入整理成一批可以逐个送入 `LLMEngine` 的请求。

整体流程：

```text
prompts
SamplingParams / PoolingParams
LoRARequest
priority
tokenization kwargs
multimodal kwargs
        ↓
批量参数对齐
        ↓
逐个 prompt 预处理 / 渲染
        ↓
EngineInput
        ↓
_render_and_add_requests()
        ↓
_add_request()
        ↓
LLMEngine.add_request()
```

---

## 6. 批量输入对齐

用户输入的形式比较灵活：

- `prompts` 可以是单个输入，也可以是一批输入；
- 参数可以是单个对象，也可以是与 prompts 一一对应的序列；
- LoRA、priority 同样需要与请求数量对齐。

核心代码：

```python
seq_prompts = prompt_to_seq(prompts)

seq_params = self._params_to_seq(
    params,
    len(seq_prompts),
)

seq_lora_requests = self._lora_request_to_seq(
    lora_request,
    len(seq_prompts),
)

seq_priority = self._priority_to_seq(
    priority,
    len(seq_prompts),
)
```

这一步解决的问题可以概括为：

```text
灵活的用户输入
       ↓
统一成等长序列
       ↓
request 0: prompt[0] + params[0] + lora[0] + priority[0]
request 1: prompt[1] + params[1] + lora[1] + priority[1]
...
```

如果用户显式传入参数序列，而长度和 prompt 数量不一致，会在请求进入引擎之前直接报错。

### 当前学习边界

这里需要理解“为什么要对齐”和“对齐后的数据形态”。

暂时不需要深入：

- 所有参数类型的完整泛型实现；
- 每种 LoRA / multimodal 参数的具体语义；
- 所有输入合法性检查。

---

## 7. Prompt 预处理与渲染

### 7.1 `_preprocess_cmpl_one()`

在提交给 `LLMEngine` 之前，每个 prompt 会先被预处理。

调用关系：

```text
PromptType
    ↓
_preprocess_cmpl_one()
    ↓
EngineInput
```

在当前章节，只需要理解它的边界：

> 它把用户侧的 prompt 表示转换为后续引擎能够消费的输入表示。

这里可能涉及文本、token、多模态信息以及 tokenization 相关覆盖参数，但这些细节当前不展开。

### 7.2 为什么这里使用生成器

核心结构：

```python
return self._render_and_add_requests(
    prompts=(
        self._preprocess_cmpl_one(
            prompt,
            tokenization_kwargs,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        for prompt in maybe_tqdm(
            seq_prompts,
            use_tqdm=use_tqdm,
        )
    ),
    params=seq_params,
    lora_requests=seq_lora_requests,
    priorities=seq_priority,
)
```

这里不是先：

```text
全部 prompt
    ↓
全部转成 EngineInput
    ↓
再统一提交
```

而是更接近：

```text
prompt 0
 ↓
render
 ↓
add

prompt 1
 ↓
render
 ↓
add
```

也就是“边产生、边消费”的方式。

当前只需要记住这个执行形态，不必研究 Python generator 本身。

---

## 8. `_render_and_add_requests()`：可靠地逐个提交

核心代码：

```python
added_request_ids = []

try:
    for i, prompt in enumerate(prompts):
        request_id = self._add_request(
            prompt,
            params[i],
            lora_request=self._resolve_mm_lora(
                prompt,
                lora_requests[i],
            ),
            priority=priorities[i],
        )
        added_request_ids.append(request_id)

except Exception as e:
    if added_request_ids:
        self.llm_engine.abort_request(
            added_request_ids,
            internal=True,
        )
    raise e
```

这段代码最重要的不是循环，而是：

> **批量提交过程具有异常回滚逻辑。**

例如：

```text
request 0  添加成功
request 1  添加成功
request 2  渲染 / 提交失败
```

如果什么都不做：

```text
Engine 中还残留 request 0、1
但本次 API 已经抛异常
```

调用者可能无法正常追踪这些请求。

因此代码会：

```text
异常发生
   ↓
abort 已经成功加入的 request
   ↓
重新抛出异常
```

---

## 9. `_add_request()`：真正跨入 `LLMEngine`

核心代码：

```python
if isinstance(params, SamplingParams):
    params.output_kind = RequestOutputKind.FINAL_ONLY

request_id = str(next(self.request_counter))

return self.llm_engine.add_request(
    request_id,
    prompt,
    params,
    lora_request=lora_request,
    priority=priority,
)
```

这里有三个关键点。

### 9.1 离线 `generate()` 使用 `FINAL_ONLY`

```text
SamplingParams
    ↓
output_kind = FINAL_ONLY
```

当前离线 `generate()` 最终只关心每个请求完成后的最终结果，因此这里把输出模式设置成 `FINAL_ONLY`。

流式输出等其他模式留到 online serving / output 章节再看。

### 9.2 生成 request ID

```python
request_id = str(next(self.request_counter))
```

概念上：

```text
第 1 个请求 → "0"
第 2 个请求 → "1"
第 3 个请求 → "2"
...
```

这个 ID 后面既用于标识 request，也用于恢复最终输出顺序。

### 9.3 真正进入 `LLMEngine`

```python
self.llm_engine.add_request(...)
```

这是当前这一阶段的重要边界。

到这里：

```text
OfflineInferenceMixin
        ↓
LLMEngine.add_request()
```

**本篇先停。**

不要现在追：

```text
LLMEngine.add_request()
    ↓
EngineCoreRequest
    ↓
EngineCoreClient
```

这是下一篇 `llm-engine.md` 的内容。

---

## 10. 第二阶段：驱动执行

请求全部加入之后：

```text
_run_completion()
       ↓
_run_engine()
```

核心结构：

```python
outputs = []

while self.llm_engine.has_unfinished_requests():
    step_outputs = self.llm_engine.step()

    for output in step_outputs:
        if output.finished:
            outputs.append(output)

return sorted(
    outputs,
    key=lambda x: int(x.request_id),
)
```

这段代码直接回答：

> **离线模式下，到底是谁不停调用 `LLMEngine.step()`？**

答案就是：

```text
OfflineInferenceMixin._run_engine()
```

---

## 11. `_run_engine()` 的循环语义

把代码翻译成执行过程：

```text
是否还有未完成 request？
        │
       yes
        ↓
  LLMEngine.step()
        ↓
得到本轮 step_outputs
        ↓
只收集 finished output
        ↓
再次检查
        │
       yes
        └──────────────┐
                       │
                       ▼
                 下一次 step

        no
        ↓
按 request_id 排序
        ↓
return
```

所以：

```text
LLMEngine.step()
```

不是“整个请求一次执行完”，而是：

> **推动底层引擎向前运行一个 step，并取得这一轮产生的前端输出。**

至于一个 step 内：

```text
Scheduler 选谁？
执行多少 token？
GPU 做什么？
```

当前全部不展开。

---

## 12. 为什么只收集 `finished` output

代码：

```python
for output in step_outputs:
    if output.finished:
        outputs.append(output)
```

当前离线 `generate()` 的目标是最终返回：

```text
list[RequestOutput]
```

因此 `_run_engine()` 不需要把每一轮中间状态都加入最终返回列表，只在某个 request 完成时保存它的最终输出。

例如：

```text
step 1
A: 未完成
B: 未完成

step 2
A: 完成   → 收集 A
B: 未完成

step 3
B: 完成   → 收集 B
```

---

## 13. 为什么最后还要排序

核心代码：

```python
return sorted(
    outputs,
    key=lambda x: int(x.request_id),
)
```

因为：

```text
提交顺序
A → B → C
```

不意味着：

```text
完成顺序
A → B → C
```

例如可能是：

```text
提交：
0 → 1 → 2

完成：
1 → 0 → 2
```

但用户调用：

```python
outputs = llm.generate(prompts, ...)
```

通常期望结果仍然与输入 prompt 的顺序对应。

因此：

```text
完成顺序
1, 0, 2
    ↓
按 request_id 排序
    ↓
0, 1, 2
    ↓
返回用户
```

---

## 14. 与 `enqueue()` / `wait_for_completion()` 的关系

这一点只作为辅助理解，不需要深入。

概念上：

```text
enqueue()
    ↓
_add_completion_requests()


wait_for_completion()
    ↓
_run_engine()
```

而普通：

```text
generate()
    ↓
_run_completion()
    ↓
_add_completion_requests()
    +
_run_engine()
```

因此这两个 API 非常直观地展示：

```text
提交请求
    ≠
驱动执行
```

但它们不是本篇主线，知道这层对应关系即可。

---

## 15. `OfflineInferenceMixin` 到底负责什么

可以把职责压缩成五件事：

```text
1. 批量输入对齐
        ↓
2. Prompt 预处理 / 渲染
        ↓
3. 逐个可靠提交 request
        ↓
4. 循环调用 LLMEngine.step()
        ↓
5. 收集完成结果并恢复输入顺序
```

---

## 16. 它明确不负责什么

### 不负责 Scheduler

它只会调用：

```python
self.llm_engine.step()
```

不会决定：

```text
本轮调度哪个 request
本轮执行多少 token
是否 preempt
```

### 不负责模型执行

它不会直接进行：

```text
model.forward()
attention()
sampling kernel
```

### 不负责 KV Cache

它不会决定：

```text
block 怎么分配
prefix cache 是否命中
block 什么时候释放
```

### 不负责 EngineCore 通信细节

它只面向：

```text
LLMEngine
```

至于：

```text
LLMEngine
    ↓
EngineCoreClient
    ↓
EngineCore
```

下一篇再追。

---

## 17. 与前后文件的关系

### 上游

```text
vllm/entrypoints/llm.py

LLM.generate()
    ↓
_run_completion()
```

### 当前层

```text
vllm/entrypoints/offline_utils.py

OfflineInferenceMixin
├── _add_completion_requests()
├── _preprocess_cmpl_one()
├── _render_and_add_requests()
├── _add_request()
├── _run_completion()
└── _run_engine()
```

### 下游

```text
vllm/v1/engine/llm_engine.py

LLMEngine.add_request()
LLMEngine.has_unfinished_requests()
LLMEngine.step()
```

所以完整边界可以画成：

```text
LLM.generate()
      │
      ▼
OfflineInferenceMixin
      │
      ├──────────────┐
      ▼              ▼
add_request 路径   step 路径
      │              │
      └──────┬───────┘
             ▼
          LLMEngine
```

---

## 18. 本篇必须掌握 / 知道即可 / 暂不展开

### 必须掌握

- `LLM.generate()` 下一步进入 `_run_completion()`。
- `_run_completion()` = `_add_completion_requests()` + `_run_engine()`。
- “请求提交”和“请求执行”是两个独立阶段。
- `_add_completion_requests()` 会先统一批量输入，再逐个渲染并提交。
- `_add_request()` 是进入 `LLMEngine.add_request()` 的最后一层。
- `_run_engine()` 循环调用 `LLMEngine.step()`。
- 只收集 `finished` 输出。
- 最终按数字 `request_id` 恢复输入顺序。
- `OfflineInferenceMixin` 是编排层，不是 Scheduler / ModelExecutor。

### 知道存在即可

- `_params_to_seq()`
- `_lora_request_to_seq()`
- `_priority_to_seq()`
- `_resolve_mm_lora()`
- progress bar
- throughput statistics
- pooling output 的共用逻辑
- `enqueue()` / `wait_for_completion()` 与两个阶段的对应关系

### 暂不展开

- Renderer 的内部实现
- tokenizer 具体调用链
- multimodal processor
- LoRA 内部加载与执行
- `LLMEngine.add_request()` 内部
- `LLMEngine.step()` 内部
- Scheduler
- KV Cache
- ModelExecutor
- Sampling

---

## 19. 当前阶段检查清单

学完本篇后，不看代码应该能够画出：

```text
LLM.generate()
    ↓
_run_completion()
    │
    ├── _add_completion_requests()
    │       ↓
    │   参数对齐
    │       ↓
    │   prompt 渲染
    │       ↓
    │   _render_and_add_requests()
    │       ↓
    │   _add_request()
    │       ↓
    │   LLMEngine.add_request()
    │
    └── _run_engine()
            ↓
      while has_unfinished_requests()
            ↓
        LLMEngine.step()
            ↓
       收集 finished
            ↓
       按 request_id 排序
            ↓
           return
```

并能够口述回答：

1. `_run_completion()` 自己做模型推理吗？
2. 为什么 `_add_completion_requests()` 和 `_run_engine()` 要分开？
3. `EngineInput` 在哪一层出现？
4. 谁真正调用 `LLMEngine.add_request()`？
5. 谁不断调用 `LLMEngine.step()`？
6. 为什么 request 的完成顺序可能和提交顺序不同？
7. 为什么最终还要排序？

如果这些问题都能回答，本篇就可以结束，不需要继续横向扩展。

---

## 20. 一句话总结

`OfflineInferenceMixin` 是 `LLM.generate()` 与 `LLMEngine` 之间的同步编排层：它先把批量公开输入规范化、渲染并可靠地提交为请求，再循环调用 `LLMEngine.step()` 直到所有请求完成，最后按 request ID 恢复输入顺序并返回结果。

下一篇只需要带着两个问题进入 `llm_engine.py`：

```text
LLMEngine.add_request()
到底如何把前端输入变成核心引擎请求？

LLMEngine.step()
到底如何从 EngineCore 取得一轮输出并处理？
```
