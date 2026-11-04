# 02 · 离线推理流程编排

本篇跟踪 `OfflineInferenceMixin`：如何将一批 prompts 和参数对齐、逐条提交，再持续收集最终结果。

## 1. 职责与边界

`OfflineInferenceMixin` 提供离线推理流程方法，由 `LLM` 继承。它使用 `LLM` 上的 renderer、引擎和计数器，把不同任务的公共提交与收集逻辑串起来。

```text
_run_completion
├─ _add_completion_requests
│  ├─ 统一输入与逐请求参数
│  └─ _render_and_add_requests
│     ├─ 消费生成器：预处理一条输入
│     └─ _add_request：设置输出模式、分配 ID、提交
└─ _run_engine
   └─ 重复 step → 收集 finished → 恢复输入顺序
```

这里的“批量”是用户提交的一批请求。GPU 每轮实际执行哪些请求，由核心调度器决定。

## 2. 源码定位与关键状态

| 文件 | 类或作用域 | 重点函数 |
| --- | --- | --- |
| [entrypoints/offline_utils.py](../../../vllm/vllm/entrypoints/offline_utils.py) | `OfflineInferenceMixin` | `_run_completion`、`_params_to_seq`、`_add_completion_requests`、`_render_and_add_requests`、`_add_request`、`_run_engine` |
| [renderers/inputs/preprocess.py](../../../vllm/vllm/renderers/inputs/preprocess.py) | 模块级函数 | `prompt_to_seq` |

`request_counter` 是跨调用的入口计数器；`added_request_ids` 只记录当前提交过程中已经成功添加的内部 ID；`outputs` 只收集本次等待循环取到的完成结果。三者的用途和生命周期不同。

## 3. 实现分析

### 3.1 _run_completion：先提交，再收集

```python
# 文件：vllm/entrypoints/offline_utils.py
# 类：OfflineInferenceMixin
# 函数：_run_completion
# 源码范围：L326–L349（连续节选）
def _run_completion(
    self,
    prompts: PromptType | Sequence[PromptType],
    params: SamplingParams
    | PoolingParams
    | Sequence[SamplingParams | PoolingParams],
    output_type: type[_O],
    *,
    use_tqdm: bool | Callable[..., tqdm] = True,
    lora_request: Sequence[LoRARequest] | LoRARequest | None = None,
    priority: list[int] | None = None,
    tokenization_kwargs: dict[str, Any] | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
):
    self._add_completion_requests(
        prompts=prompts,
        params=params,
        use_tqdm=use_tqdm,
        lora_request=lora_request,
        priority=priority,
        tokenization_kwargs=tokenization_kwargs,
        mm_processor_kwargs=mm_processor_kwargs,
    )
    return self._run_engine(use_tqdm=use_tqdm, output_type=output_type)
```

对入口调用线程而言，两步有明确先后顺序。但后台 Engine Core 可以在第一个请求提交后开始工作，无需等入口完成整批提交。

### 3.2 将输入与参数对齐

`prompt_to_seq` 先把单个字符串、字典或一条 Token ID 列表包装成请求序列；已经是请求序列的输入继续保留批量形式。

```python
# 文件：vllm/entrypoints/offline_utils.py
# 类：OfflineInferenceMixin
# 函数：_params_to_seq
# 源码范围：L242–L256（连续节选）
def _params_to_seq(
    self,
    params: _P | Sequence[_P],
    num_requests: int,
) -> Sequence[_P]:
    if isinstance(params, Sequence):
        if len(params) != num_requests:
            raise ValueError(
                f"The lengths of prompts ({num_requests}) "
                f"and params ({len(params)}) must be the same."
            )

        return params

    return [params] * num_requests
```

一份参数对象可广播给所有输入；参数序列则必须与输入数量一致。广播使用 `[params] * num_requests`，各项引用同一对象，并没有在这一层克隆。请求执行参数的克隆发生在 `InputProcessor` 中。LoRA 参数和优先级也在此完成数量对齐。

```python
# 文件：vllm/entrypoints/offline_utils.py
# 类：OfflineInferenceMixin
# 函数：_add_completion_requests
# 源码范围：L303–L324（连续节选）
seq_prompts = prompt_to_seq(prompts)
seq_params = self._params_to_seq(params, len(seq_prompts))
seq_lora_requests = self._lora_request_to_seq(lora_request, len(seq_prompts))
seq_priority = self._priority_to_seq(priority, len(seq_prompts))

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
            desc="Rendering prompts",
        )
    ),
    params=seq_params,
    lora_requests=seq_lora_requests,
    priorities=seq_priority,
)
```

传给 `_render_and_add_requests` 的 `prompts` 是生成器。循环每取出一项，才调用一次 `_preprocess_cmpl_one`，随后立即提交这一条输入；因此预处理和提交在入口侧逐条交替发生。

### 3.3 逐条提交与异常回收

```python
# 文件：vllm/entrypoints/offline_utils.py
# 类：OfflineInferenceMixin
# 函数：_render_and_add_requests
# 源码范围：L523–L550（连续节选）
def _render_and_add_requests(
    self,
    prompts: Iterable[EngineInput],
    params: Sequence[SamplingParams | PoolingParams],
    *,
    lora_requests: Sequence[LoRARequest | None] | None = None,
    priorities: Sequence[int] | None = None,
) -> list[str]:
    added_request_ids: list[str] = []

    try:
        for i, prompt in enumerate(prompts):
            request_id = self._add_request(
                prompt,
                params[i],
                lora_request=self._resolve_mm_lora(
                    prompt,
                    None if lora_requests is None else lora_requests[i],
                ),
                priority=0 if priorities is None else priorities[i],
            )
            added_request_ids.append(request_id)
    except Exception as e:
        if added_request_ids:
            self.llm_engine.abort_request(added_request_ids, internal=True)
        raise e

    return added_request_ids
```

成功添加的请求 ID 被逐条保存。如果后续某条预处理或提交抛出异常，这个函数会取消已经添加成功的请求，再把异常交还调用者。

这里使用 `internal=True`：`LLMEngine.add_request` 返回的是经过内部标识处理后的 ID。普通文本下 `_resolve_mm_lora` 会返回传入的 LoRA 配置；多模态适配器选择是扩展分支。

### 3.4 设置 FINAL_ONLY，分配外部 ID

```python
# 文件：vllm/entrypoints/offline_utils.py
# 类：OfflineInferenceMixin
# 函数：_add_request
# 源码范围：L552–L571（连续节选）
def _add_request(
    self,
    prompt: EngineInput,
    params: SamplingParams | PoolingParams,
    lora_request: LoRARequest | None = None,
    priority: int = 0,
) -> str:
    if isinstance(params, SamplingParams):
        # We only care about the final output
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

这段实现直接修改传入 `SamplingParams.output_kind`，没有先克隆，因此共享参数对象也会受到这个修改。它要求离线生成只返回最终 `RequestOutput`；内部仍持续生成与处理 Token。

计数器产生的字符串如 `"0"`、`"1"` 是入口分配的外部请求 ID。进入 `LLMEngine` 后，还会生成供核心和状态表使用的内部 ID。两种 ID 的对应关系见第 04、05 篇。

### 3.5 等待循环：完成后收集，最后排序

下面只截取结果收集主线，进度条与速度统计留在原文件：

```python
# 文件：vllm/entrypoints/offline_utils.py
# 类：OfflineInferenceMixin
# 函数：_run_engine
# 源码范围：L591–L599（连续节选）
outputs: list[_O] = []
total_in_toks = 0
total_out_toks = 0
while self.llm_engine.has_unfinished_requests():
    step_outputs = self.llm_engine.step()
    for output in step_outputs:
        assert isinstance(output, output_type)
        if output.finished:
            outputs.append(output)  # type: ignore[arg-type]
```

`LLMEngine.step()` 可以返回空列表；例如 `FINAL_ONLY` 模式下，当前处理到的请求还未完成。循环继续与否依据引擎的未完成状态，不能由“本轮没有用户输出”推断执行已经结束。

收集结束后，函数使用外部请求 ID 恢复输入顺序：

```python
# 文件：vllm/entrypoints/offline_utils.py
# 类：OfflineInferenceMixin
# 函数：_run_engine
# 源码范围：L626–L626（连续节选）
return sorted(outputs, key=lambda x: int(x.request_id))
```

`RequestOutput.request_id` 已由输出处理恢复为外部 ID，所以这里可转换为整数排序。这个实现依赖 `LLM` 离线入口使用递增数字 ID，不能直接推广成所有 API 的通用请求排序算法。

## 4. 手工推演：B 比 A 更早完成

```text
输入顺序：A、B
入口外部 ID：A="0"，B="1"

提交 A → 提交 B → 进入收集循环
某轮：B 完成，收集 RequestOutput("1")
后续：A 完成，收集 RequestOutput("0")
排序：返回 [A 的结果，B 的结果]
```

完成顺序由执行进度决定，返回列表仍与输入对应。如果提交 B 前的预处理失败，A 已提交的请求会进入取消清理流程，本次调用向用户抛出异常。

## 5. 设计原因与自测

批量归一化减少后续函数对单条与多条输入的重复判断；逐条渲染并提交减少等待整批预处理的要求；请求 ID 将乱序完成与有序返回连接起来；异常回收处理部分提交成功的情形。

读完后应能解释：生成器在哪一步被消费？为什么共享参数会被设为 `FINAL_ONLY`？本轮没有 `RequestOutput` 时为什么仍需继续循环？

[章节目录](../README.md) · [上一篇：LLM 与初始化](01-LLM-and-Initialization.md) · [下一篇：Renderer 与输入表示](03-Renderer-and-EngineInput.md)
