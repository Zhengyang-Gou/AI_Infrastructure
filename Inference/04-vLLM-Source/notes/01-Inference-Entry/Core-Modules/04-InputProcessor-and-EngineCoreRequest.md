# 04 · InputProcessor 与请求构造

本篇跟踪输入处理的第二道边界：将已经准备好的 `EngineInput` 与请求参数结合，校验并构造 `EngineCoreRequest`。

## 1. 职责与边界

`InputProcessor` 持有模型、调度、缓存和生成相关配置，判断当前输入与参数能否用于执行。生成主线由 `LLMEngine.add_request` 调用它。

```text
EngineInput + 外部请求 ID + SamplingParams
    → process_inputs
        → 校验任务、生成参数与模型输入
        → 克隆并补全执行侧参数
        → 构造 EngineCoreRequest
    → LLMEngine 调用 assign_request_id
    → 可以注册输出状态并提交核心
```

内容转换和请求构造分成两步：renderer 解决输入形式，processor 结合模型约束与生成意图建立请求。

## 2. 源码定位与关键数据

| 文件 | 类 | 重点内容 |
| --- | --- | --- |
| [v1/engine/input_processor.py](../../../vllm/vllm/v1/engine/input_processor.py) | `InputProcessor` | `process_inputs`、`_validate_params`、`_validate_model_inputs`、`_validate_model_input`、`_validate_prompt_len`、`assign_request_id` |
| [v1/engine/__init__.py](../../../vllm/vllm/v1/engine/__init__.py) | `EngineCoreRequest` | 核心请求的字段契约与 `params` 属性 |
| [sampling_params.py](../../../vllm/vllm/sampling_params.py) | `SamplingParams` | `clone`、`verify`、`update_from_generation_config`、`update_from_tokenizer` |

| 核心请求字段 | 意义 |
| --- | --- |
| `request_id`、`external_req_id` | 内部执行标识与外部请求标识；在内部 ID 分配前后发生变化 |
| `prompt_token_ids`、`prompt_embeds` | 模型输入内容；普通文本主线使用 Token IDs |
| `sampling_params`、`pooling_params` | 经过处理的任务参数，两类任务使用不同参数 |
| `arrival_time` | 沿输入路径传来的到达时间 |
| `priority`、`data_parallel_rank` | 调度优先级与可选的执行路由信息 |
| `cache_salt`、`lora_request` | 缓存隔离及适配器信息 |
| `mm_features` | 多模态扩展信息；本篇普通文本路径中为空 |

`EngineCoreRequest` 使用 `msgspec.Struct` 定义，是跨引擎边界传递的结构。进入核心后还会形成调度侧的 `Request`；两者承担的职责不同，后者在 Engine Core 章节展开。

## 3. 实现分析

### 3.1 先校验参数与输入语义

`process_inputs` 首先调用 `_validate_params` 和 `_validate_lora`，还会检查显式指定的数据并行 rank 是否在允许范围。

生成参数分支会检查模型支持的任务，并调用参数对象的验证方法：

```python
# 文件：vllm/v1/engine/input_processor.py
# 类：InputProcessor
# 函数：_validate_params
# 源码范围：L97–L109（连续节选）
if isinstance(params, SamplingParams):
    supported_generation_tasks = [
        task for task in supported_tasks if task in GENERATION_TASKS
    ]
    if not supported_generation_tasks:
        raise VLLMValidationError("This model does not support generation")

    params.verify(
        self.model_config,
        self.speculative_config,
        self.structured_outputs_config,
        self.tokenizer,
    )
```

参数合法性不只来自 `SamplingParams` 构造时的检查，还依赖当前模型、tokenizer 与引擎功能是否支持这些选项。

主线中传入的 prompt 已带 `type` 字段，processor 直接使用这个 `EngineInput`。源码仍保留原始 prompt 的兼容预处理分支，并给出弃用提示；理解当前入口时，应沿 renderer 已完成转换的路径追踪。

随后调用平台校验，拆分 encoder/decoder 输入，并进入 `_validate_model_inputs`。普通 decoder-only 文本主要检查 decoder 侧。

### 3.2 长度校验需要给生成留出空间

```python
# 文件：vllm/v1/engine/input_processor.py
# 类：InputProcessor
# 函数：_validate_prompt_len
# 源码范围：L396–L441（连续节选）
def _validate_prompt_len(
    self,
    prompt_len: int,
    prompt_type: Literal["encoder", "decoder"],
):
    if self.skip_prompt_length_check:
        return

    if prompt_len == 0 and prompt_type == "decoder":
        raise VLLMValidationError(f"The {prompt_type} prompt cannot be empty")

    model_config = self.model_config
    max_prompt_len = (
        model_config.max_model_len
        if prompt_type == "decoder"
        else self.mm_encoder_cache_size
    )
    if prompt_len > max_prompt_len:
        if self.supports_mm_inputs:
            suggestion = (
                "Make sure that `max_model_len` is no smaller than the "
                "number of text tokens plus multimodal tokens. For image "
                "inputs, the number of image tokens depends on the number "
                "of images, and possibly their aspect ratios as well."
            )
        else:
            suggestion = (
                "Make sure that `max_model_len` is no smaller than the "
                "number of text tokens."
            )

        raise VLLMValidationError(
            f"The {prompt_type} prompt (length {prompt_len}) is "
            f"longer than the maximum model length of {max_prompt_len}. "
            f"{suggestion}"
        )
    elif prompt_len == max_prompt_len and model_config.runner_type == "generate":
        suggestion = (
            "Make sure that `max_model_len` is no smaller than the "
            "number of text tokens (prompt + requested output tokens)."
        )
        raise VLLMValidationError(
            f"The {prompt_type} prompt (length {prompt_len}) plus the number of "
            f"requested output tokens (at least 1) is longer than the maximum "
            f"model length of {max_prompt_len}. {suggestion}"
        )
```

普通文本生成场景可以将这里的条件归纳为：

| Prompt 长度 | 处理结果 |
| --- | --- |
| 0 | decoder 输入为空，拒绝 |
| 大于 `max_model_len` | 超出模型长度，拒绝 |
| 等于 `max_model_len` | 生成任务没有空间再产生至少一个 Token，拒绝 |
| 小于 `max_model_len` | 通过本项检查，继续其他校验 |

`_validate_model_input` 还会检查 Token ID 是否超过 tokenizer 与模型词表允许的上界。多模态和 embedding 路径具有相应的输入长度计算方式。

这里直接检查的是 Prompt 长度及至少一个输出位置，并非对所有显式 `max_tokens` 都简单执行“Prompt 长度加请求输出长度超限就报错”。执行时仍受模型总长度上限约束，需要在后续调度章节继续追踪。

### 3.3 克隆参数并补全执行配置

```python
# 文件：vllm/v1/engine/input_processor.py
# 类：InputProcessor
# 函数：process_inputs
# 源码范围：L320–L339（连续节选）
sampling_params = None
pooling_params = None
if isinstance(params, SamplingParams):
    # TODO: can we avoid cloning here in multiproc case?
    sampling_params = params.clone()
    # If unset max tokens, then generate up to the max_model_len.
    if sampling_params.max_tokens is None:
        seq_len = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        sampling_params.max_tokens = self.model_config.max_model_len - seq_len

    sampling_params.update_from_generation_config(
        self.generation_config_fields,
        self.renderer.get_eos_token_id(),
    )
    if self.tokenizer is not None:
        sampling_params.update_from_tokenizer(self.tokenizer)
else:
    pooling_params = params.clone()
```

克隆之后，执行侧有了自己的参数对象。`clone` 默认使用深拷贝，`skip_clone=True` 时使用浅拷贝；它与上一篇 mixin 广播同一对象是不同步骤。

当 `max_tokens is None` 时，输出预算按模型长度减输入长度补齐。`update_from_generation_config` 在这份源码中主要补充 EOS 与停止 Token 相关配置；`update_from_tokenizer` 可将 bad words 等文本约束转成 Token 形式。这与 `LLM.get_default_sampling_params` 从默认生成配置创建参数对象有不同作用。

### 3.4 构造核心请求

经过验证、内容提取与参数补全后，函数返回：

```python
# 文件：vllm/v1/engine/input_processor.py
# 类：InputProcessor
# 函数：process_inputs
# 源码范围：L379–L394（连续节选）
return EngineCoreRequest(
    request_id=request_id,
    prompt_token_ids=prompt_token_ids,
    prompt_embeds=prompt_embeds,
    prompt_is_token_ids=prompt_is_token_ids,
    mm_features=mm_features,
    sampling_params=sampling_params,
    pooling_params=pooling_params,
    arrival_time=arrival_time,
    lora_request=lora_request,
    cache_salt=decoder_inputs.get("cache_salt"),
    priority=priority,
    data_parallel_rank=data_parallel_rank,
    trace_headers=trace_headers,
    resumable=resumable,
)
```

输入内容现在与请求标识、生成意图、到达时间和执行元数据组合在一起。processor 到这里仍未将请求插入调度队列，也没有执行模型；提交由 `LLMEngine` 继续完成。

### 3.5 为执行分配内部 ID

```python
# 文件：vllm/v1/engine/input_processor.py
# 类：InputProcessor
# 函数：assign_request_id
# 源码范围：L231–L249（连续节选）
@staticmethod
def assign_request_id(request: EngineCoreRequest):
    """Replace the externally supplied request ID with an internal request ID
    that adds 8 random characters in order to ensure uniqueness.
    """
    if request.external_req_id is not None:
        raise ValueError(
            "The external_req_id field should not be set on EngineCoreRequests"
            " passed to vLLM; use the request_id field."
        )
    request.external_req_id = request.request_id
    if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
        logger.warning_once(
            "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set and will be "
            "removed in a future release. Duplicate externally-provided "
            "request IDs may cause failures and/or subtle correctness errors."
        )
    else:
        request.request_id = f"{request.external_req_id}-{random_uuid():.8}"
```

这个静态方法属于 `InputProcessor`，但调用位置在 `LLMEngine.add_request` 中，发生在 `process_inputs` 返回之后。

```text
入口提交：request_id="0"，external_req_id=None
内部编号后：request_id="0-<随机后缀>"，external_req_id="0"
核心和输出状态表：使用内部 ID
最终 RequestOutput：恢复外部 ID "0"
```

默认内部编号用于减少重复外部 ID 对核心状态管理的干扰；源码也保留关闭随机化的环境开关。普通离线入口自己生成的外部 ID 已经递增，内部机制仍会统一应用。

## 4. 手工推演：长度与输出预算

假设普通文本 Prompt 编码后有 800 个 Token，`max_model_len=1024`，用户显式设置 `max_tokens=None`：

```text
800 < 1024 → 通过本项 Prompt 长度检查
克隆 SamplingParams
补全 max_tokens = 1024 - 800 = 224
构造 EngineCoreRequest
保留外部 ID，分配内部 ID
返回 LLMEngine 继续提交
```

如果 Prompt 已有 1024 个 Token，生成任务会在长度校验中被拒绝；如果前面 tokenizer 执行了截断，应以实际转换后的 Token 数判断。

## 5. 设计原因与自测

输入形式可以有很多种，但合法的执行请求需要共同遵守模型与任务约束。独立 processor 将这些验证集中在提交边界；参数克隆区分用户配置与请求执行配置；内外 ID 则连接用户可识别结果与执行侧唯一标识。

读完后应能解释：`process_inputs` 返回之后还缺少哪些提交动作？`assign_request_id` 谁定义、谁调用？`max_tokens=None` 与显式给定长度的处理有何不同？

[章节目录](../README.md) · [上一篇：Renderer 与输入表示](03-Renderer-and-EngineInput.md) · [下一篇：LLMEngine 与核心边界](05-LLMEngine-and-Core-Boundary.md)
