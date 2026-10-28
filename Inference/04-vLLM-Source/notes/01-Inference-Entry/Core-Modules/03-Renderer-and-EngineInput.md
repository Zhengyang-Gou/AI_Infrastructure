# 03 · Renderer 与输入表示

本篇跟踪输入处理的第一道边界：用户传入的文本或 Token IDs，如何变成统一的 `EngineInput`。

## 1. 职责与边界

Overview 中的 Renderer 表示输入渲染组件；本地源码的公共基类名为 `BaseRenderer`，具体类型由配置和注册表选择。普通文本路径的大部分处理逻辑就在基类中。

```text
用户 Prompt
    → parse_model_prompt：解析并统一字典形式
    → render_cmpl
        → render_prompts：准备输入表示
        → tokenize_prompts：需要时分词
        → process_for_engine：构造引擎输入
    → EngineInput
```

输入解析与 renderer 协作完成内容转换。请求 ID、采样参数和执行合法性继续交给 `InputProcessor` 与 `LLMEngine`；这里还未创建一次完整的核心执行请求。

## 2. 源码定位与数据表示

| 文件 | 类或作用域 | 重点内容 |
| --- | --- | --- |
| [entrypoints/offline_utils.py](../../../vllm/vllm/entrypoints/offline_utils.py) | `OfflineInferenceMixin` | `_preprocess_cmpl`、`_preprocess_cmpl_one` |
| [renderers/inputs/preprocess.py](../../../vllm/vllm/renderers/inputs/preprocess.py) | 模块级函数 | `parse_model_prompt`、`parse_dec_only_prompt` |
| [renderers/base.py](../../../vllm/vllm/renderers/base.py) | `BaseRenderer` | `render_cmpl`、`_tokenize_singleton_prompt`、`_tokenize_prompt`、`_process_tokens`、`process_for_engine` |
| [renderers/registry.py](../../../vllm/vllm/renderers/registry.py) | 模块级函数、`RendererRegistry` | `renderer_from_config`、具体 renderer 的选择 |
| [inputs/llm.py](../../../vllm/vllm/inputs/llm.py) | 类型定义 | `PromptType`、`TextPrompt`、`TokensPrompt` |
| [inputs/engine.py](../../../vllm/vllm/inputs/engine.py) | 类型定义、模块级函数 | `EngineInput`、`TokensInput`、`tokens_input` |

| 表示 | 表达的内容 | 普通文本例子（示意） |
| --- | --- | --- |
| `PromptType` | 用户可传入的多种输入形式 | 字符串 `"Hello"` |
| `TextPrompt` | 标准字典形式的文本输入 | `prompt` 字段保存文本 |
| `TokensPrompt` | 已获得 Token IDs 的输入 | `prompt_token_ids`，可能还保留原文 |
| `TokensInput` | 带类型标识的引擎输入 | `type="token"`、Token IDs、到达时间等 |
| `EngineInput` | 引擎输入类型的联合 | 普通文本路径中的具体值是 `TokensInput` |

`EngineInput` 在这条路径中是带字段的输入表示，并不是已经批处理好的 GPU 张量。Token ID 也取决于实际 tokenizer，下面推演不假定某段文字对应固定整数。

## 3. 实现分析

### 3.1 Mixin 准备解析结果与分词参数

```python
# 文件：vllm/entrypoints/offline_utils.py
# 类：OfflineInferenceMixin
# 函数：_preprocess_cmpl
# 源码范围：L124–L143（连续节选）
renderer = self.renderer
model_config = self.model_config

parsed_prompts = [
    parse_model_prompt(model_config, prompt) for prompt in prompts
]
tok_params = renderer.default_cmpl_tok_params.with_kwargs(
    **(tokenization_kwargs or {})
)
prompt_extras = (
    None
    if mm_processor_kwargs is None
    else {"mm_processor_kwargs": mm_processor_kwargs}
)

return renderer.render_cmpl(
    parsed_prompts,
    tok_params,
    prompt_extras=prompt_extras,
)
```

这一步先根据模型类型解析 prompts，再从 renderer 的默认参数派生本次分词参数。`tokenization_kwargs` 可以覆盖分词选项；`mm_processor_kwargs` 则通过额外字段继续向后传递。

普通 decoder-only 输入的解析包括：字符串转换成 `TextPrompt`，整数列表转换成 `TokensPrompt`，字典则检查其包含受支持的输入字段。encoder-decoder 模型会进入对应的解析分支。

`_preprocess_cmpl_one` 将单条输入包装成长度为一的序列，复用上面的批量方法，再取回唯一结果。它是上一篇生成器逐条渲染时的调用入口。

### 3.2 render_cmpl 组织三步转换

```python
# 文件：vllm/renderers/base.py
# 类：BaseRenderer
# 函数：render_cmpl
# 源码范围：L985–L1006（连续节选）
def render_cmpl(
    self,
    prompts: Sequence[DictPrompt | bytes],
    tok_params: TokenizeParams | None = None,
    *,
    prompt_extras: dict[str, Any] | None = None,
    skip_mm_cache: bool = False,
):
    arrival_time = time.time()

    if tok_params is None:
        tok_params = self.default_cmpl_tok_params

    dict_prompts = self.render_prompts(prompts)
    tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

    self._apply_prompt_extras(tok_prompts, prompt_extras)

    return [
        self.process_for_engine(prompt, arrival_time, skip_mm_cache=skip_mm_cache)
        for prompt in tok_prompts
    ]
```

这一层的执行顺序是先准备所有传入本次 renderer 调用的 prompts，再分词、附加额外信息、构造 `EngineInput`。在离线生成主线中，上游通常通过 `_preprocess_cmpl_one` 每次传一条。

`arrival_time` 在进入渲染时记录，随后传递给请求处理和统计。它不是 GPU 开始执行的时间。

### 3.3 已有 Token IDs 时可以复用

```python
# 文件：vllm/renderers/base.py
# 类：BaseRenderer
# 函数：_tokenize_singleton_prompt
# 源码范围：L516–L536（连续节选）
def _tokenize_singleton_prompt(
    self,
    prompt: SingletonDictPrompt,
    params: TokenizeParams,
) -> SingletonTokPrompt:
    if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
        if not isinstance(prompt.get("prompt"), str):
            raise TypeError(
                "Expected prompt['prompt'] to be a string before tokenization; "
                "use 'prompt_token_ids' for token ID inputs"
            )
        prompt = params.apply_pre_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]
        prompt = self._tokenize_prompt(prompt, params)

    if params.needs_detokenization and "prompt" not in prompt:
        if "prompt_token_ids" not in prompt:
            raise RuntimeError("Cannot run detokenization on embeddings")

        prompt = self._detokenize_prompt(prompt)  # type: ignore[arg-type]

    return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]
```

只有既没有 `prompt_token_ids`、也没有 `prompt_embeds` 的输入，才需要把文本交给 tokenizer。已有 Token IDs 的路径仍可能执行前后处理或按需还原原文，所以“跳过文本分词”并不意味着整个 renderer 都被跳过。

真正对文本编码的代码在 `_tokenize_prompt` 中：

```python
# 文件：vllm/renderers/base.py
# 类：BaseRenderer
# 函数：_tokenize_prompt
# 源码范围：L472–L487（连续节选）
def _tokenize_prompt(
    self,
    prompt: TextPrompt,
    params: TokenizeParams,
) -> TokensPrompt:
    tokenizer = self.get_tokenizer()
    want_offsets = self._wants_offsets(prompt, params)
    kwargs = params.get_encode_kwargs()
    if want_offsets:
        kwargs = {**kwargs, "return_offsets_mapping": True}
    encoding = tokenizer(prompt["prompt"], **kwargs)
    return self._build_tokens_prompt(
        encoding["input_ids"],
        prompt,
        offset_mapping=encoding["offset_mapping"] if want_offsets else None,
    )
```

编码结果中的 `input_ids` 被放入 `TokensPrompt`；按需计算的字符偏移也可以一同保存。特殊 Token、截断和大小写等行为由 tokenizer 与 `TokenizeParams` 共同决定。分词后，Token 数才是后续长度校验与 Prefill 工作量的主要依据。

### 3.4 _process_tokens 创建带类型的输入

```python
# 文件：vllm/renderers/base.py
# 类：BaseRenderer
# 函数：_process_tokens
# 源码范围：L769–L803（连续节选）
def _process_tokens(
    self,
    prompt: TokensPrompt,
    *,
    skip_mm_cache: bool = False,
) -> TokensInput | MultiModalInput:
    """Process token inputs, with multimodal preprocessing offloaded
    to the shared thread pool in the async variant.
    """
    prompt_token_ids = prompt["prompt_token_ids"]

    engine_input: TokensInput | MultiModalInput
    if multi_modal_data := prompt.get("multi_modal_data"):
        engine_input = self._process_multimodal(
            prompt_token_ids,
            multi_modal_data,
            mm_processor_kwargs=prompt.get("mm_processor_kwargs"),
            tokenization_kwargs=None,  # Tokenization already done in Step 2
            mm_uuids=prompt.get("multi_modal_uuids"),
            skip_mm_cache=skip_mm_cache,
        )
    else:
        engine_input = tokens_input(prompt_token_ids)

    if prompt_text := prompt.get("prompt"):
        engine_input["prompt"] = prompt_text
    if cache_salt := prompt.get("cache_salt"):
        engine_input["cache_salt"] = cache_salt
    # Narrow the union — `prompt_token_offsets` is only on TokensInput.
    if engine_input["type"] == "token" and (
        (offsets := prompt.get("prompt_token_offsets")) is not None
    ):
        engine_input["prompt_token_offsets"] = offsets

    return engine_input
```

普通文本分支调用 `tokens_input(prompt_token_ids)`，得到带有 `type="token"` 的字典；有多模态数据时则进入多模态处理路径。原始文本、缓存隔离盐和字符偏移按存在情况继续保留。

`tokens_input` 是模块级构造函数，不属于 `BaseRenderer`：

```python
# 文件：vllm/inputs/engine.py
# 类：无（模块级函数）
# 函数：tokens_input
# 源码范围：L53–L70（连续节选）
def tokens_input(
    prompt_token_ids: list[int],
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
) -> TokensInput:
    """
    Construct [`TokensInput`][vllm.inputs.engine.TokensInput]
    from optional values.
    """
    inputs = TokensInput(type="token", prompt_token_ids=prompt_token_ids)

    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt

    return inputs
```

这说明 `TextPrompt`、`TokensPrompt` 和 `TokensInput` 虽然都可以表现为字典，但字段契约及面向的下游阶段不同。

### 3.5 process_for_engine 补充引擎侧公共信息

```python
# 文件：vllm/renderers/base.py
# 类：BaseRenderer
# 函数：process_for_engine
# 源码范围：L945–L960（连续节选）
def process_for_engine(
    self,
    prompt: TokPrompt,
    arrival_time: float,
    *,
    skip_mm_cache: bool = False,
) -> EngineInput:
    engine_input: EngineInput
    if "encoder_prompt" in prompt:
        engine_input = self._process_enc_dec(prompt, skip_mm_cache=skip_mm_cache)  # type: ignore[arg-type]
    else:
        engine_input = self._process_singleton(prompt, skip_mm_cache=skip_mm_cache)

    engine_input["arrival_time"] = arrival_time

    return engine_input
```

decoder-only 路径通过 `_process_singleton` 进入 Token 或 embedding 分支；encoder-decoder 路径则构造相应的双侧输入。到达时间最终写入返回的 `EngineInput`。

## 4. 手工推演：文本输入与 Token 输入汇合

```text
路径 A：字符串 "Hello"
    → TextPrompt
    → tokenizer 编码，得到 IDs（由模型 tokenizer 决定）
    → TokensPrompt → TokensInput

路径 B：用户直接提供同一组合法 IDs
    → TokensPrompt
    → 复用 IDs，并执行需要的参数处理
    → TokensInput

汇合：两者都可以交给 InputProcessor
    → 但是否保留原文、特殊 Token 与分词参数是否一致，需要分别检查
```

`generate` 接收普通 prompts；`chat` 则先经过消息与模板处理，再进入相应的输入转换流程。聊天模板会影响实际 Token 序列，因此同一句可见文本不保证得到相同的模型输入。

## 5. 设计原因与自测

将内容转换独立成 renderer，使不同输入形式可以汇合到统一的下游接口；模型相关的 tokenizer 与多模态处理器也可以在这一层选择和复用。执行请求的校验和参数封装则继续由下一道边界处理。

读完后应能解释：已有 Token IDs 时哪些处理仍会发生？为什么 `EngineInput` 还不是核心请求？原文、Token IDs 与到达时间分别在哪一步产生或传递？

[章节目录](../README.md) · [上一篇：离线推理流程编排](02-Offline-Inference-Flow.md) · [下一篇：InputProcessor 与请求构造](04-InputProcessor-and-EngineCoreRequest.md)
