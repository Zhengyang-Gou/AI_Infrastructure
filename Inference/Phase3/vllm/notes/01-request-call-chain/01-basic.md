# Basic Offline Inference

## 1. 文件定位

- 文件路径：`examples/basic/offline_inference/basic.py`
- 所属层次：用户示例层
- 核心职责：展示如何创建 `LLM`、设置 `SamplingParams`、调用 `generate()` 并读取生成结果。
- 在调用链中的位置：整个离线推理请求的最外层入口。
- 官方源代码：[vLLM `basic.py`](https://github.com/vllm-project/vllm/blob/main/examples/basic/offline_inference/basic.py)

## 2. 核心对象

| 对象 | 作用 | 来源 | 传递给 |
| --- | --- | --- | --- |
| `prompts` | 保存一个或多个用户输入 | 用户代码 | `LLM.generate()` |
| `SamplingParams` | 描述温度、最大 token 数等生成参数 | 用户代码 | `LLM.generate()` |
| `LLM` | 提供离线推理 API 并持有底层引擎 | `vllm` | 用户代码 |
| `RequestOutput` | 保存一次请求的完整生成结果 | `LLM.generate()` | 用户代码 |

## 3. 主执行流程

```text
准备 prompts
→ 创建 SamplingParams
→ 初始化 LLM
→ 调用 LLM.generate()
→ 遍历 RequestOutput
→ 读取 prompt 和生成文本
```

这个示例只负责演示公开 API。请求如何被处理、调度和执行，分别由 `LLM`、`LLMEngine` 与 `EngineCore` 完成。

## 4. 输入与输出

### 输入

- 模型名称或本地模型路径。
- 一个 prompt 或由多个 prompt 组成的序列。
- 控制生成行为的 `SamplingParams`。

### 输出

`LLM.generate()` 返回 `RequestOutput` 列表。每个元素对应一个输入请求，并包含生成文本、token ID 和结束原因等信息。

### 状态变化

示例代码本身不维护请求状态。`LLM.generate()` 是同步接口，会持续运行，直到本批请求全部完成后再返回。

## 5. 关键代码解析

### `LLM(...)`

```python
llm = LLM(model="facebook/opt-125m")
```

`LLM` 是 vLLM 的同步离线推理入口。这里传入 Hugging Face 模型 ID；构造期间会解析模型配置、选择设备与执行后端、加载权重，并创建底层 `LLMEngine`。因此该语句通常是示例中开销最大的初始化步骤，而不是一次普通的轻量对象创建。

### `SamplingParams(...)`

```python
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
```

- `temperature=0.8` 对 logits 进行温度缩放；数值低于 1 会让分布更集中。
- `top_p=0.95` 使用 nucleus sampling，只在累计概率达到 0.95 的候选 token 集合内采样。

示例没有显式设置 `max_tokens` 等字段，其余行为采用当前 vLLM 版本和模型 generation config 合并后的默认值。同一个 `SamplingParams` 会应用到列表中的全部 prompt。

### `LLM.generate(...)`

```python
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
```

`generate()` 接收一批 prompt，将其转换为内部请求并加入引擎，然后由同步执行循环持续调度，直到本批请求全部完成。返回值是 `list[RequestOutput]`，顺序与输入 prompt 一致。

每个 `RequestOutput` 表示一个输入请求；`output.outputs` 是该请求的候选序列列表。示例使用默认候选数 `n=1`，所以读取 `outputs[0].text`。如果设置 `SamplingParams(n>1)`，应遍历 `output.outputs`，不能只取第一个候选。

## 6. 完整源代码

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    llm = LLM(model="facebook/opt-125m")
    outputs = llm.generate(prompts, sampling_params)

    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
```

## 7. 与其他文件的关系

- 上游：用户或业务代码。
- 下游：`vllm/entrypoints/llm.py`。
- 传递的数据：模型配置、prompt、采样参数。
- 返回的数据：`RequestOutput`。

## 8. 当前结论

`basic.py` 是用户视角的最小离线推理入口。它说明 vLLM 对外暴露什么 API，但不包含请求调度和模型执行的实现。
