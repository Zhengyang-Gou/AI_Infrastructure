# 00 · 固定配置与阅读约定

整套笔记使用以下配置，后续查阅运行方式和参数时回到本页。

## 1. 运行配置

| 项目 | 设置 |
| --- | --- |
| 接口 | 离线 `LLM.generate(prompts)`，一次提交一批输入，每条生成一个候选（n=1） |
| 模型 | 本地普通 Llama 文本模型，原生 vLLM 实现，safetensors 权重 |
| 设备 | 单张 CUDA GPU，TP=1，PP=1 |
| 核心连接 | InprocClient，入口和核心位于同一进程 |
| 执行器与 Worker | UniProcExecutor → WorkerWrapperBase → GPU Worker |
| 模型执行 | V1 GPUModelRunner |
| 调度 | 基础 Scheduler，同步调度，一批计算在途 |
| 前缀缓存与 Prompt 分块 | enable_prefix_caching=True，enable_chunked_prefill=True |
| 编译与图执行 | enforce_eager=True，关闭 torch.compile 与 CUDA Graph |
| 其他模型设置 | 使用默认权重精度，不使用量化、LoRA、推测模型与 CPU offload |

一批计算可以包含多条请求；请求在轮次之间加入或退出执行集合，形成连续批处理。

## 2. 初始化代码

`model` 换成包含模型配置、tokenizer 和权重的本地目录。

```python
# 文件：使用示例（非仓库源码）
# 类：无
# 函数：无（模块级示例）
import os

# 在导入 vLLM 前设置运行方式。
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

from vllm import LLM

llm = LLM(
    model="/path/to/local/llama-model",
    model_impl="vllm",
    load_format="safetensors",
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    distributed_executor_backend="uni",
    async_scheduling=False,
    enforce_eager=True,
    max_model_len=2048,
    max_num_seqs=4,
    max_num_batched_tokens=256,
    enable_chunked_prefill=True,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.8,
)
```

| 参数 | 值 | 含义 |
| --- | --- | --- |
| max_model_len | 2048 | 单请求允许的总长度 |
| max_num_seqs | 4 | 运行请求数量上限 |
| max_num_batched_tokens | 256 | 每轮 token 预算相关配置 |
| gpu_memory_utilization | 0.8 | 引擎计划使用的总显存比例 |
| kv_cache_memory_bytes | 未指定 | 通过显存测算确定 KV 预算 |

这些数值可按模型和设备调整。容量计算中的具体数字会在相应例子中列出。

## 3. 设置对应的实现

| 设置 | 对应结果 |
| --- | --- |
| VLLM_ENABLE_V1_MULTIPROCESSING=0 | EngineCoreClient 创建 InprocClient |
| distributed_executor_backend="uni" | Executor.get_class 返回 UniProcExecutor |
| VLLM_USE_V2_MODEL_RUNNER=0 | Worker 创建 vllm/v1/worker/gpu_model_runner.py 中的 GPUModelRunner |
| async_scheduling=False | 创建基础 Scheduler；PP=1 时 max_concurrent_batches=1 |
| enforce_eager=True | 编译模式与 CUDA Graph 模式为 NONE，仍执行显存测算、算子及采样预热 |

## 4. 源码阅读约定

源码以仓库内 `04-vLLM-Source/vllm/` 为依据。代码块开头的文件路径相对于这个目录，并标注类、函数和本地行范围。

源码节选保持连续，只调整共同缩进。使用示例、流程和计算示意会单独标明。

正文按实际调用顺序解释输入、状态变化与下一步，必要术语用简明中文说明。固定配置统一在本页维护。

[返回总目录](README.md) · [全局执行路线](Overview.md) · [开始第 01 章](01-Engine-Initialization/README.md)
