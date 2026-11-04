# 05 · 基础预热与就绪检查

## 1. 职责与边界

本篇输入是已经加载模型和实际 KV 存储的 Worker，负责 eager 基础预热、必要缓冲准备和资源就绪验收。不展开编译、图捕获/重放、采样算法或真实请求执行。

enforce_eager=True 关闭的是指定的图/编译执行路径，不代表跳过所有启动探测、dummy 调用或算子自身的首次初始化。

## 2. 源码定位与观察对象

| 文件 | 重点 |
| --- | --- |
| [gpu_worker.py](../../../vllm/vllm/v1/worker/gpu_worker.py) | compile_or_warm_up_model |
| [gpu_model_runner.py](../../../vllm/vllm/v1/worker/gpu_model_runner.py) | _dummy_run、_dummy_sampler_run 的接口 |
| [kernel_warmup.py](../../../vllm/vllm/model_executor/warmup/kernel_warmup.py) | kernel_warmup 的启动调用入口 |

本篇只说明何时需要这些调用、准备什么资源；dummy 输入如何打包、Attention 怎么执行、sampler 怎么选择 token 都停在接口。

## 3. 实现分析

### 3.1 eager 路径仍会准备基础算子

```python
# 文件：vllm/v1/worker/gpu_worker.py
# 类：Worker
# 函数：compile_or_warm_up_model
# 源码范围：L711–L717（连续节选）
# Warmup and tune the kernels used during model execution before
# cuda graph capture.
kernel_warmup(self)

cuda_graph_memory_bytes = 0
if not self.model_config.enforce_eager:
    cuda_graph_memory_bytes = self.model_runner.capture_model()
```

kernel_warmup 仍会被调用；capture_model 则被 enforce_eager 条件隔开。函数名包含 compile，并不意味着固定配置一定进行了 torch.compile 或 CUDA Graph 捕获。

源码前面按编译配置构建 warmup_sizes，那些专用预热尺寸与图优化不在本篇展开。

### 3.2 V1 路线准备末端缓冲

单卡 PP=1 是最后一段，固定 V1 Runner 路线会选择不超过 max_num_seqs 和 max_num_batched_tokens 的预热请求规模：

```python
# 文件：vllm/v1/worker/gpu_worker.py
# 类：Worker
# 函数：compile_or_warm_up_model
# 源码范围：L802–L816（连续节选）
max_num_reqs = min(
    self.scheduler_config.max_num_seqs,
    self.scheduler_config.max_num_batched_tokens,
)

# We skip EPLB here since we don't want to record dummy metrics
hidden_states, last_hidden_states = self.model_runner._dummy_run(
    num_tokens=max_num_reqs,
    skip_eplb=True,
    cudagraph_runtime_mode=CUDAGraphMode.NONE,
)
if self.model_runner.is_pooling_model:
    self.model_runner._dummy_pooler_run(hidden_states)
else:
    self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)
```

普通生成模型走 _dummy_sampler_run。它在此用于触发所需初始化并准备最大形状相关缓冲，减少真实请求第一次执行时的额外工作；本篇不展开 logits 到 token 的算法。

这里得到的 hidden states 是临时预热数据，不登记用户请求，不向用户返回生成结果。

### 3.3 恢复随机状态并完成启动收尾

```python
# 文件：vllm/v1/worker/gpu_worker.py
# 类：Worker
# 函数：compile_or_warm_up_model
# 源码范围：L818–L820（连续节选）
# Reset the seed to ensure that the random state is not affected by
# the model initialization and profiling.
set_random_seed(self.model_config.seed)
```

模型构造、探测和预热可能使用随机数，因此重新设置 seed，防止这些启动工作改变后续的随机数起点。这不等于保证所有并发环境、后端与设备间的逐位一致。

方法末尾还启用配置对应的 JIT 监控/同步检查，处理 GC 相关启动设置，并返回 CompilationTimes。该返回值是计时结果，不是用户输出，也不是全部内存配置的替代品。

## 4. 本章就绪检查

以下检查对象只用于核对启动结果，不重新推演 EngineCore 的初始化编排。

| 观察位置 | 预期证据 | 不能据此替代的检查 |
| --- | --- | --- |
| Worker.device 与 model_runner 类型 | 设备正确，固定主线选中 V1 Runner | checkpoint 参数是否全部加载 |
| Loader 的装载跟踪结果 | 必需参数按模型规则装载成功 | 仅凭“目录里有权重文件” |
| Runner.model | 实例存在，加载模板返回 eval 模式模型 | 仅凭 HF config 读成功 |
| determine_available_memory 返回值 | 字节预算可交给容量规划 | 仅凭 GPU 总显存够大 |
| KV 分配与 bind_kv_cache | 各需缓存的层取得对应存储引用 | 仅凭 num_blocks 已赋值 |
| compile_or_warm_up_model 返回 | 基础预热和末端缓冲准备已执行 | 真实请求正确性与性能基准 |

静态读源码时沿这些位置记录变量；具备可运行环境时再通过断点或日志检查实际值。此表不声称已经在当前机器完成 GPU 运行。

## 5. 设计原因与自测

- enforce_eager=True 为什么仍有 kernel_warmup 和 dummy 调用？
- profile_run 与资源就绪后的基础预热分别解决什么问题？
- 为什么最后要恢复随机种子？
- KV 已绑定与某个 token 已写入 KV 有何区别？
- 本章结束后，哪类“资源已经就绪”的证据应该能单独列出？

[章节目录](../README.md) · [本章概览](../Overview.md)
