# 05 · 模型执行器与输入组织

先认识一次前向的元数据，再看执行器如何组织输入、分配显存并执行模型。首次阅读可先抓住 prepare_prefill、prepare_decode 和 run，之后再回看多进程与 CUDA Graph。

对应源码：[context.py](../source/nano-vllm/nanovllm/utils/context.py)、[model_runner.py](../source/nano-vllm/nanovllm/engine/model_runner.py)。

## context.py

```python
@dataclass(slots=True)
class Context:
    # 标记当前执行阶段。
    is_prefill: bool = False
    # 扁平 Query 数组中各序列的累计边界。
    cu_seqlens_q: torch.Tensor | None = None
    # 各序列有效 Key 长度的累计边界，含缓存前缀。
    cu_seqlens_k: torch.Tensor | None = None
    # 本批次最大的 Query 长度。
    max_seqlen_q: int = 0
    # 本批次最大的 Key 长度。
    max_seqlen_k: int = 0
    # 本轮 Token 写入 KV Cache 的物理槽位。
    slot_mapping: torch.Tensor | None = None
    # Decode 中每条序列的有效上下文长度。
    context_lens: torch.Tensor | None = None
    # 逻辑块到物理块的映射，Decode 或缓存前缀 Prefill 使用。
    block_tables: torch.Tensor | None = None
```

**功能描述：** 保存一次前向所需的批量元数据，连接 ModelRunner 与各层 Attention。执行前通过 set_context 设置，各层通过 get_context 读取，结束后 reset_context 清空，避免逐层传递大量参数。

## model_runner.py

`ModelRunner` 连接调度系统和 GPU 模型，主要负责：

1. 初始化分布式环境、模型和权重
2. 根据显存余量分配 KV Cache
3. 把 `Sequence` 转换成 Prefill/Decode 所需的批量张量
4. 执行 Eager 或 CUDA Graph 推理
5. 在张量并行进程之间同步同一条命令

```python
def __init__(self, config, rank, event):
    ...
    # 建立 NCCL 通信组，每个 rank 对应一张 GPU。
    dist.init_process_group(
        "nccl", "tcp://localhost:2333",
        world_size=self.world_size, rank=rank,
    )
    torch.cuda.set_device(rank)
    # 保存原默认类型，再按模型配置创建 CUDA 参数。
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf_config.dtype)
    torch.set_default_device("cuda")
    self.model = Qwen3ForCausalLM(hf_config)
    # 加载当前 rank 所需的权重分片。
    load_model(self.model, config.model)
    self.sampler = Sampler()
    # 先预热并测量峰值显存，再分配 KV Cache。
    self.warmup_model()
    self.allocate_kv_cache()
    # 启用图模式时预先捕获固定形状的 Decode。
    if not self.enforce_eager:
        self.capture_cudagraph()
    # 将默认设备设回 CPU，并恢复此前的数据类型。
    torch.set_default_device("cpu")
    torch.set_default_dtype(default_dtype)
```

**功能描述：** 初始化当前 rank 的 GPU 执行环境，加载模型并按显存预算分配缓存，必要时捕获 Decode 计算图，为后续批次执行做好准备。

```python
if self.world_size > 1:
    if rank == 0:
        # rank 0 创建共享区域，供各工作进程读取命令。
        self.shm = SharedMemory(
            name="nanovllm", create=True, size=2**20
        )
        # 同步各 rank，保证连接前共享内存已经创建。
        dist.barrier()
    else:
        # 同步各 rank，保证连接前共享内存已经创建。
        dist.barrier()
        # rank 0 创建共享区域，供各工作进程读取命令。
        self.shm = SharedMemory(name="nanovllm")
        # 工作进程持续等待和执行主进程命令。
        self.loop()
```

**功能描述：** 建立多进程命令通道：rank 0 创建共享内存，其余 rank 连接后进入工作循环。命令参数通过共享内存传递，模型张量通信由 NCCL 负责。

```python
def read_shm(self):
    # 等待主进程发出新命令通知。
    self.event.wait()
    # 前 4 字节存储后续 pickle 数据的长度。
    n = int.from_bytes(self.shm.buf[0:4], "little")
    # 反序列化方法名和参数。
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    # 清除本次通知。
    self.event.clear()
    return method_name, args

# 反序列化方法名和参数。
def write_shm(self, method_name, *args):
    # 反序列化方法名和参数。
    # 将调用内容序列化到共享缓冲区。
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        # 通知每个工作进程读取命令。
        event.set()

# 反序列化方法名和参数。
def call(self, method_name, *args):
    # 多卡时仅 rank 0 负责广播命令。
    if self.world_size > 1 and self.rank == 0:
        # 反序列化方法名和参数。
        self.write_shm(method_name, *args)
    # 主进程也调用本地同名方法。
    method = getattr(self, method_name, None)
    return method(*args)
```

**功能描述：** 实现主进程向工作进程分发方法调用的协议，使各 rank 按一致顺序执行 run、exit 等操作。

```python
def warmup_model(self):
    # 释放可回收的分配器缓存，不会释放仍被引用的张量。
    torch.cuda.empty_cache()
    # 从本次预热开始统计峰值。
    torch.cuda.reset_peak_memory_stats()
    # 同时受批次 Token 预算和上下文长度约束。
    seq_len = min(max_num_batched_tokens, max_model_len)
    # 根据单条长度和并发上限计算批次大小。
    # 构造虚拟 Token 序列。
    num_seqs = min(
        max_num_batched_tokens // seq_len,
        self.config.max_num_seqs,
    )
    # 构造虚拟 Token 序列。
    seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
    for seq in seqs:
        # 将整个虚拟输入设为本轮计算范围。
        seq.num_scheduled_tokens = seq_len
    # 执行一次 Prefill；此时缓存尚未分配。
    self.run(seqs, True)
    # 释放可回收的分配器缓存，不会释放仍被引用的张量。
    torch.cuda.empty_cache()
```

**功能描述：** 用接近配置上限的虚拟 Prefill 批次预热模型并记录峰值显存，为 KV Cache 容量估算提供运行时开销依据。

```python
def allocate_kv_cache(self):
    # 查询设备当前空闲显存与总显存。
    free, total = torch.cuda.mem_get_info()
    used = total - free
    # 预热期间 PyTorch 的峰值已分配显存。
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    # 当前已分配显存；peak - current 作为临时开销预留。
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    ...
    # 一个物理块覆盖所有层的 K/V；KV Head 数已按 rank 切分。
    block_bytes = (
        2 * num_hidden_layers * block_size
        * num_kv_heads * head_dim * dtype.itemsize
    )
    # 可用预算除以每块字节数，向下取整。
    config.num_kvcache_blocks = int(
        total * gpu_memory_utilization - used - peak + current
    ) // block_bytes
    # 维度依次为 K/V、层、物理块、块内 Token、KV Head、Head 维度。
    self.kv_cache = torch.empty(
        2, num_hidden_layers, num_blocks,
        block_size, num_kv_heads, head_dim,
    )
```

**功能描述：** 在显存预算中扣除已有占用和预热测得的临时峰值开销，计算并分配当前 rank 的分页 KV Cache。随后各 Attention 层使用这块大张量中对应层的视图。

补充示意：

```
总显存 × 利用率 - 当前已用显存 - (峰值显存 - 当前 PyTorch 显存)
```

```
[K/V, layer, physical_block, token_in_block, kv_head, head_dim]
```

```python
def prepare_block_tables(self, seqs):
    # 以最长页表作为本批次的统一宽度。
    max_len = max(len(seq.block_table) for seq in seqs)
    block_tables = [
        # 较短页表用 -1 补齐无效位置。
        seq.block_table + [-1] * (max_len - len(seq.block_table))
        for seq in seqs
    ]
    return torch.tensor(
        # 使用 int32 页表和 CPU 锁页内存。
        block_tables, dtype=torch.int32, pin_memory=True
    # 发起到 GPU 的异步复制。
    ).cuda(non_blocking=True)
```

**功能描述：** 将不同长度的请求页表整理成 GPU 可读取的二维批量张量，供 Attention 定位每条序列的历史物理块。

```python
def prepare_prefill(self, seqs):
    input_ids, positions = [], []
    cu_seqlens_q, cu_seqlens_k = [0], [0]
    ...
    for seq in seqs:
        # 跳过已有 KV 的前缀，也适用于此前完成的 Prefill 分块。
        start = seq.num_cached_tokens
        # 本轮实际计算的 Query 数量。
        seqlen_q = seq.num_scheduled_tokens
        end = start + seqlen_q
        # Key 覆盖从序列起点到本轮结束位置的全部上下文。
        seqlen_k = end
        # 拼接各请求的输入，不做二维 Token Padding。
        input_ids.extend(seq[start:end])
        # 保留 Token 在原序列中的绝对位置。
        positions.extend(range(start, end))
        # 累加边界，例如长度 3、2 对应 [0, 3, 5]。
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        # Key 边界单独累计，可能与 Query 长度不同。
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
```

**功能描述：** 把变长 Prefill 输入拼成扁平 Token 数组，并生成位置和序列边界。Query 只覆盖本轮新增计算部分，Key 的有效范围包含已经缓存的历史。

补充示意：

```
扁平 Token: [A A A | B B]
cu_seqlens_q: [0, 3, 5]
```

```python
for i in range(start_block, end_block):
    # 槽位起点 = 物理块编号 × 每块 Token 数。
    slot_start = seq.block_table[i] * self.block_size
    ...
    # 加入本轮覆盖的块内位置，形成逐 Token 的写入映射。
    slot_mapping.extend(range(slot_start, slot_end))
```

**功能描述：** 为本轮新 Token 构造 KV 写入地址，将逻辑位置转换为扁平物理槽位。完整输入准备流程还会在需要读取缓存前缀时传入页表，并把元数据写入 Context。

补充示意：

```
slot = physical_block_id × block_size + offset_in_block
```

```python
def prepare_decode(self, seqs):
    for seq in seqs:
        # 只输入上轮生成的最后一个 Token。
        input_ids.append(seq.last_token)
        # 位置从 0 开始，因此为总长度减 1。
        positions.append(len(seq) - 1)
        # 当前前向可见的上下文长度，包含此次写入位置。
        context_lens.append(len(seq))
        # 末物理块起点加末块内偏移，得到写入位置。
        slot_mapping.append(
            seq.block_table[-1] * self.block_size
            + seq.last_block_num_tokens - 1
        )
```

**功能描述：** 组织 Decode 所需的单 Token 输入、位置、有效上下文长度和缓存写入槽位，使模型复用历史 KV 来计算下一个 Token。完整方法还会准备批量页表。

```python
def run_model(self, input_ids, positions, is_prefill):
    # Prefill、强制 Eager 或超过此处 512 阈值时直接前向。
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(
            self.model(input_ids, positions)
        )
    ...
    # 选择能容纳真实批次的最小已捕获图。
    graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
    ...
    # 重放前已更新静态输入，填充槽位使用 -1 等无效标记。
    graph.replay()
    # 只取前 bs 条真实序列的隐藏状态计算 logits。
    return self.model.compute_logits(graph_vars["outputs"][:bs])
```

**功能描述：** 根据阶段、配置和批次大小选择直接执行或 CUDA Graph 回放，并返回 logits。图模式使用固定缓冲区，完整实现会复制真实输入并屏蔽补齐位置，只读取有效输出。

```python
def run(self, seqs, is_prefill):
    # 按 Prefill 或 Decode 分支组织输入并设置 Context。
    input_ids, positions = (
        self.prepare_prefill(seqs)
        if is_prefill else self.prepare_decode(seqs)
    )
    # 仅 rank 0 需要采样温度。
    temperatures = (
        self.prepare_sample(seqs) if self.rank == 0 else None
    )
    # 所有 rank 参与模型计算和张量并行通信。
    logits = self.run_model(input_ids, positions, is_prefill)
    # rank 0 返回采样结果，其余 rank 返回 None。
    token_ids = (
        self.sampler(logits, temperatures).tolist()
        if self.rank == 0 else None
    )
    # 避免下一轮误用当前批次元数据。
    reset_context()
    return token_ids
```

**功能描述：** 串起一次模型执行：准备输入、完成各 rank 的前向计算，再由 rank 0 采样返回 Token ID，最后清空当前上下文。

```python
# 小批次取 1/2/4/8，较大批次按 16 的间隔捕获。
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
# 从大到小捕获，并共享图内存池。
for bs in reversed(self.graph_bs):
    graph = torch.cuda.CUDAGraph()
    set_context(...)
    # 捕获前先执行预热。
    outputs[:bs] = self.model(...)  # 预热
    # 记录模型执行到 CUDA Graph。
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(...)
    ...
```

**功能描述：** 为多种固定批次大小捕获 Decode 图，后续回放可减少 Python 调度和 CUDA Kernel 启动开销。固定输入、输出及缓存元数据缓冲区确保捕获和回放时地址与形状一致。

执行器退出时，`exit()` 负责关闭共享内存、同步 GPU、释放 CUDA Graph 并销毁 NCCL 进程组。

---

[学习目录](README.md) · [上一篇：调度与批处理](04-Scheduler-and-Batching.md) · [下一篇：Qwen3 模型与基础算子](06-Qwen3-and-Basic-Layers.md)
