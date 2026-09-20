# 05 · 模型执行器与输入组织

先认识一次前向的元数据，再看执行器如何组织输入、分配显存并执行模型。首次阅读可先抓住 prepare_prefill、prepare_decode 和 run，之后再回看多进程与 CUDA Graph。

对应源码：[context.py](../source/nano-vllm/nanovllm/utils/context.py)、[model_runner.py](../source/nano-vllm/nanovllm/engine/model_runner.py)。

## context.py

```python
@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
```
`Context` 是 `ModelRunner` 与每一层 `Attention` 之间的本轮推理元数据：

- Prefill 使用 `cu_seqlens_q/k` 和最大序列长度调用变长 FlashAttention
- Prefill、Decode 都使用 `slot_mapping` 写入 KV Cache
- Decode 使用 `context_lens` 和 `block_tables` 读取分页缓存
- 前缀缓存 Prefill 也会使用 `block_tables`

模块级 `_CONTEXT` 相当于一次前向期间的全局上下文。`set_context()` 在前向前设置，模型各层通过 `get_context()` 读取，`reset_context()` 在前向后清空。这样无需把大量缓存元数据逐层写进 `forward` 参数。

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
    dist.init_process_group(
        "nccl", "tcp://localhost:2333",
        world_size=self.world_size, rank=rank,
    )
    torch.cuda.set_device(rank)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(hf_config.dtype)
    torch.set_default_device("cuda")
    self.model = Qwen3ForCausalLM(hf_config)
    load_model(self.model, config.model)
    self.sampler = Sampler()
    self.warmup_model()
    self.allocate_kv_cache()
    if not self.enforce_eager:
        self.capture_cudagraph()
    torch.set_default_device("cpu")
    torch.set_default_dtype(default_dtype)
```
初始化 GPU 执行环境：

- 每个 rank 对应一张 GPU，使用 NCCL 建立张量并行进程组
- 临时把默认设备改为 CUDA、默认数据类型改为模型配置的 dtype，使模型参数直接创建在对应 GPU 上
- 构造模型并从 safetensors 加载本 rank 所需的权重分片
- 先预热并统计峰值显存，再使用剩余显存分配 KV Cache
- 非 Eager 模式继续捕获 Decode 的 CUDA Graph
- 最后恢复进程原来的默认设备和 dtype，避免影响其他代码

```python
if self.world_size > 1:
    if rank == 0:
        self.shm = SharedMemory(
            name="nanovllm", create=True, size=2**20
        )
        dist.barrier()
    else:
        dist.barrier()
        self.shm = SharedMemory(name="nanovllm")
        self.loop()
```
多卡时，rank 0 是控制进程，其余 rank 进入命令循环：

- 共享内存传递方法名和 Python 参数
- `Event` 通知工作进程共享内存中出现了新命令
- NCCL 负责模型内部的张量通信
- `barrier()` 保证其他 rank 只在共享内存创建完成后连接

```python
def read_shm(self):
    self.event.wait()
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args

def write_shm(self, method_name, *args):
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()

def call(self, method_name, *args):
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)
    method = getattr(self, method_name, None)
    return method(*args)
```
命令分发协议：

- 前 4 字节记录 pickle 数据长度，之后保存方法名和参数
- rank 0 写入并唤醒所有工作进程，然后自己也执行相同方法
- 其他 rank 读取后通过 `getattr()` 调用本地同名方法
- 因而所有 GPU 会以相同顺序执行 `run`、`exit` 等操作

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    seq_len = min(max_num_batched_tokens, max_model_len)
    num_seqs = min(
        max_num_batched_tokens // seq_len,
        self.config.max_num_seqs,
    )
    seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
    for seq in seqs:
        seq.num_scheduled_tokens = seq_len
    self.run(seqs, True)
    torch.cuda.empty_cache()
```
模型预热：

- 构造接近配置上限的虚拟 Prefill Batch，执行一次完整前向
- 触发 PyTorch/FlashAttention 的初始化、内核选择或编译
- `reset_peak_memory_stats()` 后执行预热，因此稍后可以读取模型运行时的峰值显存
- 清除可释放的临时缓存，为 KV Cache 留出空间

```python
def allocate_kv_cache(self):
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
    ...
    block_bytes = (
        2 * num_hidden_layers * block_size
        * num_kv_heads * head_dim * dtype.itemsize
    )
    config.num_kvcache_blocks = int(
        total * gpu_memory_utilization - used - peak + current
    ) // block_bytes
    self.kv_cache = torch.empty(
        2, num_hidden_layers, num_blocks,
        block_size, num_kv_heads, head_dim,
    )
```
计算并分配 KV Cache：

- 一个物理块必须包含所有 Transformer 层的 K 和 V，所以大小中有 `2 × num_hidden_layers`
- KV Head 已按张量并行规模切分，每个 rank 只分配自己的部分
- `peak - current` 是预热过程中出现过、但当前已经释放的临时显存；预算中需要为下一次前向重新预留
- 可用于 KV Cache 的空间近似为：
```
总显存 × 利用率 - 当前已用显存 - (峰值显存 - 当前 PyTorch 显存)
```
- 再除以单块字节数，得到可分配的物理块数量

整体 KV Cache 形状：
```
[K/V, layer, physical_block, token_in_block, kv_head, head_dim]
```

随后遍历每一层 `Attention`，让其 `k_cache`、`v_cache` 指向大张量中对应层的视图。

```python
def prepare_block_tables(self, seqs):
    max_len = max(len(seq.block_table) for seq in seqs)
    block_tables = [
        seq.block_table + [-1] * (max_len - len(seq.block_table))
        for seq in seqs
    ]
    return torch.tensor(
        block_tables, dtype=torch.int32, pin_memory=True
    ).cuda(non_blocking=True)
```
不同序列占用的块数不同，因此先用 `-1` 补齐成二维矩阵，再从锁页内存异步复制到 GPU。

`block_tables[b, i]` 表示 Batch 中第 `b` 条序列的第 `i` 个逻辑块位于哪个物理块。

```python
def prepare_prefill(self, seqs):
    input_ids, positions = [], []
    cu_seqlens_q, cu_seqlens_k = [0], [0]
    ...
    for seq in seqs:
        start = seq.num_cached_tokens
        seqlen_q = seq.num_scheduled_tokens
        end = start + seqlen_q
        seqlen_k = end
        input_ids.extend(seq[start:end])
        positions.extend(range(start, end))
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
```
组织 Prefill 输入：

- 多条变长序列不做二维 Padding，而是把本轮需要计算的 Token 拼成一维数组
- `cu_seqlens_q` 是各序列 Query 在扁平数组中的累积边界
- `cu_seqlens_k` 是各序列完整上下文的累积边界
- 无前缀缓存时 `seqlen_q == seqlen_k`
- 命中前缀缓存时，Q 只包含未缓存部分，K/V 的有效长度还要包括已缓存前缀，因此 `seqlen_k = end`

例如两条 Q 长度分别为 3 和 2：
```
扁平 Token: [A A A | B B]
cu_seqlens_q: [0, 3, 5]
```

```python
for i in range(start_block, end_block):
    slot_start = seq.block_table[i] * self.block_size
    ...
    slot_mapping.extend(range(slot_start, slot_end))
```
`slot_mapping` 把本轮每个新 Token 映射到扁平 KV Cache 槽位：
```
slot = physical_block_id × block_size + offset_in_block
```
Attention 层使用它将新计算出的 K/V 写入正确的物理位置。

如果 K 的累计长度大于 Q，说明存在复用前缀，此时额外传入 `block_tables`，让 FlashAttention 从 Paged KV Cache 中读取完整 K/V；最后把全部元数据写入全局 `Context`。

```python
def prepare_decode(self, seqs):
    for seq in seqs:
        input_ids.append(seq.last_token)
        positions.append(len(seq) - 1)
        context_lens.append(len(seq))
        slot_mapping.append(
            seq.block_table[-1] * self.block_size
            + seq.last_block_num_tokens - 1
        )
```
组织 Decode 输入：

- 每条序列只输入最后一个 Token
- 位置是当前序列长度减一
- `context_lens` 告诉 Attention 每条序列可读取多少个历史 Token
- `slot_mapping` 指向当前最后一个 Token 的 KV 写入位置
- `block_tables` 告诉 Attention 如何沿物理块读取全部历史 KV Cache

```python
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(
            self.model(input_ids, positions)
        )
    ...
    graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
    ...
    graph.replay()
    return self.model.compute_logits(graph_vars["outputs"][:bs])
```
选择执行方式：

- Prefill 形状变化大，直接 Eager 执行
- 强制 Eager 或 Decode Batch 超过捕获上限时，也直接执行
- 其他 Decode 请求选择不小于真实 Batch Size 的最小 CUDA Graph
- 把真实输入复制进 Graph 的静态缓冲区，剩余槽位通过 `slot_mapping=-1`、`context_lens=0` 屏蔽
- Replay 后只取前 `bs` 条真实输出计算 logits

```python
def run(self, seqs, is_prefill):
    input_ids, positions = (
        self.prepare_prefill(seqs)
        if is_prefill else self.prepare_decode(seqs)
    )
    temperatures = (
        self.prepare_sample(seqs) if self.rank == 0 else None
    )
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = (
        self.sampler(logits, temperatures).tolist()
        if self.rank == 0 else None
    )
    reset_context()
    return token_ids
```
一次模型执行的完整路径：

1. 准备 Prefill 或 Decode 输入
2. 只有 rank 0 准备温度并执行最终采样
3. 所有 rank 都执行模型前向和张量并行通信
4. 清空全局 Context，避免下一轮误用旧元数据
5. rank 0 把生成的 Token ID 返回调度器

```python
self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
for bs in reversed(self.graph_bs):
    graph = torch.cuda.CUDAGraph()
    set_context(...)
    outputs[:bs] = self.model(...)  # warmup
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(...)
    ...
```
`capture_cudagraph` 为一组固定 Batch Size 预先捕获 Decode 计算图：

- CUDA Graph 要求张量地址和形状固定，因此预先创建最大尺寸的输入、缓存元数据和输出缓冲区
- 小 Batch 使用 1、2、4、8，大 Batch 以 16 为间隔，兼顾显存占用与 Padding 浪费
- 从大到小捕获并共享 graph memory pool
- Replay 可以减少每个 Decode step 的 Python 和 CUDA Kernel Launch 开销

`exit()` 则负责关闭共享内存、同步 GPU、释放 CUDA Graph 并销毁 NCCL 进程组。

---

[学习目录](README.md) · [上一篇：调度与批处理](04-Scheduler-and-Batching.md) · [下一篇：Qwen3 模型与基础算子](06-Qwen3-and-Basic-Layers.md)
