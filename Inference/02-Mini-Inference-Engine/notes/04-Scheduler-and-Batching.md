# 04 · 调度与批处理

结合 Sequence 和 BlockManager，跟踪 Prefill、Decode、抢占和请求结束的状态变化。

对应源码：[scheduler.py](../source/nano-vllm/nanovllm/engine/scheduler.py)。

## scheduler.py

一个请求从加入系统到完成，大致经历：
```
add(seq)
   ↓
waiting 队列
   ↓
schedule()
   ↓
Prefill：处理输入 prompt
   ↓
running 队列
   ↓
Decode：每轮生成一个 token
   ↓
postprocess()
   ↓
遇到 EOS 或达到 max_tokens
   ↓
FINISHED
```
```python
class Scheduler:
    def __init__(self, config: Config):
        # 每轮序列数量上限。
        self.max_num_seqs = config.max_num_seqs
        # Prefill 每轮可用的 Token 预算。
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 生成此 Token 时可触发请求结束。
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        # 集中管理缓存容量、分配与前缀复用。
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
        )
        # 保存新请求、未完成 Prefill 和被抢占的请求。
        self.waiting: deque[Sequence] = deque()
        # 保存已可继续 Decode 的请求。
        self.running: deque[Sequence] = deque()
```

**功能描述：** 创建受序列数和 Token 预算约束的调度器，并维护等待、运行两个队列及共享的缓存块管理器。

```python
    def is_finished(self):
        # 等待队列与运行队列必须同时为空。
        return not self.waiting and not self.running
```

**功能描述：** 判断整个调度器是否已无待处理请求，供引擎结束批量生成循环。

```python
    def add(self, seq: Sequence):
        # 常规新请求按入队顺序等待。
        self.waiting.append(seq)
```

**功能描述：** 将新请求放到等待队列末尾，使其进入后续 Prefill 调度。

```python
    def schedule(self) -> tuple[list[Sequence], bool]:
        # 存储本轮选中的请求。
        scheduled_seqs = []
        # 累计本轮已分配的 Token 预算。
        num_batched_tokens = 0
```

**功能描述：** 初始化一轮调度的结果与 Token 计数。后续先尝试 Prefill，若未选中请求再尝试 Decode，最终返回请求列表和阶段标记。

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    # 查看队首，不提前移除尚未完成 Prefill 的请求。
    seq = self.waiting[0]
    # 扣除已调度 Token，计算剩余预算。
    remaining = self.max_num_batched_tokens - num_batched_tokens
    # 扣除已调度 Token，计算剩余预算。
    if remaining == 0:
        break
    if not seq.block_table:
        # 首次进入时检查缓存容量和可复用前缀。
        num_cached_blocks = self.block_manager.can_allocate(seq)
        # 首次进入时检查缓存容量和可复用前缀。
        if num_cached_blocks == -1:
            break
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
    else:
        # 分块 Prefill 只继续处理尚未缓存的部分。
        num_tokens = seq.num_tokens - seq.num_cached_tokens
    # 仅允许本批次第一条序列执行分块 Prefill。
    # 只有本批次首条序列允许部分 Prefill。
    if remaining < num_tokens and scheduled_seqs:
        break
    if not seq.block_table:
        # 首次调度时建立页表。
        self.block_manager.allocate(seq, num_cached_blocks)
    # 本轮执行量不超过剩余预算。
    seq.num_scheduled_tokens = min(num_tokens, remaining)
    num_batched_tokens += seq.num_scheduled_tokens
    # 本轮执行量不超过剩余预算。
    # 本轮可补齐所有输入，后续就能进入 Decode。
    if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
    scheduled_seqs.append(seq)

if scheduled_seqs:
    # True 告诉执行器和后处理逻辑本轮是 Prefill。
    return scheduled_seqs, True
```

**功能描述：** 在资源约束内组成 Prefill 批次，复用已缓存前缀并支持首条请求分块执行。能在本轮完成 Prefill 的请求转入运行队列；本轮一旦选中 Prefill，就不混合 Decode。

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    # 优先选取运行队列头部。
    seq = self.running.popleft()
    # 检查当前末尾 Token 是否需要额外缓存块。
    while not self.block_manager.can_append(seq):
        if self.running:
            # 优先释放队尾其他请求的缓存。
            self.preempt(self.running.pop())
        # 对应 while：仅未通过 break 退出时执行。
        else:
            # 无其他请求可抢占时，将当前请求也退回等待队列。
            self.preempt(seq)
            break
    # 对应 while：仅未通过 break 退出时执行。
    else:
        # Decode 每条序列只计算一个输入 Token。
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
assert scheduled_seqs
# 反转后从左侧插入，保持原有请求顺序。
self.running.extendleft(reversed(scheduled_seqs))
# False 表示 Decode。
return scheduled_seqs, False
```

**功能描述：** 在没有可执行 Prefill 时组成 Decode 批次，每条请求处理一个 Token。缓存不足时抢占队尾请求释放空间，确保当前请求有机会继续推进。

```python
    def preempt(self, seq: Sequence):
        # 重新等待调度。
        seq.status = SequenceStatus.WAITING
        # 恢复时需要重新构建 KV Cache。
        seq.is_prefill = True
        # 解除页表引用，不删除序列 Token。
        self.block_manager.deallocate(seq)
        # 放到等待队列头部，优先安排恢复。
        self.waiting.appendleft(seq)
```

**功能描述：** 将运行请求退回等待状态并释放其缓存，保留 Token 历史供恢复时重建 KV。这样可以缓解显存压力，但会引入重新 Prefill 的计算开销。

```python
    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
    ):
        # 按本轮批次顺序配对模型输出。
        for seq, token_id in zip(seqs, token_ids):
            # 先根据本轮执行范围登记新填满的块。
            self.block_manager.hash_blocks(seq)
            # 累计已计算 Token，并清零本轮计划。
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # Prompt 尚未全部处理时，丢弃中间采样结果。
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            # Prefill 完成或 Decode 时才接收生成结果。
            seq.append_token(token_id)
            # 只有未忽略 EOS 时才按结束符停止。
            reached_eos = not seq.ignore_eos and token_id == self.eos
            # 生成部分达到请求的长度上限。
            reached_limit = (
                seq.num_completion_tokens == seq.max_tokens
            )
            # 结束请求，归还缓存并从运行队列移除。
            if reached_eos or reached_limit:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```

**功能描述：** 将模型输出同步到序列、缓存和调度队列。未完成的 Chunked Prefill 只更新缓存进度，其余请求追加生成 Token，并在达到停止条件时释放资源。

---

[学习目录](README.md) · [上一篇：分页 KV Cache 与前缀缓存](03-Paged-KV-Cache-and-Prefix-Caching.md) · [下一篇：模型执行器与输入组织](05-ModelRunner-and-Context.md)
