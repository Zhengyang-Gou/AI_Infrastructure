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
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
```
构造一个调度器，初始化参数：

- 最大并发序列数量：表示一个 batch 最多可以包含多少条序列
- 最大 batch token 数量：表示一次模型执行最多处理多少个 token
- EOS token：eos 是 End Of Sequence token ID
- KV Cache block 大小：KV Cache 不是逐 token 分配，而是以 block 为单位管理
- 创建 BlockManager：传入 KV Cache block 总数量以及每个 block 可以容纳的 token 数
- 等待队列：保存尚未完成 Prefill，或者被抢占后需要重新进入 Prefill 的请求
- 运行队列：保存已经完成 Prompt Prefill、可以执行 Decode 的请求

```python
    def is_finished(self):
        return not self.waiting and not self.running
```
判断所有请求是否已经处理完毕

不是检查某一条序列是否完成，而是检查整个调度器中是否已经没有待处理请求

```python
    def add(self, seq: Sequence):
        self.waiting.append(seq)
```
把一个新请求添加到等待队列末尾

因为使用 append()，所以整体上采用 FIFO

```python
    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0
```
返回值是：scheduled_seqs, is_prefill

整体策略是：
1. 先尝试调度 Prefill
2. 只要调度到了任何 Prefill 请求，就立即返回
3. 如果没有 Prefill 可执行，再调度 Decode

初始化本轮状态:
1. scheduled_seqs 保存本轮选中的请求
2. num_batched_tokens 保存本轮已经安排的 token 总数

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    remaining = self.max_num_batched_tokens - num_batched_tokens
    if remaining == 0:
        break
    if not seq.block_table:
        num_cached_blocks = self.block_manager.can_allocate(seq)
        if num_cached_blocks == -1:
            break
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
    else:
        num_tokens = seq.num_tokens - seq.num_cached_tokens
    # Only allow chunked prefill for the first sequence.
    if remaining < num_tokens and scheduled_seqs:
        break
    if not seq.block_table:
        self.block_manager.allocate(seq, num_cached_blocks)
    seq.num_scheduled_tokens = min(num_tokens, remaining)
    num_batched_tokens += seq.num_scheduled_tokens
    if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
    scheduled_seqs.append(seq)

if scheduled_seqs:
    return scheduled_seqs, True
```
调度 Prefill 阶段的请求：

1. 只要等待队列不为空，并且本轮序列数量没有超过 `max_num_seqs`，就不断尝试加入请求
2. 每次只查看等待队列头部的序列，保证等待队列整体遵循 FIFO
3. `remaining` 表示本轮 Batch 还能容纳多少个 token；如果已经没有剩余容量，就结束本轮调度
4. 如果序列还没有 `block_table`，说明尚未分配 KV Cache：
    - 调用 `can_allocate(seq)` 检查是否有足够的物理块
    - 返回 `-1` 表示当前无法分配，停止继续调度
    - 否则用序列总 token 数减去可复用缓存块中的 token 数，得到本次真正需要计算的 token 数
5. 如果序列已经有 `block_table`，说明它可能执行过部分 Prefill，本次只处理尚未缓存的 token
6. 只有本轮第一条序列允许 Chunked Prefill；如果当前序列无法完整放入 Batch，并且前面已经选中了其他序列，就留到下一轮处理
7. 为首次进入的序列分配 KV Cache，并记录本轮实际调度的 token 数
8. 当 `已缓存 token 数 + 本轮调度 token 数` 等于序列总 token 数时，说明本轮可以完成 Prefill：
    - 将状态改为 `RUNNING`
    - 从 `waiting` 队列移除
    - 加入 `running` 队列，等待后续 Decode
9. 只要本轮选中了 Prefill 请求，就返回 `(scheduled_seqs, True)`，本轮不再混合执行 Decode

这里的 `True` 表示本轮是 Prefill，`ModelRunner` 和后处理逻辑会据此选择对应的执行方式。

```python
while self.running and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.running.popleft()
    while not self.block_manager.can_append(seq):
        if self.running:
            self.preempt(self.running.pop())
        else:
            self.preempt(seq)
            break
    else:
        seq.num_scheduled_tokens = 1
        seq.is_prefill = False
        self.block_manager.may_append(seq)
        scheduled_seqs.append(seq)
assert scheduled_seqs
self.running.extendleft(reversed(scheduled_seqs))
return scheduled_seqs, False
```
调度 Decode 阶段的请求：

1. 当本轮没有可执行的 Prefill 时，从 `running` 队列头部依次选择序列
2. Decode 每轮只为每条序列生成一个新 token，因此成功调度后将 `num_scheduled_tokens` 设置为 `1`
3. `can_append(seq)` 检查当前序列的 KV Cache 是否还能容纳下一个 token
4. 如果空间不足，就进行抢占：
    - 还有其他运行序列时，优先抢占队尾的序列，为当前序列释放 KV Cache
    - 已经没有其他序列可抢占时，只能抢占当前序列，并停止调度它
5. Python 的 `while...else` 表示只有循环没有通过 `break` 退出时才执行 `else`：
    - 将序列切换到 Decode 状态
    - 必要时为下一个 token 扩展 KV Cache
    - 把序列加入本轮执行列表
6. `assert scheduled_seqs` 确保 Decode 阶段至少成功调度了一条序列
7. `extendleft(reversed(...))` 把已调度序列按原顺序放回 `running` 队列头部，使它们下一轮仍能继续 Decode
8. 返回 `(scheduled_seqs, False)`，其中 `False` 表示本轮是 Decode

这一段通过“释放低优先级序列的缓存，让当前序列继续运行”来应对 KV Cache 空间不足。

```python
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
```
`preempt`：抢占一条正在运行的序列

1. 把序列状态从 `RUNNING` 改回 `WAITING`
2. 将 `is_prefill` 重新设置为 `True`，因为它恢复执行时需要重新构建已经释放的 KV Cache
3. 调用 `deallocate(seq)` 释放该序列占用的全部物理缓存块
4. 使用 `appendleft()` 把序列放到等待队列头部，使它能够优先重新调度

抢占不会删除序列的 token 数据，只会释放它的 KV Cache 映射。因此请求不会丢失，但恢复时会产生重新计算 Prefill 的开销。

```python
    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int],
        is_prefill: bool,
    ):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            reached_eos = not seq.ignore_eos and token_id == self.eos
            reached_limit = (
                seq.num_completion_tokens == seq.max_tokens
            )
            if reached_eos or reached_limit:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```
`postprocess`：根据模型输出更新序列和调度器状态

参数：

- `seqs`：本轮参与执行的序列
- `token_ids`：模型为各序列返回的 token ID
- `is_prefill`：本轮是否处于 Prefill 阶段

处理流程：

1. 使用 `zip(seqs, token_ids)` 将每条序列与对应的模型输出配对
2. `hash_blocks(seq)` 为已经填满的 KV Cache 块计算哈希，使相同前缀后续可以复用缓存
3. 把本轮调度的 token 数累加到 `num_cached_tokens`，然后清空 `num_scheduled_tokens`
4. 如果当前是 Chunked Prefill，并且仍有输入 token 没有写入缓存，就直接处理下一条序列：
    - 此时返回的 `token_id` 不作为生成结果
    - 当前序列会在后续轮次继续 Prefill
5. Prefill 完成或当前处于 Decode 时，将模型输出追加到序列：
    - 更新 `token_ids`
    - 更新 `last_token`
    - 增加 `num_tokens`
6. 判断请求是否应该结束：
    - 未设置 `ignore_eos`，并且模型生成了 EOS
    - 已生成的 token 数达到 `max_tokens`
7. 请求结束后：
    - 将状态设置为 `FINISHED`
    - 释放该序列占用的 KV Cache
    - 从 `running` 队列中移除

`postprocess` 完成了“模型输出 → 序列状态 → 缓存状态 → 调度队列”的同步，是每轮推理结束后的收尾步骤。

---

[学习目录](README.md) · [上一篇：分页 KV Cache 与前缀缓存](03-Paged-KV-Cache-and-Prefix-Caching.md) · [下一篇：模型执行器与输入组织](05-ModelRunner-and-Context.md)
