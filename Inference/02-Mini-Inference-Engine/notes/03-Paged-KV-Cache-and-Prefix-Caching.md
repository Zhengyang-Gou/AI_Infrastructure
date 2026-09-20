# 03 · 分页 KV Cache 与前缀缓存

先理解 CPU 侧物理块分配、引用计数和前缀复用，再看调度器如何使用这些能力。

对应源码：[block_manager.py](../source/nano-vllm/nanovllm/engine/block_manager.py)。

## block_manager.py

`BlockManager` 只管理 KV Cache 的元数据，真正的 K、V 张量由 `ModelRunner` 在 GPU 上分配。

它维护的核心关系是：
```
Sequence 中的第 i 个逻辑块
              ↓ block_table[i]
GPU KV Cache 中的物理块 block_id
```

```python
class Block:
    def __init__(self, block_id):
        # GPU 物理块的索引。
        self.block_id = block_id
        # 当前引用此块的序列数。
        self.ref_count = 0
        # 未登记完整块哈希时使用 -1。
        self.hash = -1
        # 保存块内 Token，哈希命中后用于核对内容。
        self.token_ids = []

    # 完整块产生后登记前缀链哈希及 Token。
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        # 重新分配给一个使用者；同时清除旧匹配信息。
        self.ref_count = 1
        # 未登记完整块哈希时使用 -1。
        self.hash = -1
        # 保存块内 Token，哈希命中后用于核对内容。
        self.token_ids = []
```

**功能描述：** 用 CPU 侧元数据描述一个物理缓存块，支持引用计数、前缀匹配和重新分配；实际 K/V 数据位于 GPU。

```python
def __init__(self, num_blocks: int, block_size: int):
    self.block_size = block_size
    # 为所有物理块建立元数据对象。
    self.blocks = [Block(i) for i in range(num_blocks)]
    # 从前缀哈希查找可复用的物理块。
    self.hash_to_block_id = dict()
    # 空闲队列中的块可以重新分配。
    self.free_block_ids = deque(range(num_blocks))
    # 记录至少被一条序列引用的物理块。
    self.used_block_ids = set()
```

**功能描述：** 初始化缓存块池及查找结构，为分配、回收和共享前缀提供统一管理。

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    # 使用 64 位哈希累积块内容。
    h = xxhash.xxh64()
    # 首块以外先加入前一个完整块的链式哈希。
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    # 再加入当前块 Token 的字节表示。
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```

**功能描述：** 将前缀历史与当前块 Token 一起编码成链式哈希，避免仅因局部 Token 相同而复用不同上下文的 KV。哈希命中还需配合 Token 核对。

补充示意：

```
H0 = hash(block0)
H1 = hash(H0, block1)
H2 = hash(H1, block2)
```

```python
def _allocate_block(self) -> int:
    # 从空闲队列头部借出块。
    block_id = self.free_block_ids.popleft()
    block = self.blocks[block_id]
    assert block.ref_count == 0
    # 仅删除仍指向当前块的旧哈希索引。
    if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
        del self.hash_to_block_id[block.hash]
    # 重置内容，并为新使用者设置一次引用。
    block.reset()
    self.used_block_ids.add(block_id)
    return block_id

def _deallocate_block(self, block_id: int):
    # 只有引用计数归零的块才可回收。
    assert self.blocks[block_id].ref_count == 0
    self.used_block_ids.remove(block_id)
    # 加入空闲队列，不立即擦除前缀元数据。
    self.free_block_ids.append(block_id)
```

**功能描述：** 管理物理块的底层借出与归还。归还时保留前缀信息，使尚未覆盖的空闲块仍有复用机会；重新分配时再清理旧索引。

```python
def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    # 先按全部逻辑块都需要空闲块进行预算。
    num_new_blocks = seq.num_blocks
    # 排除末块，即便它恰好填满也保留重新计算的尾部。
    for i in range(seq.num_blocks - 1):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        # 前缀链首次未命中或 Token 不符时停止复用。
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break
        num_cached_blocks += 1
        # 已使用的共享块无需额外消耗空闲块；空闲缓存块仍计入预算。
        if block_id in self.used_block_ids:
            num_new_blocks -= 1
    if len(self.free_block_ids) < num_new_blocks:
        # 当前空闲块不足以接纳请求。
        return -1
    return num_cached_blocks
```

**功能描述：** 预判新请求能否获得足够缓存，并返回可复用的前缀块数；容量不足返回 -1。该步骤只检查资源，实际建立页表由 allocate 完成。

```python
def allocate(self, seq: Sequence, num_cached_blocks: int):
    # 仅处理尚未建立页表的请求。
    assert not seq.block_table
    h = -1
    # 按前缀链顺序找到可共享的块。
    for i in range(num_cached_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id[h]
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            # 仍在使用的缓存块增加一个引用。
            block.ref_count += 1
        else:
            block.ref_count = 1
            # 将空闲但有效的前缀缓存块重新激活。
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
        seq.block_table.append(block_id)
    # 为剩余逻辑块分配新的物理空间。
    for i in range(num_cached_blocks, seq.num_blocks):
        seq.block_table.append(self._allocate_block())
    # 把命中块数换算成可跳过的 Token 数。
    seq.num_cached_tokens = num_cached_blocks * self.block_size
```

**功能描述：** 为请求建立逻辑块到物理块的页表，复用命中前缀并分配剩余块，同时记录可以跳过的 Prefill 长度。

```python
def deallocate(self, seq: Sequence):
    # 逆序释放序列引用的块。
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        # 减少本序列占用的一次引用。
        block.ref_count -= 1
        # 最后一个使用者释放后才真正回收。
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    # 清空此请求的缓存进度与映射。
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

**功能描述：** 解除一条序列对缓存块的引用并清空页表，只有无人引用的物理块才回到空闲池，避免破坏其他请求共享的前缀。

```python
def can_append(self, seq: Sequence) -> bool:
    # 余数为 1 表示末尾 Token 刚进入新块，需要一个空闲块；布尔值按 0/1 比较。
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq: Sequence):
    if len(seq) % self.block_size == 1:
        # 仅在跨块时追加物理块，其余位置沿用末块。
        seq.block_table.append(self._allocate_block())
```

**功能描述：** 在 Decode 前检查并按需扩展缓存，为当前末尾 Token 的 K/V 写入预留空间。序列此时已包含上轮生成的 Token。

```python
def hash_blocks(self, seq: Sequence):
    # 执行前缓存边界所在的逻辑块。
    start = seq.num_cached_tokens // self.block_size
    # 向下取整，仅覆盖本轮结束后已完整的块。
    end = (
        seq.num_cached_tokens + seq.num_scheduled_tokens
    ) // self.block_size
    # 执行前缓存边界所在的逻辑块。
    # 没有新完整块就无需更新哈希。
    if start == end:
        return
    # 从前一完整块的哈希继续构建前缀链。
    h = (
        self.blocks[seq.block_table[start - 1]].hash
        if start > 0 else -1
    )
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        # 保存 Token 和哈希，再建立查询索引。
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block.block_id
```

**功能描述：** 将本轮计算后刚填满的块登记为可复用前缀。新请求随后可通过 can_allocate 查找这些块，并在 allocate 中共享它们。

补充示意：

```
旧请求完成一个整块
    ↓ hash_blocks 登记
新请求进入 waiting
    ↓ can_allocate 查找相同前缀
allocate 共享物理块
    ↓
Prefill 从 num_cached_tokens 之后开始计算
```

---

[学习目录](README.md) · [上一篇：请求状态与生命周期](02-Sequence-and-Lifecycle.md) · [下一篇：调度与批处理](04-Scheduler-and-Batching.md)
