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
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
```
`Block`：一个物理 KV Cache 块在 CPU 侧的描述对象

- `block_id`：物理块编号，也是访问 GPU KV Cache 的索引
- `ref_count`：引用计数；多个拥有相同前缀的序列可以共享同一个块
- `hash`：当前完整块及其前缀链的哈希
- `token_ids`：保存该块对应的 Token，用于在哈希相同时再次核对内容，避免哈希碰撞造成错误复用
- `update()`：块填满后登记哈希和 Token
- `reset()`：块被重新分配时清除旧内容，并将引用计数设为 1

```python
def __init__(self, num_blocks: int, block_size: int):
    self.block_size = block_size
    self.blocks = [Block(i) for i in range(num_blocks)]
    self.hash_to_block_id = dict()
    self.free_block_ids = deque(range(num_blocks))
    self.used_block_ids = set()
```
`BlockManager` 初始化：

- `blocks`：全部物理块的元数据
- `hash_to_block_id`：前缀哈希到物理块 ID 的索引，用于前缀缓存查找
- `free_block_ids`：空闲块队列
- `used_block_ids`：正在被至少一条序列引用的块集合

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()
```
计算链式哈希：

第 `i` 块的哈希不仅包含本块 Token，还包含第 `i-1` 块的哈希：
```
H0 = hash(block0)
H1 = hash(H0, block1)
H2 = hash(H1, block2)
```
所以只有从 Prompt 开头起完全相同的块才能命中，不能错误复用出现在其他位置的相同 Token 块。

```python
def _allocate_block(self) -> int:
    block_id = self.free_block_ids.popleft()
    block = self.blocks[block_id]
    assert block.ref_count == 0
    if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
        del self.hash_to_block_id[block.hash]
    block.reset()
    self.used_block_ids.add(block_id)
    return block_id

def _deallocate_block(self, block_id: int):
    assert self.blocks[block_id].ref_count == 0
    self.used_block_ids.remove(block_id)
    self.free_block_ids.append(block_id)
```
物理块的底层分配与回收：

- 分配时从空闲队列头部取块
- 如果这个空闲块还保留着旧的前缀缓存索引，重新使用前先删除索引
- 回收时只把引用计数已经归零的块放回空闲队列
- 回收不会立即清除 `hash` 和 `token_ids`，因此空闲但尚未被覆盖的块仍可作为前缀缓存重新激活

```python
def can_allocate(self, seq: Sequence) -> int:
    h = -1
    num_cached_blocks = 0
    num_new_blocks = seq.num_blocks
    for i in range(seq.num_blocks - 1):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id.get(h, -1)
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            break
        num_cached_blocks += 1
        if block_id in self.used_block_ids:
            num_new_blocks -= 1
    if len(self.free_block_ids) < num_new_blocks:
        return -1
    return num_cached_blocks
```
检查一条新序列是否能够分配：

1. 按顺序检查已经填满的 Prompt 块是否命中前缀缓存
2. 最后一个块不参与匹配，因为它可能尚未填满，后续 Decode 还会继续写入
3. 哈希命中后还要比较 `token_ids`，防止哈希碰撞
4. 已经在使用的命中块只需增加引用，不占用新的空闲块
5. 已回到空闲队列的缓存块需要从空闲队列重新取出，所以仍计入 `num_new_blocks`
6. 空闲块不足返回 `-1`，否则返回可复用的完整块数量

```python
def allocate(self, seq: Sequence, num_cached_blocks: int):
    assert not seq.block_table
    h = -1
    for i in range(num_cached_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block_id = self.hash_to_block_id[h]
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1
        else:
            block.ref_count = 1
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)
        seq.block_table.append(block_id)
    for i in range(num_cached_blocks, seq.num_blocks):
        seq.block_table.append(self._allocate_block())
    seq.num_cached_tokens = num_cached_blocks * self.block_size
```
为序列建立 `block_table`：

- 命中的前缀块直接共享；正在使用的块增加引用计数，空闲缓存块则重新激活
- 未命中的逻辑块分配新物理块
- 最后将命中块数换算成已缓存 Token 数，后续 Prefill 可以跳过这些 Token

```python
def deallocate(self, seq: Sequence):
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self._deallocate_block(block_id)
    seq.num_cached_tokens = 0
    seq.block_table.clear()
```
释放序列占用的缓存：

- 逆序减少每个物理块的引用计数
- 只有最后一个使用者退出时才真正回收到空闲队列
- 清空序列的缓存进度和逻辑到物理块映射

```python
def can_append(self, seq: Sequence) -> bool:
    return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

def may_append(self, seq: Sequence):
    if len(seq) % self.block_size == 1:
        seq.block_table.append(self._allocate_block())
```
为 Decode 的下一个 Token 检查和扩展空间。

调度发生在新 Token 生成之前，但 `len(seq)` 已经包含当前最后一个 Token。余数为 `1` 说明当前 Token 是一个新逻辑块的第一个 Token，需要提前分配物理块；Python 中布尔值可作为 `0/1` 参与比较。

```python
def hash_blocks(self, seq: Sequence):
    start = seq.num_cached_tokens // self.block_size
    end = (
        seq.num_cached_tokens + seq.num_scheduled_tokens
    ) // self.block_size
    if start == end:
        return
    h = (
        self.blocks[seq.block_table[start - 1]].hash
        if start > 0 else -1
    )
    for i in range(start, end):
        block = self.blocks[seq.block_table[i]]
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h)
        block.update(h, token_ids)
        self.hash_to_block_id[h] = block.block_id
```
把本轮刚刚填满的块加入前缀缓存：

1. 根据执行前的已缓存 Token 数得到起始块
2. 根据本轮执行后的缓存边界得到结束块
3. 只处理跨过完整块边界的部分，不缓存未填满的块
4. 从上一个块的哈希继续计算链式哈希
5. 保存块内容，并建立哈希索引

由此，前缀缓存的完整流程是：
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
