# 02 · 请求状态与生命周期

理解请求如何保存 Token、生成进度和缓存块映射，为缓存管理与调度打基础。

对应源码：[sequence.py](../source/nano-vllm/nanovllm/engine/sequence.py)。

## sequence.py

在 vLLM 一类推理框架中，一次生成请求通常可以抽象成：
```
Prompt Tokens
      ↓
Prefill 阶段：处理全部输入 token，构建 KV Cache
      ↓
Decode 阶段：每次生成一个新 token
      ↓
达到 EOS 或 max_tokens
      ↓
Finished
```
Sequence 就是这个请求在推理系统中的状态载体

```python
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
```
序列状态，三个状态分别表示：
- WAITING：等待调度
- RUNNING：正在执行推理
- FINISHED：生成结束

auto() 会自动为枚举成员分配值，业务代码不需要关心具体数字

```python
class Sequence:
    block_size = 256
    counter = count()
```
每个逻辑缓存块容纳 256 个 token

itertools.count() 会不断产生递增整数，每创建一个 Sequence，就会获得唯一的序列 ID

```python
    def __init__(
        self,
        token_ids: list[int],
        sampling_params=SamplingParams(),
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
```
1. 新序列拥有唯一 ID，初始状态为等待调度
2. token_ids：当前序列中的所有 token
3. last_token：最后一个 token
4. num_tokens：当前 token 总数
5. num_prompt_tokens：原始提示词 token 数
6. num_cached_tokens：已有 KV Cache 的 token 数量
7. num_scheduled_tokens：本轮被调度执行的 token 数量
8. is_prefill：是否仍处于 Prefill 阶段
9. block_table：逻辑块到物理 KV Cache 块的映射表
10. temperature：控制采样随机性
11. max_tokens：最大生成 token 数
12. ignore_eos：是否忽略结束符 EOS

```python
    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]
```
Python 容器协议

支持 len(seq), 支持下标访问, seq[0]

```python
    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]
```
常用属性：

- 是否结束
- 已生成 token 数量
- Prompt token：返回原始输入部分
- Completion token：返回模型后续生成部分

```python
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[
            i * self.block_size : (i + 1) * self.block_size
        ]
```
KV Cache 分块计算：

- 当前需要多少个块
- 最后一个块有多少 token
- 获取第 i 个逻辑块

```python
    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
```
添加生成 token：

- Decode 阶段每生成一个 token，就可以调用：
    - 将 token 加入列表
    - 更新最后一个 token
    - 增加 token 总数

```python
    def __getstate__(self):
        last_state = (
            self.last_token if not self.is_prefill else self.token_ids
        )
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            last_state,
        )
```
自定义序列化：

- 导出状态：针对两个阶段做了优化
    - Prefill 阶段：Prefill 需要处理完整 Prompt，因此发送所有 token
    - Decode 阶段：Decode 每轮通常只需要最新 token，所以不再发送完整 token 列表，可以减少进程间通信数据量

```python
    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            last_state,
        ) = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
```
恢复状态：判断最后一个字段的类型

- 收到列表：表示这是 Prefill 数据
- 收到整数：表示这是 Decode 数据

---

[学习目录](README.md) · [上一篇：入口配置与生成主循环](01-Entry-and-Generation.md) · [下一篇：分页 KV Cache 与前缀缓存](03-Paged-KV-Cache-and-Prefix-Caching.md)
