# Nano-vLLM
## 一次生成如何执行
### example.py
```python
def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
```
- tokenizer = AutoTokenizer.from_pretrained(path)：从指定路径读取配置和词表，初始化该模型对应的分词器，负责文本与 Token ID 之间的转换
- llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)：初始化推理引擎
    - path：指定模型权重路径
    - enforce_eager=True：强制使用 PyTorch 的 Eager 模式（即不使用 CUDA Graph 等图捕获加速），这在调试或轻量设备上更稳定、占用显存更小
    - tensor_parallel_size=1：张量并行度设置为 1，代表只使用 1 张 GPU 运行模型
- sampling_params = ...：配置文本生成的采样参数：
    - temperature=0.6：温度参数，值越高生成的文本越富有创造性/随机；值越低越确定
    - max_tokens=256：模型单次生成的最大 Token 数量限制
- prompts = [...]：定义输入的原始文本列表
- 使用列表推导式遍历所有的 prompt，将其转换为指令微调模型能识别的格式
    - [{"role": "user", "content": prompt}]：构造单轮对话的 Message 列表
    - tokenize=False：只输出格式化后的文本字符串（例如插入 <|im_start|> 等特殊标记），先不转成数字 ID
    - add_generation_prompt=True：在格式化文本的末尾自动加上模型的回答引导头（如 <|im_start|>assistant\n），提示模型开始回答
- llm.generate(...)：将格式化后的提示词列表和采样参数传入引擎，批量执行模型推理，返回生成结果的列表 outputs

### llm.py
```python
from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    pass
```
把 LLMEngine 包装成一个对外暴露的 LLM 类

LLM 会自动拥有 LLMEngine 的全部方法和属性

### config.py
```python
@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(
            self.max_model_len,
            self.hf_config.max_position_embeddings,
        )
```
- model: str：模型权重文件在本地的路径或 Hugging Face ID（这里是必填属性，没有默认值）
- max_num_batched_tokens: int = 16384：一个 Batch 中所有序列的 Token 数量之和的最大值，用于防止推理时的显存溢出
- max_num_seqs: int = 512：一个 Batch 中最多同时并行处理的请求（序列）数量
- max_model_len: int = 4096：模型单条序列允许的最大上下文长度（Prompt + 生成的 Token）
- gpu_memory_utilization: float = 0.9：允许推理引擎预先占用的单张 GPU 显存比例（这里指 90% 的显存用于模型权重和 KV Cache）
- tensor_parallel_size: int = 1：张量并行度（GPU 数量）
- enforce_eager: bool = False：是否强制启用 PyTorch Eager 模式
- hf_config: AutoConfig | None = None：保存 Hugging Face 模型配置对象的字段
- eos: int = -1：结束符 Token ID（End-of-sequence ID），默认初始值为 -1
- kvcache_block_size: int = 256：PagedAttention 机制中每个 KV Cache 物理块包含的 Token 数量
- num_kvcache_blocks: int = -1：系统可以分配的总 KV Cache 物理块数量，初始为 -1，通常后续会在显存分析后动态计算并填充

- 断言 1：确保传入的 self.model 必须是一个合法的本地文件夹路径
- 断言 2：要求 kvcache_block_size 必须是 256 的整数倍
- 断言 3：限制张量并行度 tensor_parallel_size 必须在 1 到 8 之间
- self.hf_config = AutoConfig.from_pretrained(self.model)：自动读取模型路径下的 config.json，获取模型元数据
- self.max_model_len = min(...)：防错保护。将用户设定的 max_model_len 与模型架构本身支持的最大位置嵌入长度（max_position_embeddings）取最小值，确保模型不会因越界导致位置编码异常

### sampling_params.py
```python
@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
```
- temperature：控制输出随机性的超参数
- max_tokens：限制本次请求最多能生成的 Token 数量，防止模型无休止地生成下去导致显存爆满
- ignore_eos (是否忽略终止符)：
    - False：如果模型输出了终止符，推理就会提前结束
    - True：强行忽略 EOS 标记，必须把 max_tokens 生成满才停止

### llm_engine.py
```python
class LLMEngine:
```
LLMEngine 类，这个类是用户操作推理系统的主要入口

```python
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(
                target=ModelRunner,
                args=(config, i, event),
            )
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)
```
__init__：初始化引擎

参数：
- model：模型名称或本地模型路径
- kwargs：其他配置，例如张量并行规模、KV Cache 块大小等

1. 过滤配置参数，返回这些字段的描述对象，只保留 Config 支持的参数
2. 设置 Sequence 的 KV Cache 块大小
3. 保存工作进程和同步事件
4. 创建 spawn 多进程上下文
5. 启动张量并行工作进程
6. 在主进程创建 rank 0 ModelRunner
7. 加载 tokenizer
8. 设置结束 token
9. 创建调度器
10. 注册退出清理

```python
    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
```
exit：关闭引擎

负责关闭模型执行器和子进程

1. 通过主 ModelRunner 向各工作进程发送退出命令
2. 删除主进程对模型执行器的引用
3. 等待每个子进程退出

```python
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
```
add_request：添加一条生成请求

prompt 支持两种输入："你好" 或者已经编码好的 token ID：[1, 345, 782, 29]

1. 文本编码：当输入是字符串时，调用 tokenizer 转为 token ID；当输入已经是 list[int] 时，跳过编码
2. 构造 Sequence：seq = Sequence(prompt, sampling_params)
3. 加入调度器：此时并没有立即运行模型，请求只是进入调度队列，等待后续决定何时被执行，这种设计支持连续批处理

```python
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = (
            sum(seq.num_scheduled_tokens for seq in seqs)
            if is_prefill
            else -len(seqs)
        )
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [
            (seq.seq_id, seq.completion_token_ids)
            for seq in seqs
            if seq.is_finished
        ]
        return outputs, num_tokens
```
step：执行一轮推理

一次 step() 只完成一轮调度和模型执行，并不保证一条请求完全生成结束

1. 调度请求：调度器返回
    - seqs：本轮需要处理的序列
    - is_prefill：本轮是否是 Prefill
2. 计算本轮 token 数量
    - Prefill 时：计算本轮实际处理的输入 token 总数
    - Decode 时：-len(seqs)，绝对值表示本轮 token 数
3. 执行模型：把本轮序列发送给模型执行器，返回的 token_ids 通常对应每条序列新生成的 token
4. 更新调度器状态
5. 收集已完成请求：只返回本轮刚处理且已经结束的请求，没有完成的序列不会返回，但仍然留在调度器中继续生成

```python
    def is_finished(self):
        return self.scheduler.is_finished()
```
is_finished：判断是否全部结束

真正的判断逻辑在 Scheduler 中

```python
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(
            total=len(prompts),
            desc="Generating",
            dynamic_ncols=True,
            disable=not use_tqdm,
        )
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
            }
            for token_ids in outputs
        ]
        return outputs
```
generate：批量生成完整结果

参数：
- prompts：多条文本，或者多组 token ID
- sampling_params：一套共享参数，或者每条请求独立参数
- se_tqdm：是否显示进度条

1. 创建进度条
2. 扩展共享采样参数：用户可以为全部请求传同一套参数
3. 添加所有请求：把每条 prompt 和对应的采样参数组成请求
4. 准备结果容器：结果先按 seq_id 保存，请求完成顺序不一定等于提交顺序，使用字典按 seq_id 存储，最后再排序，就能恢复原输入顺序
5. 记录最近一次 Prefill 和 Decode 的吞吐量
6. 主生成循环：只要调度器里还有未完成请求，就不断执行推理
7. 开始计时
8. 执行一步：拿到本轮完成的请求，本轮处理 token 数，并通过正负表示阶段
9. 计算 Prefill 吞吐量
10. 计算 Decode 吞吐量
11. 更新进度条信息
12. 保存本轮完成结果
13. 按请求 ID 恢复顺序
14. token 解码

## 调度系统
### sequence.py
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

### scheduler.py
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

## Paged KV Cache 和前缀缓存
### block_manager.py
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

## 执行器和模型输入组织
### model_runner.py
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

## 模型和 Attention
### context.py
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

### qwen3.py
模型结构：
```
input_ids
   ↓ VocabParallelEmbedding
N × Qwen3DecoderLayer
   ├─ RMSNorm → Attention → 残差
   └─ RMSNorm → MLP       → 残差
   ↓ RMSNorm
hidden_states
   ↓ ParallelLMHead
logits
```

```python
class Qwen3Attention(nn.Module):
    def __init__(...):
        tp_size = dist.get_world_size()
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = num_kv_heads // tp_size
        self.head_dim = (
            head_dim or hidden_size // num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
```
每个 rank 只负责一部分 Q Head 和 KV Head。Qwen3 使用 GQA 时，`num_kv_heads` 可以小于 `num_heads`，多组 Q Head 共享 K/V Head。

```python
qkv = self.qkv_proj(hidden_states)
q, k, v = qkv.split(
    [self.q_size, self.kv_size, self.kv_size], dim=-1
)
q = q.view(-1, self.num_heads, self.head_dim)
k = k.view(-1, self.num_kv_heads, self.head_dim)
v = v.view(-1, self.num_kv_heads, self.head_dim)
if not self.qkv_bias:
    q = self.q_norm(q)
    k = self.k_norm(k)
q, k = self.rotary_emb(positions, q, k)
o = self.attn(q, k, v)
return self.o_proj(o.flatten(1, -1))
```
Attention 前向：

1. 使用合并的列并行线性层一次计算 Q、K、V
2. 按当前 rank 的 Head 数拆分并恢复 Head 维度
3. 无 QKV Bias 的 Qwen3 配置对每个 Q/K Head 单独做 RMSNorm
4. 对 Q、K 应用 RoPE
5. Attention 核心完成 KV Cache 写入与注意力计算
6. 合并 Head，通过行并行输出层并在 rank 间 All-Reduce

```python
class Qwen3MLP(nn.Module):
    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        return self.down_proj(x)
```
MLP 使用 SwiGLU：
```
down_proj(SiLU(gate_proj(x)) * up_proj(x))
```
`gate_proj` 和 `up_proj` 被合并进一个列并行矩阵乘，激活后再通过行并行 `down_proj` 恢复隐藏维度。

```python
def forward(self, positions, hidden_states, residual):
    if residual is None:
        hidden_states, residual = (
            self.input_layernorm(hidden_states), hidden_states
        )
    else:
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual
        )
    hidden_states = self.self_attn(positions, hidden_states)
    hidden_states, residual = self.post_attention_layernorm(
        hidden_states, residual
    )
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```
Decoder Layer 使用延迟残差融合：

- 第一层先保存 Embedding 输出作为残差
- 后续 `add_rms_forward` 在做 RMSNorm 前顺便完成上一个子层的残差加法
- Attention 输出的残差加法融合进 `post_attention_layernorm`
- MLP 输出的残差加法延迟到下一层 `input_layernorm`
- 最后一层尚未合并的 MLP 残差由模型末尾的 `norm` 完成

这种写法减少了独立的加法 Kernel 和中间张量读写。

```python
packed_modules_mapping = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```
Hugging Face 权重中 Q/K/V、Gate/Up 分别保存，而推理模型把它们合并成大参数。这个映射告诉 Loader 应将原权重装入哪个合并参数的哪个分片。

`Qwen3ForCausalLM.forward()` 只返回隐藏状态，`compute_logits()` 再调用 LM Head。若配置启用 `tie_word_embeddings`，LM Head 与输入 Embedding 直接共享底层权重数据。

### linear.py
该文件实现张量并行线性层。PyTorch 线性层的权重形状为：
```
[output_size, input_size]
```

```python
class ColumnParallelLinear(LinearBase):
    def __init__(...):
        super().__init__(
            input_size, output_size // tp_size, bias, 0
        )

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
```
列并行按权重的输出维度切分：
```
完整 W = [W0; W1; ...]
每个 rank: Yi = X · Wi^T
```
每个 rank 产生不同的输出特征，不需要立即通信，适合 QKV 投影和 MLP 的 Gate/Up 投影。

`weight_loader()` 按 rank 从完整权重的第 0 维截取对应分片。

```python
class RowParallelLinear(LinearBase):
    def __init__(...):
        super().__init__(
            input_size // tp_size, output_size, bias, 1
        )

    def forward(self, x):
        y = F.linear(
            x, self.weight,
            self.bias if self.tp_rank == 0 else None,
        )
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```
行并行按输入维度切分：
```
X = [X0, X1, ...]
W = [W0, W1, ...]
Y = Σ Xi · Wi^T
```
每个 rank 先计算部分结果，再 All-Reduce 求和。Bias 只在 rank 0 添加一次，随后求和即可广播到最终结果。

`MergedColumnParallelLinear` 把多个输出矩阵合并到一个参数中。加载时先根据 `loaded_shard_id` 找到合并参数内部的区域，再取该区域属于当前 TP rank 的分片。

`QKVParallelLinear` 与之类似，但 Q Head 数和 KV Head 数可能不同，因此分别计算 Q、K、V 在当前 rank 中的大小和偏移。

`ReplicatedLinear` 不切分权重，各 rank 各自保存完整副本；当前 Qwen3 主干主要使用列并行和行并行层。

### embed_head.py
```python
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings_per_partition = (
            num_embeddings // self.tp_size
        )
        self.vocab_start_idx = (
            self.num_embeddings_per_partition * self.tp_rank
        )
        self.vocab_end_idx = (
            self.vocab_start_idx
            + self.num_embeddings_per_partition
        )
```
Embedding 按词表维度切分，每个 rank 只保存连续的一段词表。

```python
if self.tp_size > 1:
    mask = (
        (x >= self.vocab_start_idx)
        & (x < self.vocab_end_idx)
    )
    x = mask * (x - self.vocab_start_idx)
y = F.embedding(x, self.weight)
if self.tp_size > 1:
    y = mask.unsqueeze(1) * y
    dist.all_reduce(y)
```
每个 Token 只会落在一个 rank 的词表区间：

1. 本 rank 范围内的 Token 转为局部下标
2. 范围外 Token 临时映射到 0，查表后再用 Mask 清零
3. All-Reduce 后，每个位置只剩下真正所属 rank 的 Embedding

```python
class ParallelLMHead(VocabParallelEmbedding):
    def forward(self, x):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
```
Prefill 会得到所有输入 Token 的隐藏状态，但生成下一个 Token 只需要每条序列最后一个位置，所以通过累计长度取出各序列末尾，避免为所有 Prompt Token 计算词表 logits。

每个 rank 先计算自己的局部词表 logits，之后将结果 Gather 到 rank 0，并沿词表维拼接成完整 logits。只有 rank 0 需要完整词表，因为最终采样也只在那里执行。

### layernorm.py
`RMSNorm` 的公式：
```
y = x / sqrt(mean(x²) + eps) × weight
```
与 LayerNorm 不同，它不减均值，也没有 Bias。

实现先转为 FP32 计算平方均值和归一化，减少低精度数值误差，最后转回原 dtype。`@torch.compile` 用于融合逐元素操作。

`add_rms_forward(x, residual)` 先执行：
```
residual = x + residual
```
再对合并结果做 RMSNorm，同时返回归一化输出和新的残差。这是 Decoder Layer 中残差连接与归一化融合的基础。

### activation.py
```python
class SiluAndMul(nn.Module):
    @torch.compile
    def forward(self, x):
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
```
输入是合并的 Gate 和 Up 投影结果，沿最后一维一分为二，计算 SwiGLU 激活。使用 `torch.compile` 可以将切分、SiLU 和乘法尽可能融合。

### rotary_embedding.py
```python
inv_freq = 1.0 / (
    base ** (
        torch.arange(0, rotary_dim, 2) / rotary_dim
    )
)
t = torch.arange(max_position_embeddings)
freqs = torch.einsum("i,j -> ij", t, inv_freq)
cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
```
初始化时预计算每个位置、每个频率对应的 cos/sin，注册为非持久 Buffer：

- 会随模块移动到 GPU
- 不属于模型权重，不写入 state_dict
- 推理时按 `positions` 直接查表

```python
x1, x2 = torch.chunk(x.float(), 2, dim=-1)
y1 = x1 * cos - x2 * sin
y2 = x2 * cos + x1 * sin
return torch.cat((y1, y2), dim=-1).to(x.dtype)
```
RoPE 将 Head 维度的两半视为二维向量并按位置旋转，把相对位置信息编码进 Q、K。计算临时转成 FP32，再恢复原 dtype。

`get_rope` 使用 `@lru_cache(1)` 缓存最近创建的 RoPE 模块，使相同配置的各 Transformer 层共享同一份 cos/sin Cache，避免重复占用显存。

### attention.py
```python
@triton.jit
def store_kvcache_kernel(...):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    ...
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```
自定义 Triton Kernel 将本轮产生的 K/V 写入 Paged KV Cache：

- 每个 Program 负责一个 Token
- `slot_mapping[idx]` 给出该 Token 的扁平物理槽位
- 一个 Token 的所有 KV Head 和 Head Dim 被当作长度 `D` 的连续区域复制
- `slot=-1` 用于 CUDA Graph Padding，表示该位置不应写缓存

```python
def forward(self, q, k, v):
    context = get_context()
    if k_cache.numel() and v_cache.numel():
        store_kvcache(
            k, v, k_cache, v_cache, context.slot_mapping
        )
```
Attention 每一层首先把新 K/V 写入该层自己的缓存。模型预热发生在 KV Cache 分配前，此时 Cache 是空张量，因此跳过写入。

```python
if context.is_prefill:
    if context.block_tables is not None:
        k, v = k_cache, v_cache
    o = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=context.cu_seqlens_q,
        cu_seqlens_k=context.cu_seqlens_k,
        max_seqlen_q=context.max_seqlen_q,
        max_seqlen_k=context.max_seqlen_k,
        softmax_scale=self.scale,
        causal=True,
        block_table=context.block_tables,
    )
```
Prefill 使用变长 FlashAttention：

- 普通 Prefill：直接使用本轮连续的 Q/K/V
- 命中前缀缓存：K/V 改为整个分页缓存，并通过 `block_table` 找到当前序列的历史块
- `cu_seqlens` 描述扁平 Batch 中每条序列的边界
- `causal=True` 保证 Token 只能看到自己及之前的位置

```python
else:
    o = flash_attn_with_kvcache(
        q.unsqueeze(1), k_cache, v_cache,
        cache_seqlens=context.context_lens,
        block_table=context.block_tables,
        softmax_scale=self.scale,
        causal=True,
    )
```
Decode 时每条序列只有一个 Query，使用专门的 KV Cache Attention：

- `context_lens` 限制每条序列的有效缓存长度
- `block_table` 将逻辑位置映射到非连续的物理块
- 不需要复制、拼接每条序列的历史 K/V

因此 PagedAttention 的核心并不是改变注意力公式，而是让注意力 Kernel 能通过页表直接读取离散物理块。

### sampler.py
```python
logits = logits.float().div_(temperatures.unsqueeze(1))
probs = torch.softmax(logits, dim=-1)
sample_tokens = probs.div_(
    torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
).argmax(dim=-1)
```
采样流程：

1. logits 转为 FP32，并为 Batch 中每条序列除以各自温度
2. Softmax 得到概率
3. 为每个候选 Token 生成独立的指数分布随机数
4. 取 `probability / exponential_noise` 最大的位置

最后两步属于指数竞赛（Exponential Race），与按分类分布进行 multinomial 采样等价，适合用逐元素操作和 `argmax` 实现。`@torch.compile` 可将这些操作编译优化。

温度越低，原 logits 差距被放大，输出更确定；温度越高，概率分布更平坦。项目在 `SamplingParams` 中禁止温度为 0，所以没有单独实现 Greedy Sampling。

## 工程优化
### loader.py
```python
def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(
        model, "packed_modules_mapping", {}
    )
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                ...
```
逐个打开模型目录中的 safetensors 文件，并将权重先映射到 CPU。`safe_open` 按需读取单个 Tensor，不需要一次把所有权重文件完整载入内存。

```python
for k in packed_modules_mapping:
    if k in weight_name:
        v, shard_id = packed_modules_mapping[k]
        param_name = weight_name.replace(k, v)
        param = model.get_parameter(param_name)
        weight_loader = getattr(param, "weight_loader")
        weight_loader(
            param, f.get_tensor(weight_name), shard_id
        )
        break
```
对于 Q/K/V 和 Gate/Up 等合并权重：

1. 把 Hugging Face 参数名替换为推理模型中的合并参数名
2. 取得目标参数上绑定的专用 `weight_loader`
3. 传入 `shard_id`，让 Loader 知道它属于合并参数的哪一段
4. 专用 Loader 同时完成“合并位置选择”和“张量并行 rank 切分”

```python
else:
    param = model.get_parameter(weight_name)
    weight_loader = getattr(
        param, "weight_loader", default_weight_loader
    )
    weight_loader(param, f.get_tensor(weight_name))
```
普通权重优先调用参数自己的 Loader：

- 并行 Linear、Embedding 使用专用 Loader，只复制当前 rank 的权重分片
- RMSNorm 等未绑定专用 Loader 的参数使用 `default_weight_loader`，复制完整权重

将加载逻辑绑定在参数本身，而不是在总 Loader 中硬编码所有层类型，使新增并行层时只需定义该参数应如何切分。

## 一次 Step 的完整数据流
```
Scheduler.schedule()
    ↓ 选择 Sequence，分配/扩展 block_table
ModelRunner.prepare_prefill() / prepare_decode()
    ↓ input_ids、positions、slot_mapping、block_tables
set_context()
    ↓
Qwen3 forward
    ↓ 每层 Attention 读取 Context
store_kvcache()
    ↓ 写入分页 KV Cache
FlashAttention
    ↓ hidden states
ParallelLMHead
    ↓ rank 0 获得完整 logits
Sampler
    ↓ token_ids
Scheduler.postprocess()
    ↓ 更新 Sequence、哈希完整块、结束或进入下一轮
```

这套实现的关键点是把三类状态分开：

- `Sequence`：一条请求的 Token 和生命周期状态
- `BlockManager`：CPU 侧 KV Cache 页表、引用计数和前缀哈希
- `Context`：当前一次 GPU 前向需要的临时批量元数据

三者通过 `Scheduler` 和 `ModelRunner` 串联起来，实现了连续批处理、Chunked Prefill、Paged KV Cache、前缀缓存、张量并行和 CUDA Graph Decode。
