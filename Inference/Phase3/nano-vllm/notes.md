# LLMEngine 源码讲解

## 整体定位

`LLMEngine` 是 nano-vLLM 的顶层控制器。它不直接实现 Transformer，而是组织 tokenizer、请求状态、调度器与 GPU 执行器：

```text
prompts → tokenize → Sequence → Scheduler → ModelRunner
        → 更新请求状态 → 重复 step → decode 输出
```

`nanovllm/llm.py` 中的 `LLM` 直接继承 `LLMEngine`，所以 `LLM.generate()` 实际执行的就是这里的 `generate()`。

## `__init__`：初始化引擎

### 构造配置

```python
config_fields = {field.name for field in fields(Config)}
config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
config = Config(model, **config_kwargs)
Sequence.block_size = config.kvcache_block_size
```

先取得 `Config` 的合法字段，再过滤用户传入的 `kwargs`。未在 `Config` 中定义的参数会被忽略。

`Sequence.block_size` 决定一条序列需要多少个 KV Cache block、最后一个 block 中有多少 token。它必须与 `BlockManager` 使用的 block size 一致。

### 创建 Tensor Parallel 子进程

```python
self.ps = []
self.events = []
ctx = mp.get_context("spawn")
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
    self.ps.append(process)
    self.events.append(event)
```

每个 TP rank 对应一个 `ModelRunner`：rank 0 留在主进程，rank 1 到 `TP-1` 使用 `spawn` 创建子进程。循环从 1 开始，是因为 rank 0 稍后单独创建。

`spawn` 不会复制已有 CUDA 上下文，适合 CUDA 多进程。每个 `Event` 用于通知对应 rank：共享内存中已经写入新的方法调用。

单卡时 `tensor_parallel_size=1`，循环不会执行。

### 创建核心组件

```python
self.model_runner = ModelRunner(config, 0, self.events)
self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
config.eos = self.tokenizer.eos_token_id
self.scheduler = Scheduler(config)
atexit.register(self.exit)
```

rank 0 的 `ModelRunner` 会初始化 NCCL、加载模型权重、warmup、分配 KV Cache，并按配置捕获 CUDA Graph。

加载 tokenizer 后，先把 EOS token id 写入 `config`，再创建 `Scheduler`。顺序不能交换，因为 Scheduler 初始化时就会保存 EOS，用它判断请求是否结束。

`atexit.register()` 保证 Python 正常退出时清理模型进程和分布式资源。

```text
LLMEngine
├── tokenizer
├── scheduler
│   └── block_manager
└── model_runner（rank 0）
    └── 其他 TP ModelRunner
```

## `exit`：释放资源

```python
def exit(self):
    self.model_runner.call("exit")
    del self.model_runner
    for p in self.ps:
        p.join()
```

`call("exit")` 让所有 TP rank 执行 `ModelRunner.exit()`，关闭共享内存、删除 CUDA Graph、等待 CUDA 完成并销毁 NCCL ProcessGroup。

随后删除主 `ModelRunner`，最后通过 `join()` 等待所有子进程完全结束，避免残留进程。

## `add_request`：添加请求

```python
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
    if isinstance(prompt, str):
        prompt = self.tokenizer.encode(prompt)
    seq = Sequence(prompt, sampling_params)
    self.scheduler.add(seq)
```

`prompt` 可以是字符串，也可以是已经编码好的 token id 列表。字符串会先经过 tokenizer；token id 列表直接使用，方便调用者在外部应用 chat template。

`Sequence` 保存请求的 token、采样参数、运行状态、KV Cache block table、已缓存 token 数与本轮调度 token 数。新建请求最后进入 Scheduler 的 `waiting` 队列，此时尚未执行模型。

## `step`：执行一轮推理

```python
def step(self):
    seqs, is_prefill = self.scheduler.schedule()
    num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

### 1. 调度

```python
seqs, is_prefill = self.scheduler.schedule()
```

- `seqs`：本轮组成 batch 的 Sequence
- `is_prefill`：本轮执行 prefill 还是 decode

Scheduler 如果能从 waiting 队列调度请求，就执行 prefill；否则从 running 队列调度 decode。因此一个 step 只包含一个阶段。

### 2. 统计 token

Prefill 中一条序列可能计算多个 token，所以累加 `num_scheduled_tokens`。Decode 中每条序列只计算一个 token，所以数量等于 `len(seqs)`。

Decode 数量被设成负数：

```text
num_tokens > 0：prefill
num_tokens < 0：decode
```

这样 `generate()` 不需要额外返回阶段标志即可分别统计吞吐量。

### 3. 执行模型

```python
token_ids = self.model_runner.call("run", seqs, is_prefill)
```

`ModelRunner.run()` 根据阶段调用 `prepare_prefill()` 或 `prepare_decode()`，然后执行模型 forward、计算 logits 并采样。返回的 `token_ids` 与 `seqs` 一一对应。

### 4. 后处理

```python
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

后处理会：

- 更新已缓存 token 数
- 为完整 KV block 建立 prefix-cache hash
- 将采样 token 追加到 Sequence
- 判断 EOS 或 `max_tokens`
- 标记完成请求并释放它的 KV Cache

Chunked prefill 如果只完成 prompt 的一部分，不会立即追加采样 token，而会等待 prompt 全部计算完成。

### 5. 返回完成请求

```python
outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
```

只返回本轮刚完成的请求；未完成请求继续留在 Scheduler 中。

```python
([], 1024)                   # prefill 了 1024 tokens，无请求完成
([(3, [100, 200, 300])], -8) # decode batch=8，seq 3 完成
```

## `is_finished`：判断全部完成

```python
def is_finished(self):
    return self.scheduler.is_finished()
```

Scheduler 只有在 `waiting` 和 `running` 两个队列都为空时才返回 `True`：

```text
waiting 非空：还有请求等待或继续 prefill
running 非空：还有请求正在 decode
两者都为空：全部完成
```

## `generate`：完成一批生成

```python
def generate(
    self,
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
    use_tqdm: bool = True,
) -> list[str]:
```

这是同步离线推理入口。源码返回注解写成 `list[str]`，但实际返回的是包含 `text` 和 `token_ids` 的字典列表。

### 创建进度条与统一参数

```python
pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
if not isinstance(sampling_params, list):
    sampling_params = [sampling_params] * len(prompts)
```

进度条按请求数计算。如果只传一个 `SamplingParams`，所有 prompt 共用它。这里复制的是同一对象引用，但推理过程只读取参数，所以不会相互影响。

### 将请求加入 Scheduler

```python
for prompt, sp in zip(prompts, sampling_params):
    self.add_request(prompt, sp)
```

每个 prompt 与采样参数配对后进入 waiting 队列。需要注意：两者长度不同时，`zip()` 会在较短列表结束，多出的元素不会处理。

### 循环执行 step

```python
outputs = {}
prefill_throughput = decode_throughput = 0.
while not self.is_finished():
    t = perf_counter()
    output, num_tokens = self.step()
```

只要 Scheduler 中还有请求，就持续执行 `step()`。结果暂存在以 `seq_id` 为 key 的字典中，因为短请求可能比排在前面的长请求更早结束。

### 计算吞吐量

```python
if num_tokens > 0:
    prefill_throughput = num_tokens / (perf_counter() - t)
else:
    decode_throughput = -num_tokens / (perf_counter() - t)
```

公式为：

```text
tokens/s = 本轮 token 数 / 本轮耗时
```

这里显示的是最近一个 step 的瞬时吞吐量，不是整个生成任务的平均吞吐量。

### 收集并恢复顺序

```python
for seq_id, token_ids in output:
    outputs[seq_id] = token_ids
    pbar.update(1)

outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
```

每完成一个请求，进度增加 1。由于 `Sequence.counter` 按请求加入顺序递增，最终按 `seq_id` 排序即可恢复 prompts 的原始顺序。

### 解码结果

```python
outputs = [
    {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
    for token_ids in outputs
]
return outputs
```

解码的只有 completion tokens，不包含输入 prompt。返回格式为：

```python
[{"text": "Nano-vLLM is ...", "token_ids": [123, 456, 789]}]
```

## 完整调用链

```text
LLMEngine.generate
├── add_request
│   ├── tokenizer.encode
│   ├── Sequence(...)
│   └── Scheduler.add
│
└── while not is_finished
    └── step
        ├── Scheduler.schedule
        ├── ModelRunner.call("run")
        │   ├── prepare_prefill / prepare_decode
        │   ├── model forward
        │   └── sampler
        └── Scheduler.postprocess
            ├── 更新 KV Cache
            ├── 追加 token
            ├── 判断结束条件
            └── 释放完成请求的 KV Cache
```

## 核心设计总结

| 模块 | 职责 |
|---|---|
| `LLMEngine` | 控制完整生成循环 |
| `Sequence` | 保存单个请求状态 |
| `Scheduler` | 决定本轮运行哪些请求 |
| `BlockManager` | 管理物理 KV Cache block |
| `ModelRunner` | 准备输入并执行 GPU 模型 |
| `Sampler` | 从 logits 选择下一个 token |

`generate()` 的本质就是不断执行：

```text
schedule → run → postprocess
```

每个 step 都重新组成 batch，请求可以动态进入和退出运行集合，这是 continuous batching 的基础。Prefill 与 Decode 共用同一控制循环，具体差异由 `is_prefill` 传递到 Scheduler 和 ModelRunner。
