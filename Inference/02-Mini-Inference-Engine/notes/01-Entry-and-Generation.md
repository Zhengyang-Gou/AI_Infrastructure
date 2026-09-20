# 01 · 入口配置与生成主循环

先建立一次 generate 调用的整体印象；遇到调度器和执行器时先了解职责，后续章节再展开。

对应源码：[example.py](../source/nano-vllm/example.py)、[llm.py](../source/nano-vllm/nanovllm/llm.py)、[config.py](../source/nano-vllm/nanovllm/config.py)、[sampling_params.py](../source/nano-vllm/nanovllm/sampling_params.py)、[llm_engine.py](../source/nano-vllm/nanovllm/engine/llm_engine.py)。

## example.py

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

## llm.py

```python
from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    pass
```
把 LLMEngine 包装成一个对外暴露的 LLM 类

LLM 会自动拥有 LLMEngine 的全部方法和属性

## config.py

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

## sampling_params.py

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

## llm_engine.py

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

---

[学习目录](README.md) · [下一篇：请求状态与生命周期](02-Sequence-and-Lifecycle.md)
