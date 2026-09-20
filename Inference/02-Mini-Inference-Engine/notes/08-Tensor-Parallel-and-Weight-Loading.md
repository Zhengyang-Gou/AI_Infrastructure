# 08 · 张量并行与权重加载

理解线性层和词表如何切分，以及原始权重如何装入合并参数和各 rank 分片。

对应源码：[linear.py](../source/nano-vllm/nanovllm/layers/linear.py)、[embed_head.py](../source/nano-vllm/nanovllm/layers/embed_head.py)、[loader.py](../source/nano-vllm/nanovllm/utils/loader.py)。

## linear.py

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

## embed_head.py

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

## loader.py

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

---

[学习目录](README.md) · [上一篇：Attention 与 KV Cache 读写](07-Attention-and-Cache-Access.md) · [下一篇：采样与输出 Token](09-Sampling.md)
