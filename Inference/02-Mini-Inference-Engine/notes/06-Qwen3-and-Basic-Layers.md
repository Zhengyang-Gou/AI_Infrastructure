# 06 · Qwen3 模型与基础算子

沿模型前向理解 Attention、MLP、残差、RMSNorm 和 RoPE；并行层的切分与通信在第 08 篇展开。

对应源码：[qwen3.py](../source/nano-vllm/nanovllm/models/qwen3.py)、[layernorm.py](../source/nano-vllm/nanovllm/layers/layernorm.py)、[activation.py](../source/nano-vllm/nanovllm/layers/activation.py)、[rotary_embedding.py](../source/nano-vllm/nanovllm/layers/rotary_embedding.py)。

## qwen3.py

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

## layernorm.py

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

## activation.py

```python
class SiluAndMul(nn.Module):
    @torch.compile
    def forward(self, x):
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
```
输入是合并的 Gate 和 Up 投影结果，沿最后一维一分为二，计算 SwiGLU 激活。使用 `torch.compile` 可以将切分、SiLU 和乘法尽可能融合。

## rotary_embedding.py

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

---

[学习目录](README.md) · [上一篇：模型执行器与输入组织](05-ModelRunner-and-Context.md) · [下一篇：Attention 与 KV Cache 读写](07-Attention-and-Cache-Access.md)
