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
        # 获取张量并行规模。
        tp_size = dist.get_world_size()
        # 按 rank 划分 Q Head。
        self.num_heads = num_heads // tp_size
        # KV Head 数也按 rank 划分，可小于 Q Head 数。
        self.num_kv_heads = num_kv_heads // tp_size
        # 优先采用显式 head_dim，否则由隐藏维度推导。
        self.head_dim = (
            head_dim or hidden_size // num_heads
        )
        # 本 rank 的 Q 投影宽度。
        self.q_size = self.num_heads * self.head_dim
        # 本 rank 的 K 或 V 投影宽度。
        self.kv_size = self.num_kv_heads * self.head_dim
        # 缩放点积使用 1 / sqrt(head_dim)。
        self.scaling = self.head_dim ** -0.5
```

**功能描述：** 确定当前张量并行 rank 负责的注意力头及投影大小，供 QKV 拆分和缩放点积注意力使用。GQA 中多个 Q Head 可以共享 K/V Head。

```python
# 合并线性层一次生成本 rank 的 Q/K/V。
qkv = self.qkv_proj(hidden_states)
# 按 Q 与 KV 各自宽度拆开。
q, k, v = qkv.split(
    [self.q_size, self.kv_size, self.kv_size], dim=-1
)
# 恢复 Token、Head、Head 维度三个轴。
q = q.view(-1, self.num_heads, self.head_dim)
k = k.view(-1, self.num_kv_heads, self.head_dim)
v = v.view(-1, self.num_kv_heads, self.head_dim)
# 此配置分支对 Q 和 K 的每个 Head 进行 RMSNorm。
if not self.qkv_bias:
    q = self.q_norm(q)
    k = self.k_norm(k)
# 为 Q/K 注入 RoPE 位置信息。
q, k = self.rotary_emb(positions, q, k)
# 写入新 KV，并执行对应阶段的 Attention。
o = self.attn(q, k, v)
# 合并 Head，再由行并行投影汇总各 rank 的结果。
return self.o_proj(o.flatten(1, -1))
```

**功能描述：** 完成注意力子层的前向计算：将隐藏状态投影为 Q/K/V，加入位置编码并结合缓存计算注意力，最后投影回模型隐藏空间。

```python
class Qwen3MLP(nn.Module):
    def forward(self, x):
        # 一次列并行矩阵乘同时计算 Gate 与 Up。
        gate_up = self.gate_up_proj(x)
        # 执行 SiLU(gate) × up。
        x = self.act_fn(gate_up)
        # 行并行投影回隐藏维度并汇总结果。
        return self.down_proj(x)
```

**功能描述：** 实现 SwiGLU 前馈网络，将 Gate 和 Up 合并投影，经门控激活后再降维回隐藏空间。

补充示意：

```
down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

```python
def forward(self, positions, hidden_states, residual):
    # 首层保存 Embedding 输出作为残差。
    if residual is None:
        hidden_states, residual = (
            self.input_layernorm(hidden_states), hidden_states
        )
    # 后续层在归一化前合并上一层 MLP 输出与残差。
    else:
        hidden_states, residual = self.input_layernorm(
            hidden_states, residual
        )
    # 执行注意力子层。
    hidden_states = self.self_attn(positions, hidden_states)
    # 合并 Attention 输出与残差，再归一化。
    hidden_states, residual = self.post_attention_layernorm(
        hidden_states, residual
    )
    # MLP 输出暂不相加，留给下一次归一化处理。
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

**功能描述：** 执行一层 Decoder，并将残差加法与后续 RMSNorm 合并。延迟合并 MLP 残差可减少独立加法 Kernel 和中间读写，最后一层残差由模型末尾的 norm 收尾。

```python
packed_modules_mapping = {
    # Q/K/V 分别写入 qkv_proj 的对应区域。
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    # Gate/Up 分别对应 gate_up_proj 的第 0、1 段。
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
}
```

**功能描述：** 定义原始权重名到合并参数及内部片段的映射，使 Loader 能把独立保存的 Q/K/V 和 Gate/Up 装入推理模型的合并权重。

模型整体接口中，`Qwen3ForCausalLM.forward()` 返回隐藏状态，`compute_logits()` 再调用 LM Head；启用 `tie_word_embeddings` 时，LM Head 与输入 Embedding 共享权重数据。

## layernorm.py

`RMSNorm` 的公式：
```
y = x / sqrt(mean(x²) + eps) × weight
```
与 LayerNorm 不同，它不减均值，也没有 Bias。

实现先转为 FP32 计算平方均值和归一化，减少低精度数值误差，最后转回原 dtype。`@torch.compile` 用于融合逐元素操作。

`add_rms_forward(x, residual)` 先执行：
```python
# 将当前子层输出合并到累积残差，随后再执行 RMSNorm。
residual = x + residual
```

**功能描述：** 完成归一化前的残差更新。完整的 `add_rms_forward` 随后返回归一化结果和新残差，供后续子层继续使用。

## activation.py

```python
class SiluAndMul(nn.Module):
    # 编译优化切分与逐元素运算。
    @torch.compile
    def forward(self, x):
        # 最后一维拆成 Gate 与 Up 两半。
        x, y = x.chunk(2, -1)
        # 对 Gate 应用 SiLU，再与 Up 逐元素相乘。
        return F.silu(x) * y
```

**功能描述：** 实现 SwiGLU 的门控激活，接收合并投影后的张量，输出激活后的前馈中间特征。

## rotary_embedding.py

```python
# 构造各旋转维度对应的逆频率。
inv_freq = 1.0 / (
    base ** (
        torch.arange(0, rotary_dim, 2) / rotary_dim
    )
)
# 枚举模型支持的位置索引。
t = torch.arange(max_position_embeddings)
# 位置与逆频率外积，得到每个位置的旋转角。
freqs = torch.einsum("i,j -> ij", t, inv_freq)
# 拼接 cos 和 sin，供前向查表。
cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
```

**功能描述：** 预计算各位置的 RoPE 三角函数表，供每层 Attention 按位置查表使用。完整实现将其注册为非持久 Buffer，随模型迁移设备但不写入权重文件。

```python
# 暂转 FP32，并将 Head 两半作为成对坐标。
x1, x2 = torch.chunk(x.float(), 2, dim=-1)
# 应用二维旋转矩阵的第一行。
y1 = x1 * cos - x2 * sin
# 应用二维旋转矩阵的第二行。
y2 = x2 * cos + x1 * sin
# 拼回 Head 维度并恢复原始数据类型。
return torch.cat((y1, y2), dim=-1).to(x.dtype)
```

**功能描述：** 按位置旋转 Q/K 的成对维度，将位置信息融入注意力计算。get_rope 还通过缓存模块实例，让相同配置的各层共享 cos/sin 表。

---

[学习目录](README.md) · [上一篇：模型执行器与输入组织](05-ModelRunner-and-Context.md) · [下一篇：Attention 与 KV Cache 读写](07-Attention-and-Cache-Access.md)
