# Online Softmax
## 普通 Softmax → Stable Softmax
### 普通 Softmax
给定一个向量：
$$x = [x_1, x_2, \dots, x_n]$$
普通 Softmax 的定义为：
$$softmax(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

执行了两个主要操作：
1. 指数化：对每个元素进行指数运算
2. 归一化：除以所有指数值的和

输出性质：
1. $0 < softmax(x_i) < 1$
2. $\sum_{i} softmax(x_i) = 1$

因此，Softmax 可以将原始的 logits 转化为概率分布

### 普通 Softmax 的数值问题
在计算机中，浮点数的表示范围有限。当 $x_i$ 很大时，$e^{x_i}$ 会因为数值过大而导致溢出，变成 inf

### Stable Softmax
为了解决上述溢出问题，利用 Softmax 的一个重要数学性质：平移不变性

给所有元素同时减去同一个常数 $c$，Softmax 的结果保持不变：
$$softmax(x_i) = softmax(x_i - c)$$

为了确保数值稳定，我们取 $c = m$，其中 $m = \max_{j} x_j$（输入向量中的最大值）

Stable Softmax 的计算公式为：
$$softmax(x_i) = \frac{e^{x_i - m}}{\sum_{j} e^{x_j - m}}$$

通过减去最大值 $m$：
- 对于所有元素，$x_i - m \le 0$
- 因此，$e^{x_i - m} \le e^0 = 1$

这样，所有指数项的最大值被限制在 1 以内，从而彻底避免了数值溢出的问题

```Python
def naive_softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x))

def stable_softmax(x):
    x_max = torch.max(x)
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x)
```

## Online Softmax for 1D Vector
### Online Softmax 的问题设定
假设我们已经处理了前 $t$ 个元素：
$$x_1, x_2, \dots, x_t$$

已经知道：
$$m_t = \max(x_1, \dots, x_t)$$
$$l_t = \sum_{j=1}^{t} e^{x_j-m_t}$$

现在来了一个新元素：
$$x_{t+1}$$

我们希望不用重新扫描前面的所有元素，就得到：
$m_{t+1}$ 和 $l_{t+1}$

### 最大值更新
$$m_{t+1} = \max(m_t, x_{t+1})$$

### 分母 $l$ 更新
旧分母是：
$$l_t = \sum_{j=1}^{t} e^{x_j-m_t}$$

新分母应该是：
$$l_{t+1} = \sum_{j=1}^{t+1} e^{x_j-m_{t+1}}$$

$$\sum_{j=1}^{t} e^{x_j-m_{t+1}} = e^{m_t-m_{t+1}} \sum_{j=1}^{t} e^{x_j-m_t}$$

所以：
$$\sum_{j=1}^{t} e^{x_j-m_{t+1}} = e^{m_t-m_{t+1}}l_t$$

因此：
$$l_{t+1} = e^{m_t-m_{t+1}}l_t + e^{x_{t+1}-m_{t+1}}$$

这就是 Online Softmax for 1D vector 的核心公式

```Python
def stable_softmax(x):
    m = torch.max(x)
    l = torch.sum(torch.exp(x - m))
    return torch.exp(x - m) / l

def online_softmax_1d(x):
    m = torch.tensor(float("-inf"), dtype=x.dtype)
    l = torch.tensor(0.0, dtype=x.dtype)

    for i in range(x.numel()):
        xi = x[i]

        m_new = torch.maximum(m, xi)
        l_new = torch.exp(m - m_new) * l + torch.exp(xi - m_new)

        m = m_new
        l = l_new

    return torch.exp(x - m) / l
```

## block-wise online softmax
### element-wise to block-wise
element-wise:
$$m_{new} = \max(m_{old}, x_i)$$
$$l_{new} = e^{m_{old}-m_{new}}l_{old} + e^{x_i-m_{new}}$$

如果一次来一个 block：
$$B = [x_a, x_{a+1}, \dots, x_b]$$
那么先在 block 内部算两个量：
$$m_{blk} = \max_{j \in blk} x_j$$
$$l_{blk} = \sum_{j \in blk} e^{x_j-m_{blk}}$$
然后把旧状态和新 block 合并

### Block-wise Online Softmax 核心公式
旧状态：
$$m_{old}$$
$$l_{old} = \sum_{j \in old} e^{x_j-m_{old}}$$

新 block 状态：
$$m_{blk}$$
$$l_{blk} = \sum_{j \in blk} e^{x_j-m_{blk}}$$

合并后的最大值：
$$m_{new} = \max(m_{old}, m_{blk})$$

合并后的分母：
$$l_{new} = e^{m_{old}-m_{new}}l_{old} + e^{m_{blk}-m_{new}}l_{blk}$$

```Python
import torch

def stable_softmax(x):
    m = torch.max(x)
    l = torch.sum(torch.exp(x - m))
    return torch.exp(x - m) / l

def blockwise_online_softmax_1d(x, block_size):
    m = torch.tensor(float("-inf"), dtype=x.dtype, device=x.device)
    l = torch.tensor(0.0, dtype=x.dtype, device=x.device)

    for start in range(0, x.numel(), block_size):
        block = x[start:start + block_size]

        m_blk = torch.max(block)
        l_blk = torch.sum(torch.exp(block - m_blk))

        m_new = torch.maximum(m, m_blk)

        l_new = (
            torch.exp(m - m_new) * l
            +
            torch.exp(m_blk - m_new) * l_blk
        )

        m = m_new
        l = l_new

    return torch.exp(x - m) / l
```

## Online Softmax Attention
### 先只看单个 query 的 attention
$$\boldsymbol{q} \in \mathbb{R}^d$$

有一组 key/value：
$$\boldsymbol{K} = [\boldsymbol{k}_1, \boldsymbol{k}_2, \dots, \boldsymbol{k}_N]$$
$$\boldsymbol{V} = [\boldsymbol{v}_1, \boldsymbol{v}_2, \dots, \boldsymbol{v}_N]$$

attention score 是：
$$s_j = \frac{\boldsymbol{q}\boldsymbol{k}_j^T}{\sqrt{d}}$$

attention 输出是：
$$\boldsymbol{o} = \sum_{j=1}^{N} softmax(s)_j \boldsymbol{v}_j$$
$$\boldsymbol{o} = \frac{\sum_{j=1}^{N} e^{s_j - m} \boldsymbol{v}_j}{\sum_{j=1}^{N} e^{s_j - m}}$$

其中：
$$m = \max_j s_j$$

### 不仅维护分母，还维护分子
前面 softmax 只维护：
$$m$$
$$l = \sum_{j} e^{s_j - m}$$

现在 attention 输出还需要分子：
$$\boldsymbol{n} = \sum_{j} e^{s_j - m} \boldsymbol{v}_j$$

最后：
$$\boldsymbol{o} = \frac{\boldsymbol{n}}{l}$$

所以 online softmax attention 要维护三个状态：
1. $m = \text{当前最大} score$
2. $l = \text{当前} softmax \text{分母}$
3. $\boldsymbol{n} = \text{当前} attention \text{未归一化输出分子}$

### 单元素 Online Attention 递推公式
假设已经处理了旧部分：
$$m_{old}$$
$$l_{old} = \sum_{j \in old} e^{s_j - m_{old}}$$
$$\boldsymbol{n}_{old} = \sum_{j \in old} e^{s_j - m_{old}} \boldsymbol{v}_j$$

现在来了一个新元素：
$$s_i$$
$$\boldsymbol{v}_i$$

新的最大值：
$$m_{new} = \max(m_{old}, s_i)$$

分母更新：
$$l_{new} = e^{m_{old} - m_{new}} l_{old} + e^{s_i - m_{new}}$$

分子更新：
$$\boldsymbol{n}_{new} = e^{m_{old} - m_{new}} \boldsymbol{n}_{old} + e^{s_i - m_{new}} \boldsymbol{v}_i$$

最后：
$$\boldsymbol{o} = \frac{\boldsymbol{n}}{l}$$

### Block-wise Online Attention 公式
一个 block 里有：
$$\boldsymbol{K}_{blk}$$
$$\boldsymbol{V}_{blk}$$

先算当前 block 的 scores：
$$\boldsymbol{s}_{blk} = \boldsymbol{q}\boldsymbol{K}_{blk}^T$$

block 内最大值：
$$m_{blk} = \max(\boldsymbol{s}_{blk})$$

block 内分母：
$$l_{blk} = \sum_{j \in blk} e^{s_j - m_{blk}}$$

block 内分子：
$$\boldsymbol{n}_{blk} = \sum_{j \in blk} e^{s_j - m_{blk}} \boldsymbol{v}_j$$

然后和旧状态合并：
$$m_{new} = \max(m_{old}, m_{blk})$$
$$l_{new} = e^{m_{old} - m_{new}} l_{old} + e^{m_{blk} - m_{new}} l_{blk}$$
$$\boldsymbol{n}_{new} = e^{m_{old} - m_{new}} \boldsymbol{n}_{old} + e^{m_{blk} - m_{new}} \boldsymbol{n}_{blk}$$

最后：
$$\boldsymbol{o} = \frac{\boldsymbol{n}_{new}}{l_{new}}$$

```Python
import math
import torch

def online_attention_one_query_blockwise(q, K, V, block_size):
    """
    q: [d]
    K: [N, d]
    V: [N, dv]
    return: [dv]
    """
    d = q.shape[0]
    N = K.shape[0]
    dv = V.shape[1]

    m = torch.tensor(float("-inf"), dtype=q.dtype, device=q.device)
    l = torch.tensor(0.0, dtype=q.dtype, device=q.device)
    n = torch.zeros(dv, dtype=q.dtype, device=q.device)

    for start in range(0, N, block_size):
        K_blk = K[start:start + block_size]   # [B, d]
        V_blk = V[start:start + block_size]   # [B, dv]

        # 当前 block 的 scores
        scores_blk = K_blk @ q / math.sqrt(d)  # [B]

        # 当前 block 的局部 stable softmax 统计量
        m_blk = torch.max(scores_blk)
        p_blk = torch.exp(scores_blk - m_blk)  # [B], 未归一化概率
        l_blk = torch.sum(p_blk)               # scalar

        # 当前 block 的未归一化 attention 分子
        n_blk = p_blk @ V_blk                  # [dv]

        # 合并 old state 和 block state
        m_new = torch.maximum(m, m_blk)

        alpha = torch.exp(m - m_new)
        beta = torch.exp(m_blk - m_new)

        l_new = alpha * l + beta * l_blk
        n_new = alpha * n + beta * n_blk

        m = m_new
        l = l_new
        n = n_new

    out = n / l
    return out
```

## mini FlashAttention forward
### 我们要实现什么
标准 attention 是：
$$S = QK^T / \sqrt{d}$$
$$P = softmax(S)$$
$$O = PV$$

其中：
$$Q \in \mathbb{R}^{N \times d}$$
$$K \in \mathbb{R}^{N \times d}$$
$$V \in \mathbb{R}^{N \times d_v}$$

标准实现会显式生成：
$$S \in \mathbb{R}^{N \times N}$$
和：
$$P \in \mathbb{R}^{N \times N}$$

FlashAttention 的核心是：不保存完整 $S$ 和 $P$，而是按 block 计算，并对每一行 query 维护 $m, l, n$

### mini FlashAttention 的核心状态
对于一个 query block:
$$Q_i \in \mathbb{R}^{B_r \times d}$$

遍历所有 key/value block:
$$K_j \in \mathbb{R}^{B_c \times d}$$
$$V_j \in \mathbb{R}^{B_c \times d_v}$$

当前 score block 是：
$$S_{ij} = Q_i K_j^T / \sqrt{d}$$

shape 是：
$$S_{ij} \in \mathbb{R}^{B_r \times B_c}$$

对 $Q_i$ 里的每一行，都维护：
$$m \in \mathbb{R}^{B_r}$$
$$l \in \mathbb{R}^{B_r}$$
$$n \in \mathbb{R}^{B_r \times d_v}$$

其中：

$m_r =$ 当前第 $r$ 个 query 已见过的最大 score

$l_r =$ 当前第 $r$ 个 query 的 softmax denominator

$n_r =$ 当前第 $r$ 个 query 的 attention numerator

最后：
$$O_i = n / l$$