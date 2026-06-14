# FlashAttrntion-1
## 标准 Attention 的 IO 问题
FlashAttention-1 论文的核心判断是：传统 attention 缺少 IO-aware 视角，也就是没有认真考虑 GPU 不同层级内存之间的读写成本。论文提出 FlashAttention 的目标就是减少 HBM 和片上 SRAM 之间的读写，而不是把 attention 近似掉

标准 Attention: 
$$S = QK^T$$
$$P = \text{softmax}(S)$$
$$O = PV$$

其中：
$$Q,K,V,O \in \mathbb{R}^{N \times d}$$

但：
$$S,P \in \mathbb{R}^{N \times N}$$

标准实现大概会这样：

```
读 Q,K → 算 S → 写 S
读 S → 算 softmax → 写 P
读 P,V → 算 O → 写 O
```

也就是说，中间的 $S$ 和 $P$ 会被写入 HBM，再读出来

假设：
$$N = 4096,\quad d = 64$$

那么：
$$Q : 4096 \times 64 = 262K$$
$$S : 4096 \times 4096 = 16.7M$$

所以：
$$S \text{ 比 } Q \text{ 大 } 64 \text{ 倍}$$
fp16 下：$$S \approx 32MB$$
$$P \approx 32MB$$
单 head 就有几十 MB 的中间矩阵读写

FlashAttention 的动机：不把完整的 $S$ 和 $P$ 存到 HBM

它不是少算 attention，而是少搬数据

一句话：
$$\text{标准 Attention 的瓶颈} = N^2 \text{ 中间矩阵的 HBM IO}$$

## Online Softmax to Online Attention
### 普通 softmax
对一个向量：
$$x = [x_1, x_2, \dots, x_N]$$

softmax 是：
$$p_j = \frac{e^{x_j}}{\sum_{t=1}^N e^{x_t}}$$

为了数值稳定，写成：
$$m = \max_j x_j$$
$$l = \sum_{j=1}^N e^{x_j - m}$$

于是：
$$p_j = \frac{e^{x_j - m}}{l}$$

Online Softmax 的重点是：即使我们不是一次看到完整的 $x$，而是一块一块看到，也能维护正确的：
$$m$$
$$l$$

### Attention 其实就是 softmax 后再加权求和
对某一个 query:
$$q_i$$

它和所有 key 做点积：
$$s_j = q_i \cdot k_j$$

得到一行 score:
$$s = [s_1, s_2, \dots, s_N]$$

标准 attention 是：
$$p_j = \frac{e^{s_j}}{\sum_{t=1}^N e^{s_t}}$$

然后用这个概率加权 $V$:
$$o_i = \sum_{j=1}^N p_j v_j$$

也就是：
$$o_i = \sum_{j=1}^N \frac{e^{s_j}}{\sum_{t=1}^N e^{s_t}} v_j$$

数值稳定写法：
$$m = \max_j s_j$$
$$l = \sum_{j=1}^N e^{s_j - m}$$
$$o_i = \frac{\sum_{j=1}^N e^{s_j - m} v_j}{l}$$

这里出现了一个非常关键的东西：
$$\sum_{j=1}^N e^{s_j - m} v_j$$

它是 softmax 分子乘 $V$ 后的累加

### Online Softmax 到 Online Attention 的变化
Online Softmax 维护两个量：
$$m$$
$$l$$

Online Attention 维护三个量：
$$m$$
$$l$$
$$o$$

其中：
- $m$: 当前看到的最大 score
- $l$: 当前 softmax denominator
- $o$: 当前 attention 输出

但为了推导更清楚，我们先不用 $o$，而是维护一个未归一化的累加量：
$$a = \sum_{j} e^{s_j - m} v_j$$

最后：
$$o = \frac{a}{l}$$

所以 Online Attention 的核心其实是维护：
$$m, \quad l, \quad a$$

最后再算：
$$o = a / l$$

### 单个新元素的更新
假设当前已经处理了一部分 score，维护了：
$$m_{\text{old}}$$
$$l_{\text{old}}$$
$$a_{\text{old}}$$

其中：
$$l_{\text{old}} = \sum_{\text{old}} e^{s_j - m_{\text{old}}}$$
$$a_{\text{old}} = \sum_{\text{old}} e^{s_j - m_{\text{old}}} v_j$$

现在来了一个新 score:
$$s_{\text{new}}$$

以及对应的 value:
$$v_{\text{new}}$$

新的最大值是：
$$m_{\text{new}} = \max(m_{\text{old}}, s_{\text{new}})$$

旧的累加量要从基准 $m_{\text{old}}$ 换到基准 $m_{\text{new}}$

所以：
$$l_{\text{old}}$$

要乘：
$$e^{m_{\text{old}} - m_{\text{new}}}$$

因为：
$$e^{s_j - m_{\text{old}}} \cdot e^{m_{\text{old}} - m_{\text{new}}} = e^{s_j - m_{\text{new}}}$$

因此 denominator 更新为：
$$l_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} l_{\text{old}} + e^{s_{\text{new}} - m_{\text{new}}}$$

同理 numerator 更新为：
$$a_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} a_{\text{old}} + e^{s_{\text{new}} - m_{\text{new}}} v_{\text{new}}$$

最后：
$$o_{\text{new}} = \frac{a_{\text{new}}}{l_{\text{new}}}$$
这就是从 Online Softmax 到 Online Attention 的第一步

### 从单个元素扩展到一个 block
FlashAttention 不会一个元素一个元素处理，而是一块一块处理

假设当前处理一个 query:
$$q_i$$

现在来了一个 block:
$$K_b \in \mathbb{R}^{B \times d}$$
$$V_b \in \mathbb{R}^{B \times d}$$

先算这个 block 的 score:
$$s_b = q_i K_b^T$$

其中：
$$s_b \in \mathbb{R}^B$$

这个 block 内部的最大值：
$$m_b = \max(s_b)$$

block 内部的 denominator:
$$l_b = \sum e^{s_b - m_b}$$

block 内部的 value 加权和：
$$a_b = \sum e^{s_j - m_b} v_j$$

也可以写成矩阵形式：
$$a_b = e^{s_b - m_b} V_b$$

### Block-wise Online Attention 更新公式
现在有旧状态：
$$m_{\text{old}}, \quad l_{\text{old}}, \quad a_{\text{old}}$$

新 block 有：
$$m_b, \quad l_b, \quad a_b$$

新的全局最大值：
$$m_{\text{new}} = \max(m_{\text{old}}, m_b)$$

旧状态缩放因子：
$$\alpha = e^{m_{\text{old}} - m_{\text{new}}}$$

新 block 缩放因子：
$$\beta = e^{m_b - m_{\text{new}}}$$

于是：
$$l_{\text{new}} = \alpha l_{\text{old}} + \beta l_b$$
$$a_{\text{new}} = \alpha a_{\text{old}} + \beta a_b$$

最后：
$$o_{\text{new}} = \frac{a_{\text{new}}}{l_{\text{new}}}$$

这组公式就是 Online Attention 的核心

## 单行 FlashAttention 推导
### 标准做法
标准 attention 对一个 query 是：
```
1. 算所有 score:
   s = q @ K.T

2. 对 s 做 softmax:
   p = softmax(s)

3. 加权求和:
   o = p @ V
```

数学上：
$$s_j = q \cdot k_j$$
$$p_j = \frac{e^{s_j}}{\sum_t e^{s_t}}$$
$$o = \sum_j p_j v_j$$

FlashAttention 的想法是：

不等完整 $s$ 生成出来，而是一块一块扫过 K/V，边扫边更新 softmax 和输出

### 把 K/V 分块
假设把 $K, V$ 按行分成多个 block：
$$K = [K_1; K_2; \dots; K_T]$$
$$V = [V_1; V_2; \dots; V_T]$$

每个 block：
$$K_b \in \mathbb{R}^{B \times d}$$
$$V_b \in \mathbb{R}^{B \times d}$$

对于当前 query:
$$q \in \mathbb{R}^d$$

每次只算一块 score:
$$s_b = qK_b^T$$

其中：
$$s_b \in \mathbb{R}^B$$

### 每个 block 内先算局部信息
对一个 block：
$$s_b = qK_b^T$$

先算 block 内最大值：
$$m_b = \max(s_b)$$

再算 block 内 softmax 分母：
$$l_b = \sum_{j \in b} e^{s_j - m_b}$$

再算 block 内 value 加权和：
$$a_b = \sum_{j \in b} e^{s_j - m_b} v_j$$

也可以写成：
$$a_b = e^{s_b - m_b} V_b$$

注意：
$$e^{s_b - m_b} \in \mathbb{R}^B$$
$$V_b \in \mathbb{R}^{B \times d}$$

所以：
$$a_b \in \mathbb{R}^d$$

### 全局维护三个状态
扫过 K/V block 的过程中，我们维护：
$$m, \quad l, \quad a$$

含义是：
- $m = \text{目前见过的最大 score}$
- $l = \sum_{\text{seen}} e^{s_j - m}$
- $a = \sum_{\text{seen}} e^{s_j - m} v_j$

最终输出：
$$o = \frac{a}{l}$$

初始化：
$$m = -\infty$$
$$l = 0$$
$$a = 0$$

### 核心更新公式
假设旧状态是：
$$m_{\text{old}}, l_{\text{old}}, a_{\text{old}}$$

当前 block 的局部信息是：
$$m_b, l_b, a_b$$

新的最大值：
$$m_{\text{new}} = \max(m_{\text{old}}, m_b)$$

旧状态缩放：
$$\alpha = e^{m_{\text{old}} - m_{\text{new}}}$$

当前 block 缩放：
$$\beta = e^{m_b - m_{\text{new}}}$$

更新 denominator:
$$l_{\text{new}} = \alpha l_{\text{old}} + \beta l_b$$

更新 numerator:
$$a_{\text{new}} = \alpha a_{\text{old}} + \beta a_b$$

然后替换：
$$m \leftarrow m_{\text{new}}$$
$$l \leftarrow l_{\text{new}}$$
$$a \leftarrow a_{\text{new}}$$

最后：
$$o = a / l$$

### 单行 FlashAttention 伪代码
```
输入:
  q: [d]
  K: [N, d]
  V: [N, d]

初始化:
  m = -inf
  l = 0
  a = zeros([d])

for each block K_b, V_b:

  s_b = q @ K_b.T     # [B]

  m_b = max(s_b)      # scalar

  p_b = exp(s_b - m_b) # [B]

  l_b = sum(p_b)      # scalar

  a_b = p_b @ V_b     # [d]

  m_new = max(m, m_b)

  alpha = exp(m - m_new)
  beta  = exp(m_b - m_new)

  l = alpha * l + beta * l_b

  a = alpha * a + beta * a_b

  m = m_new

输出:
  o = a / l
```

## Block-wise FlashAttention Forward
扩展到一组 query：
$$Q_i \in \mathbb{R}^{B_r \times d}$$
一次处理 $B_r$ 行 query

$$S_{ij} = Q_i K_j^T$$

其中：
$$Q_i \in \mathbb{R}^{B_r \times d}$$
$$K_j \in \mathbb{R}^{B_c \times d}$$

所以：
$$S_{ij} \in \mathbb{R}^{B_r \times B_c}$$

现在每次不是算一个 score vector，而是算一个 score block

### Block-wise Attention 的整体结构
```
for each Q block Q_i:
    初始化 m_i, l_i, O_i
    
    for each K/V block K_j, V_j:
        算 S_ij = Q_i @ K_j.T
        用 S_ij 更新 m_i, l_i, O_i
        
    写回 O_i
```
这里：
$$m_i \in \mathbb{R}^{B_r}$$
$$l_i \in \mathbb{R}^{B_r}$$
$$O_i \in \mathbb{R}^{B_r \times d}$$

每一行 query 都有自己的 $m, l, O$

### 每个 block 内算什么
对当前 block：
$$S_{ij} = Q_i K_j^T$$

其中：
$$S_{ij} \in \mathbb{R}^{B_r \times B_c}$$

对每一行做 row-wise max：
$$m_{ij} = \text{rowmax}(S_{ij})$$

所以：
$$m_{ij} \in \mathbb{R}^{B_r}$$

然后算：
$$P_{ij} = e^{S_{ij} - m_{ij}}$$

注意这里的 $m_{ij} \text{ 要按行 broadcast}$：
$$P_{ij}[r, c] = e^{S_{ij}[r, c] - m_{ij}[r]}$$

然后：
$$l_{ij} = \text{rowsum}(P_{ij})$$

所以：
$$l_{ij} \in \mathbb{R}^{B_r}$$

再算当前 block 对输出的贡献：
$$A_{ij} = P_{ij} V_j$$

其中：
$$P_{ij} \in \mathbb{R}^{B_r \times B_c}$$
$$V_j \in \mathbb{R}^{B_c \times d}$$

所以：
$$A_{ij} \in \mathbb{R}^{B_r \times d}$$

### 状态更新公式
旧状态：
$$m_i, \quad l_i, \quad O_i$$

当前 block：
$$m_{ij}, \quad l_{ij}, \quad A_{ij}$$

新的 max：
$$m_i^{\text{new}} = \max(m_i, m_{ij})$$

缩放因子：
$$\alpha = e^{m_i - m_i^{\text{new}}}$$
$$\beta = e^{m_{ij} - m_i^{\text{new}}}$$

新的 denominator：
$$l_i^{\text{new}} = \alpha l_i + \beta l_{ij}$$

如果维护未归一化 accumulator $A_i$，公式是：
$$A_i^{\text{new}} = \alpha A_i + \beta A_{ij}$$

最后：
$$O_i^{\text{new}} = \frac{A_i^{\text{new}}}{l_i^{\text{new}}}$$

但实际论文/实现里常直接维护归一化后的 $O_i$，所以公式写成：
$$O_i^{\text{new}} = \frac{\alpha l_i O_i + \beta A_{ij}}{l_i^{\text{new}}}$$

这和维护 $A_i$ 是等价的，因为：
$$A_i = l_i O_i$$

### 维度
```
Q_i [B_r, d]
K_j [B_c, d]
V_j [B_c, d]

S_ij = Q_i @ K_j.T     -> [B_r, B_c]
P_ij = exp(...)        -> [B_r, B_c]
A_ij = P_ij @ V_j      -> [B_r, d]
O_i update             -> [B_r, d]
```

### Forward 伪代码
```
for each Q block Q_i:

    O_i = 0
    m_i = -inf
    l_i = 0

    for each K/V block K_j, V_j:

        S_ij = Q_i @ K_j.T

        m_ij = rowmax(S_ij)

        P_ij = exp(S_ij - m_ij[:, None])

        l_ij = rowsum(P_ij)

        A_ij = P_ij @ V_j

        m_new = max(m_i, m_ij)

        alpha = exp(m_i - m_new)
        beta  = exp(m_ij - m_new)

        l_new = alpha * l_i + beta * l_ij

        O_i = (
            alpha[:, None] * l_i[:, None] * O_i
            +
            beta[:, None] * A_ij
        ) / l_new[:, None]

        m_i = m_new
        l_i = l_new

    write O_i
```

## FA1 Forward 伪代码
### 输入和分块
设：
$$Q, K, V \in \mathbb{R}^{N \times d}$$

分块：
$$Q_i \in \mathbb{R}^{B_r \times d}$$
$$K_j, V_j \in \mathbb{R}^{B_c \times d}$$

每次计算一个小 score block:
$$S_{ij} = Q_i K_j^T$$

维度是：
$$S_{ij} \in \mathbb{R}^{B_r \times B_c}$$

### 每个 Q block 维护什么
对一个 $Q_i$，维护：
$$m_i \in \mathbb{R}^{B_r}$$
$$l_i \in \mathbb{R}^{B_r}$$
$$O_i \in \mathbb{R}^{B_r \times d}$$

初始化：
$$m_i = -\infty$$
$$l_i = 0$$
$$O_i = 0$$

这里 $m_i, l_i$ 是每一行 query 各自一份

### 一个 block 的计算流程
当前处理：
$$Q_i, K_j, V_j$$

先算：
$$S_{ij} = Q_i K_j^T$$

然后对每一行做局部 max:
$$m_{ij} = \text{rowmax}(S_{ij})$$

再算未归一化 softmax:
$$P_{ij} = e^{S_{ij} - m_{ij}}$$

这里 $m_{ij} \text{ 按行 broadcast}$

然后：
$$l_{ij} = \text{rowsum}(P_{ij})$$
$$A_{ij} = P_{ij} V_j$$

其中：
$$A_{ij} \in \mathbb{R}^{B_r \times d}$$

### Online 更新
旧状态：
$$m_i, l_i, O_i$$

当前 block:
$$m_{ij}, l_{ij}, A_{ij}$$

新的 max:
$$m_i^{\text{new}} = \max(m_i, m_{ij})$$

缩放因子：
$$\alpha = e^{m_i - m_i^{\text{new}}}$$
$$\beta = e^{m_{ij} - m_i^{\text{new}}}$$

新的 denominator:
$$l_i^{\text{new}} = \alpha l_i + \beta l_{ij}$$

新的输出：
$$O_i^{\text{new}} = \frac{\alpha l_i O_i + \beta A_{ij}}{l_i^{\text{new}}}$$

然后更新：
$$m_i \leftarrow m_i^{\text{new}}$$
$$l_i \leftarrow l_i^{\text{new}}$$
$$O_i \leftarrow O_i^{\text{new}}$$

### FA1 Forward 伪代码
```
for each Q block Q_i:

    O_i = 0
    m_i = -inf
    l_i = 0

    for each K/V block K_j, V_j:

        S_ij = Q_i @ K_j.T

        m_ij = rowmax(S_ij)

        P_ij = exp(S_ij - m_ij[:, None])

        l_ij = rowsum(P_ij)

        A_ij = P_ij @ V_j

        m_new = max(m_i, m_ij)

        alpha = exp(m_i - m_new)
        beta  = exp(m_ij - m_new)

        l_new = alpha * l_i + beta * l_ij

        O_i = (
            alpha[:, None] * l_i[:, None] * O_i
            +
            beta[:, None] * A_ij
        ) / l_new[:, None]

        m_i = m_new
        l_i = l_new

    write O_i
```

## IO Complexity
### 标准 Attention 的 IO 路径
标准 attention：
$$S = QK^T$$
$$P = \text{softmax}(S)$$
$$O = PV$$

典型执行路径：
```
读 Q,K → 写 S
读 S → 写 P
读 P,V → 写 O
```

最大的问题是：
$$S,P \in \mathbb{R}^{N \times N}$$

所以会产生大量 HBM traffic：
```
写 S: N²
读 S: N²
写 P: N²
读 P: N²
```

也就是至少：$4N^2$ 个元素级别的中间矩阵读写

### FlashAttention 的 IO 路径
FA1 分块计算：$S_{ij} = Q_i K_j^T$，但：$S_{ij}$ 只在片上内存中临时存在，用完就丢

同样：$P_{ij} = e^{S_{ij} - m_{ij}}$ 也只临时存在，用来算：$P_{ij}V_j$ 然后丢掉

FA1 主要写回的是：$O \in \mathbb{R}^{N \times d}$ 以及 softmax 统计量：$m, l \in \mathbb{R}^N$

所以它避免了完整：$S,P \in \mathbb{R}^{N \times N}$ 的 HBM 读写

## Backward + Recomputation
为什么反向传播要重算 $P$

Forward 已经知道：
$$O = \text{softmax}(QK^T)V$$

标准 attention 会保存：
$$P = \text{softmax}(QK^T)$$

但 FA1 forward 不保存完整 $P$

因为：
$$P \in \mathbb{R}^{N \times N}$$
太大

所以 FA1 backward 的核心是：

反向传播时重新按 block 计算 $S_{ij}$ 和 $P_{ij}$，这叫 recomputation

### 标准 Attention Backward 公式
先看普通 attention：
$$S = QK^T$$
$$P = \text{softmax}(S)$$
$$O = PV$$

给定上游梯度：
$$dO$$

先有：
$$dV = P^T dO$$

因为：
$$O = PV$$

然后：
$$dP = dO V^T$$

接着 softmax backward：
$$dS = P \odot (dP - \text{rowsum}(dP \odot P))$$

最后：
$$dQ = dS K$$
$$dK = dS^T Q$$

如果 forward 里有 scale：
$$S = \frac{QK^T}{\sqrt{d}}$$

那么：
$$dQ = \frac{dS K}{\sqrt{d}}$$
$$dK = \frac{dS^T Q}{\sqrt{d}}$$

### FA1 的做法
FA1 forward 不保存完整 $P$

它只保存：
$$O$$

以及 softmax 统计量，通常可以是：
$$m, l$$

或者更常见的：
$$L = \log \sum_j e^{S_{ij}}$$
也就是每一行的 logsumexp

Backward 时，对每个 block 重新算：
$$S_{ij} = Q_i K_j^T$$

然后用保存的 softmax 统计量恢复：
$$P_{ij}$$

如果保存的是 $m, l$，那么：
$$P_{ij} = \frac{e^{S_{ij} - m_i}}{l_i}$$

如果保存的是 logsumexp：
$$L_i = m_i + \log l_i$$

那么：
$$P_{ij} = e^{S_{ij} - L_i}$$

所以不需要保存完整 $P$，只需要能重建当前 block 的 $P_{ij}$

### Block-wise backward 核心流程
对每个 block pair：
$$Q_i, K_j, V_j$$

重新计算：
$$S_{ij} = Q_i K_j^T$$

恢复：
$$P_{ij} = \text{softmax block}$$

然后贡献梯度：
$$dV_j += P_{ij}^T dO_i$$
$$dP_{ij} = dO_i V_j^T$$
$$dS_{ij} = P_{ij} \odot (dP_{ij} - D_i)$$
$$dQ_i += dS_{ij} K_j$$
$$dK_j += dS_{ij}^T Q_i$$

这里最重要的是：$$D_i$$

### $D_i$ 是什么？
softmax backward 中需要：
$$\text{rowsum}(dP \odot P)$$

也就是每一行：
$$D_i = \sum_j dP_{ij} P_{ij}$$
看起来这也需要完整 $P$

但可以化简，因为：
$$dP = dO V^T$$

所以：
$$dP_{ij} = dO_i \cdot v_j$$

于是：
$$D_i = \sum_j P_{ij} (dO_i \cdot v_j)$$

把 $dO_i$ 提出来：
$$D_i = dO_i \cdot \sum_j P_{ij} v_j$$

而：
$$\sum_j P_{ij} v_j = O_i$$

所以：
$$D_i = dO_i \cdot O_i$$

也就是：
$$D_i = \text{rowsum}(dO_i \odot O_i)$$

因为 $O_i$ 是 forward 输出，已经保存了；$dO_i$ 是上游梯度

所以 backward 不需要完整 $P$ 来算 $D_i$

### recomputation
用更多计算，换更少 HBM IO

标准 backward：
```
forward 保存完整 P
backward 读取完整 P
```

FA1 backward：
```
forward 不保存 P
backward 重新计算 P_ij block
```

代价：
$$QK^T$$
会被重新算一部分

收益：
$$P \in \mathbb{R}^{N \times N}$$
不用写入 HBM，也不用从 HBM 读出

在 GPU 上，这通常是划算的，因为 HBM traffic 很贵