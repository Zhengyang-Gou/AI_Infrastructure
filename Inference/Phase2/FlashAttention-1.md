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