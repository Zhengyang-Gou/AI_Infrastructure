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
