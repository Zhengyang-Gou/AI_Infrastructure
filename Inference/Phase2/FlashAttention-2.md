# FlashAttention-2
## FA2 到底改了 FA1 的什么
FA1 主要解决 HBM IO 问题；FA2 主要解决 GPU 并行度和计算划分问题

FA2 没有推翻 FA1，它继承 FA1 的核心：
```
分块读 K/V
计算局部 QK^T
online softmax 更新 m, l, O
不保存完整 S / P
```
但 FA2 发现：FA1 的 IO 已经优化得很好了，瓶颈开始转移到 GPU kernel 内部的并行效率

