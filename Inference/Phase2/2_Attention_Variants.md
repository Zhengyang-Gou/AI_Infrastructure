# Attention Variants
MHA：每个 Query 头都有自己独立的 Key、Value

MQA：所有 Query 头共享同一套 Key、Value

GQA：若干 Query 头组成一组，每组共享一套 Key、Value

MLA：不直接缓存完整 Key、Value，而是缓存一个低维潜变量，各个头再用不同投影读取它

它们的演化，本质上是在解决同一个问题：

如何在尽量保持多头注意力能力的同时，减少自回归生成时庞大的 KV Cache 和显存带宽消耗

## Attention
![alt text](image.png)

## MHA
Multi-Head Attention，多头注意力，它解决的问题是：对序列中的每个 token，如何根据当前上下文，动态地从其他 token 中提取有用信息

输入 X 经过三个线性变换：
