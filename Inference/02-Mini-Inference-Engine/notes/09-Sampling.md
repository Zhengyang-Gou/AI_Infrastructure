# 09 · 采样与输出 Token

接上 LM Head 的 logits，理解温度缩放与采样如何得到下一 Token。

对应源码：[sampler.py](../source/nano-vllm/nanovllm/layers/sampler.py)。

## sampler.py

```python
logits = logits.float().div_(temperatures.unsqueeze(1))
probs = torch.softmax(logits, dim=-1)
sample_tokens = probs.div_(
    torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
).argmax(dim=-1)
```
采样流程：

1. logits 转为 FP32，并为 Batch 中每条序列除以各自温度
2. Softmax 得到概率
3. 为每个候选 Token 生成独立的指数分布随机数
4. 取 `probability / exponential_noise` 最大的位置

最后两步属于指数竞赛（Exponential Race），与按分类分布进行 multinomial 采样等价，适合用逐元素操作和 `argmax` 实现。`@torch.compile` 可将这些操作编译优化。

温度越低，原 logits 差距被放大，输出更确定；温度越高，概率分布更平坦。项目在 `SamplingParams` 中禁止温度为 0，所以没有单独实现 Greedy Sampling。

---

[学习目录](README.md) · [上一篇：张量并行与权重加载](08-Tensor-Parallel-and-Weight-Loading.md) · [下一篇：一次 Step 的完整数据流](10-End-to-End-Step.md)
