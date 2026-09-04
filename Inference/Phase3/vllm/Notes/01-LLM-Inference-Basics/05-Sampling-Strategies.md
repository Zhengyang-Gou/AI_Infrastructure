# Sampling Strategies

> 状态：已完成初稿 ｜ 更新：2026-09-04

## 1. Sampling 在生成流程中的位置

Transformer 并不直接输出文字，而是为词表中的每个 Token 输出一个分数，即 `logits`。Sampling 的任务是把这组分数转换成下一个 Token：

```text
hidden state
    ↓ LM Head
logits [batch_size, vocab_size]
    ↓ 约束与惩罚
processed logits
    ↓ Temperature / Top-k / Top-p / Min-p
candidate distribution
    ↓ Greedy 或随机采样
next token id
    ↓ 追加到上下文，进入下一轮 Decode
```

Sampling 不会改变模型已经学到的知识，但会改变模型如何在候选 Token 之间取舍，因此直接影响：

- **确定性**：相同 Prompt 是否产生相同结果。
- **多样性**：模型是否愿意选择非最高概率 Token。
- **可靠性**：是否容易跑题、重复或生成不合法内容。
- **性能**：过滤、排序、随机数、返回 logprobs 和并行生成都会产生额外开销。

## 2. 从 Logits 到概率

设词表大小为 `V`，模型输出 `z ∈ R^V`。Softmax 将 logits 转为概率分布：

```text
pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```

实际实现会先减去最大 logit，避免 `exp` 溢出。因为 Softmax 保持大小关系，所以：

```text
argmax(softmax(z)) = argmax(z)
```

纯 Greedy Decoding 不需要先物化完整概率分布，直接对 logits 做 `argmax` 即可。

### Temperature

Temperature 在 Softmax 前缩放 logits：

```text
pᵢ(T) = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)
```

- `0 < T < 1`：放大 logit 差异，分布更尖锐，输出更稳定。
- `T = 1`：保持原始分布。
- `T > 1`：缩小 logit 差异，分布更平坦，随机性更强。
- `T = 0`：公式本身无定义；工程实现通常将它解释为 Greedy Decoding。

Temperature 调整的是候选 Token 之间的相对概率，并不负责移除低概率 Token。

## 3. 常见解码策略

### 3.1 Greedy Decoding

每一步都选择当前概率最高的 Token：

```text
token = argmax(z)
```

优点是速度快、结果稳定，适合分类式回答、信息抽取、工具调用和低随机性的代码任务。缺点是只做局部最优选择，不能保证整条序列概率最高，也容易陷入重复或过于模板化的输出。

### 3.2 Random Sampling

直接按完整 Softmax 分布随机抽样。它保留了所有 Token，包含大量极低概率候选；若不配合过滤，可能选中明显不合理的长尾 Token。

因此开放式生成通常组合 Temperature 与一种或多种候选集截断策略。

### 3.3 Top-k Sampling

只保留 logits 最大的 `k` 个 Token，其余位置设为 `-∞`，再归一化并采样：

```text
C = k 个最高概率 Token
p'ᵢ = pᵢ / Σⱼ∈C pⱼ,  i ∈ C
```

- `k` 小：输出更集中，但可能过早排除合理候选。
- `k` 大：多样性更高，但固定数量无法适应分布形状。

例如模型非常确定时，前 3 个 Token 可能已经覆盖 99% 概率；模型不确定时，前 3 个 Token 可能只覆盖 30%。Top-k 在两种情况下都保留相同数量，这是它的主要局限。

### 3.4 Top-p / Nucleus Sampling

将 Token 按概率从高到低排序，保留累计概率首次达到 `p` 的最小候选集合：

```text
C = 最小集合，使 Σᵢ∈C pᵢ ≥ p
```

Top-p 的候选数会随分布动态变化：模型确定时保留较少 Token，模型不确定时保留更多 Token。`top_p` 越低，结果通常越保守。

### 3.5 Min-p Sampling

以当前最高概率为基准，只保留满足以下条件的 Token：

```text
pᵢ ≥ min_p × maxⱼ(pⱼ)
```

它使用相对阈值而非累计概率。最高概率很高时过滤更严格；最高概率较低时允许更多候选。Min-p 不会删除最高概率 Token，因此不会改变 Greedy 的结果。

### 3.6 Beam Search

Beam Search 不是随机采样，而是同时维护 `beam_width` 条累计分数最高的候选序列，并在每一步扩展和裁剪。

它适合翻译等更强调序列整体得分的任务，但有明显系统代价：

- 每个 Beam 都需要维护自己的状态与 KV Cache。
- 计算量和显存占用近似随 Beam 数增长。
- 开放式对话中容易偏向安全、常见和较短的序列。

需要区分 **Beam Search** 与 `n > 1` 的 **Parallel Sampling**：前者的候选会相互竞争并按累计分数裁剪，后者只是为同一个 Prompt 独立生成多条样本。

## 4. Logits Processor 与重复惩罚

在真正采样前，系统通常会先修改或屏蔽部分 logits。

### Presence 与 Frequency Penalty

对已经在输出中出现过的 Token，常见定义为：

```text
z'ᵢ = zᵢ - presence_penalty × 1[countᵢ > 0]
          - frequency_penalty × countᵢ
```

- **Presence Penalty** 只关心“是否出现过”，鼓励引入新 Token。
- **Frequency Penalty** 与出现次数成正比，更直接抑制反复出现。

### Repetition Penalty

Repetition Penalty 通常对 Prompt 和已生成文本中出现过的 Token 做乘法式、符号感知的缩放。大于 `1` 时抑制重复，小于 `1` 时鼓励重复。它和前两种惩罚的定义不同，不应把三个参数当成等价旋钮叠加到很大。

### 约束类处理

- `allowed_token_ids`：只允许白名单 Token。
- `bad_words`：屏蔽会补全禁用词的 Token。
- `logit_bias`：对指定 Token 的 logit 加偏置。
- `min_tokens`：达到最短长度前屏蔽 EOS 或停止 Token。
- Structured Output：根据 JSON Schema、Grammar 等状态，只保留当前步骤合法的 Token。

约束解码的本质是动态修改候选集合。它能保证语法或格式合法，但不能保证字段内容在语义上正确。

## 5. 参数组合与执行顺序

以当前仓库中的 vLLM V1 Sampler 为例，主流程可概括为：

```text
raw logits
  → 转 FP32
  → allowed_token_ids / bad_words
  → 可能改变 argmax 的 Logits Processor
  → repetition / frequency / presence penalties
  → Greedy 分支（全是 Greedy 请求时可提前返回）
  → temperature
  → 不改变 argmax 的 Processor（如 min_p）
  → top_k / top_p
  → 按过滤后的分布随机采样
  → 收集 sampled token 与可选 logprobs
```

顺序很重要。Temperature 会改变概率分布的形状，因此会间接改变 Top-p 和 Min-p 的候选集合；Top-k 只依赖 logit 排名，正温度缩放不会改变它保留的 Token。

当 Top-k 与 Top-p 同时开启时，最终候选必须同时满足两种过滤条件。参数叠加过强可能只剩一个 Token，使“随机采样”实际退化为确定性输出。

## 6. 如何选择参数

以下只是起点，最终应以具体模型的推荐配置和任务评测为准：

| 场景 | 建议起点 | 关注点 |
| --- | --- | --- |
| 事实问答、抽取、分类 | `temperature=0` | 稳定、便于回归测试 |
| 代码与工具调用 | 低 Temperature；配合结构化约束 | 格式正确性比多样性重要 |
| 通用对话 | 中等 Temperature + Top-p | 在稳定性与自然度之间折中 |
| 创意写作、候选生成 | 较高 Temperature + Top-p/Min-p，或 `n>1` | 同时检查跑题与成本 |
| 可复现实验 | 固定模型、参数、Seed 和运行环境 | Seed 不是跨后端完全一致的保证 |

调参原则：

1. 先使用模型发布方推荐参数作为基线。
2. 优先只调 Temperature 和一种截断策略，避免无法判断效果来源。
3. 用任务指标和人工评估比较，而不是凭单个样例下结论。
4. 如果已经退化、跑题，降低随机性；如果输出僵化、重复，再逐步增加候选空间。

## 7. 确定性、质量与性能

### Seed 不等于绝对复现

固定 Seed 能控制随机数流，但结果还可能受模型版本、Tokenizer、Batch 排布、并行方式、采样 Kernel、硬件和浮点误差影响。不同实现可以保证统计分布等价，却不保证逐 Token 相同。

### Sampling 通常不是 Decode 的最大瓶颈

相比模型前向，Sampling 计算较小，但大词表和高并发下仍需处理 `[batch, vocab_size]` 的 logits。下列设置会放大开销：

- Top-p 的排序或扫描。
- `logprobs`，尤其返回整个词表概率。
- 每个请求使用独立 Seed，导致随机数状态更复杂。
- `n > 1` 或 Beam Search，增加活跃序列及 KV Cache。
- 复杂的 Grammar、Bad Words 或自定义 Logits Processor。

Greedy 可直接 `argmax`，也能跳过不影响 argmax 的过滤器，因此通常是最简单、最便宜的路径。

### 不要混淆采样质量与模型置信度

低 Temperature 只是让输出更确定，并不会让错误答案自动变正确；高 Temperature 也不是让模型“更有创造力”的独立能力，而是提高低概率候选被选中的机会。

## 8. 面试快速回答

**Temperature、Top-k 和 Top-p 有什么区别？** Temperature 调整整个概率分布的尖锐程度；Top-k 固定保留概率最高的 `k` 个 Token；Top-p 动态保留累计概率达到阈值的最小集合。前者改变相对概率，后两者缩小候选集合。

**为什么 `temperature=0` 不是代入 Softmax 公式？** 因为除以 0 没有定义。推理框架将其作为特殊分支，直接对 logits 做 `argmax`。

**Greedy 能找到概率最高的完整序列吗？** 不能。它只保证每一步选择局部最高概率 Token，早期选择会改变后续条件分布，整条序列不一定全局最优。

**Top-p 为什么比固定 Top-k 更自适应？** 它按概率质量而非固定数量决定候选集；模型确定时集合自动变小，不确定时集合自动变大。

**固定 Seed 为什么仍可能无法完全复现？** 批处理顺序、GPU Kernel、并行策略和浮点误差都可能改变随机数消耗或临界 Token 排名。Seed 只解决随机数来源的一部分问题。

**`n > 1` 对 Serving 有什么影响？** 它会为一个 Prompt 维护多条输出序列，增加 Decode 工作量与 KV Cache 占用，并降低系统可承载的并发请求数。

## 总结

Sampling 的核心是两步：**先加工并裁剪 logits，再决定选最大值还是按概率随机抽样。** Greedy 强调确定性，Temperature 控制分布形状，Top-k、Top-p 和 Min-p 控制候选集合，Penalty 与 Grammar 则注入历史和业务约束。对 Serving 系统而言，还要同时考虑随机数、全词表处理、logprobs、并行候选及 KV Cache 带来的性能代价。
