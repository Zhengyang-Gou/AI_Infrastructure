# Proposers

## 1. 文件定位

- 路径：`vllm/v1/spec_decode/ngram_proposer.py`、`ngram_proposer_gpu.py`、`suffix_decoding.py`。
- 职责：不运行独立语言模型，直接从请求历史或共享前缀中寻找可复用的后续 token。

## 2. 候选来源

- N-gram：用当前序列末尾的 n-gram 在历史 token 中寻找相同片段，并复制其后续 token。
- GPU N-gram：将匹配与抽取移动到 GPU，降低批量请求下的 CPU 开销和同步。
- Suffix Decoding：跨序列维护后缀索引，从已见序列中查找可延续模式。

这些方法没有 draft logits，优点是成本低；候选质量依赖文本重复度，代码、固定模板和长上下文通常更容易命中。

## 3. 批处理流程

```text
读取每个请求的 token history
→ 选择允许的 n-gram 长度
→ 搜索最长匹配
→ 截取最多 K 个后续 token
→ 对无匹配请求返回空候选
→ 将不同长度候选交给验证批次
```

需要特别记录有效候选长度，padding 不能被当成真正 draft token。

## 4. 关键代码解析

### `NgramProposer.batch_propose()`

### `NgramProposer.propose()`

### `batch_propose_numba()`

### `NgramProposerGPU.propose()`

### `NgramProposerGPU.update_token_ids_ngram()`

### `SuffixDecodingProposer.propose()`

## 5. 与其他文件的关系

- 输入来自 Model Runner 保存的请求 token 状态。
- 输出被标准化为 draft token IDs，供 Scheduler 在下一轮安排验证。
- 无 draft logits 时，验证路径使用适合该 proposer 的接受逻辑。

## 6. 当前结论

无模型 proposer 用“历史中已经出现过的局部规律”换取几乎免费的候选，适合先理解推测解码的数据结构与状态流。
