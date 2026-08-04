# Vocab-Parallel Embedding and LM Head

## 1. 文件定位

- 文件路径：`vllm/model_executor/layers/vocab_parallel_embedding.py`
- 所属层次：模型 embedding 与输出权重层
- 核心职责：按 vocabulary 维度切分 token embedding 和 LM Head 权重，并处理词表 padding、LoRA 新增 token、权重加载和 TP 通信。
- 在调用链中的位置：Llama forward 的输入端使用 `VocabParallelEmbedding`，logits 计算端使用 `ParallelLMHead`。

Embedding 和 LM Head 的权重形状相同，都是 `[vocab_size, hidden_size]`，因此可以共享词表并行的分片与加载逻辑。

## 2. 核心类与组件

| 类 / 组件 | 作用 | 关注点 |
| --- | --- | --- |
| `UnquantizedEmbeddingMethod` | 创建并使用未量化 embedding 权重 | `F.embedding` 与权重共享 |
| `VocabParallelEmbeddingShardIndices` | 保存本 rank 的原始词表、新增词表和 padding 区间 | TP 分片边界 |
| `VocabParallelEmbedding` | vocabulary-parallel token embedding | mask、lookup、all-reduce |
| `ParallelLMHead` | vocabulary-parallel 输出权重 | 由 `LogitsProcessor` 使用 |
| `get_masked_input_and_mask()` | 将全局 token ID 映射成本地索引 | 非本 rank token 置 mask |
| `pad_vocab_size()` | 将词表补齐到指定倍数 | 保证可被 TP size 均分 |

## 3. 词表分片布局

词表先分别处理原始词表和新增词表的 padding，再按 TP rank 切分：

```text
全局逻辑布局
[原始词表] [原始 padding] [新增词表] [新增 padding]
```

`VocabParallelEmbeddingShardIndices` 同时保存：

- padding 后的本 rank 原始词表区间；
- padding 后的本 rank 新增词表区间；
- 去掉 padding 后的真实原始词表区间；
- 去掉 padding 后的真实新增词表区间。

这种布局保证 LoRA 新增 embedding 总在每个分片的原始词表区域之后，同时能兼容不同 checkpoint 加载方式。

## 4. Embedding Forward

```text
全局 token IDs
→ get_masked_input_and_mask()
→ 属于本 rank 的 token 转成本地索引
→ 其他 token 映射到安全索引并记录 mask
→ 本地 embedding lookup
→ 非本 rank 位置清零
→ tensor parallel all-reduce
→ 完整 embedding
```

每个 token 只会在拥有对应词表分片的 rank 上产生非零 embedding。all-reduce 求和后，每个 rank 都得到完整 embedding Tensor。

TP size 为 1 时不需要 mask 和 all-reduce，直接使用输入 token ID 查询本地权重。

## 5. 权重加载

```text
checkpoint embedding weight
→ VocabParallelEmbedding.weight_loader()
→ 根据本 rank 的 org_vocab_start / end 取切片
→ 写入本地参数前部
→ 将本地 padding 区域清零
```

若参数没有 `output_dim` 元数据，说明它不按词表维度切分，loader 会把完整参数复制到所有 rank。对于打包量化权重，切分偏移与长度还会按 pack factor 调整。

## 6. Parallel LM Head

`ParallelLMHead` 继承 `VocabParallelEmbedding`，复用词表分片、权重创建和加载逻辑，但不执行 embedding lookup：

```text
hidden_states
→ LlamaForCausalLM.compute_logits()
→ LogitsProcessor(ParallelLMHead, hidden_states)
→ 使用 LM Head 分片权重计算 logits
```

其 `forward()` 会直接报错，提醒调用方应由 logits 处理器使用权重。若模型配置启用 `tie_word_embeddings`，`tie_weights()` 会让 LM Head 与输入 embedding 共享同一个权重参数。

## 7. Logits 重排

各 rank 的分片中夹有原始词表 padding 和新增词表 padding。`get_sharded_to_full_mapping()` 构造一个重排索引，使 gather 后的 logits 恢复为：

```text
[所有真实原始 token] [所有真实新增 token] [所有 padding]
```

这样最终有效 logits 的索引重新与全局 token ID 一一对应，padding 被移动到尾部。

## 8. 输入与输出

### 输入

- 全局 token IDs 或 logits 计算所需 hidden states。
- `num_embeddings`、`embedding_dim`、原始词表大小和 padding size。
- TP rank、TP size、量化配置和 checkpoint 权重。

### 输出

- `VocabParallelEmbedding.forward()` 返回完整 hidden-size embedding。
- `get_sharded_to_full_mapping()` 返回 gather logits 的重排索引，TP size 为 1 时返回 `None`。
- `ParallelLMHead` 提供分片权重给 logits processor，而不直接执行普通 forward。

### 状态变化

- 初始化阶段计算本 rank 词表区间并创建权重分片。
- 权重加载阶段复制本 rank 的真实词表行，并将 padding 行清零。
- 权重共享时，LM Head 的 `weight` 指向 input embedding 的同一参数。

## 9. 关键代码解析

### `UnquantizedEmbeddingMethod.create_weights()`

### `UnquantizedEmbeddingMethod.embedding()`

### `UnquantizedEmbeddingMethod.tie_weights()`

### `pad_vocab_size()`

### `get_masked_input_and_mask()`

### `VocabParallelEmbedding.__init__()`

### `VocabParallelEmbedding._get_indices()`

### `VocabParallelEmbedding.get_sharded_to_full_mapping()`

### `VocabParallelEmbedding.weight_loader()`

### `VocabParallelEmbedding.forward()`

### `ParallelLMHead.__init__()`

### `ParallelLMHead.tie_weights()`

### `ParallelLMHead.forward()`

## 10. 与其他文件的关系

- 上游模型：`LlamaModel` 创建 `VocabParallelEmbedding`，`LlamaForCausalLM` 创建 `ParallelLMHead`。
- logits 计算：`LogitsProcessor` 使用 `ParallelLMHead` 的分片权重。
- 分布式通信：embedding forward 依赖 tensor-parallel all-reduce。
- 权重加载：Model Loader 通过参数上的 `weight_loader` 写入当前 rank 分片。
- LoRA：词表布局为新增 embedding 预留独立且可 padding 的区域。

## 11. 当前结论

`vocab_parallel_embedding.py` 在 vocabulary 维度切分输入与输出权重：输入端通过 mask、局部 lookup 和 all-reduce 恢复完整 embedding，输出端则把相同分片权重交给 logits 处理与采样路径使用。
