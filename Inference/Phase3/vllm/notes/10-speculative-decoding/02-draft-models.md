# Draft Models

## 1. 文件定位

- 路径：`vllm/v1/spec_decode/llm_base_proposer.py`、`draft_model.py`、`eagle.py`、`medusa.py`、`dflash.py`。
- Worker 集成：`vllm/v1/worker/gpu/spec_decode/speculator.py`、`autoregressive/speculator.py`。
- 职责：加载或构造低成本模型组件，准备 attention/KV 状态并连续生成 draft tokens。

## 2. 主要实现类型

| 类型 | 候选依据 | 主要成本 |
| --- | --- | --- |
| 独立 Draft Model | 小语言模型的自回归分布 | 额外模型权重和多步 forward |
| EAGLE/MTP | 目标模型 hidden states 与轻量预测层 | hidden state 对齐和专用 KV/metadata |
| Medusa | 多个预测 head | 一次 head 计算和候选组织 |
| DFlash 等 | 专用并行草稿结构 | 特定模型与 attention 实现 |

## 3. 状态准备

- Draft 与 Target 的词表、token IDs 和最大长度必须兼容，必要时通过 vocab mapping 转换。
- EAGLE/MTP 读取 Target hidden states，并维护自己下一位置的 position、slot mapping 和 attention metadata。
- Draft Model 可以共享 embedding 或 LM head，但共享必须遵守参数布局与量化限制。
- 不同请求接受长度不同，因此下一轮 draft 状态必须按实际接受结果更新。

## 4. 关键代码解析

### `SpecDecodeBaseProposer.propose()`

### `SpecDecodeBaseProposer.prepare_inputs()`

### `SpecDecodeBaseProposer.load_model()`

### `DraftModelProposer._create_draft_vllm_config()`

### `DraftModelProposer._get_model()`

### `EagleProposer.__init__()`

### `MedusaProposer.propose()`

### `DraftModelSpeculator.load_model()`

### `AutoRegressiveSpeculator.propose()`

## 5. 与 Target Model 的边界

- Target Model 负责给候选位置产生权威 logits。
- Proposer 只负责候选及可选 draft logits，不能直接决定最终输出。
- Model Runner 统一安排 hidden states、采样结果和 proposer 调用顺序。

## 6. 当前结论

模型型 proposer 的难点不只是加载一个小模型，而是让两套自回归状态在变长批次、KV block 和实际接受长度上保持同步。
