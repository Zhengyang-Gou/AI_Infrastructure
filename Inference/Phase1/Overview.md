# Overview
## 推理流程
### 总体流程
```
用户请求
↓
API Server 接收请求
↓
Tokenizer：文本转 token
↓
Scheduler：请求排队、调度、组 batch
↓
KV Cache 分配
↓
Prefill：处理输入 prompt
↓
Decode：逐 token 生成
↓
Sampling：采样下一个 token
↓
Detokenizer：token 转文本
↓
Streaming 返回给用户
↓
请求结束，释放 KV Cache
```
这条链路里最核心的是两个阶段：
- Prefill 阶段：处理输入 prompt，产生第一个 token 前的计算
- Decode 阶段：一个 token 一个 token 地生成后续内容
几乎所有推理指标，都是围绕这两个阶段展开的

### 请求进入服务
用户发送一段 prompt，例如：请解释一下 KV Cache 的作用

如果是 ChatGPT / API 场景，真实输入通常不只是用户文本，而是带有系统提示词、历史对话、工具信息、模板格式的完整 prompt
```
<system>
你是一个 helpful assistant
</system>

<user>
请解释一下 KV Cache 的作用。
</user>

<assistant>
```
这一层会影响输入长度，也会直接影响推理性能

### Tokenizer：文本转 token
大模型不能直接处理文字，而是处理 token ID

例如："KV Cache 很重要"，会被 tokenizer 转成类似：[21834, 1237, 5021, 873]，这一步通常在 CPU 上完成

### 请求进入调度器
在真实服务中，不是一个请求独占一张 GPU

调度器会把多个请求组织起来一起跑，例如：
```
Request A: 输入 200 tokens，要生成 100 tokens
Request B: 输入 1000 tokens，要生成 50 tokens
Request C: 输入 50 tokens，要生成 500 tokens
```
推理框架需要决定：
- 哪些请求先进入 GPU
- 哪些请求可以组成 batch
- KV Cache 显存够不够
- 是否要等待更多请求组成更大的 batch
- 是否要抢占、暂停、换出某些请求
- prefill 和 decode 如何混合调度

这就是 vLLM、SGLang、TensorRT-LLM 等推理框架的重要工作

### Prefill 阶段：处理输入 prompt
假设输入 prompt 有 1000 个 token，Prefill 阶段会一次性处理这 1000 个 token，模型会计算：
```
token 1 → hidden state
token 2 → hidden state
...
token 1000 → hidden state
```
并且为每一层 Transformer 生成对应的 KV Cache

Transformer 的 attention 需要 Q、K、V，对于输入 prompt 中的每一个 token，模型都会计算对应的 Key 和 Value，这些 K/V 会被缓存起来，后续生成 token 时不用重复计算历史 token 的 K/V，这就是 KV Cache

Prefill 阶段主要输出两个东西：
1. prompt 对应的 KV Cache
2. 第一个生成 token 的 logits
logits 可以理解成模型对下一个 token 的概率分布

Prefill 的主要特点是：输入长，一次性并行计算，计算量大，但 GPU 利用率通常较高

因为 prompt 的所有 token 可以并行处理，所以 prefill 通常是 compute-bound，也就是更依赖算力，输入越长，prefill 越慢，所以长上下文场景中，TTFT 往往会变大

### Decode 阶段：逐 token 生成
Prefill 结束后，模型开始生成输出，假设模型要回答：KV Cache 是一种用于加速大模型推理的缓存机制。它不是一次性生成整句话，而是一个 token 一个 token 地生成，流程类似：
```
Step 1: 生成 "KV"
Step 2: 生成 " Cache"
Step 3: 生成 " 是"
Step 4: 生成 " 一种"
Step 5: 生成 " 用于"
...
```
每一步 decode 都会：
```
读取历史 KV Cache
计算当前 token 的 Q/K/V
用当前 Q 和历史 K/V 做 attention
经过 MLP
得到 logits
采样或贪心选择下一个 token
把当前 token 的 K/V 追加进 KV Cache
```
Decode 的问题是：强串行，第 10 个 token 必须等第 9 个 token 生成之后才能生成，所以 decode 阶段不能像 prefill 那样把所有输出 token 一次性并行算完

Decode 阶段的特点是：每次只处理一个新 token，但要读取越来越长的 KV Cache。随着输出变长，序列长度变长，attention 需要访问的历史 KV 也越来越多。所以 decode 通常更容易受：
- 显存带宽
- KV Cache 读取效率
- batch 调度效率
- attention kernel 性能
- cache locality

影响，很多推理优化，例如 PagedAttention、FlashAttention、MLA、MQA、GQA、RadixAttention、Prefix Cache，本质上都和降低 decode 成本有关

### 一个完整推理请求的时间组成
假设用户请求模型回答一个问题，完整时间可以拆成：
```
总延迟
= 排队时间
+ tokenization 时间
+ prefill 时间
+ decode 时间
+ detokenization 时间
+ 网络传输时间
```

## 推理相关指标
大模型推理指标大致分成几类：
| 类别   | 指标                              | 主要回答的问题   |
| ---- | ------------------------------- | --------- |
| 延迟指标 | TTFT、TPOT、ITL、E2E Latency       | 用户等得久不久   |
| 吞吐指标 | tokens/s、requests/s、QPS         | 系统能处理多少请求 |
| 显存指标 | KV Cache 占用、峰值显存、batch capacity | 能并发多少请求   |
| 资源指标 | GPU 利用率、显存带宽、MFU                | GPU 有没有跑满 |
| 服务指标 | p95/p99、SLA、Goodput             | 线上稳定性如何   |
| 成本指标 | tokens/$、requests/GPU/hour      | 单位成本是否划算  |

### 延迟指标 Latency
#### E2E Latency：端到端延迟
E2E Latency 是用户从发出请求到收到完整回答的总时间

E2E Latency = 请求结束时间 - 请求开始时间

包含所有环节：
```
排队
tokenization
prefill
decode
detokenization
网络传输
```

#### TTFT：Time To First Token
TTFT 是首 token 延迟，表示用户发出请求后，多久能看到第一个生成 token

TTFT = 第一个输出 token 返回时间 - 请求到达时间

TTFT 主要由这些因素决定：

TTFT ≈ 排队时间 + tokenization 时间 + prefill 时间 + 第一次 decode 时间

如果 prompt 很长，prefill 时间会变长，如果服务很忙，请求在队列里等很久，TTFT 也会变长

在流式输出场景中，用户体验高度依赖 TTFT，聊天机器人、Copilot、客服系统都非常关注 TTFT

#### TPOT / ITL：每个输出 token 的生成延迟
TPOT：Time Per Output Token，表示平均生成一个输出 token 需要多久

TPOT = Decode 总时间 / 输出 token 数

例如：输出 100 个 token，decode 阶段耗时 5 秒，那么 TPOT = 5 / 100 = 0.05 秒/token = 50 ms/token，对应生成速度是 1 / TPOT = 20 tokens/s

ITL：Inter-Token Latency，ITL 是相邻两个 token 之间的间隔

TPOT 通常是平均值，ITL 更细，可以看到 token 生成是否稳定

### 吞吐指标 Throughput
#### Request Throughput：请求吞吐
Request Throughput 表示单位时间内完成多少个请求

Request Throughput = 完成请求数 / 时间

#### Token Throughput：token 吞吐
LLM 推理更常看 token 吞吐，Token Throughput 表示单位时间内处理多少 token，通常分成：
```
Input token throughput
Output token throughput
Total token throughput
```
输入 token 吞吐：Input Token Throughput = 处理的输入 token 数 / 时间

主要对应 prefill 能力，长 prompt 场景中，这个指标很重要

输出 token 吞吐：Output Token Throughput = 生成的输出 token 数 / 时间

主要对应 decode 能力，聊天、代码生成、长文本生成都非常关注这个指标

总 token 吞吐：Total Token Throughput = (输入 tokens + 输出 tokens) / 时间

### 服务指标
#### Goodput：满足 SLA 的有效吞吐
Throughput 只看处理了多少请求，Goodput 看的是：在满足服务质量要求的前提下，系统能完成多少请求

例如 SLA 规定：
```
TTFT < 1s
TPOT < 100ms
E2E Latency < 10s
```
如果系统每秒完成 100 个请求，但其中 40 个请求 TTFT 超标，那么有效吞吐不是 100 RPS，而是 60 RPS

Goodput = 满足 SLA 的请求数 / 时间

#### Tail Latency：p50 / p90 / p95 / p99
线上系统不能只看平均值，平均值可能看起来还可以，但少数用户体验非常差

看分位数

- p50 是中位数：50% 的请求延迟小于等于这个值
- p95 表示：95% 的请求延迟小于等于这个值
- p99 表示：99% 的请求延迟小于等于这个值

线上服务通常非常关注：
- p95 TTFT
- p99 TTFT
- p95 TPOT
- p99 E2E latency

### 并发、Batch 和调度指标
#### Concurrency：并发数
Concurrency 表示同一时刻系统中有多少请求正在被处理

并发数越高，系统吞吐通常越高，但单请求延迟也可能变差

#### Batch Size：批大小
LLM 推理里 batch size 有两种常见理解

Prefill batch size，表示一次 prefill 处理多少个请求
Decode batch size，表示一次 decode step 处理多少个活跃请求

#### Continuous Batching
传统 batching 是：
```
收集一批请求
一起开始
一起结束
```
问题是：有的请求短，有的请求长，短请求结束后 GPU 资源会浪费

Continuous Batching 是：
```
每一步 decode 都动态维护 batch
有请求结束，就移除
有新请求进来，就加入
```
这样可以显著提高 GPU 利用率和吞吐，vLLM、SGLang 等框架都强依赖 continuous batching

### KV Cache 相关指标
KV Cache 是 LLM 推理优化的核心

#### KV Cache 是什么
每一层 Transformer 都需要保存历史 token 的 Key 和 Value，对一个请求而言，KV Cache 大小大致和这些因素成正比

层数 × 序列长度 × KV heads 数 × head dimension × 数据类型大小

更具体地说：KV Cache ≈ 2 × num_layers × seq_len × num_kv_heads × head_dim × bytes_per_element

这里的 2 是因为有 K 和 V 两份缓存

#### 为什么 KV Cache 很重要
因为 decode 阶段每生成一个 token，都需要读取历史 KV Cache

KV Cache 会影响：
- 显存占用
- 最大上下文长度
- 最大并发数
- decode 速度
- batch capacity

#### KV Cache Hit Rate
在 prefix cache、prompt cache、radix cache 场景中，会关注 cache hit rate

Cache Hit Rate = 命中的缓存 token 数 / 可复用的 token 数

例如很多用户请求都有相同系统 prompt：你是一个智能助手……

这部分前缀可以复用，减少 prefill 计算，Cache hit rate 越高，prefill 成本越低，TTFT 通常越好

### 显存指标
#### Model Weight Memory
模型权重占用显存
```
7B FP16 模型 ≈ 14GB
70B FP16 模型 ≈ 140GB
```
因为 FP16 每个参数 2 bytes，如果使用 INT8、INT4 量化，权重显存会下降

#### KV Cache Memory
KV Cache 是动态显存，随着请求数和序列长度变化，它大致由以下因素决定：
- 并发请求数
- 输入长度
- 输出长度
- 层数
- KV heads 数
- head_dim
- dtype

线上推理经常不是权重放不下，而是 KV Cache 不够

#### Peak Memory
Peak Memory 是运行过程中出现过的峰值显存占用，包括：
- 模型权重
- KV Cache
- 激活临时 buffer
- attention workspace
- CUDA graph buffer
- 通信 buffer
- 框架管理开销

优化系统时要留安全余量，不能只看理论值

### GPU 资源指标
#### GPU Utilization
GPU Utilization 表示 GPU 计算单元忙碌程度，但这个指标容易误导，不一定说明系统性能好，因为 GPU 可能在忙着搬数据，而不是高效做矩阵乘，所以还需要看：
- SM utilization
- Memory bandwidth utilization
- Tensor Core utilization
- Kernel occupancy

#### Compute-bound vs Memory-bound
推理优化里经常说：
- Prefill 更偏 compute-bound
- Decode 更偏 memory-bound

意思是：Compute-bound 瓶颈在算力，GPU 算得不够快

典型场景：
- 长 prompt prefill
- 大 batch GEMM
- MLP 计算

Memory-bound 瓶颈在显存读写，GPU 等数据，不是等计算

典型场景：
- decode attention 读取 KV Cache
- 小 batch decode
- 频繁访问长上下文

#### MFU：Model FLOPs Utilization
MFU 表示模型理论计算量中，有多少被硬件有效执行

MFU = 实际模型 FLOPs / GPU 理论峰值 FLOPs

MFU 越高，说明越接近硬件极限

### 通信指标：多 GPU / 多机推理
当模型太大，一张 GPU 放不下，或者为了提升吞吐，需要多 GPU 推理

常见并行策略：
- TP：Tensor Parallel
- PP：Pipeline Parallel
- DP：Data Parallel
- EP：Expert Parallel

#### 通信延迟
例如 TP 中，每一层可能需要 AllReduce，通信慢会直接拖慢每个 decode step

#### 通信带宽
多卡之间通过 NVLink、PCIe、InfiniBand 通信

大模型推理中常见通信操作：
- AllReduce
- AllGather
- ReduceScatter
- Broadcast
- Send / Recv

#### 通信计算重叠
高性能推理系统会尝试：一边计算，一边通信