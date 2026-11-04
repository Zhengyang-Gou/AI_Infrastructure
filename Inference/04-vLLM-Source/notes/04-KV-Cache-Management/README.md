# 04 · KV Cache Management

**本章回答：KV 块怎样定容量、分配、复用、淘汰与回收？**

状态：Overview 与 5 篇 Core-Modules 已完成，依据本地源码快照。

## 1. 职责边界

| 项目 | 契约 |
| --- | --- |
| 输入 | 各层 KVCacheSpec、可用缓存字节预算；请求的 token 范围、已有块与申请/释放指令 |
| 输出 | KVCacheConfig；前缀命中长度与块引用；新增块 ID、分配成功/失败和资源交还结果 |
| 本章负责 | 规格、分组和容量规则，块池与引用计数，请求持有块，前缀查询与登记，淘汰与实际回收的安全条件 |
| 终点 | 交出配置、块引用、块 ID 或失败结果；不决定请求的调度策略，不创建实际 GPU KV 张量 |

| 内容 | 唯一归属 |
| --- | --- |
| 选择谁执行、谁被抢占，重置 Request 进度和核心判停 | 03 Scheduler |
| 探测可用显存、按配置创建实际 GPU KV 张量 | 05 Model and Weight Loading |
| 组装 block table、slot_mapping 和执行输入 | 06 Model Execution |
| KV 物理维度、Attention 元数据与 kernel 读写 | 07 Attention Backend |

固定主线沿单卡、单个 FullAttention 分组，无量化、无跨层 KV 共享、无远端 KV；配置见 [总目录](../README.md#1-固定运行配置)。其他缓存类型只说明它们在哪个接口改变规则。

## 2. 正文范围与阅读顺序

先读 [Overview](Overview.md)，再按下表阅读。每篇只展开所列状态和实现。

| 顺序 | 笔记 | 唯一负责的内容 |
| --- | --- | --- |
| 01 | [规格、分组与容量](Core-Modules/01-Spec-Groups-and-Capacity.md) | Spec → Group → Config；字节预算到块容量 |
| 02 | [块池与引用计数](Core-Modules/02-Block-Pool-and-References.md) | 单个块的元数据、空闲队列、借出/touch/归还 |
| 03 | [请求块与增量分配](Core-Modules/03-Allocation-and-Request-Blocks.md) | req_to_blocks、申请预测、命中块接入与新增块 |
| 04 | [前缀缓存与淘汰](Core-Modules/04-Prefix-Cache-and-Eviction.md) | hash、连续前缀命中、登记、复用与淘汰 |
| 05 | [释放与块账本](Core-Modules/05-Release-and-Block-Ledger.md) | 请求账本退出、延迟交还契约与案例 B |

01 不探测显存；02 不计算请求需要几个块；03 不实现 hash 匹配；04 不重讲引用计数；05 不重讲调度退场和 GPU 张量初始化。

## 3. 案例 B：块账本

完整条件、三轮块数表与释放后的状态只在 [Core 05 §4](Core-Modules/05-Release-and-Block-Ledger.md#4-案例-b唯一的完整块账本) 维护。各轮计算位置由外部给定，本章只核对块账，不推导调度轮次。

## 4. 验收

- 区分每层 page、缓存分组、共享块池和实际 GPU 张量。
- 从给定规格与字节预算算出块数，并识别 null block 的保留。
- 解释空闲队列中的块为何仍可能保留前缀内容。
- 对照请求已有块，算出一次增量申请和命中块的容量影响。
- 解释命中查询、取得引用、登记 hash 与 GPU 写入完成的区别。
- 独立复算案例 B，并说明释放请求为何不等于释放 GPU 大张量。

[源码纵览](../00-Overview.md) · [返回模块索引](../README.md)
