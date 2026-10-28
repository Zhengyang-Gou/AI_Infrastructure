# 01 · Inference Entry

本章学习 vLLM 的推理入口：**一次用户调用，如何成为引擎请求，又如何取得生成结果？**

以仓库内 `04-vLLM-Source/vllm/` 保存的源码为依据，主线固定为 decoder-only 普通文本、离线 `LLM.generate(...)`、每条输入生成一个候选（`n=1`）。主线之外的接口和分支在相关笔记中说明衔接位置。

## 1. 本章在学习路线中的位置

```text
用户调用
    ↓
Inference Entry：接口、输入准备、请求提交、结果收集  ← 本章
    ↓↑
Engine Core：调度、缓存管理、模型执行              ← 后续章节
```

前置知识是 [Nano-vLLM 的生成主循环](../../../02-Mini-Inference-Engine/notes/01-Entry-and-Generation.md)、请求生命周期和增量生成。本章掌握入口侧的完整交接，再到 [02 · Engine Core](../02-Engine-Core/) 追踪核心执行闭环。

## 2. 章节框架

| 顺序 | 主题 | 核心问题 |
| --- | --- | --- |
| 01 | LLM 与初始化 | 用户创建引擎时准备什么？公共调用从哪里进入？ |
| 02 | 离线推理流程编排 | 一批输入如何对齐参数、提交并收集结果？ |
| 03 | Renderer 与输入表示 | 文本或 Token IDs 如何成为统一的 EngineInput？ |
| 04 | InputProcessor 与请求构造 | 输入与生成参数如何组成合法的执行请求？ |
| 05 | LLMEngine 与核心边界 | 如何注册输出状态、提交核心并取回结果？ |
| 06 | 输出处理与文本还原 | 增量 Token 如何解码、判停并构造用户结果？ |

## 3. 文档组织与阅读顺序

```text
01-Inference-Entry/
├── README.md
├── Overview.md
└── Core-Modules/
    ├── 01-LLM-and-Initialization.md
    ├── 02-Offline-Inference-Flow.md
    ├── 03-Renderer-and-EngineInput.md
    ├── 04-InputProcessor-and-EngineCoreRequest.md
    ├── 05-LLMEngine-and-Core-Boundary.md
    └── 06-OutputProcessor-and-Detokenization.md
```

- **README** 介绍本章范围、主题划分与学习导航。
- **Overview** 解释组件框架、数据变化与整条执行路线。
- **Core-Modules** 按「职责与边界 → 源码定位与关键数据 → 实现分析 → 手工推演 → 设计原因与自测」进入具体代码。

先读 [Overview](Overview.md)，再顺序阅读六篇 Core，最后回到 Overview，用同一条请求复盘全过程。

| 笔记 | 主要源码与学习重点 |
| --- | --- |
| [01 · LLM 与初始化](Core-Modules/01-LLM-and-Initialization.md) | LLM、EngineArgs、引擎工厂、默认参数、公共接口 |
| [02 · 离线推理流程编排](Core-Modules/02-Offline-Inference-Flow.md) | OfflineInferenceMixin、批量归一化、生成器提交、异常回收、结果排序 |
| [03 · Renderer 与输入表示](Core-Modules/03-Renderer-and-EngineInput.md) | BaseRenderer、解析、分词、TokensInput、EngineInput |
| [04 · InputProcessor 与请求构造](Core-Modules/04-InputProcessor-and-EngineCoreRequest.md) | 参数和长度校验、参数克隆、EngineCoreRequest、内外请求 ID |
| [05 · LLMEngine 与核心边界](Core-Modules/05-LLMEngine-and-Core-Boundary.md) | 请求提交、状态注册、step、核心客户端与取消 |
| [06 · 输出处理与文本还原](Core-Modules/06-OutputProcessor-and-Detokenization.md) | RequestState、增量解码、停止字符串、输出模式、结果与清理 |

## 4. 源码阅读约定

所有函数代码块开头使用 `# 文件`、`# 类`、`# 函数` 注明位置，并给出本地源码行范围。文件路径相对于 `04-vLLM-Source/vllm/`；模块级函数标为「类：无」。展示的代码均为连续源码节选，省略部分通过前后说明连接；行号对应本地这份源码。

普通文本主线先理解请求提交、状态推进和输出返回，再按需要展开 chat、多个候选、多模态、在线异步入口与数据并行。核心内部的调度、KV 管理和模型执行会在后续章节分别学习。

## 5. 本章学习目标

- 解释 LLM、OfflineInferenceMixin、Renderer、InputProcessor、LLMEngine 与 OutputProcessor 的职责及协作关系。
- 追踪 `Prompt → EngineInput → EngineCoreRequest → EngineCoreOutput → RequestOutput` 的变化。
- 说明外部与内部请求 ID 如何连接提交、状态查找、取消和结果排序。
- 解释同步 generate 如何封装增量执行，以及 FINAL_ONLY 为什么仍需持续处理输出。
- 区分入口侧 step、核心执行步骤和单请求生成进度。
- 追踪正常完成、输出侧停止与部分提交失败时的清理路线。

[开始阅读 Overview](Overview.md) · [返回推理学习路线](../../../Roadmap.md)
