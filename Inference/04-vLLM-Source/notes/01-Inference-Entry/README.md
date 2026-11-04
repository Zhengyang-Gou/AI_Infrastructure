# 01 · Inference Entry

本章学习 vLLM 的推理入口：**一次用户调用，如何成为引擎请求，又如何取得生成结果？**

固定场景（decoder-only 普通文本、离线 `LLM.generate(...)`、每条输入生成一个候选）见[顶层 README](../README.md) 第 1 节，本章不再复述。主线之外的接口和分支在相关笔记中说明衔接位置。

## 职责边界

| 项目 | 契约 |
| --- | --- |
| 输入 | 用户 Prompt 与生成参数；返回方向的 EngineCoreOutputs |
| 输出 | EngineCoreRequest、内部取消 ID；用户侧 RequestOutput |
| 本章负责 | 离线入口、输入转换与校验、请求 ID、入口输出状态、解码、停止字符串与结果格式 |
| 不展开 | 客户端与核心循环（02）、调度与核心判停（03）、KV 管理（04）、执行/Attention/采样算法（06–08）、在线连接交付（11） |

同章分工也固定：01 公共入口与默认参数，02 批量编排，03 内容转换，04 合法请求与内部 ID，05 调用顺序，06 输出状态与解码。其他篇只使用其接口，不重复展开实现。

## 1. 本章在学习路线中的位置

```text
用户调用
    ↓
Inference Entry：接口、输入准备、请求提交、结果收集  ← 本章
    ↓↑
Engine Core：接收请求、组织执行、返回增量          ← 02
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
- **Overview** 解释入口侧的组件框架、数据变化与调用路线。
- **Core-Modules** 按「职责与边界 → 源码定位与关键数据 → 实现分析 → 手工推演 → 设计原因与自测」进入具体代码。

先读 [Overview](Overview.md)，再顺序阅读六篇 Core，最后回到 Overview，用同一条请求复盘全过程。

| 笔记 | 主要源码与学习重点 |
| --- | --- |
| [01 · LLM 与初始化](Core-Modules/01-LLM-and-Initialization.md) | LLM、EngineArgs、引擎工厂、默认参数、公共接口 |
| [02 · 离线推理流程编排](Core-Modules/02-Offline-Inference-Flow.md) | OfflineInferenceMixin、批量归一化、生成器提交、异常回收、结果排序 |
| [03 · Renderer 与输入表示](Core-Modules/03-Renderer-and-EngineInput.md) | BaseRenderer、解析、分词、TokensInput、EngineInput |
| [04 · InputProcessor 与请求构造](Core-Modules/04-InputProcessor-and-EngineCoreRequest.md) | 参数和长度校验、参数克隆、EngineCoreRequest、内外请求 ID |
| [05 · LLMEngine 与核心边界](Core-Modules/05-LLMEngine-and-Core-Boundary.md) | 请求提交、状态注册、step、客户端接口与取消转交 |
| [06 · 输出处理与文本还原](Core-Modules/06-OutputProcessor-and-Detokenization.md) | RequestState、增量解码、停止字符串、输出模式、结果与清理 |

## 4. 源码阅读约定

所有函数代码块开头使用 `# 文件`、`# 类`、`# 函数` 注明位置，并给出本地源码行范围。文件路径相对于 `04-vLLM-Source/vllm/`；模块级函数标为「类：无」。展示的代码均为连续源码节选，省略部分通过前后说明连接；行号对应本地这份源码。

普通文本主线只展开本章六篇的职责。chat 和多候选属于入口变体；在线连接与异步交付归 11，数据并行归 10。本章到 EngineCoreClient 的提交/结果接口为止。

## 5. 本章学习目标

- 解释 LLM、OfflineInferenceMixin、Renderer、InputProcessor、LLMEngine 与 OutputProcessor 的职责及协作关系。
- 追踪 `Prompt → EngineInput → EngineCoreRequest → EngineCoreOutput → RequestOutput` 的变化。
- 说明外部与内部请求 ID 如何连接提交、状态查找、取消和结果排序。
- 解释同步 generate 如何封装增量执行，以及 FINAL_ONLY 为什么仍需持续处理输出。
- 区分入口侧 step、核心执行步骤和单请求生成进度。
- 追踪正常完成、输出侧停止与部分提交失败时的入口状态清理及取消指令。

[开始阅读 Overview](Overview.md) · [返回推理学习路线](../../../Roadmap.md)
