# 第一章源码索引

本目录保存“请求调用链”章节阅读时使用的源码快照。文件沿用 vLLM 仓库中的原始相对路径，便于与笔记、import 路径和调用栈互相对应。

| 笔记 | 源码快照 | 阅读重点 |
| --- | --- | --- |
| `01-basic.md` | `examples/basic/offline_inference/basic.py` | 离线推理入口 |
| `02-llm.md` | `vllm/entrypoints/llm.py` | `LLM` 初始化与 `generate()` |
| `03-offline-utils.md` | `vllm/entrypoints/offline_utils.py` | 请求编排和同步执行循环 |
| `04-llm-engine.md` | `vllm/v1/engine/llm_engine.py` | 输入处理、核心请求提交与输出处理 |
| `05-core-client.md` | `vllm/v1/engine/core_client.py` | 进程内及多进程客户端 |
| `06-engine-core.md` | `vllm/v1/engine/core.py` | 核心调度与执行循环 |
| `07-end-to-end-trace.md` | 上述全部文件 | 完整请求链路 |
| `08-configuration-and-initialization.md` | `vllm/engine/arg_utils.py` | `EngineArgs` 与配置创建入口 |
| `08-configuration-and-initialization.md` | `vllm/config/vllm.py` | `VllmConfig` 聚合配置 |
| `08-configuration-and-initialization.md` | `vllm/platforms/interface.py` | 通用平台接口和默认配置 |
| `08-configuration-and-initialization.md` | `vllm/platforms/cuda.py` | CUDA 平台校验与后端选择 |

## 使用说明

- 这里的文件是学习快照，不参与 vLLM 包的实际运行；运行时仍使用仓库根目录下的原始源码。
- 阅读笔记时，可以从对应快照中搜索类名或函数名，再沿调用关系跳到下一文件。
- 若根目录源码版本发生变化，应重新同步快照，避免笔记与实现不一致。
