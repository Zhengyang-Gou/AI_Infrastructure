# 02 · Mini Inference Engine

以 Nano-vLLM 为学习案例，从生成入口读到请求调度、分页缓存和模型执行，理解最小推理引擎的组成。

- [学习笔记与阅读顺序](notes/README.md)：按依赖关系拆分的 10 篇笔记。
- [Nano-vLLM 源码](source/nano-vllm/)：完整项目，包含模型实现、示例和基准测试。
- [原项目说明](source/nano-vllm/README.md)：安装与使用方法。

```text
02-Mini-Inference-Engine/
├── README.md
├── notes/                 # 分主题学习笔记
└── source/
    └── nano-vllm/         # 源码、示例、依赖配置与原项目资源
```

阅读顺序：入口与生成 → 请求状态 → KV Cache → 调度 → 执行器 → Qwen3 与基础算子 → Attention → 张量并行与权重加载 → 采样 → 完整 Step。

运行示例或安装项目时，以 `source/nano-vllm/` 为项目工作目录。

[上一阶段：推理基础](../01-LLM-Inference-Basics/README.md) · [推理学习路线](../Roadmap.md)
