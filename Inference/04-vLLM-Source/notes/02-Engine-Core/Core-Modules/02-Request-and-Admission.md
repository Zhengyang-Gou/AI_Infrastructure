# 02 · Request 与请求接收

## 1. 职责与边界

本篇从收到 EngineCoreRequest 开始，到调用 Scheduler.add_request 为止。负责转换工厂的调用与核心接收，不展开 Request 初始化字段、队列实现或生命周期。

```text
EngineCoreRequest
    → preprocess_add_request：调用转换工厂，取得 Request
    → EngineCore.add_request：检查核心接收条件
    → Scheduler.add_request：将对象交给调度器
```

## 2. 数据契约

| 数据 | 在本篇的含义 |
| --- | --- |
| EngineCoreRequest | 入口已准备好的 ID、输入与请求参数 |
| Request | 转换工厂返回的核心请求；内部字段由 03 负责 |
| request_wave | 随请求传递的协调信息；单核心主线不解释其跨 rank 语义 |

核心沿用收到的内部 ID，不再分词或重新编号。

## 3. 核心接收实现

### 3.1 调用转换工厂

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：preprocess_add_request
# 源码范围：L983–L991（连续节选）
req = Request.from_engine_core_request(request, self.request_block_hasher)
if req.use_structured_output:
    # Note on thread safety: no race condition.
    # `grammar_init` is only invoked in input processing thread. For
    # `structured_output_manager`, each request is independent and
    # grammar compilation is async. Scheduler always checks grammar
    # compilation status before scheduling request.
    self.structured_output_manager.grammar_init(req)
return req, request.current_wave
```

转换工厂接收提交数据与预先准备的 hasher，返回 Request。核心不在这里解释 token 历史和进度字段如何初始化。

返回值是 `(req, request.current_wave)`。结构化输出时还会准备 grammar 依赖；普通文本主线直接交出请求。

### 3.2 转交 Scheduler

EngineCore.add_request 检查请求 ID 类型、任务支持情况及传输配置后，调用：

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：add_request
# 源码范围：L479–L479（连续节选）
self.scheduler.add_request(request)
```

输入是转换完成的 Request；调用成功表示调度器已接收。核心不在此选择运行名额或执行 token。

## 4. 最小调用记录

```text
收到提交数据 Q
    → 转换工厂返回 Request R
    → 核心检查通过
    → Scheduler.add_request(R)
```

记录到这里结束，不继续推演调度队列与运行轮次。

## 5. 阅读检查

- 哪个调用完成提交数据到 Request 的转换？
- 核心接收与调度器选择执行为什么是不同接口？
- 此处是否重新分词、分配内部 ID？

[章节目录](../README.md) · [上一篇：EngineCore 与初始化](01-EngineCore-and-Initialization.md) · [下一篇：Step 与执行闭环](03-Step-and-Execution-Loop.md)
