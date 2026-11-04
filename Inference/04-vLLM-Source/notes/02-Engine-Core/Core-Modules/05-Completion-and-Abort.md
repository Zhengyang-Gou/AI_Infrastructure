# 05 · 完成与取消的核心交接

## 1. 职责与边界

本篇只负责正常完成信息的返回、取消 ID 的转交和取消队列的处理时机。

| 输入 | 核心动作 | 交出什么 |
| --- | --- | --- |
| Scheduler 返回的核心输出 | 随本轮结果返回 | 给客户端的 EngineCoreOutputs |
| 内部取消 ID | 调用 Scheduler 的外部结束接口 | request_ids 与 FINISHED_ABORTED |
| 执行期间积累的取消 ID | 在结果更新前批量处理 | 同一取消接口调用 |

结束条件、RequestStatus 的完整对应、队列退出与释放指令属于 03；实际 KV 回收属于 04；入口输出状态属于 01。本篇不展开它们。

## 2. 正常完成信息的返回

核心将本轮计划与模型结果交给 Scheduler.update_from_output，取得按客户端索引组织的 EngineCoreOutputs。完成标记已经在这份结果中，核心负责交还客户端，不重复执行判停。

输出的直接返回与消息发送由本章第 03、04 篇负责；本篇到“结束信息已随输出交出”为止。

## 3. 取消 ID 的转交

入口已经把取消请求转换为内部 ID。核心的取消接口执行：

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：abort_requests
# 源码范围：L491–L491（连续节选）
self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)
```

这里的 FINISHED_ABORTED 是交给调度器的目标状态。查找有效请求、移出队列和资源交还由该接口完成；核心不进入其实现。

## 4. 执行期间到达的取消

多进程输入线程接收 ABORT 时，会同时写入核心 `input_queue` 和 `aborts_queue`。普通输入队列维持消息处理顺序，单独的取消队列让核心能在模型执行返回后尽快处理取消。

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCore
# 函数：_process_aborts_queue
# 源码范围：L741–L749（连续节选）
def _process_aborts_queue(self):
    if not self.aborts_queue.empty():
        request_ids = []
        while not self.aborts_queue.empty():
            ids = self.aborts_queue.get_nowait()
            # Should be a list here, but also handle string just in case.
            request_ids.extend((ids,) if isinstance(ids, str) else ids)
        # More efficient to abort all as a single batch.
        self.abort_requests(request_ids)
```

`step()` 和 batch queue 路径都在 `update_from_output()` 之前调用它。已经下发的 GPU 工作可能完成，但核心可以先终止请求，再处理返回结果；实际回收安全条件由缓存管理模块负责，本篇只确认先取消、后更新的调用顺序。

如果某条取消消息先到取消队列，而对应 ADD 仍在等待处理，普通输入队列中的那份 ABORT 仍能按消息顺序再次生效。两个队列与重复取消处理共同避免遗漏。

## 5. 最小顺序检查

```text
下发模型工作
    → 收到取消 ID，记录在取消队列
    → 模型调用返回
    → 取出取消 ID，调用 Scheduler.finish_requests
    → 调用 Scheduler.update_from_output
```

取消请求本身不表示已撤销 GPU 调用。核心只保证取消指令在结果更新前得到处理。

## 6. 阅读检查

- 核心对正常完成信息做了哪些转交？
- 为什么在 update_from_output 之前处理取消？
- 普通输入队列与单独取消队列各自保证什么？

[章节目录](../README.md) · [上一篇：核心客户端与进程循环](04-Core-Client-and-Process-Loop.md)
