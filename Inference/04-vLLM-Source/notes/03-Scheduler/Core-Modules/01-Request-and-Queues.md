# 01 · Request 与调度队列

## 1. 职责与边界

本篇负责 Request 的初始状态、调度器登记与队列归属。输入是核心已构造的 Request；输出是已登记、可参与调度的请求。核心如何接收消息不在本篇展开。

| 对象 | 本篇中的作用 |
| --- | --- |
| Request / RequestStatus | 保存 token 历史、调度进度与生命周期 |
| requests | 通过内部 ID 查找 Request |
| waiting / skipped_waiting | 保存普通等待与等待额外依赖的请求 |
| running | 保存已接纳的请求 |

源码：`vllm/v1/request.py`、`vllm/v1/core/sched/scheduler.py`、`vllm/v1/core/sched/request_queue.py`。路径相对于源码根目录。

## 2. Request 为跨轮执行准备状态

新的普通请求先进入等待状态：

```python
# 文件：vllm/v1/request.py
# 类：Request
# 函数：__init__
# 源码范围：L97–L99（连续节选）
self.status = RequestStatus.WAITING
self.events: list[EngineCoreEvent] = []
self.stop_reason: int | str | None = None
```

输入长度、已生成部分和完整 token 历史也在构造时准备：

```python
# 文件：vllm/v1/request.py
# 类：Request
# 函数：__init__
# 源码范围：L140–L148（连续节选）
self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
    prompt_token_ids, prompt_embeds
)
self._output_token_ids: list[int] = []
self._all_token_ids: list[int] = (
    self.prompt_token_ids.copy()
    if self.prompt_token_ids is not None
    else [0] * self.num_prompt_tokens
)
```

以 4 个输入 token 的请求为例，初始状态可以这样理解：

| 字段 | 初始内容与作用 |
| --- | --- |
| prompt_token_ids | 4 个输入 token 的 ID |
| num_prompt_tokens | 4，输入长度 |
| output_token_ids | 空序列，尚未生成输出 |
| num_computed_tokens | 0，调度侧执行进度的起点 |
| status | WAITING，等待安排 |
| max_tokens | 本请求允许生成的最大输出 token 数 |

后续生成的新 token 会保存在 Request 中，下一轮可以继续使用。`num_computed_tokens` 表示调度侧推进的进度，调度时就可能增加；异步执行时它可能包含在途计算，不能直接当成已经全部完成的 GPU 计算量。

## 3. Scheduler 登记与入队

以下是新的非流式输入请求的登记分支；请求构造已经完成。

```python
# 文件：vllm/v1/core/sched/scheduler.py
# 类：Scheduler
# 函数：add_request
# 源码范围：L2230–L2235（连续节选）
self._enqueue_waiting_request(request)
self.requests[request.request_id] = request
if self.connector is not None:
    self.connector.on_new_request(request)
if self.log_stats:
    request.record_event(EngineCoreEventType.QUEUED)
```

其中两个动作最重要：

- `_enqueue_waiting_request(request)`：放入等待队列；普通 `WAITING` 请求进入 `waiting`，需要额外依赖的请求有相应等待队列。
- `self.requests[request.request_id] = request`：按内部 ID 保存 Request，执行结果返回时可以找到同一个对象。

队列与请求表承担不同作用：队列组织调度顺序，请求表支持按 ID 查找。两处引用的是同一个 Request 对象。

## 4. 生命周期与计算阶段

```text
WAITING：等待调度
    ↓ Scheduler 接纳并安排
RUNNING：进入运行集合，经历一轮或多轮执行
    ├─ 继续：更新进度，参加后续调度
    ├─ 被抢占：进入 PREEMPTED 等待恢复
    └─ 结束：进入相应 FINISHED 状态
```

`RUNNING` 请求可能仍在处理 Prompt 分块，也可能正在 Decode。生命周期状态和 Prefill / Decode 计算阶段是两个不同维度。

## 5. 阅读检查

- 请求表与队列为什么同时保存同一个 Request 的引用？
- Request 的初始 token 历史与输出序列有何区别？
- 为什么 RUNNING 不能直接解释为 Decode？

[章节目录](../README.md) · [本章概览](../Overview.md)
