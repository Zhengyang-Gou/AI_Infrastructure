# 04 · 核心客户端与进程循环

**这一篇回答：谁接收新请求，谁反复调用核心 step，结果又怎样返回入口。** 同进程靠调用方推进，多进程靠核心后台循环推进。

## 1. 职责与边界

`LLMEngine` 通过 EngineCoreClient 联系核心。同样是 `add_request()` 和 `get_output()`，不同客户端的内部路线不同。

| 路径 | 核心在哪里 | 谁推进执行 |
| --- | --- | --- |
| InprocClient | 与入口在同一进程 | get_output 调用核心 step_fn |
| SyncMPClient | 核心在后台进程 | EngineCoreProc.run_busy_loop |

这里的“同进程”指入口与核心的关系；Executor 的 Worker 布置还由执行配置决定。它不直接等同于所有模型工作都在一个进程。

本篇读清调用、队列和循环。消息序列化格式、启动握手、在线异步接口及数据并行协调，可以结合各自场景阅读。

## 2. 源码定位与队列

| 文件 | 类 | 关注内容 |
| --- | --- | --- |
| [vllm/v1/engine/core_client.py](../../../vllm/vllm/v1/engine/core_client.py) | InprocClient | add_request、get_output |
| 同上 | SyncMPClient | add_request、get_output、_send_input |
| [vllm/v1/engine/core.py](../../../vllm/vllm/v1/engine/core.py) | EngineCoreProc | 输入输出线程、run_busy_loop、_process_input_queue、_process_engine_step |

多进程路径中有两侧各自的队列，注意属性名：

| 名称 | 所在侧 | 保存什么 |
| --- | --- | --- |
| input_queue | 核心进程 | 输入线程接收并准备好的消息，等待核心循环处理 |
| output_queue | 核心进程 | 核心产生的结果，等待输出线程发送 |
| outputs_queue | 客户端进程 | 从核心收到的结果，供 get_output 取出 |

跨进程传递经过通信通道；这些 Queue 本身是对应进程里线程之间的交接点。

## 3. 实现分析

### 3.1 同进程：直接接收，取输出时推进

```python
# 文件：vllm/v1/engine/core_client.py
# 类：InprocClient
# 函数：add_request
# 源码范围：L327–L329（连续节选）
def add_request(self, request: EngineCoreRequest) -> None:
    req, request_wave = self.engine_core.preprocess_add_request(request)
    self.engine_core.add_request(req, request_wave)
```

`preprocess_add_request()` 返回 `(Request, request_wave)`，客户端取出二者，再调用核心接收方法。

取得输出时：

```python
# 文件：vllm/v1/engine/core_client.py
# 类：InprocClient
# 函数：get_output
# 源码范围：L319–L322（连续节选）
def get_output(self) -> EngineCoreOutputs:
    outputs, model_executed = self.engine_core.step_fn()
    self.engine_core.post_step(model_executed=model_executed)
    return outputs and outputs.get(0) or EngineCoreOutputs()
```

这里先执行 `step_fn()`，再调用 `post_step()`。核心返回按客户端索引组织的结果字典，这条路径取索引 0；没有对应输出时，返回空的 `EngineCoreOutputs`。

```text
LLMEngine.step
    → InprocClient.get_output
    → EngineCore.step_fn
    → EngineCore.post_step
    → 将核心输出交回 LLMEngine
```

因此，同进程的 `get_output()` 同时承担推进核心和取得结果的作用。

### 3.2 多进程：提交是发消息，取输出是读接收队列

```python
# 文件：vllm/v1/engine/core_client.py
# 类：SyncMPClient
# 函数：add_request
# 源码范围：L904–L907（连续节选）
def add_request(self, request: EngineCoreRequest) -> None:
    if self.is_dp:
        self.engines_running = True
    self._send_input(EngineCoreRequestType.ADD, request)
```

`_send_input()` 将消息类型与编码后的请求发送到核心。核心接收线程解码 ADD 消息，在那里调用 `preprocess_add_request()` 创建 Request，再将 `(Request, request_wave)` 放入核心 `input_queue`。

客户端取结果的代码是：

```python
# 文件：vllm/v1/engine/core_client.py
# 类：SyncMPClient
# 函数：get_output
# 源码范围：L872–L882（连续节选）
def get_output(self) -> EngineCoreOutputs:
    # If an exception arises in process_outputs_socket task,
    # it is forwarded to the outputs_queue so we can raise it
    # from this (run_output_handler) task to shut down the server.
    outputs = self.outputs_queue.get()

    if isinstance(outputs, Exception):
        raise self._format_exception(outputs) from None
    if outputs.wave_complete is not None:
        self.engines_running = False
    return outputs
```

这里没有直接调用核心 step，而是等待客户端接收队列中的结果。接收线程转交的后台异常也会在这里抛出。`wave_complete` 对应数据并行协调，单核心主线先关注结果的取得。

### 3.3 后台循环先处理输入，再执行核心

`EngineCoreProc` 继承 EngineCore，复用其调度与执行能力，并增加进程通信和循环：

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCoreProc
# 函数：run_busy_loop
# 源码范围：L1378–L1389（连续节选）
def run_busy_loop(self):
    """Core busy loop of the EngineCore."""
    while self._handle_shutdown():
        # 1) Poll the input queue until there is work to do.
        self._process_input_queue()
        # Publish request counts before and after GPU step to ensure freshness.
        self._maybe_publish_request_counts()
        # 2) Step the engine core and return the outputs.
        self._process_engine_step()
        self._maybe_publish_request_counts()

    raise SystemExit
```

`_process_input_queue()` 的作用分为两部分：没有工作时等待输入，有工作时再处理已经排队的消息。ADD、ABORT 等消息交给 `_handle_client_request()` 分派：

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCoreProc
# 函数：_handle_client_request
# 源码范围：L1512–L1520（连续节选）
if request_type == EngineCoreRequestType.WAKEUP:
    return
elif request_type == EngineCoreRequestType.ADD:
    req, request_wave = request
    if self._reject_add_in_shutdown(req):
        return
    self.add_request(req, request_wave)
elif request_type == EngineCoreRequestType.ABORT:
    self.abort_requests(request)
```

ADD 消息里的请求已经由输入线程预处理为 Request，循环中继续调用核心 `add_request()`；ABORT 消息交给核心取消方法。

循环允许在执行轮次之间接收新请求和取消。它不需要等一条请求生成完整回答后，才接收另一条请求。

### 3.4 执行一轮，把结果交给输出线程

```python
# 文件：vllm/v1/engine/core.py
# 类：EngineCoreProc
# 函数：_process_engine_step
# 源码范围：L1435–L1444（连续节选）
def _process_engine_step(self) -> bool:
    """Called only when there are unfinished local requests."""

    # Step the engine core.
    outputs, model_executed = self.step_fn()
    # Put EngineCoreOutputs into the output queue.
    for output in outputs.items() if outputs else ():
        self.output_queue.put_nowait(output)
    # Post-step hook.
    self.post_step(model_executed)
```

`step_fn()` 返回的字典按客户端索引组织。遍历 `outputs.items()` 得到 `(client_index, EngineCoreOutputs)`，将它们放入核心 `output_queue`，输出线程据此发送给对应客户端。

`has_work()` 除了查看 Scheduler 的工作，还会检查 batch queue。即使没有新的请求可安排，在途批次仍可能需要取回和处理。

消息等待、队列读写和空闲检查共同决定循环何时等待输入、何时继续推进；不必将名字里的 busy 理解为始终空转计算。

## 4. 用一条多进程请求串起两侧

```text
入口进程                              核心进程

LLMEngine.add_request
    → SyncMPClient 发送 ADD          → 输入线程解码并创建 Request
                                        → input_queue
                                        → 后台循环处理 ADD
                                        → Scheduler 保存请求

                                    后台循环反复调用 step_fn
                                        → 调度、执行、更新
                                        → output_queue
                                        → 输出线程发送

客户端接收线程                      ← 核心增量结果
    → outputs_queue
    → SyncMPClient.get_output
    → LLMEngine.step
    → OutputProcessor 处理
```

后台执行可以与入口提交其他请求、处理输出交错进行。因此，入口调用一次 `LLMEngine.step()` 与核心执行一次 step 不保证严格一一对应。

## 5. 阅读检查

- 为什么同名的 get_output，在两条路径里做的事情不同？
- Request 是在核心输入线程还是后台循环里构造的？
- 核心 output_queue 与客户端 outputs_queue 分别连接哪两段？
- 谁调用 step_fn，谁调用 post_step？

[章节目录](../README.md) · [上一篇：Step 与执行闭环](03-Step-and-Execution-Loop.md) · [下一篇：完成、取消与清理](05-Completion-and-Abort.md)
