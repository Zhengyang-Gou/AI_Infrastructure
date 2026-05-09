# GPU/CUDA 入门
## GPU Performance Background
### GPU 架构基础
GPU 是一个高度并行的处理器架构，主要由处理单元和内存层级组成，从宏观上看，NVIDIA的GPU主要由以下三个核心部分构成：
- SM (Streaming Multiprocessors，流多处理器)
- L2 Cache (二级缓存)
- DRAM (高带宽内存)

在现代神经网络中，乘加运算是最频繁的操作，它是全连接层和卷积层的基础构建块

评估一个 GPU 算力的公式如下：
- 单个SM每个时钟周期的浮点运算次数 = 乘加运算次数 × 2
- 整个GPU的峰值算力 = 单个SM每时钟周期的浮点运算次数 × SM的总数 × SM的时钟频率

一台拥有108个SM、主频为1.41 GHz的A100 GPU，它的峰值稠密吞吐量可以达到 156 TF32 TFLOPS 和 312 FP16 TFLOPS

Tensor Cores vs. CUDA Cores

GPU内部执行计算的主要分两类，它们各司其职：
- Tensor Cores：自Volta架构开始引入，专门为了加速机器学习和科学计算中的矩阵乘法和累加操作而生
  - 它们不是逐个元素计算，而是直接对小矩阵块（例如4x4的矩阵块）进行操作
  - 混合精度计算： Tensor Cores 可以在计算和累加时，使用比输入数据更高的精度。比如，输入数据是FP16（半精度）以节省内存和加速读取，但在相乘后，它可以把结果用FP32（单精度）累加起来，从而避免了精度丢失
- CUDA Cores：
  - 当数学运算无法被公式化为矩阵运算时，就会被分配给其他CUDA Cores来执行

像FP16级别的计算，既可以在Tensor Cores上执行，也可以在CUDA Cores上执行

### GPU 执行模型
GPU 执行函数时，并不是简单启动一个线程，而是使用两级结构：

Thread → Thread Block
```
一个 GPU function / kernel
        ↓
被拆成多个 thread blocks
        ↓
每个 thread block 包含多个 threads
```
一个函数的所有线程会被分组到 equally-sized thread blocks 中，然后这些 thread blocks 被一起 launch 执行

当某些线程因为等待数据或等待前序指令结果而暂时不能继续执行时，GPU 不会一直空等，而是切换去执行其他已经准备好的线程，所以**GPU 需要远多于硬件执行单元数量的线程**

#### Thread Block 与 SM 的关系
Thread Block 被放置到某个 SM 上执行
- 一个 block 会被分配给一个 SM
- block 内的 threads 在这个 SM 上执行
- 同一个 block 内的 threads 可以高效通信和同步
- 不同 block 通常相互独立，由 GPU 调度到不同 SM 上

每个 SM 支持：
- 多线程执行
- shared memory
- synchronization

所以同一个 thread block 内的线程可以使用 SM 提供的机制进行通信和同步

#### Wave：一批同时运行的 thread blocks
wave 是一批同时在 GPU 上运行的 thread blocks

假设一块 GPU 有 8 个 SM，并且每个 SM 同一时间只能执行 1 个 block，那么：一个 wave 最多包含 8 个 blocks

如果总共有 12 个 blocks，那么执行过程会分成两波：第 1 wave：8 个 blocks；第 2 wave：4 个 blocks

#### Tail Effect：尾部效应
kernel 执行接近结束时，只剩少量 thread blocks 还在运行，导致部分 SM 没有工作

最有效的是启动能执行 several waves 的函数，这样 tail wave 占用的时间比例更小

### 理解性能
#### 决定性能的三大瓶颈
一个函数在处理器上执行时，其性能通常受限于以下三个因素之一：
- 数学计算带宽 (Math bandwidth)： GPU 算数字的速度
- 内存带宽 (Memory bandwidth)： GPU 搬运数据的速度
- 延迟 (Latency)： 等待指令或数据的时间

#### Math-limited
如果：T_math > T_memory，说明执行数学计算所需时间更长，这时 kernel 的瓶颈是数学计算吞吐

这类操作的典型特征是：
- 每读入一些数据，会做大量计算
- 数据复用率高
- 计算单元很忙
- Tensor Cores 或 CUDA Cores 是主要瓶颈

典型例子：大矩阵乘法、大 batch 的 Linear layer、大规模 Convolution

#### Memory-limited
如果：T_memory > T_math，说明访问内存所需时间更长，这时 kernel 的瓶颈是内存带宽

这类操作的典型特征是：
- 读写大量数据
- 对每个数据元素只做很少计算
- 计算单元可能等数据
- 即使 GPU 有很高 FLOPS，也用不满

典型例子：ReLU、elementwise add、pooling、LayerNorm、SoftMax 的部分实现

#### 算数强度 Arithmetic Intensity
算术强度 = 计算操作数 / 内存访问字节数（FLOPS/B）

表示每访问 1 byte 数据，执行多少次浮点操作

Arithmetic Intensity 衡量的是：一个算法算得多还是搬得多

| Arithmetic Intensity | 含义                | 典型瓶颈               |
| -------------------- | ----------------- | ------------------ |
| 高                    | 每读取/写入一点数据，就做很多计算 | 更可能 math-limited   |
| 低                    | 搬运很多数据，但只做很少计算    | 更可能 memory-limited |

#### Ops:byte ratio
Arithmetic intensity 是算法或 kernel 的属性，但要判断瓶颈，还需要知道 GPU 本身的能力比例

ops:byte ratio = GPU 的计算带宽 / GPU 的内存带宽

表示GPU 每能搬运 1 byte 数据，理论上可以执行多少计算操作，这个值越高，说明 GPU 的计算能力相对于内存带宽更强

| 比较关系                                  | 结论             |
| ------------------------------------- | -------------- |
| arithmetic intensity > ops:byte ratio | math-limited   |
| arithmetic intensity < ops:byte ratio | memory-limited |

#### Latency-limited
Latency-limited 通常发生在：workload 不够大，或者没有足够并行性

| 现象                 | 说明                       |
| ------------------ | ------------------------ |
| thread blocks 数量太少 | 无法填满所有 SM                |
| threads 太少         | 无法隐藏指令或访存延迟              |
| kernel 很小          | launch overhead 和调度开销占比高 |
| tail effect 严重     | 最后一波 blocks 只占用少数 SM     |
| batch size 太小      | DNN 推理中常见                |

## Easier introduction to CUDA
### 从 C++ 代码开始
在这一步，核心目标是：写一个程序，把两个包含 100 万个浮点数的数组加起来

如果只用 CPU 来做这件事，标准的 C++ 代码是非常直观的：

```C++
#include <iostream>
#include <math.h>

void add(int n, float *x, float *y) {
  for (int i = 0; i < n; i++) {
    y[i] = x[i] + y[i];
  }
}

int main(void) {
  int N = 1<<20; 

  // 分配内存
  float *x = new float[N];
  float *y = new float[N];

  // 初始化数组
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // 在 CPU 上调用函数
  add(N, x, y);

  // 释放内存
  delete[] x;
  delete[] y;

  return 0;
}
```
这段代码在 CPU 上运行得很好，但是，如果要处理的数据不是 100 万，而是 10 亿呢？CPU 串行计算就会变得很慢

在 CUDA 编程中，有一个非常核心的概念：Host（主机） 和 Device（设备）
- Host (主机)：指的是你的 CPU 以及系统的内存
- Device (设备)：指的是你的 GPU 以及 GPU 上的显存

要做的第一件事，就是把上面那个普通的 C++ add 函数，变成一个可以在 GPU 上运行的函数，在 CUDA 中，这种在 GPU 上执行的函数被称为核函数 (Kernel)

```C++
// 加上 __global__，它就变成了一个 CUDA 核函数
__global__ 
void add(int n, float *x, float *y) {
  for (int i = 0; i < n; i++) {
    y[i] = x[i] + y[i];
  }
}
```
__global__ 告诉 CUDA 编译器：
- 这个函数将在 GPU 上执行
- 这个函数可以被 CPU 调用

在 C++ 中，调用函数是这样的：

add(N, x, y);

因为 add 现在是一个 GPU 核函数，当从 CPU 调用它时，我们必须告诉 GPU 需要分配多少个线程来执行这个任务

为了传递这个信息，CUDA 扩展了 C++ 的语法，引入了三对尖括号 <<< ... >>>，这被称为执行配置

add<<<1, 1>>>(N, x, y);

- 第一个 1 代表：启动 1 个线程块
- 第二个 1 代表：这个线程块里面包含 1 个线程

普通的 C++ 代码通常保存为 .cpp 文件，用 g++ 或 clang 编译，CUDA 代码通常保存为 .cu 文件，需要用 NVIDIA 提供的 CUDA 编译器 nvcc 来编译

编译命令：nvcc add.cu -o add_cuda

### CUDA 内存管理
CPU 和 GPU 的内存是物理隔离的，如果直接把 CPU 分配的内存指针传给 GPU，程序会直接崩溃

在早期的 CUDA 编程中，为了让 GPU 能访问数据，必须做以下繁琐的步骤
- 在 CPU 上分配内存 (malloc / new)
- 在 GPU 上分配显存 (cudaMalloc)
- 把数据从 CPU 拷贝到 GPU (cudaMemcpy HostToDevice)
- GPU 计算...
- 把结果从 GPU 拷贝回 CPU (cudaMemcpy DeviceToHost)
- 释放两边的内存

为了让 CUDA 变得更简单，NVIDIA 引入了 统一内存 (Unified Memory) 技术

统一内存提供了一个单一的内存地址空间，这个空间里的数据，CPU 和 GPU 都可以直接访问，底层是由系统自动处理数据的迁移

有了统一内存，我们只需要把标准 C++ 的内存分配函数，替换成 CUDA 提供的统一内存分配函数即可

#### 第一步 替换 new 和 delete
```C++
float *x, *y;
cudaMallocManaged(&x, N * sizeof(float));
cudaMallocManaged(&y, N * sizeof(float));
```
cudaMallocManaged 接收两个参数，第一个是指针的地址，第二个是需要分配的字节数

#### 释放内存：用 cudaFree 替换 delete
```C++
cudaFree(x);
cudaFree(y);
```

#### 异步执行(Asynchronous)
核函数启动是异步的，当执行 add<<<1, 1>>>(N, x, y); 时，CPU 只是给 GPU 发送了一个指令去执行这个函数。发送完指令后，CPU 不会等待 GPU 计算完成，而是立刻执行下一行代码

需要让 CPU 停下来，等 GPU 计算完，CUDA 提供了一个函数叫 cudaDeviceSynchronize()
```C++
#include <iostream>
#include <math.h>

// 1. 加上 __global__ 变成核函数
__global__
void add(int n, float *x, float *y) {
  for (int i = 0; i < n; i++) {
    y[i] = x[i] + y[i];
  }
}

int main(void) {
  int N = 1<<20;
  float *x, *y;

  // 2. 使用统一内存分配，CPU和GPU都能访问
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // 在 CPU 上初始化数据
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // 3. 启动核函数 (1个线程块，1个线程)
  add<<<1, 1>>>(N, x, y);

  // 4. 让 CPU 等待 GPU 计算完成
  cudaDeviceSynchronize();

  // 检查结果
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // 5. 释放统一内存
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
```

### 引入线程实现并行化
#### 第一步 修改执行配置
把调用代码改成：add<<<1, 256>>>(N, x, y);（2的幂次方在 GPU 上运行效率更高）

这行代码的意思是：启动 1 个线程块，在这个线程块里给我安排 256 个线程同时工作

#### 第二步 改造核函数
现在，有 256 个线程同时冲进了 add 这个核函数里

如果不修改原来的核函数代码，这 256 个线程都会去执行 for (int i = 0; i < n; i++)

这意味着，这 100 万次加法，被 256 个线程各自重复做了一遍

为了实现分工，CUDA 为每个线程提供了两个极其重要的内置变量
- threadIdx.x：当前线程在队伍里的编号（从 0 到 255）
- blockDim.x：这个队伍里总共有多少个线程（这里是 256）

#### 步长循环
```C++
__global__
void add(int n, float *x, float *y) {
  // 1. 获取当前线程的编号 (0 到 255)
  int index = threadIdx.x;
  
  // 2. 获取每次跨越的步长 (256)
  int stride = blockDim.x;
  
  // 3. 核心：每个线程只处理属于自己的那部分数据！
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}
```

### 跨越线程块：网格与多处理器
前面部分只用了一个线程块，导致 GPU 上只有一个流式多处理器 (SM) 在工作，其他的都在闲置

为了启动多个线程块，CUDA 引入了网格的概念，一个网格包含多个线程块，一个线程块包含多个线程

那么，处理 100 万个元素，我们需要多少个线程块呢？

假设每个块还是 256 个线程，需要确保总线程数至少能覆盖 100 万个元素

计算公式很简单：1,000,000 / 256 = 3906.25，线程块的数量必须是整数，所以需要向上取整

#### 第一步 修改执行配置
```C++
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;

add<<<numBlocks, blockSize>>>(N, x, y);
```

#### 网格步长循环
CUDA 提供了另外两个内置变量：
- blockIdx.x：当前线程所在的线程块的编号
- gridDim.x：网格中总共有多少个线程块

计算一个线程的全局唯一编号：当前块的编号 * 每个块的线程数 + 块内的线程编号

计算整个网格的总跨度：每个块的线程数 * 网格中的总块数
```C++
__global__
void add(int n, float *x, float *y) {
  // 1. 计算当前线程在整个网格中的全局唯一索引
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  
  // 2. 计算整个网格的总线程数，作为步长
  int stride = blockDim.x * gridDim.x;
  
  // 3. 网格步长循环
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}
```