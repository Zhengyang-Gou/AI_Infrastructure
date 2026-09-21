# 03 · Paged KV Cache

状态：待学习。

## 核心问题

- 连续分配 KV Cache 会遇到哪些容量和碎片问题？
- 逻辑块、物理块与 Block Table 如何对应？
- Token 的位置如何映射到物理块和块内偏移？
- 请求增长或结束时如何分配和回收块？
- 块大小如何影响碎片、管理开销与访问方式？

## 阅读入口

- [已有笔记：分页 KV Cache 与前缀缓存](../02-Mini-Inference-Engine/notes/03-Paged-KV-Cache-and-Prefix-Caching.md)
- [已有笔记：Attention 与 KV Cache 读写](../02-Mini-Inference-Engine/notes/07-Attention-and-Cache-Access.md)

## 手工推演

假设每块容纳 4 个 Token，为长度分别为 3、7、10 的请求建立 Block Table，并追踪追加 Token 和请求结束后的块变化。

## 学习记录

待补充：机制说明、源码位置与版本、推演过程、疑问及总结。

[返回学习目录](README.md)
