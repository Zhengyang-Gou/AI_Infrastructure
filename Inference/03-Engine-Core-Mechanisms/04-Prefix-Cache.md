# 04 · Prefix Cache

状态：待学习。

## 核心问题

- 哪些条件下可以安全复用已有前缀的 KV Cache？
- 如何通过 Token 和前缀上下文识别可复用的缓存块？
- 完整块与未填满的尾块分别如何处理？
- 缓存共享、引用计数、淘汰与写入如何配合？
- 如何记录命中率、实际减少的计算量与缓存占用？

## 阅读入口

- [已有笔记：分页 KV Cache 与前缀缓存](../02-Mini-Inference-Engine/notes/03-Paged-KV-Cache-and-Prefix-Caching.md)
- [前置主题：Paged KV Cache](03-Paged-KV-Cache.md)

## 手工推演

构造两个共享前缀的 Prompt，标出可复用的块与仍需计算的部分；修改前缀中的一个 Token，再追踪命中范围。

## 学习记录

待补充：机制说明、源码位置与版本、推演过程、疑问及总结。

[返回学习目录](README.md)
