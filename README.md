# useful_cache

vllm-gr 代码走读分析文档集。基于 `vllm-gr/` 和 `vllm/` 实际代码逐行核对。

## 文档索引

### BeamSearch 专题(`vllm-gr_analysis/BeamSearch/`)

| 文档 | 内容 | 适用 |
|---|---|---|
| [`beam_search_walkthrough.md`](vllm-gr_analysis/BeamSearch/beam_search_walkthrough.md) | **概览**(Draft v1.0)。整体架构、文件拓扑、全链路图、与原生对比。⚠️ 有勘误,见 errata.md | 建立全局认知 |
| [`online_pipeline.md`](vllm-gr_analysis/BeamSearch/online_pipeline.md) | **在线链路精读 ❶–❽**。patch 注入 → 参数 → BeamForkRequest → 主循环 → AsyncLLM → AsyncMPClient → 子进程入口 → template-clone。带核心例子和行号锚点 | 逐层走读在线服务 |
| [`scheduling_cascade.md`](vllm-gr_analysis/BeamSearch/scheduling_cascade.md) | **调度与 Cascade Attention**。beam_prefix_groups 诞生与传递、cascade 三阶段、LSE 合并数学、Triton kernel | 走读调度与计算层 |
| [`errata.md`](vllm-gr_analysis/BeamSearch/errata.md) | **原文档勘误表**。A 类事实错误(11)/ B 类简化失真(6)/ C 类概念遗漏(4)/ 行号核对 | 读原文档前先看 |
| [`gr_specifics.md`](vllm-gr_analysis/BeamSearch/gr_specifics.md) | **GR 特有机制**。Catalog/Trie(推荐候选约束)、begin/end_token(session 标记)、GR 类型扩展链路(gr_features 透传) | 做生成式推荐必读 |

## 推荐阅读路径

1. **建立全局**:读 `beam_search_walkthrough.md` §1-2(架构 + 文件拓扑)+ 本目录的 `errata.md`(知道原文档哪里不能信)
2. **在线链路**(核心):按 `online_pipeline.md` ❶→❽ 逐章走读
3. **调度与计算**:读 `scheduling_cascade.md`(阶段 A/B/C)
4. **GR 专章**(推荐场景):读 `gr_specifics.md`(Catalog 是推荐场景核心)

## 核心结论速查

**vllm-gr beam search 的"共享前缀"思想在三个层次各实现一次**:

| 层次 | 机制 | 省 |
|---|---|---|
| 存储 | template-clone + beam_cache | N 份 KV → 1 份共享 + N 份增量 |
| 调度 | beam_prefix_groups(computed_blocks 缓存) | N 次 hash 遍历 → 1 次 |
| 计算 | cascade attention(prefix 算一次 + LSE 合并) | N 次前缀 attention → 1 次 |

**心智模型**:ADD_BATCH ≈ Prefill(冷启动建 KV)、BEAM_FORK ≈ Decode(增量复用父 KV)。
