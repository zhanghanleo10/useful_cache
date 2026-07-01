# useful_cache

vllm-gr 代码走读分析文档集。基于 `vllm-gr/` 和 `vllm/` 实际代码逐行核对。

## 文档索引

### BeamSearch 专题(`vllm-gr_analysis/BeamSearch/`)

> ⚠️ **机制演进提示**:旧文档(`online_pipeline.md` / `scheduling_cascade.md` / `beam_search_walkthrough.md`)基于 **BEAM_FORK + template-clone** 机制,该机制已被重构取代。当前代码用 **Mega(`MegaRequestStepUpdate` + `ADD_BATCH` + `MEGA_REQUEST_STEP_UPDATE`)** 机制。走读当前代码请优先读 `online_pipeline_mega.md`;两代对照见该文末「与旧 BEAM_FORK 机制对照」。

| 文档 | 内容 | 适用 |
|---|---|---|
| [`online_pipeline_mega.md`](vllm-gr_analysis/BeamSearch/online_pipeline_mega.md) | ⭐ **当前版本·在线链路走读(6 单元)**。基于 Mega 机制:patch 装配 → 首步 ADD_BATCH → MEGA 合成 → 调度/Worker(position 重写 + InputBatchProxy)→ cascade attention → 采样 remux。带行号锚点 | **走读当前在线服务(首选)** |
| [`mega_deep_dive.md`](vllm-gr_analysis/BeamSearch/mega_deep_dive.md) | ⭐ **Mega 机制难点深化(数值与张量级)**。用贯穿例子(W=3/prefix=4/decode=2)把 position 重写、cascade 索引、_remux 的张量形状与索引逐值推演;附易踩坑速查 | **深入最难点(配套上一份)** |
| [`beam_search_walkthrough.md`](vllm-gr_analysis/BeamSearch/beam_search_walkthrough.md) | **概览**(Draft v1.0)。整体架构、文件拓扑、全链路图、与原生对比。⚠️ 旧版本,有勘误见 errata.md | 建立全局认知 |
| [`online_pipeline.md`](vllm-gr_analysis/BeamSearch/online_pipeline.md) | **在线链路精读**(旧 BEAM_FORK / template-clone 机制)。patch 注入 → BeamForkRequest → AsyncLLM → template-clone。⚠️ 机制已过时,仅作演进参照 | 旧版本对照 |
| [`scheduling_cascade.md`](vllm-gr_analysis/BeamSearch/scheduling_cascade.md) | **调度与 Cascade Attention**。⚠️ 部分概念(beam_prefix_groups)属旧机制;cascade 三阶段 + LSE 合并数学仍适用 | 走读调度与计算层 |
| [`errata.md`](vllm-gr_analysis/BeamSearch/errata.md) | **原文档勘误表**。A 类事实错误(11)/ B 类简化失真(6)/ C 类概念遗漏(4)/ 行号核对 | 读旧文档前先看 |
| [`gr_specifics.md`](vllm-gr_analysis/BeamSearch/gr_specifics.md) | **GR 特有机制**。Catalog/Trie(推荐候选约束)、begin/end_token(session 标记)、GR 类型扩展链路(gr_features 透传) | 做生成式推荐必读 |

## 推荐阅读路径

1. **走读当前代码(首选)**:按 `online_pipeline_mega.md` 6 个单元逐层走读(基于当前 Mega 机制)
2. **深入最难点**:读 `mega_deep_dive.md`(position 重写 / cascade / remux 的数值与张量级展开)
3. **建立全局**:读 `beam_search_walkthrough.md` §1-2(架构 + 文件拓扑)+ `errata.md`(旧文档勘误)
4. **旧版本对照**:读 `online_pipeline.md` 了解 BEAM_FORK/template-clone 机制演进
5. **GR 专章**(推荐场景):读 `gr_specifics.md`(Catalog 是推荐场景核心)

## 核心结论速查

**vllm-gr beam search 的"共享前缀"思想在三个层次各实现一次**(旧机制表述,当前 Mega 机制见下):

| 层次 | 机制(旧) | 当前 Mega 对应 | 省 |
|---|---|---|---|
| 存储 | template-clone + beam_cache | `beam_cache` + 继承 `block_hashes` | N 份 KV → 1 份共享 + N 份增量 |
| 调度 | beam_prefix_groups(computed_blocks 缓存) | scheduler 裁 `prefix_len` + 透传 `mega_data` | N 次 hash 遍历 → 1 次 |
| 计算 | cascade attention(prefix 算一次 + LSE 合并) | 同(cascade prefix 非因果 + suffix 因果 + merge) | N 次前缀 attention → 1 次 |

**心智模型**:
- 旧(BEAM_FORK):ADD_BATCH ≈ Prefill(冷启动建 KV)、BEAM_FORK ≈ Decode(增量复用父 KV)。
- **当前(Mega)**:ADD_BATCH ≈ 首步 Prefill(建 `beam_cache` + 展开 1→W beam);MEGA_REQUEST_STEP_UPDATE ≈ 后续 Decode(拼单条 Mega Request 复用前缀 KV + cascade attention,W 行采样后 `_remux` 压回 1 行)。
