# DeepSeek-V4 in vLLM — 分析文档集

基于 `vllm/vllm/models/deepseek_v4/` 实际代码逐文件核对。所有引用形如 `file:line`,可点击跳转源码。

> 代码位置:`vllm/vllm/models/deepseek_v4/`(平台隔离:`nvidia/` · `amd/` · `xpu/` + 公共 `common/`)
> 配置类:`DeepseekV4Config`,`model_type = "deepseek_v4"`,默认 `max_position_embeddings = 1048576`(**1M 上下文**)。

## 文档索引

| 文档 | 内容 | 适用 |
|---|---|---|
| [`overview.md`](overview.md) | ⭐ **总览**。一句话定义 + 核心特性矩阵 + 平台隔离 + 数据流梗概 + 心智模型 | 建立全局认知(首选入口) |
| [`architecture.md`](architecture.md) | **代码地图**。类层次、目录拓扑、DecoderLayer/Model 前向数据流、权重加载映射、PP/EP/TP 并行 | 找代码位置、理清调用链 |
| [`mhc.md`](mhc.md) | ⭐ **MHC 多流隐式扩容**。`hc_mult` 展开 → Sinkhorn 混合 → `hc_head` 折叠;V4 最独特的结构创新 | 理解 V4 vs V3 的根本差异 |
| [`mla.md`](mla.md) | **MLA 机制原理**。KV 联合压缩 → 矩阵吸收 → RoPE 解耦;为什么省显存;V2/V3/V4 通用 | 读 Sparse MLA 前的前置原理 |
| [`sparse_mla_indexer.md`](sparse_mla_indexer.md) | ⭐ **稀疏潜在注意力**。Sparse MLA + Compressor + Lightning Indexer;C4A/C128A 双模式;KV cache 布局 | 长上下文(1M)能跑的关键 |
| [`moe.md`](moe.md) | **MoE 体系**。MegaMoE(fp8×fp4)/ FusedMoE + Hash MoE + sqrtsoftplus 路由 + EPLB + swiglu_limit | 高吞吐 MoE 全貌 |
| [`quantization.md`](quantization.md) | **量化体系**。FP8/MXFP4/NVFP4 + `expert_dtype` 派发 + KV cache 三种格式 + e8m0 scale 处理 | 量化路径与权重点映射 |
| [`mtp.md`](mtp.md) | **MTP 投机解码**。draft 结构、`e_proj`/`h_proj`、`hc_head` 投影、target residual stash | 投机解码链路 |

## 学习路径(15 单元细粒度)

> 想系统学 forward?进入 [`learning/`](learning/README.md) — 把 forward 拆成 **15 个细粒度单元**(L1 入口范式 → L15 MegaMoE),每单元带「🎯 学习目标 + 📖 讲解 + 代码锚点 + 🔑 小结 + ✅ 自测(答案折叠)」,按数据流排序、带依赖图与三条推荐路线。
> 与上层特性文档**互补**:分析文档讲「**是什么**」,学习单元讲「**怎么一步步跑**」。

## 推荐阅读路径

1. **建立全局**:读 `overview.md`(一句话心智 + 特性矩阵)
2. **定位代码**:读 `architecture.md`(类层次 + 前向数据流,带行号锚点)
3. **逐特性深入**(按兴趣顺序):
   - `mhc.md` — V4 最独特的结构创新(多流隐式扩容)
   - `mla.md` — MLA 机制原理(前置:latent 压缩 + 矩阵吸收 + RoPE 解耦)
   - `sparse_mla_indexer.md` — 1M 长上下文的实现核心
   - `moe.md` — 高吞吐 MoE 全家桶
   - `quantization.md` — 全链路量化与权重点映射
   - `mtp.md` — 投机解码

## 核心结论速查

**DeepSeek-V4 = 五大特性叠加**:

| 特性 | 一句话 | 关键代码 |
|---|---|---|
| **MHC** | 1 个 token 的计算承载 `hc_mult` 个 hidden stream,层间用 Sinkhorn 混合收敛 | `nvidia/model.py:1018`(repeat)、`1048`(hc_head) |
| **Sparse MLA** | MLA latent 压缩 + Compressor 写压缩 KV + Lightning Indexer 选 topk token 做稀疏注意力 | `attention.py:98`、`compressor.py:177`、`attention.py:661` |
| **MoE** | Hash 路由(浅层)+ sqrtsoftplus 路由(深层)+ MegaMoE(fp8×fp4,SM100)/ FusedMoE + EPLB | `nvidia/model.py:478` |
| **量化** | Linear/Attn FP8 block;专家 MXFP4 / FP8 / NVFP4;KV cache `fp8_ds_mla`(UE8M0) | `quant_config.py:29` |
| **MTP** | 投机解码 draft,分离 `e_proj`/`h_proj`,`hc_head` 超压缩词表投影 | `nvidia/mtp.py:260` |

**心智模型**:
- V4 是为 **Blackwell SM100 + 超长上下文(1M)+ 超大规模 MoE** 量身设计的架构。
- 相比 V3:① 引入 MHC 多流扩容(表达力 ↑);② 用 Lightning Indexer 把全局注意力稀疏化(长序列成本 ↓);③ Hash MoE + MegaMoE 提升浅层路由与专家计算吞吐;④ 全链路 FP8/MXFP4 量化。
- 工程上把**多流 CUDA stream overlap**、**融合 kernel(tilelang/triton/cutedsl)**、**breakable cudagraph** 压榨到极致。

---

> 注:本目录是「分析沉淀」,非「在线走读」。如需逐行走读某条链路(类似 `vllm-gr_analysis/BeamSearch/READING_PROGRESS.md` 的进度追踪),可在此基础上追加进度文件。
