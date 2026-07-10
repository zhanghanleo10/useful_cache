# DeepSeek-V4 forward 学习路径

> 把 DSV4 的 forward 拆成 **15 个细粒度单元**,每个独立可学。基于 `vllm/vllm/models/deepseek_v4/` 实际代码。
> 与上层 [`../`](../) 分析文档**互补**:分析文档按「特性」组织(是什么),本路径按「forward 数据流」组织(怎么一步步跑)。

## 每个单元的统一格式

```
🎯 学习目标      学完能回答什么
📖 讲解 + 代码锚点  file:line 可点击跳转
🔑 一句话小结
✅ 自测问题       1-2 个,<details> 折叠答案
```

## 单元清单(15 个,按 forward 数据流)

| # | 文件 | 主题 | 阶段 |
|---|------|------|------|
| L1 | [`L01_entry_pattern.md`](L01_entry_pattern.md) | vLLM 入口范式(forward 返回 hidden / compute_logits 分离 / PP 接口) | ① 骨架 |
| L2 | [`L02_model_forward.md`](L02_model_forward.md) | Model.forward 主干(embed→repeat→layers→hc_head→norm) | ① 骨架 |
| L3 | [`L03_mhc_math.md`](L03_mhc_math.md) | MHC 思想与数学(torch.py 三段切分 + Sinkhorn) | ② MHC |
| L4 | [`L04_mhc_state_machine.md`](L04_mhc_state_machine.md) | MHC 状态机(跨层 mhc_pre/fused_post_pre/post 张量流) | ② MHC |
| L5 | [`L05_decoder_layer.md`](L05_decoder_layer.md) | DecoderLayer 结构(MHC 包夹 attn/ffn) | ③ 单层 |
| L6 | [`L06_mla_principle.md`](L06_mla_principle.md) | MLA 原理(latent 压缩 + 吸收 + RoPE 解耦) | ④ 注意力 |
| L7 | [`L07_attn_projection_chain.md`](L07_attn_projection_chain.md) | attn 投影链(fused_wqa_wkv/wq_b/wo_a/wo_b) | ④ 注意力 |
| L8 | [`L08_attn_gemm_overlap.md`](L08_attn_gemm_overlap.md) | attn_gemm_parallel_execute(3 路 aux stream overlap) | ④ 注意力 |
| L9 | [`L09_fused_qnorm_rope_kv_insert.md`](L09_fused_qnorm_rope_kv_insert.md) | _fused_qnorm_rope_kv_insert(三 dtype 分支) | ④ 注意力 |
| L10 | [`L10_compressor.md`](L10_compressor.md) | Compressor(compress→norm→rope→quant→store) | ④ 注意力 |
| L11 | [`L11_forward_mqa.md`](L11_forward_mqa.md) | forward_mqa(SWA + extra_k_cache 两路合并) | ④ 注意力 |
| L12 | [`L12_o_proj.md`](L12_o_proj.md) | _o_proj(inverse-RoPE + wo_a/wo_b FP8 einsum) | ④ 注意力 |
| L13 | [`L13_indexer.md`](L13_indexer.md) | Lightning Indexer(MQA 选 topk) | ④ 注意力 |
| L14 | [`L14_moe_routing.md`](L14_moe_routing.md) | MoE 路由(Hash / sqrtsoftplus / noaux_tc) | ⑤ MoE |
| L15 | [`L15_mega_moe.md`](L15_mega_moe.md) | MegaMoE 专家计算(fp8×fp4 DeepGEMM + EPLB) | ⑤ MoE |

## 依赖图

```
骨干:  L1 ──► L2 ──► L5 ──────────────────────────────────────► (收尾:hc_head,在 L2 内)
                 │
MHC:             └──► L3 ──► L4   (理解 L5 的 MHC 包夹需先看)

注意力链:                 L6 ──► L7 ──► L8 ──► L9 ──► L10 ──► L11 ──► L12
                                                                    ▲
                              L13(填 topk_indices_buffer,喂给 L11)──┘

MoE:                                                                   L14 ──► L15
```

- **L1→L2** 是必经主干。
- **L3→L4(MHC)** 可在 L5 前独立学;L5 之后回头也行。
- **L6→L12(注意力)** 是最长一条链,严格顺序最佳。
- **L13(Indexer)** 与 L11 配对(它产出 L11 消费的 topk),可紧跟 L11。
- **L14→L15(MoE)** 完全独立,随时可学。

## 推荐学习路径

| 路线 | 适合 | 顺序 |
|------|------|------|
| **A. 先全景再回填**(快) | 想快速建立全局认知 | L1→L2→L5→L6→L11→L14,再按需回填细节 |
| **B. 严格顺序**(全) | 系统学习、不跳步 | L1→L2→…→L15 |
| **C. 按兴趣跳读** | 已有基础、查特定点 | 直接跳目标单元,按依赖图补前置 |

## 前置知识

- Transformer 基础:self-attention、MLP/SwiGLU、RMSNorm、RoPE
- vLLM 基本概念:KV cache、paged attention、Pipeline/Tensor Parallel、cudagraph capture
- (可选)DeepSeek-V2/V3 的 MLA 背景(本路径 L6 会从零讲)

## 与上层分析文档的对应

| 学习单元 | 配套分析文档(深入背景) |
|---------|------------------------|
| L3-L4 | [`../mhc.md`](../mhc.md) |
| L6-L13 | [`../mla.md`](../mla.md)(原理)+ [`../sparse_mla_indexer.md`](../sparse_mla_indexer.md)(实现) |
| L14-L15 | [`../moe.md`](../moe.md) |
| 全局定位 | [`../overview.md`](../overview.md)、[`../architecture.md`](../architecture.md) |

> 学习单元是「**怎么跑**」的逐步走读;分析文档是「**是什么/为什么**」的特性总结。两者交叉阅读效果最好。
