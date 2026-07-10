# DeepSeek-V4 总览

> 配置:`vllm/vllm/transformers_utils/configs/deepseek_v4.py`
> 主实现(默认平台):`vllm/vllm/models/deepseek_v4/nvidia/model.py`

## 一句话定义

DeepSeek-V4 是一个为 **Blackwell SM100 + 1M 超长上下文 + 超大规模 MoE** 优化的 Transformer-MoE 架构,叠加五项核心创新:**MHC 多流隐式扩容**、**Sparse MLA(Compressor + Lightning Indexer)**、**Hash/sqrtsoftplus 路由 + MegaMoE**、**全链路 FP8/MXFP4 量化**、**MTP 投机解码**。

## 配置要点(`DeepseekV4Config`)

```python
# configs/deepseek_v4.py:13
max_position_embeddings = 1048576   # 1M 上下文
rope_theta = 10000.0
# 其余结构参数(hidden_size / num_hidden_layers / n_routed_experts /
# hc_mult / compress_ratios / q_lora_rank / o_lora_rank / head_dim /
# index_topk / num_hash_layers / expert_dtype / ...)从 hf_config 读取
```

## 平台隔离

入口 `vllm/vllm/models/deepseek_v4/__init__.py:17-25` 按 `current_platform` 分发:

| 平台 | 模型实现 | 注意力后端 |
|------|---------|-----------|
| NVIDIA(默认) | `nvidia/model.py` → `DeepseekV4FlashMLAAttention` / `DeepseekV4FlashInferMLAAttention` | `FLASHMLA_SPARSE_DSV4` / `FLASHINFER_MLA_SPARSE_DSV4` |
| AMD ROCm | `amd/model.py` → `DeepseekV4ROCMAiterMLAAttention` | ROCm Aiter |
| Intel XPU | `xpu/model.py` | XPU sparse |

公共逻辑抽到 `common/`(`rope.py`、`ops/`),平台无关。

## 核心特性矩阵

| 特性 | V3 有? | V4 实现 | 代表代码 |
|------|:---:|---|---|
| **MHC 多流扩容** | ✗ | token → `hc_mult` streams,层间 Sinkhorn 混合,`hc_head` 折叠 | `nvidia/model.py:1018,1048` |
| **MLA 潜在压缩** | ✓(基础) | `q_lora_rank`/`o_lora_rank`,`fused_wqa_wkv` + `wq_b` + `wo_a/wo_b`,`head_dim=512` | `attention.py:194-230` |
| **GPT-J RoPE** | 部分 | `is_neox_style=False`(interleaved),disable mscale,deepseek_yarn | `common/rope.py:35` |
| **Lightning Indexer** | ✗ | MQA 选 topk token,主 MLA 只算稀疏子集;C4A 才有 | `attention.py:661`,`sparse_attn_indexer.py` |
| **Compressor** | ✗ | compress→norm→rope→quant→store 融合,带 APE + score state | `compressor.py:177` |
| **C4A / C128A 双模式** | ✗ | 逐层 `compress_ratios`:4(稀疏)或 128(全局近似) | `attention.py:179` |
| **attn_sink** | ✗ | 每 head 一个 attention sink 参数(初值 -inf) | `attention.py:189` |
| **Hash MoE** | ✗ | 前 `num_hash_layers` 层用 `tid2eid` 查表路由 | `nvidia/model.py:530-544` |
| **sqrtsoftplus 路由** | ✗ | `vllm_topk_softplus_sqrt`,配合 `e_score_correction_bias` | `fused_topk_bias_router.py:223` |
| **MegaMoE** | ✗ | SM100 专用 fp8 激活 × fp4 权重(`fp8_fp4_mega_moe`),须 EP | `nvidia/model.py:464` |
| **EPLB** | 部分 | redundant experts + logical→physical 动态重平衡 | `nvidia/model.py:171,365` |
| **swiglu_limit** | ✗ | SiLU + clamp 激活限幅 | `nvidia/model.py:506` |
| **MXFP4 专家** | ✗ | `expert_dtype="fp4"`,ue8m0(e8m0fnu)scales | `quant_config.py:142-152` |
| **`fp8_ds_mla` KV cache** | ✓(V3 MLA) | UE8M0 block-scaled fp8 打包 uint8,576B 对齐,584B/token | `sparse_mla.py:102` |
| **MTP 投机解码** | ✓ | 分离 `e_proj`/`h_proj`(V3 是 fused `eh_proj`),`hc_head` 投影 | `nvidia/mtp.py:89-104` |

## 顶层结构

```
DeepseekV4ForCausalLM           # nvidia/model.py:1251
├── model: DeepseekV4Model      # :888
│   ├── embed_tokens            # VocabParallelEmbedding(首 PP rank)
│   ├── layers × N              # DeepseekV4DecoderLayer (:740)
│   │   ├── attn                # DeepseekV4Attention (Sparse MLA)
│   │   ├── ffn                 # DeepseekV4MoE
│   │   ├── attn_norm / ffn_norm
│   │   └── hc_* mixing 参数     # MHC Sinkhorn 混合
│   ├── norm                    # 末 PP rank
│   ├── hc_head_*               # 多流折叠参数
│   └── _mtp_hidden_buffer      # 给 MTP draft 的 residual stash
├── lm_head
└── logits_processor
```

## 前向数据流(梗概)

```
input_ids
  → embed → repeat 到 hc_mult 个 stream          # model.py:1018
  → for each layer:                              # model.py:1027
       mhc_pre/fused_post_pre (Sinkhorn 混合 attn) # 融合 attn_norm + MHC mix
       → attn (Sparse MLA + Indexer + Compressor)
       mhc_fused_post_pre (Sinkhorn 混合 ffn)     # 融合 ffn_norm + MHC mix
       → ffn (MoE: Hash/sqrtsoftplus 路由 + MegaMoE/FusedMoE)
  → mhc_post (收尾混合)
  → stash pre-hc_head residual 给 MTP            # model.py:1046
  → hc_head_fused (折叠 hc_mult → hidden_size)   # model.py:1048
  → norm → lm_head → logits
```

详细数据流见 [`architecture.md`](architecture.md)。

## 一句话心智模型

> V4 用 **MHC** 在 token 内扩容表达力,用 **Lightning Indexer** 把全局注意力稀疏化以支撑 1M 上下文,用 **Hash MoE + MegaMoE(fp8×fp4)** 提升路由与计算吞吐,并全链路 FP8/MXFP4 量化 —— 一切围绕「**更长上下文、更高吞吐、更省显存**」三个目标,且把多 stream overlap / 融合 kernel / cudagraph break 的工程优化做到极致。

各特性展开见专题文档。
