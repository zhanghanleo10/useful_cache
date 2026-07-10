# DeepSeek-V4 代码地图

> 目的:快速定位「某个东西在哪个文件哪一行」。所有锚点基于 `vllm/vllm/models/deepseek_v4/`。

## 目录拓扑

```
vllm/vllm/models/deepseek_v4/
├── __init__.py              # 平台分发入口(L17-25)
├── quant_config.py          # DeepseekV4FP8Config(expert_dtype 派发)
├── attention.py             # DeepseekV4Attention(MLA 主体)+ DeepseekV4Indexer ★
├── compressor.py            # DeepseekCompressor + CompressorStateCache
├── sparse_mla.py            # FlashMLA sparse 后端 + 元数据 + C128A topk kernel
├── common/
│   ├── rope.py              # build_deepseek_v4_rope(GPT-J style)
│   └── ops/                 # 融合算子(triton/cutedsl)
│       ├── fused_qk_rmsnorm.py
│       ├── fused_indexer_q.py
│       ├── fused_compress_quant_cache.py
│       ├── fused_inv_rope_fp8_quant.py
│       ├── fused_mtp_input_rmsnorm.py
│       ├── save_partial_states.py
│       └── cache_utils.py
├── nvidia/                  # 默认平台实现
│   ├── model.py             # DeepseekV4ForCausalLM + Model + DecoderLayer + MoE ★
│   ├── mtp.py               # DeepSeekV4MTP
│   ├── flashmla.py          # DeepseekV4FlashMLAAttention
│   ├── flashinfer_sparse.py # DeepseekV4FlashInferMLAAttention
│   └── ops/                 # prepare_megamoe / o_proj / sparse_attn_compress_cutedsl
├── amd/                     # ROCm:model.py / mtp.py / rocm.py
└── xpu/                     # Intel XPU:model.py / mtp.py / xpu_*.py
```

## 类层次

```
DeepseekV4ForCausalLM(nn.Module, SupportsPP, DeepseekV4MixtureOfExperts)  # nvidia/model.py:1251
├── .model: DeepseekV4Model(nn.Module)                                     # :888
│   ├── .embed_tokens: VocabParallelEmbedding | PPMissingLayer
│   ├── .layers: list[DeepseekV4DecoderLayer]                             # :933
│   ├── .norm: RMSNorm | PPMissingLayer
│   └── ._mtp_hidden_buffer: Tensor | None  # 末 PP rank 才有(:974)
├── .lm_head: ParallelLMHead | PPMissingLayer
└── .logits_processor: LogitsProcessor

DeepseekV4DecoderLayer(nn.Module)                                          # :740
├── .attn: DeepseekV4Attention(平台子类)
├── .ffn: DeepseekV4MoE                                                    # :760
├── .attn_norm / .ffn_norm: RMSNorm
└── .hc_* : MHC 混合参数(attn/ffn 各一组 fn/scale/base,:770-811)

DeepseekV4MoE(nn.Module)                                                   # :478
├── .gate: GateLinear
├── .shared_experts: DeepseekV4MLP | None                                 # :551
└── .experts:
     ├── use_mega_moe → DeepseekV4MegaMoEExperts                          # :140
     └── else         → FusedMoE                                          # :639

DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC)                    # attention.py:98
├── .fused_wqa_wkv: MergedColumnParallelLinear  (WQ_A + WKV)              # :194
├── .q_norm / .kv_norm: RMSNorm
├── .wq_b: ColumnParallelLinear                                            # :203
├── .wo_a: ColumnParallelLinear (bmm, o_groups)                            # :213
├── .wo_b: RowParallelLinear                                               # :223
├── .rotary_emb: RotaryEmbedding (GPT-J)
├── .attn_sink: Parameter(n_local_heads, 初值 -inf)                        # :189
├── .indexer: DeepseekV4Indexer | None   (仅 compress_ratio==4)            # :252
├── .compressor: DeepseekCompressor | None (compress_ratio>1)              # :308
├── .swa_cache_layer: DeepseekV4SWACache                                  # :288
└── .aux_stream_list / .ln_events  (3 个 aux CUDA stream)                  # :266-270

DeepseekV4Indexer(nn.Module)                                              # attention.py:661
├── .wq_b / .weights_proj: ReplicatedLinear
├── .k_cache: DeepseekV4IndexerCache
├── .compressor: DeepseekCompressor
└── .indexer_op: SparseAttnIndexer                                        # :746

DeepseekCompressor(nn.Module)                                             # compressor.py:177
├── .ape: Parameter(compress_ratio, coff*head_dim)
├── .fused_wkv_wgate: MergedColumnParallelLinear
├── .norm: RMSNorm
└── .state_cache: CompressorStateCache                                    # :240
```

## DecoderLayer 前向数据流(`nvidia/model.py:813-885`)

```
forward(x, positions, input_ids, post_mix, res_mix, residual)
│
├─ 若 residual is None(首层):
│     mhc_pre_tilelang(x, hc_attn_*, rms_norm)   # :827
│     → 同时算 attn_norm + MHC 混合 → 输出 residual/post_mix/res_mix/x
│  否则(中间层):
│     mhc_fused_post_pre_tilelang(...)            # :841  融合:上轮 ffn 后混合 + 本轮 attn 前混合 + attn_norm
│
├─ x = self.attn(positions, x, None)             # :861  Sparse MLA
│
├─ mhc_fused_post_pre_tilelang(..., hc_ffn_*)     # :865  ffn_norm + MHC 混合
│
├─ x = self.ffn(x, input_ids)                    # :884  MoE
│
└─ return x, residual, post_mix, res_mix
```

> 关键:**RMSNorm 不单独调用**,而是融合进 `mhc_*` tilelang kernel(同时做归一化 + MHC Sinkhorn 混合)。`attn_norm`/`ffn_norm` 只作为权重容器(`model.py:860,863`)。

## Model 前向数据流(`nvidia/model.py:1006-1057`)

```
forward(input_ids, positions, intermediate_tensors, inputs_embeds)
│
├─ 首 PP rank:
│    hidden = embed(input_ids)
│    hidden = hidden.unsqueeze(-2).repeat(1, hc_mult, 1)   # :1018  展开成 hc_mult 流
│  非首 PP rank:
│    hidden = intermediate_tensors["hidden_states"]         # 形状已含 hc_mult 维
│
├─ for layer in layers[start:end]:                          # :1027
│    hidden, residual, post_mix, res_mix = layer(hidden, positions, input_ids,
│                                               post_mix, res_mix, residual)
│
├─ hidden = mhc_post_tilelang(hidden, residual, post_mix, res_mix)  # :1037 收尾混合
│
├─ 非末 PP rank → return IntermediateTensors({"hidden_states": hidden})  # :1042
│
├─ 末 PP rank:
│    _mtp_hidden_buffer[:T].copy_(hidden.flatten(1))        # :1046  stash 给 MTP
│    hidden = hc_head_fused_kernel_tilelang(hidden, ...)    # :1048  折叠 hc_mult → hidden_size
│    hidden = self.norm(hidden)                             # :1056
│    return hidden
```

> PP 中间张量形状:`(num_tokens, hc_mult, hidden_size)`(`model.py:996`)—— V4 把多流维度带进 pipeline。

## 权重加载映射(`nvidia/model.py:1177-1211`)

`_make_deepseek_v4_weights_mapper(expert_dtype)` 构建 `WeightsMapper`,处理 checkpoint → 模型的命名差异:

| checkpoint 名 | 模型名 | 说明 |
|---|---|---|
| `layers.X.` | `model.layers.X.` | 前缀加 `model.` |
| `embed.` / `norm.` / `hc_head` / `mtp.` | 加 `model.` 前缀 | |
| `head.weight` | `lm_head.weight` | |
| `embed.weight` | `embed_tokens.weight` | |
| `.ffn.gate.bias` | `.ffn.gate.e_score_correction_bias` | noaux_tc 偏置 |
| `.shared_experts.w2` | `.shared_experts.down_proj` | |
| `.experts.{i}.w{123}.scale` | (fp4)`.w{123}_weight_scale` / (fp8)`.weight_scale_inv` | **按 expert_dtype 分支**(`model.py:1183-1193`) |
| `.scale`(非专家) | `.weight_scale_inv` | FP8 block scale |

**stacked 参数**(`model.py:1060-1068`):
- `w1`+`w3` → `gate_up_proj`(MLP / shared experts)
- `attn.wq_a`+`attn.wkv` → `attn.fused_wqa_wkv`
- `compressor.wkv`+`compressor.wgate` → `compressor.fused_wkv_wgate`

> ⚠️ e8m0 scale 特殊处理(`model.py:1106`):checkpoint 里是 `float8_e8m0fnu`,模型参数是 `uint8`,必须 `.view(torch.uint8)` 保留原始指数字节 —— 直接 `copy_()` 会做数值转换(如 2^-7 → 0)破坏数据。MTP 加载同样处理(`nvidia/mtp.py:406`)。

## 并行策略

| 并行 | 适用 | 约束 |
|------|------|------|
| **TP**(张量) | 注意力 `wq_b`/`wo_a`/`wo_b`、MLP、embed | `n_heads % tp_size == 0`(`attention.py:165`) |
| **EP**(专家) | MegaMoE **必须**(`model.py:493`) | `n_physical_experts % ep_size == 0`(`model.py:587`) |
| **PP**(流水) | 跨 stage 切层 | 中间张量 `(T, hc_mult, hidden_size)`,首/末 rank 处理 embed/norm/lm_head |

**MegaMoE 专家所有权**(`model.py:230-280`):逻辑专家 ID → 物理 slot 映射,`weight_loader` 按 `_map_global_expert_id` 把权重写到本 rank 对应的副本 slot(EPLB redundant experts)。

## 入口注册

- 模型注册:`DeepseekV4ForCausalLM`(`__init__.py:24` 导出)
- 量化方法:`DeepseekV4FP8Config`,`get_name() → "deepseek_v4_fp8"`(`quant_config.py:117`)
- 量化方法覆盖:`override_quantization_method`(`quant_config.py:120`)识别 `model_type == "deepseek_v4"` 或 `user_quant == "deepseek_v4_fp8"`
- MTP:`DeepSeekV4MTP`(`__init__.py:25`)
