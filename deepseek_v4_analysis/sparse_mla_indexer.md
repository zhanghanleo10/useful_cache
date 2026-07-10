# Sparse MLA + Lightning Indexer(稀疏潜在注意力)

> 主体:`vllm/vllm/models/deepseek_v4/attention.py`
> 压缩器:`vllm/vllm/models/deepseek_v4/compressor.py`
> 后端/元数据:`vllm/vllm/models/deepseek_v4/sparse_mla.py`
> Indexer 算子:`vllm/vllm/model_executor/layers/sparse_attn_indexer.py`

这是 V4 支撑 **1M 长上下文**的核心:延续 V3 的 MLA(KV latent 压缩),并叠加**两级稀疏注意力**(Compressor 压缩 KV + Lightning Indexer 选 topk token)。

## 总览:三件套

| 组件 | 作用 | 存在条件 | 代码 |
|------|------|---------|------|
| **MLA 主体** | latent 压缩 + 主注意力 | 总在 | `DeepseekV4Attention`(`attention.py:98`) |
| **Compressor** | KV latent → 压缩缓存(融合 quant+store) | `compress_ratio > 1` | `DeepseekCompressor`(`compressor.py:177`,`attention.py:308`) |
| **Lightning Indexer** | MQA 选 topk token,给主 MLA 做稀疏注意力 | `compress_ratio == 4`(C4A) | `DeepseekV4Indexer`(`attention.py:661`,`attention.py:252`) |

另有 **SWA cache**(`DeepseekV4SWACache`,`attention.py:288`):局部滑窗 KV,所有层都有。

## MLA 主体(`DeepseekV4Attention`)

延续 V3 的 **latent 压缩**思想(KV 不存全部维度,存压缩 latent):

| 组件 | 形状/参数 | 作用 | 代码 |
|------|----------|------|------|
| `fused_wqa_wkv` | MergedColumn `[q_lora_rank, head_dim]`,**disable_tp(replicated)** | query 下投影(WQ_A)+ KV latent 生成(WKV),融合成一矩阵 | `attention.py:194` |
| `q_norm` / `kv_norm` | RMSNorm | latent 归一化 | `:202,212` |
| `wq_b` | ColumnParallel `q_lora_rank → n_heads·head_dim` | query 上投影到多头 | `:203` |
| `wo_a` | ColumnParallel,`bmm`(按 `o_groups` 分组) | output 分组下投影 | `:213` |
| `wo_b` | RowParallel `n_groups·o_lora_rank → hidden` | output 上投影回 hidden | `:223` |
| `attn_sink` | `(n_local_heads,)`,初值 `-inf` | 每 head 一个 attention sink | `:189` |
| `rotary_emb` | GPT-J style(`is_neox=False`) | RoPE | `common/rope.py:35` |

**关键维度**:`head_dim = 512` = 448 NoPE + 64 RoPE(`sparse_mla.py:78-79` 注释,`attention.py:170-171`)。RoPE 只作用最后 64 维;`qk_rope_head_dim=64`。mscale 被关闭(`rope.py:27-28`)。

### 前向(`attention.py:318-364`)

```
forward(positions, hidden_states)
├─ o_padded = empty(N, padded_heads, head_dim)        # 预分配输出(:327)
├─ attn_gemm_parallel_execute(hidden)                 # :336  ★ 多流并行算输入 GEMM
│    返回 qr_kv, kv_score, indexer_kv_score, indexer_weights
├─ qr, kv = qr_kv.split([q_lora_rank, head_dim])      # :339
├─ qr, kv = fused_q_kv_rmsnorm(qr, kv, q_norm, kv_norm)  # :340 融合 RMSNorm
├─ attention_impl(...)  @eager_break_during_capture   # :351  cudagraph 在此 break
│    ├─ wq_b + _fused_qnorm_rope_kv_insert(q, kv)     # :452  Q 上投影 + qnorm + rope + KV 写 SWA cache
│    ├─ indexer(...)   (若 C4A)                        # :464  选 topk token
│    ├─ compressor(...)(若 compress_ratio>1)           # :472  写压缩 KV cache
│    └─ forward_mqa(q, kv, positions, out)            # :505  平台子类的稀疏 MLA 前向
├─ o = o_padded[:, :n_local_heads]                     # :361 去 pad
└─ _o_proj(o, positions)                               # :364  inverse-RoPE + wo_a + wo_b
```

> `padded_heads`(`get_padded_num_q_heads`):FlashMLA 要求 Q 头数对齐(如 64/128 倍数),用 `-inf` sink 填充无效槽(`attention.py:186-192`)。

## Compressor(`DeepseekCompressor`)

把 KV latent 写入**压缩缓存**,融合流水:

```
compress → RMSNorm → RoPE → FP8/MXFP4 量化 → KV cache 写入   (compressor.py:331-399)
```

| 部分 | 说明 | 代码 |
|------|------|------|
| `ape` | absolute position embedding,`(compress_ratio, coff·head_dim)`,fp32 | `compressor.py:220` |
| `fused_wkv_wgate` | MergedColumn(replicated),产出 kv + score 两路 | `:229` |
| `state_cache` | `CompressorStateCache`,存 kv_state + score_state,共享 KV page | `:240` |
| `save_partial_states` | 先存 KV/score(带 APE)进 state_cache | `:318` |
| `compress_norm_rope_store` | 再做 compress→norm→rope→quant→写 KV cache | `:375` |

**两档压缩比**(`compressor.py:140-155`):

| 模式 | compress_ratio | block_size | 滑窗 | 用途 |
|------|:---:|:---:|:---:|------|
| **C4A** | 4 | 4 | 8 | 局部 SWA + **全局稀疏**(配 Indexer) |
| **C128A** | 128 | 8 | 16 | 1/128 压缩,**近似全局注意力** |

> `coff = 1 + (compress_ratio==4)`(`compressor.py:217`):C4A 有 overlap 系数。逐层 `compress_ratios[layer_id]` 决定模式(`attention.py:179`)。

**kernel 选择**(`compressor.py:356-373`):head_dim=512 且 CUDA → cutedsl(`compress_norm_rope_store_cutedsl`);head_dim=128(indexer)/AMD/XPU → triton。两条路径签名不同(前者多 `store_full_kv/store_full_fp8/fp8_scale` 参数)。

> ⚠️ PDL(Programmatic Dependent Launch)被关闭(`compressor.py:306-310`):这些 kernel 依赖前序 GEMM/state_cache 输出但不发 PDL grid dependency,`launch_pdl=True` 会触发 RAW race。

## Lightning Indexer(`DeepseekV4Indexer`,仅 C4A)

**目标**:对每个 query,从历史 compressed KV 里选出 `index_topk` 个最相关 token,主 MLA 只对这些 token 做精确注意力 —— 把全局注意力从 O(L) 降到 O(topk)。

| 部分 | 说明 | 代码 |
|------|------|------|
| `wq_b` | ReplicatedLinear `q_lora_rank → head_dim·n_head` | `attention.py:693` |
| `weights_proj` | ReplicatedLinear `hidden → n_head` | `:700` |
| `k_cache` | `DeepseekV4IndexerCache`,`head_dim=132`(128 fp8 + 4 fp32 scale),uint8 | `:727` |
| `compressor` | 自己的 DeepseekCompressor(head_dim=128) | `:735` |
| `indexer_op` | `SparseAttnIndexer`,`skip_k_cache_insert=True` | `:746` |

**维度**:`index_n_heads = 64`,`index_head_dim = 128`,`qk_rope_head_dim = 64`(`attention.py:681-683`)。

**前向**(`attention.py:766-800`):

```
forward(hidden_states, qr, compressed_kv_score, indexer_weights, positions, rotary_emb)
├─ wq_b_and_q_quant():                                  # 并行分支 1
│    q = wq_b(qr).view(-1, n_head, head_dim)
│    q = fused_indexer_q_rope_quant(positions, q, cos_sin, weights, scale, use_fp4)
│       # 融合:RoPE + 量化(FP8/MXFP4)+ 应用 weights_proj 的注意力权重
├─ compressor(compressed_kv_score, positions, rotary_emb)  # 并行分支 2:写 indexer KV cache
└─ indexer_op(hidden_states, q_quant, k, weights)          # :800 DeepGEMM MQA 选 topk
     → 写 topk_indices_buffer(全模型共享,attention.py:917 预分配)
```

**topk 选择算子**(`sparse_attn_indexer.py`):用 DeepGEMM 的 `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` 算每 query 对历史 KV 的 MQA logit,再选 topk。`RADIX_TOPK_WORKSPACE_SIZE = 1MB`,`MXFP4_BLOCK_SIZE = 32`。

## C4A vs C128A 对比

| 维度 | C4A(`compress_ratio=4`) | C128A(`compress_ratio=128`) |
|------|--------------------------|------------------------------|
| Indexer | ✓(Lightning Indexer 选 topk) | ✗ |
| 全局注意力 | **稀疏**(只算 indexer 选中的 topk) | **近似全局**(1/128 压缩后全算) |
| 元数据 | 标准 MLA slot_mapping | 额外 C128A topk metadata(`sparse_mla.py:167-296`,triton kernel 算 global slot ids / decode lens / prefill local) |
| 适用 | 长序列精确性优先 | 极长序列、近似可接受 |
| 压缩 KV cache | C4 layout | C128 layout |

## KV Cache 布局(三种格式)

由 `_resolve_dsv4_kv_cache_dtype`(`attention.py:64-95`)按 backend 的 `use_flashmla_fp8_layout` ClassVar + `--kv-cache-dtype` 决定:

| 格式 | layout | dtype | 对齐 | 每 token | 后端 |
|------|--------|-------|------|---------|------|
| **`fp8_ds_mla`** | FlashMLA/ROCm paged | **UE8M0 block-scaled fp8 打包成 uint8** | **576B** | **584B**(448 NoPE + 128 RoPE + 8 fp8 scale,`sparse_mla.py:102`) | FlashMLA / ROCm Aiter |
| bf16 | FlashInfer paged | `bfloat16` | 无 | head_dim·2 | FlashInfer |
| per-tensor fp8 | FlashInfer paged | `float8_e4m3fn` | 无 | head_dim + scale | FlashInfer |

> `MLAAttentionSpec.alignment=576`(FlashMLA,`attention.py:615`):indexer/compressor page 也对齐 576B,以便与 FlashMLA 的 UE8M0 page 打包共享(`attention.py:652`,`compressor.py:168`)。

**Indexer cache**(`attention.py:723-734`):`head_dim=132`(128 + scale),uint8;FP8 与 MXFP4 占同样内存,后者只用前半。compressor state cache 与 KV block 共享物理 tensor(`compressor.py:142-149`),故 block_size 受 KV page 约束(C4=4,C128=8)。

## 多流并行 overlap(`attention.py:366-478`)

输入 GEMM 阶段(`attn_gemm_parallel_execute`)用 **3 个 aux CUDA stream** + default stream 并行:

| stream | 任务 | 条件 |
|--------|------|------|
| default | `fused_wqa_wkv`(最重) | 总在 |
| aux[0] | `compressor.fused_wkv_wgate`(kv_score) | compressor 存在 |
| aux[1] | `indexer.weights_proj` | indexer 存在 |
| aux[2] | `indexer.compressor.fused_wkv_wgate` | indexer 存在 |

attention_impl 阶段再做 **3-way overlap**(TRT-LLM PR #14142 Level 1):default 跑 wq_b+kv_insert,aux[0] 跑 indexer,aux[1] 跑 compressor(`attention.py:461-478`)。受 `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` 控制(`attention.py:421`),小 batch 关闭以省 stream 开销;ROCm 无 aux stream,退化为串行。

> `@eager_break_during_capture`(`attention.py:426`):attention op 在 cudagraph capture 时 eager break —— 元数据相关部分无法被捕获。

## 一句话心智模型

> Sparse MLA = **MLA 把 KV 压成 latent**(省显存)+ **Compressor 把 latent 再量化写进压缩 page**(FP8/MXFP4,省带宽)+ **Lightning Indexer 用 MQA 先粗筛 topk token**(把全局注意力稀疏化,长序列成本 ↓)+ **主 MLA 只算被选中的稀疏子集**(精确)。C4A 是「稀疏精确」,C128A 是「超压缩近似」,逐层可混用。
