# L10 · Compressor(KV 压缩写入)

> 单元④注意力。L9 把 KV 写进 **SWA cache**(局部滑窗全量);本篇讲 **Compressor** 怎么把 KV latent 压缩 + 量化写进 **compressed cache**(供全局稀疏注意力用)。

## 🎯 学习目标

学完能回答:
1. SWA cache 和 compressed cache 各存什么?为什么需要两个?
2. Compressor 的融合流水是什么(compress→norm→rope→quant→store)?
3. C4A 和 C128A 两种压缩比有什么区别?
4. `ape`(absolute position embedding)和 `state_cache` 是什么?

## 📖 讲解

### 为什么需要两个 KV cache

| cache | 内容 | 用途 |
|-------|------|------|
| **SWA cache**(L9 写) | 局部滑窗的**全量** KV(每个 token 一份) | 近期 token 的**精确**注意力 |
| **compressed cache**(Compressor 写) | 历史 KV 的**压缩 + 量化**表示 | 全局**稀疏**注意力(只取 indexer 选的 topk) |

> 长序列里大部分历史 token 不需要逐个精确算 —— 压缩成低维表示,稀疏取用即可。这就是 Compressor 存在的意义。

### 结构(`compressor.py:177-245`)

```python
class DeepseekCompressor(nn.Module):
    self.ape              # (compress_ratio, coff·head_dim) 绝对位置嵌入
    self.fused_wkv_wgate  # MergedColumn(replicated) → 产 kv + score 两路
    self.norm             # RMSNorm(head_dim)
    self.state_cache      # CompressorStateCache: 存 kv_state + score_state
```

`ape`:absolute position embedding,加在 score 上注入位置信息(`compressor.py:220`)。
`state_cache`:存压缩过程中的中间状态(kv_state + score_state),和 KV page **共享物理 tensor**。

### forward 融合流水(`compressor.py:274-399`)

```
kv_score (N, 2·coff·head_dim)
   │ split → kv, score
   ↓ save_partial_states(kv, score, ape, ...)        # 存 kv/score(带 APE)进 state_cache
   │
   ↓ compress_norm_rope_store(...)                     # ★ 融合:compress → RMSNorm → RoPE → 量化 → 写 KV cache
   │
   → 写进 compressed k_cache(给 forward_mqa 的 extra_k_cache 用)
```

**融合算子** `compress_norm_rope_store` 一次做完:压缩 → 归一化 → RoPE → FP8/MXFP4 量化 → paged cache 写入。

### C4A vs C128A(`compressor.py:140-155`)

| 模式 | compress_ratio | block_size | 滑窗 | 用途 |
|------|:---:|:---:|:---:|------|
| **C4A** | 4 | 4 | 8 | 局部 SWA + **全局稀疏**(配 Indexer) |
| **C128A** | 128 | 8 | 16 | 1/128 压缩,**近似全局** |

逐层 `compress_ratios[layer_id]` 决定模式。`coff = 1 + (compress_ratio==4)`(`:217`)。

### kernel 选择(`compressor.py:356-373`)

| 条件 | kernel |
|------|--------|
| CUDA 且 head_dim=512 | `compress_norm_rope_store_cutedsl`(cutedsl,支持 full_kv/full_fp8 标志) |
| head_dim=128(indexer)或 AMD/XPU | `compress_norm_rope_store_triton`(triton,签名不同) |

> PDL(Programmatic Dependent Launch)被关闭(`:306-310`):这些 kernel 依赖前序 GEMM/state_cache 输出但不发 PDL grid dependency,`launch_pdl=True` 会触发 RAW race。

## 🔑 小结

> Compressor = **`fused_wkv_wgate` 产 kv+score → `save_partial_states` 存带 APE 的中间状态 → `compress_norm_rope_store` 融合做 compress+norm+rope+quant+写 compressed cache**。产出供 forward_mqa 的 `extra_k_cache`(全局稀疏);C4A/C128A 控制压缩比。

## ✅ 自测

<details>
<summary><b>Q1</b>:L9 已经把 KV 写进 SWA cache 了,为什么还要 Compressor 再写一份进 compressed cache?</summary>

两个 cache 服务不同注意力路径:SWA cache 存**近期 token 的全量精确 KV**,供 forward_mqa 的 `k_cache`(滑窗全算);compressed cache 存**历史 token 的压缩量化 KV**,供 `extra_k_cache`(只取 indexer 选的 topk 稀疏算)。长序列里若所有历史 token 都存全量,显存爆;压缩后稀疏取用,O(topk) 覆盖全局。
</details>

<details>
<summary><b>Q2</b>:C4A 和 C128A 都压缩历史 KV,为什么压缩比差这么多(4 vs 128)?</summary>

设计目标不同:C4A 压缩比小(4),保留较多信息,配 **Lightning Indexer** 精确选 topk,做「稀疏但精确」的全局注意力;C128A 压缩比大(128),信息损失多但每条很便宜,做「近似全局」(把整个序列压到 1/128 后近似全算)。逐层按 `compress_ratios` 选,可混用。
</details>

**下一步**:L10 把 compressed KV 准备好;L11 讲 forward_mqa 怎么用 SWA + compressed 两路一起算注意力。
