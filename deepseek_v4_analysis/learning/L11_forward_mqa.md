# L11 · forward_mqa(SWA + 稀疏全局两路合并)

> 单元④注意力(计算核心)。这是 MLA **真正算注意力**的地方。FlashMLA kernel 一次调用同时算两路:SWA 局部 + compressed 稀疏全局。

## 🎯 学习目标

学完能回答:
1. forward_mqa 同时算哪两路注意力?各用什么 K cache?
2. `flash_mla_with_kvcache` 的 `k_cache` 和 `extra_k_cache` 分别是什么?
3. topk_indices 从哪来(C4A vs C128A 不同)?
4. decode 和 prefill 为什么走不同路径?

## 📖 讲解

### 两路注意力(核心概念)

```
forward_mqa(q, kv, positions, output)
       │
       │  q (n_heads, head_dim) 已吸收 W_UKᵀ;  kv 是 latent(吸收形式,见 L7)
       │
   ┌───┴────────────────────────────────────────────┐
   │  FlashMLA kernel 一次算两路并合并(LSE):        │
   │                                                 │
   │  路 A (SWA 局部):  k_cache = swa_cache          │  ← L9 写的局部全量 KV
   │                     indices  = swa_indices(滑窗)│     每个最近 window 个 token 全算
   │                                                 │
   │  路 B (稀疏全局): extra_k_cache = compressed    │  ← L10 写的压缩 KV
   │                     extra_indices = topk_indices │     indexer 选的 topk token
   └─────────────────────────────────────────────────┘
       ↓
   output (写进预分配 buffer)
```

> 「局部精确 + 全局稀疏」混合:近期 token 用滑窗全量算保精度,远期 token 用 indexer 选 topk 稀疏算控成本。两路 output 在 kernel 内按 LSE(log-sum-exp)合并。

### decode 路径(`flashmla.py:145-235`,`_forward_decode`)

```python
flash_mla_with_kvcache(
    q=q,                                    # pre-padded 到 padded_heads
    k_cache=swa_cache,                      # ★ 路 A:SWA 滑窗
    head_dim_v=512, is_fp8_kvcache=True,
    indices=swa_indices, topk_length=swa_lens,
    attn_sink=self.attn_sink,               # attention sink(配 padding head)
    extra_k_cache=kv_cache if not swa_only else None,   # ★ 路 B:compressed cache
    extra_indices_in_kvcache=topk_indices,                # indexer 选的 topk
    extra_topk_length=topk_lens,
    out=output,
)
```

**topk_indices 来源**(`:164-178`):
- **C4A**:从 indexer 填的 `topk_indices_buffer` 算 global slot ids(`compute_global_topk_indices_and_lens`)
- **C128A**:用 metadata 预算的 `c128a_global_decode_topk_indices`(见 L4 提到的 C128A topk kernel)

### prefill 路径(`flashmla.py:237-355`,`_forward_prefill`)

Prefill 是长 query,不走 paged `extra_k_cache`,而是**先 gather 到连续 workspace 再算**:

```
分 chunk(PREFILL_CHUNK_SIZE=4):
  for each chunk:
    dequantize_and_gather_k_cache(compressed_k_cache)   # 压缩 KV → 连续 bf16
    dequantize_and_gather_k_cache(swa_k_cache)          # SWA KV → 连续
    combine_topk_swa_indices(...)                        # 合并 topk + SWA 索引
    flash_mla_sparse_fwd(q, kv, indices=combined, ...)  # 算 sparse MLA
```

> prefill 用连续 workspace:query 长,paged gather + 连续 kernel 比直接 paged attention 高效。

### tile scheduler 共享(`sparse_swa.py:28-33,195-200`)

三种 layer type(`swaonly`/`c4a`/`c128a`)各一个 FlashMLA tile-scheduler plan,**同 type 的 ~60 层共享一个 plan**(首个该 type 层触发 planner,后续复用)。前提:同 type 层的 batch/s_q/h_q/page_block_size/topk 都相同。

## 🔑 小结

> forward_mqa = **FlashMLA 一次算两路**:SWA(`k_cache`,滑窗全量局部)+ compressed(`extra_k_cache` + `extra_indices`,稀疏全局 topk),LSE 合并写 output。decode 走 paged 双 cache;prefill 走 gather + 连续 workspace。topk_indices:C4A 来自 indexer,C128A 预算。

## ✅ 自测

<details>
<summary><b>Q1</b>:为什么不在主 MLA 外单独算一遍 SWA attention 和一遍稀疏 attention 再相加,而要塞进一个 kernel?</summary>

两路 attention 的 output 合并需要 **LSE(log-sum-exp)合并**,不是简单相加 —— 要把两路的 softmax 分数统一归一,才能正确组合两批不同来源的 attention 概率。塞进一个 kernel,kernel 内部同时拿到两路的 logits 做联合 softmax+合并,避免中间结果落盘和二次归一。分离算则要把两路完整 attention output 都存下来再合并,显存和带宽翻倍。
</details>

<details>
<summary><b>Q2</b>:decode 用 paged extra_k_cache,prefill 却 gather 成连续,为什么 prefill 不也用 paged?</summary>

prefill 的 query 长(成百上千 token),对每个 query 都要做 topk gather,paged 随机访问模式下 gather 开销大且不连续;而 prefill 一次性处理整段序列,gather 到连续 workspace 后,FlashMLA 的 sparse prefill kernel 能用连续内存高效算。decode 是单/少 token query,paged + extra_indices 的稀疏访问开销可控,且省去 gather。两种路径各自优化各自的访问模式。
</details>

**下一步**:L11 算出 attention output `o`;L12 讲 `_o_proj` 怎么把它投影回 hidden(inverse-RoPE + wo_a + wo_b)。
