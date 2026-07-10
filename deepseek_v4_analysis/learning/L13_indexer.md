# L13 · Lightning Indexer(MQA 选 topk)

> 单元④注意力(稀疏来源)。L11 的稀疏全局路需要 `topk_indices`(选哪些历史 token);本篇讲 **Lightning Indexer** 怎么选出它们。仅 C4A 层有。

## 🎯 学习目标

学完能回答:
1. Indexer 解决什么问题?为什么只有 C4A 有?
2. 它怎么用 MQA 选 topk token?(DeepGEMM 的角色)
3. `topk_indices_buffer` 是什么?为什么全模型共享一个?
4. Indexer 和 Compressor 怎么协作?

## 📖 讲解

### 职责定位

长序列里,对每个 query 做全局精确注意力太贵。Indexer 先用**廉价的 MQA**(单 KV head)算每个 query 对历史 compressed KV 的相关性分数,选出 **top-k** 个最相关 token,主 MLA(forward_mqa)只对这些选中的 token 做精确稀疏注意力。

```
query ──┐
        ├─→ Indexer(MQA 粗筛)─→ topk_indices ──→ forward_mqa 的 extra_indices
history ┘  (compressed KV)                        (主 MLA 只算这 topk)
```

> 为什么只有 C4A:C128A 是 1/128 超压缩近似全算,不需要选 topk;只有 C4A(compress_ratio=4)走「稀疏精确」路线才需要 Indexer。`attention.py:244`:`if self.compress_ratio == 4: self.indexer = ...`。

### 结构(`attention.py:661-757`)

```python
class DeepseekV4Indexer(nn.Module):
    self.wq_b          # ReplicatedLinear: q_lora_rank → head_dim·n_head(64 head × 128 dim)
    self.weights_proj  # ReplicatedLinear: hidden → n_head  (每 token 的注意力权重)
    self.k_cache       # DeepseekV4IndexerCache: head_dim=132(128 fp8 + 4 fp32 scale)
    self.compressor    # DeepseekCompressor(head_dim=128):写 indexer 自己的 KV cache
    self.indexer_op    # SparseAttnIndexer: 选 topk 的算子
```

- `index_n_heads=64`,`index_head_dim=128`(比主 MLA 的 512 小很多 → 廉价)
- `topk_tokens = config.index_topk`

### forward 流程(`attention.py:766-800`)

```
forward(hidden_states, qr, compressed_kv_score, indexer_weights, positions, rotary_emb)
   │
   ├─ 并行两路(maybe_execute_in_parallel):
   │   ① wq_b_and_q_quant:
   │        q = wq_b(qr).view(-1, 64, 128)
   │        q = fused_indexer_q_rope_quant(positions, q, cos_sin, weights, scale, use_fp4)
   │          # 融合:RoPE + 量化(FP8/MXFP4)+ 应用 weights_proj 权重
   │   ② compressor(compressed_kv_score, positions, rotary_emb)
   │        # 写 indexer 自己的 KV cache(供 indexer_op 读)
   │
   └─ indexer_op(hidden_states, q_quant, k, weights)   # ★ DeepGEMM MQA 选 topk
        → 写 topk_indices_buffer
```

### topk 选择算子(`sparse_attn_indexer.py`)

`SparseAttnIndexer`(`attention.py:746`)用 DeepGEMM 的:

- `fp8_fp4_mqa_logits`:算每 query 对历史 compressed KV 的 MQA 注意力 logit
- `fp8_fp4_paged_mqa_logits`:paged 版本

然后选 topk logit 对应的 token index,写入 `topk_indices_buffer`。

> 关键:Indexer 自己也是 MLA 风格(单 KV latent + MQA),但 head 小(64×128),是「**廉价粗筛**」。它选出的 topk 才是主 MLA 精确算的对象。

### `topk_indices_buffer`(全模型共享)

`DeepseekV4Model.__init__` 预分配(`model.py:917`):

```python
self.topk_indices_buffer = torch.empty(max_num_batched_tokens, config.index_topk, dtype=torch.int32)
```

> 所有 C4A 层**共享同一个 buffer**(同一 step 内各层的 topk 选择互不干扰,因为逐层串行 forward)。地址固定 → cudagraph 友好。forward_mqa(`flashmla.py:166-174`)从这里读 topk。

## 🔑 小结

> Lightning Indexer(仅 C4A)= **用廉价 MQA(64 head × 128 dim)对历史 compressed KV 算相关性 logit → DeepGEMM 选 topk → 写 `topk_indices_buffer`(全模型共享)**。产出的 topk 喂给 forward_mqa 的 `extra_indices`,主 MLA 只对这些 token 精确算。是「全局稀疏」的来源。

## ✅ 自测

<details>
<summary><b>Q1</b>:Indexer 为什么用 MQA(单 KV head)而不是和主 MLA 一样的多头?</summary>

Indexer 是**粗筛**(选 topk 候选),不需要主 MLA 那么高的精度。用单 KV head + 小 head_dim(64×128 vs 主 MLA 的 512)大幅降低计算成本 —— 它要为每个 query 扫一遍整个历史 KV 算 logit,这步本身就是 O(seq_len),必须廉价。粗筛完,精确注意力只对 topk 做,这才是省的关键。
</details>

<details>
<summary><b>Q2</b>:为什么所有 C4A 层共享一个 <code>topk_indices_buffer</code> 而不各层各一份?</summary>

同一 step 内各层是**串行** forward 的(一个 layer 算完才下一个),所以同一时刻只有一个层在写 buffer,不会冲突。共享一份省显存(否则几十层各一份 topk 数组)。且 buffer 地址固定,cudagraph capture/replay 时指针不变,正确性有保证。
</details>

**下一步**:L13 完成整条注意力链。L14 进入 MoE 子链 —— 路由怎么选专家。
