# L9 · _fused_qnorm_rope_kv_insert(KV 写入路径)

> 单元④注意力。L8 产出 `qr/kv` 等 latent;本篇讲这些 latent 如何做 **norm + RoPE + 量化 + 写 KV cache**,且 Q 侧和 KV 侧融合进一个算子。

## 🎯 学习目标

学完能回答:
1. `_fused_qnorm_rope_kv_insert` 同时做 Q 侧和 KV 侧的哪些事?
2. 为什么按 KV cache dtype 分三条分支?三者的 Q/KV 处理有何不同?
3. padding head 槽是什么?为什么 Q 要 zero-fill padding?
4. 为什么把 norm/rope/quant/cache-write 全融合?

## 📖 讲解

### 算子职责(`attention.py:507-594`)

这个融合算子同时处理 **Q 侧**和 **KV 侧**,产出处理后的 q(喂给 forward_mqa)并把 KV 写进 SWA cache:

| 侧 | 操作 |
|----|------|
| **Q 侧** | per-head RMSNorm + GPT-J RoPE + (必要时)zero-fill padding head 槽 |
| **KV 侧** | GPT-J RoPE + FP8/bf16 量化 + paged cache 写入(slot_mapping 定位) |

> KV 在这里写进 **SWA cache**(`swa_cache_layer.kv_cache`),不是 compressed cache —— SWA 是局部滑窗的全量 KV。Compressed cache 由 Compressor 单独写(L10)。

### 三 dtype 分支(`:541-594`)

按 `swa_kv_cache.dtype` 分支,对应不同 backend 的 KV cache 格式:

| KV cache dtype | backend | Q 侧 | KV 侧 | C op |
|---------------|---------|------|-------|------|
| `uint8`(UE8M0) | FlashMLA / ROCm | per-head RMSNorm + RoPE,**kernel 内部分配并 zero-fill padded q** | RoPE + **UE8M0 FP8 量化** + paged insert | `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`(`:548`) |
| `bfloat16` | FlashInfer | RoPE,**原地改写 q** | RoPE + 写 bf16 cache | `..._full_cache_bf16_insert`(`:567`) |
| `float8_e4m3fn` | FlashInfer(per-tensor fp8) | RoPE,产**单独 `q_fp8`** | RoPE + per-tensor 量化写 fp8 cache | `..._full_cache_fp8_insert`(`:581`) |

### padding head 槽(为什么 Q 要 pad)

回顾 [L7/L11]:FlashMLA 的 FP8 decode kernel **只接受 `h_q ∈ {64, 128}`**(`flashmla.py:58-66` 的 `get_padded_num_q_heads`)。若模型实际 head 数不匹配(如 96),pad 到 128:

- **uint8 分支**:kernel 内部分配 padded q,zero-fill 多余槽位(`:546-547` 注释)
- 配合 `attn_sink=-inf`(`:189`):padding head 在 attention 里贡献 -inf,不影响输出

### 为什么全融合

Q norm → RoPE → KV RoPE → quant → cache write 是 **5 个连续小算子**:
- 中间结果(q_normed、q_rope、kv_rope、kv_quant)都是临时张量,不融合要反复 alloc/read/write
- 融合成 1 个 kernel:省 4 次 launch + 省中间张量显存带宽
- 尤其 quant + cache write 必须融合(否则要先存量化后的全量 KV 再写 page,翻倍带宽)

### GPT-J RoPE 细节(`:331-337` 注释)

- `is_neox_style=False`:**interleaved pairs**(相邻两元素为一对旋转),不是 split-half
- `cos_sin_cache` 布局:`[max_pos, rope_head_dim]`,前半 cos 后半 sin
- 只作用 head_dim 的**最后 `rope_head_dim`(64)维**
- KV 侧用的 position 是 `(positions // compress_ratio) * compress_ratio`(压缩对齐位置)

## 🔑 小结

> `_fused_qnorm_rope_kv_insert` 把 **Q 侧(norm+rope+pad)+ KV 侧(rope+quant+paged insert)** 融成一个 kernel,按 cache dtype 分三条路径(uint8 UE8M0 / bf16 / fp8)。融合省 launch + 省中间张量带宽;Q padding 配合 attn_sink 应对 FlashMLA 的 head 数硬约束。

## ✅ 自测

<details>
<summary><b>Q1</b>:为什么 bf16 分支「原地改写 q」,而 fp8 分支「产单独 q_fp8」?</summary>

bf16 的 q 和 cache 同 dtype,直接在原 q tensor 上做 RoPE 即可,下游 forward_mqa 直接用。fp8 的 cache 是 e4m3,但 q 原本是 bf16/higher precision,不能原地转成 fp8(会丢精度且不可逆),所以要单独分配 `q_fp8` 存量化后的 q,下游用 q_fp8。per-tensor fp8 还需要 scale(`_flashinfer_fp8_q_scale_inv`,`:590`)。
</details>

<details>
<summary><b>Q2</b>:如果 quant 和 cache write 不融合(先量化出全量 KV,再单独写 page),代价是什么?</summary>

① 多一个全量量化 KV 的临时张量(显存);② 多一次「写量化结果 → 再读它写 page」的往返带宽(量化结果要先落盘再被 page-write 读)。融合后量化结果在寄存器/共享内存里直接 scatter 到 page,省掉这个中间落盘。对带宽敏感的 decode,这是关键优化。
</details>

**下一步**:L9 把 KV 写进 SWA cache(局部)。L10 讲 **Compressor** 怎么把 KV latent 压缩 + 量化写进 compressed cache(全局稀疏用)。
