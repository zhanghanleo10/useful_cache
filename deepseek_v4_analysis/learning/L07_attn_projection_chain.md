# L7 · MLA 投影链

> 单元④注意力。L6 讲了 MLA 的抽象概念(latent/吸收/解耦);本篇看代码里 **4 个投影矩阵**怎么落地这些概念。

## 🎯 学习目标

学完能回答:
1. `fused_wqa_wkv` / `wq_b` / `wo_a` / `wo_b` 各对应 MLA 的哪个概念?
2. 为什么 `WQ_A` 和 `WKV` 要融合成一个矩阵?为什么它 `disable_tp`?
3. `wo_a` 为什么用 `bmm`?`o_groups` 是什么?
4. 「吸收形式」在代码里怎么体现?

## 📖 讲解

### 投影链全景(`attention.py:194-230, 318-364`)

```
hidden (N, hidden)
   │ fused_wqa_wkv (WQ_A ‖ WKV 融合, disable_tp)
   ↓ split
qr (N, q_lora_rank)        kv (N, head_dim)         ← kv 就是 c_KV latent!
   │ fused_q_kv_rmsnorm     │ (KV 侧在 L9 写进 cache)
   ↓ wq_b (吸收 W_UKᵀ)
q (N, n_heads, head_dim)
   │ ... forward_mqa(L11) ...
   ↓
o (N, n_heads, head_dim)
   │ wo_a (吸收 W_UV, bmm by o_groups) → wo_b
   ↓
hidden (N, hidden)
```

### 4 个投影矩阵详解

| 模块 | 类型 / 形状 | MLA 角色 | 工程要点 | 代码 |
|------|------------|---------|---------|------|
| `fused_wqa_wkv` | MergedColumn `[q_lora_rank, head_dim]` | `WQ_A`(query 下投影)+ `WKV`(KV latent 生成)融合 | **`disable_tp=True`**(replicated) | `:194` |
| `wq_b` | Column `q_lora_rank → n_heads·head_dim` | query 上投影(**已吸收 `W_UKᵀ`**) | 标准 TP(按 head 切) | `:203` |
| `wo_a` | Column,`is_bmm=True` | output 下投影(**已吸收 `W_UV`**) | 按 `o_groups` 分组 bmm | `:213-222` |
| `wo_b` | Row `n_groups·o_lora_rank → hidden` | output 上投影 | 标准 RowParallel | `:223` |

### 关键设计点

**① 为什么 `WQ_A`+`WKV` 融合?** 两者输入都是同一个 `hidden`,输出拼接成 `[qr, kv]`。融合成一个 MergedColumnParallelLinear,一次 GEMM 同时算出 query latent 和 KV latent —— 省 launch + 复用 hidden 的读取。

**② 为什么 `fused_wqa_wkv` 用 `disable_tp`?** query latent(`q_lora_rank`)和 KV latent(`head_dim`)是**压缩后的共享表示**,不应该按 TP 切(切了就不是完整 latent)。真正的 TP 切分发生在后面的 `wq_b`(把 query 按 head 切到各 TP rank)。所以 WQ_A/WKV 在所有 TP rank 上 replicated。

**③ `wo_a` 的 bmm 和 `o_groups`:** output 下投影把 `n_heads·head_dim` 压到 `n_groups·o_lora_rank`。它按 `o_groups` 分组(`is_bmm=True`,`bmm_batch_size=n_local_groups`,`:221-222`),即每组的若干 head 共用一个下投影 —— 这是 GQA 风格的分组,平衡表达力和参数量。

**④ 吸收形式在代码里怎么体现:** 看 `forward`(`:318`)—— 它传给 `forward_mqa` 的 K 是 **`kv`(latent)**,不是多头 K(`:351,505`)。这正是吸收后的形态:`wq_b` 已经吸收了 `W_UKᵀ`,所以推理时 K 不用上投影成多头,直接用 latent 算分数。

### 配套的 RMSNorm(`:202,212`)

- `q_norm`:对 `qr`(query latent)归一化
- `kv_norm`:对 `kv`(KV latent)归一化

实际调用走融合算子 `fused_q_kv_rmsnorm`(`:340`,见 L8/L9),不单独调。

## 🔑 小结

> 投影链 = **`fused_wqa_wkv`(WQ_A‖WKV 融合,replicated)产 latent → `wq_b`(吸收 W_UKᵀ,TP 切 head)上投影 query → MLA 注意力 → `wo_a`(吸收 W_UV,bmm 分组)+ `wo_b` 回 hidden**。融合省 GEMM,disable_tp 保 latent 完整,bmm 分组省参数,吸收让 K 存 latent。

## ✅ 自测

<details>
<summary><b>Q1</b>:如果 <code>fused_wqa_wkv</code> 不 disable_tp(让它按 TP 切),会出什么问题?</summary>

`q_lora_rank` / `head_dim` 是**压缩 latent 的维度**,切分后每个 TP rank 拿到的就不是完整的 latent(只是 latent 的一段),语义被破坏 —— latent 是「整个 hidden 压成的摘要」,不能按维切。TP 切分应发生在 latent 展开之后(`wq_b` 按 head 切)。所以 WQ_A/WKV 必须 replicated(disable_tp)。
</details>

<details>
<summary><b>Q2</b>:「吸收」让 K 存 latent,那 attention 计算时 q 和 latent 的维度怎么对得上?</summary>

靠 `wq_b` 吸收了 `W_UKᵀ`。标准 MLA 是 `q·(W_UK·c_KV)ᵀ`;吸收后等价于 `(q·W_UKᵀ)·c_KVᵀ`。`wq_b` 的输出 q 已经是「乘过 `W_UKᵀ` 的 q」,维度被调整成和 latent `c_KV`(head_dim)可内积的形状。所以代码里 q 是 `(n_heads, head_dim)`,latent 也是 `head_dim`,直接 `q·kvᵀ` 即可,不需要显式多头 K。
</details>

**下一步**:L7 看到 `fused_wqa_wkv` 等 GEMM;L8 讲这些 GEMM 怎么用多 CUDA stream 并行算。
