# L12 · _o_proj(inverse-RoPE + output 投影)

> 单元④注意力(收尾)。L11 产出 attention output `o`;本篇讲 `_o_proj` 怎么把它**逆旋转 + 投影回 hidden_size**。

## 🎯 学习目标

学完能回答:
1. `_o_proj` 做哪三件事?为什么顺序是这样?
2. 为什么需要 inverse-RoPE?attention output 的 RoPE 部分怎么了?
3. `wo_a`/`wo_b` 怎么对应 MLA 的 output 投影(吸收 W_UV)?
4. `compute_fp8_einsum_recipe` 预计算什么?

## 📖 讲解

### 三件事(`flashmla.py:42-56`,`attention.py:363` 注释)

```python
def _o_proj(self, o, positions):
    return deep_gemm_fp8_o_proj(
        o, positions, self.rotary_emb.cos_sin_cache,
        self.wo_a, self.wo_b,
        n_groups, heads_per_group, nope_dim, rope_dim, o_lora_rank,
        einsum_recipe=self._einsum_recipe,
        tma_aligned_scales=self._tma_aligned_scales,
    )
```

| 步骤 | 操作 | 为什么 |
|------|------|--------|
| ① **inverse-RoPE** | 对 `o` 的 RoPE 部分(64 维)做逆旋转 | attention 在「旋转后空间」算的,输出要回到原空间才能投影 |
| ② **wo_a**(FP8 einsum) | `o (n_heads·head_dim)` → `n_groups·o_lora_rank`,按组 bmm | 下投影(吸收了 `W_UV`) |
| ③ **wo_b** | `n_groups·o_lora_rank` → `hidden` | 上投影回 hidden |

### 为什么需要 inverse-RoPE

回顾 MLA 解耦(L6):query 拆成 `q_C`(NoPE)+ `q_R`(RoPE),K 同理。attention 算 `q_C·k_Cᵀ + q_R·k_Rᵀ`,其中 `q_R·k_Rᵀ` 是在 **RoPE 旋转后的空间**算的。但 attention output `o` 要喂给后续的 `wo_a` 投影(回 hidden),`wo_a` 的权重是在**原空间**训练的 → 所以 `o` 的 RoPE 部分必须先**逆旋转**回原空间,才能正确投影。

```
o (n_heads, head_dim=512)
   = [ o_nope (448)  ;  o_rope (64) ]
                         ↑
                    这部分需要 inverse-RoPE
```

> inverse-RoPE 用同样的 `cos_sin_cache` + `positions`,反向旋转。这就是 `_o_proj` 接收 `positions` 和 `cos_sin_cache` 的原因。

### wo_a / wo_b 与吸收

- `wo_a`(`attention.py:213`):按 `o_groups` 分组下投影,**已吸收 `W_UV`**(把 V 的上投影矩阵吸收进 output 投影,对应 L6 的吸收思想)。`is_bmm=True`,每个 group 一组权重。
- `wo_b`(`:223`):上投影回 `hidden_size`。

### FP8 einsum 预计算(`flashmla.py:40`)

```python
self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()
```

- `_einsum_recipe`:wo_a 的分组 bmm 的 FP8 einsum 配方(形状/stride/分块)
- `_tma_aligned_scales`:TMA(Tensor Memory Accelerator,Blackwell)对齐的 FP8 scale

> 在 `__init__` 预计算,避免每次 forward 重算;TMA 对齐保证 Blackwell 上高效 DMA。这是把 FP8 分组 einsum 压榨到硬件极限的工程优化。

## 🔑 小结

> `_o_proj` = **inverse-RoPE(o 的 64 维旋转部分逆回原空间)→ wo_a(吸收 W_UV 的分组 FP8 einsum 下投影)→ wo_b 上投影回 hidden**。inverse-RoPE 补回解耦带来的旋转;einsum recipe + TMA 对齐 scale 在 init 预计算,跑满 Blackwell FP8。

## ✅ 自测

<details>
<summary><b>Q1</b>:如果去掉 inverse-RoPE,直接把 attention output <code>o</code> 喂给 <code>wo_a</code>,会怎样?</summary>

`o` 的 64 维 RoPE 部分处在「旋转后空间」,而 `wo_a` 权重在原空间训练。不逆旋转直接投影,等于用错误的坐标系做线性变换,结果完全错误(相当于给每层注入一个位置相关的错误旋转)。inverse-RoPE 把这部分转回原空间,保证 `wo_a` 投影的正确性。
</details>

<details>
<summary><b>Q2</b>:<code>wo_a</code> 「吸收了 W_UV」是什么意思?为什么 V 不显式上投影?</summary>

标准 MLA:`v = W_UV·c_KV`,attention output `o = softmax(q·kᵀ)·v`,再 `wo = W_O·o`。吸收:`o = softmax·(W_UV·c_KV) = (softmax·W_UV)·c_KV`,把 `W_UV` 合并到 attention 后的投影里。V 不显式上投影成多头,而是吸收进 `wo_a`(等价于 `W_O·W_UV` 的合并矩阵)。这样 KV cache 只存 latent `c_KV`,省掉多头 V 的存储和计算。
</details>

**下一步**:L12 完成 MLA output。L13 讲 **Lightning Indexer** —— 它在 C4A 层填 `topk_indices_buffer`,正是 L11 `extra_indices` 的来源。
