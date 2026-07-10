# MHC — Multi-stream Hidden Computation(多流隐式扩容)

> 参数与调用:`vllm/vllm/models/deepseek_v4/nvidia/model.py`(DecoderLayer / Model)
> 融合 kernel:`vllm/vllm/model_executor/kernels/mhc/`
> **数学参考实现**:`vllm/vllm/model_executor/kernels/mhc/torch.py`(纯 PyTorch,本文公式基于此)

## 是什么

MHC 是 V4 相对 V3 最独特的结构创新。每个 token 的 hidden state 不是 1 条流,而是 **`hc_mult` 条并行流**;层间通过一组**学习到的混合权重**(经 Sinkhorn 归一化)控制流之间的信息交换。效果:用「单 token 的注意力/FFN 计算量」承载「多流的表达容量」。

## 数据流总览

```
                          hc_mult 条 hidden stream
                              ┌──────────────┐
embed → repeat(hc_mult) ───► │ stream_0..N-1 │  (每条 = hidden_size)
                              └──────┬───────┘
                  ┌──────────────────┴──────────────────┐
                  ▼ 每层 DecoderLayer(model.py:1027)    ▼
   ┌─ mhc_pre / mhc_fused_post_pre ──────────────────────┐
   │  ① RMSNorm(融合)                                     │
   │  ② fn 投影 → 三段切分:pre_mix / post_mix / comb_mix  │
   │  ③ Sinkhorn 归一化 comb_mix                            │
   │  ④ layer_input = Σ(pre_mix · stream)  ← 坍缩成单流     │
   └──────────────────┬───────────────────────────────────┘
                      ▼
              attn (单流计算 Sparse MLA)    ← 省算!
                      │
   ┌─ mhc_fused_post_pre ─────────────────────────────────┐
   │  ⑤ ffn_norm + 同样混合                                │
   └──────────────────┬───────────────────────────────────┘
                      ▼
              ffn (单流计算 MoE)          ← 省算!
                      │
   ┌─ mhc_post ───────────────────────────────────────────┐
   │  ⑥ new_residual = comb_mix·residual + post_mix·x      │  ← 展开回 hc_mult 流
   └──────────────────────────────────────────────────────┘
                      ▼
                  下一层(hc_mult 流)
                      
... 末层后 ...
hc_head_fused: hc_mult 流 → 1 流 hidden_size   (model.py:1048)
```

> **核心洞察**:attn 和 ffn 只在「坍缩后的单流」上计算(layer_input),而多流表达力通过 pre/post/comb 混合权重在 residual 层面流转。这就是「**用单 token 的算力承载多流容量**」的实现方式。

## 数学细节(基于 `torch.py`)

### `mhc_pre` / `mhc_fused_post_pre`(`torch.py:6-91`)

输入:`residual (..., hc_mult, hidden_size)`,`fn (hc_mult3, hc_mult*hidden_size)`,`hc_scale (3,)`,`hc_base (hc_mult3,)`,其中:

```
hc_mult3 = hc_mult·2 + hc_mult²      # = (2 + hc_mult) · hc_mult,即 model.py:768 的 mix_hc
hc_dim   = hc_mult · hidden_size      # model.py:769
```

**步骤**:

```python
# 1. 展平 + 投影 + RMSNorm(融合的归一化)
x = residual.view(N, hc_mult * hidden_size).float()
mixes = x @ fn.T                                  # (N, hc_mult3)
sqrsum = x.square().sum(-1, keepdim=True)
mixes = mixes * rsqrt(sqrsum / (hc_mult*hidden) + rms_eps)   # ← 这就是 attn_norm/ffn_norm!

# 2. 三段切分 mixes(N, hc_mult3) → 三组权重
pre_logits  = mixes[:, :hc_mult]              * hc_scale[0] + hc_base[:hc_mult]
post_logits = mixes[:, hc_mult:2*hc_mult]      * hc_scale[1] + hc_base[hc_mult:2*hc_mult]
comb_logits = mixes[:, 2*hc_mult:].view(N, hc_mult, hc_mult) * hc_scale[2] + hc_base[2*hc_mult:]

pre_mix  = sigmoid(pre_logits)  + hc_pre_eps          # (N, hc_mult)   pre 混合权重
post_mix = sigmoid(post_logits) * hc_post_mult_value  # = * 2.0(model.py:767 hc_post_alpha)
comb_mix = softmax(comb_logits, dim=-1) + hc_sinkhorn_eps   # (N, hc_mult, hc_mult) 组合矩阵

# 3. Sinkhorn:交替行列归一化 sinkhorn_repeat 轮
for _ in range(sinkhorn_repeat - 1):
    comb_mix = comb_mix / (comb_mix.sum(dim=-1) + hc_sinkhorn_eps)   # 行归一
    comb_mix = comb_mix / (comb_mix.sum(dim=-2) + hc_sinkhorn_eps)   # 列归一

# 4. pre_mix 加权求和 → 单流 layer_input(喂给 attn/ffn)
layer_input = Σ_k(pre_mix[:,k] · residual[:,k]).to(bf16)    # (N, hidden_size)
```

**返回三件套**(传到 post 阶段):
- `post_mix (N, hc_mult, 1)`
- `comb_mix (N, hc_mult, hc_mult)`
- `layer_input (N, hidden_size)`

### `mhc_post`(`torch.py:94-106`)

```python
# 把 attn/ffn 单流输出 x 展开回 hc_mult 流,并与旧 residual 混合
mixed_residual = einsum("...ij,...ih->...jh", comb_mix, residual)  # 用组合矩阵混合旧 residual
post_term      = post_mix · x.unsqueeze(-2)                        # post_mix 缩放新输出,广播到 hc_mult
new_residual   = mixed_residual + post_term
```

> comb_mix 是 `(hc_mult, hc_mult)` 双随机矩阵(Sinkhorn 产物),`einsum` 让 hc_mult 条流之间互相混合;post_mix 决定新计算结果 `x` 注入每条流的强度。

### `hc_head_fused`(`model.py:1048`)

末层后,把 hc_mult 条 hidden_size 流折叠成单流(`hc_head_fn (hc_mult, hc_mult*hidden)`、`hc_head_scale (1,)`、`hc_head_base (hc_mult,)`)。具体公式在 tilelang fused kernel 内,接口见 `model.py:1048-1055`。MTP draft 在 `compute_logits` 里同样调用(`nvidia/mtp.py:243`)。

## 参数结构(每层 DecoderLayer)

`nvidia/model.py:770-811`:

| 参数 | 形状 | 用途 |
|------|------|------|
| `hc_attn_fn` / `hc_ffn_fn` | `(hc_mult3, hc_mult*hidden)`,fp32 | attn / ffn 各一组投影矩阵(产出三段 mixes) |
| `hc_attn_scale` / `hc_ffn_scale` | `(3,)`,fp32 | 三段各一个缩放(pre/post/comb) |
| `hc_attn_base` / `hc_ffn_base` | `(hc_mult3,)`,fp32 | 三段各一组偏置 |
| `hc_head_fn` | `(hc_mult, hc_mult*hidden)`,fp32 | 末层折叠投影(模型级,`model.py:949`) |
| `hc_head_scale` | `(1,)` | 折叠缩放 |
| `hc_head_base` | `(hc_mult,)` | 折叠偏置 |

超参:`hc_mult`、`hc_eps`(=hc_pre_eps=hc_sinkhorn_eps)、`hc_sinkhorn_iters`、`hc_post_alpha=2.0`。

## 为什么 RMSNorm 融进 MHC

观察 `mhc_pre` 第 1 步:`mixes` 在投影后立即用 `x` 的平方和做 RMS 归一化 —— 这等价于把 `attn_norm`/`ffn_norm` 的 RMSNorm 融合进混合权重计算。因此 DecoderLayer 里 `self.attn_norm`/`self.ffn_norm` **不单独 forward**,只作为权重容器(`model.py:860,863` 把 `.weight.data` 传给 tilelang kernel)。实际归一化发生在融合 kernel 内。

> 注意:`hc_*` 参数和 norm 权重都是 `requires_grad=False` —— 推理时这些是固定的预训练权重。

## PP 中的多流

Pipeline 中间张量形状是 `(num_tokens, hc_mult, hidden_size)`(`model.py:996,1018`),即 V4 把多流维度跨 stage 传递。首 rank 做 `repeat(hc_mult)`,末 rank 做 `hc_head` 折叠。

## 一句话心智模型

> MHC = **把 1 个 token 展开成 `hc_mult` 条 hidden 流**(扩容),**但 attn/ffn 只在坍缩后的单流上算**(省算),**流间信息靠 pre/post/comb 三个学习权重 + Sinkhorn 归一化在 residual 层面流转**(表达力)。本质上是用「便宜的混合」替代「昂贵的多头计算」来换取更大的隐式容量。
