# L4 · MHC 状态机(跨层流转)

> 单元②MHC。L3 讲了一次 `mhc_pre` + `mhc_post` 的数学;本篇讲这些调用在 forward 里**怎么串成状态机** —— 这是 L2 里「为什么循环要传 `post_mix`/`res_mix`」的答案。

## 🎯 学习目标

学完能回答:
1. `DecoderLayer.forward` 里有几个 `mhc_*` 调用?各在什么位置?
2. 首层为什么用 `mhc_pre`,中间层用 `mhc_fused_post_pre`?后者「融合」了哪两件事?
3. `post_mix`、`res_mix`(comb_mix)、`residual`、`x` 四个状态,在一层内怎么流转、怎么传给下一层?

## 📖 讲解

### 4 个调用点

代码:`nvidia/model.py:813-885`(DecoderLayer)+ `:1037`(Model 收尾)。

```
DeepseekV4DecoderLayer.forward(x, positions, input_ids, post_mix, res_mix, residual)
│
├─ ① attn 前
│    residual is None  →  mhc_pre_tilelang(x, hc_attn_*)              # :827  首层
│    residual is not None → mhc_fused_post_pre_tilelang(x, ..., hc_attn_*)  # :841  中间层
│    → 产出:residual, post_mix, res_mix, x(=layer_input)
│
├─ ② x = self.attn(positions, x, None)                                # :861  单流算 MLA
│
├─ ③ ffn 前
│    mhc_fused_post_pre_tilelang(x, ..., hc_ffn_*)                     # :865  融合
│    → 产出:residual, post_mix, res_mix, x(=layer_input)
│
├─ ④ x = self.ffn(x, input_ids)                                       # :884  单流算 MoE
│
└─ return x, residual, post_mix, res_mix

Model.forward 末尾:
└─ ⑤ mhc_post_tilelang(hidden, residual, post_mix, res_mix)           # :1037  收尾展开
```

### 关键洞察:`mhc_fused_post_pre` 融合了两件事

每个**非首层**的 attn/ffn 之前,`mhc_fused_post_pre` **同时做**:

```
(a) POST 展开(消费上一算子的 post_mix/comb_mix)
    把「上一个 attn/ffn 的单流输出」用 post_mix 缩放、用 comb_mix 混合进 residual
    → 更新 residual(多流)

(b) PRE 坍缩(为本算子准备输入)
    在更新后的 residual 上算本算子的 pre_mix/post_mix/comb_mix + RMSNorm
    → layer_input(单流)+ 本算子的 post_mix/comb_mix(留给下一轮 POST)
```

> 为什么融合?POST 和 PRE 都读写 `residual`(同一个多流张量),融合成一个 kernel 省一次 residual 的读写,且少一次 kernel launch。这正是「fused」的含义。

### 状态流转表(中间层一次完整循环)

| 时刻 | 消费 | 产出 | 说明 |
|------|------|------|------|
| 进 ①(attn 前) | 上一层传来的 `residual, post_mix, res_mix` + 当前 `x`(上轮 ffn 输出) | 新 `residual, post_mix(attn), res_mix(attn), x=layer_input` | POST 展开 ffn 结果 + PRE 坍缩 attn 输入 |
| ② attn | `x` | `x`(attn 单流输出) | MLA 计算 |
| 进 ③(ffn 前) | `residual, post_mix(attn), res_mix(attn)` + 当前 `x`(attn 输出) | 新 `residual, post_mix(ffn), res_mix(ffn), x=layer_input` | POST 展开 attn 结果 + PRE 坍缩 ffn 输入 |
| ④ ffn | `x` | `x`(ffn 单流输出) | MoE 计算 |
| 返回 | — | `x, residual, post_mix(ffn), res_mix(ffn)` | 传给下一层 |

下一层 ① 又消费这套 `(x, residual, post_mix, res_mix)` —— 形成跨层状态机。

### 为什么首层特殊(`mhc_pre`,`:827`)

首层没有「上一算子的输出」要展开(`residual is None`),所以:
- 直接把输入 `x` 当作 `residual`
- 只做 **PRE 坍缩**(算 attn 的三组权重 + attn_norm → layer_input),不做 POST 展开

```python
if residual is None:
    residual = x
    post_mix, res_mix, x = mhc_pre_tilelang(x, hc_attn_*, ...)   # 只有 PRE
```

### 为什么收尾要 `mhc_post`(`:1037`)

最后一层 ffn 之后,`mhc_fused_post_pre` 已经在「下一层 attn 前」才会做 POST 展开 —— 但已经没有下一层了!所以 Model 末尾显式调一次 `mhc_post`,把最后一层 ffn 的单流输出展开回多流 residual,得到最终的多流 hidden(再 stash 给 MTP + hc_head 折叠)。

## 🔑 小结

> MHC 状态机 = **首层 `mhc_pre`(只 PRE 坍缩)→ 每层 attn/ffn 前各一次 `mhc_fused_post_pre`(POST 展开 + PRE 坍缩融合)→ 末尾 `mhc_post`(只 POST 展开)**。四个状态(`x/residual/post_mix/res_mix`)跨层流转:post_mix/comb_mix 在 PRE 阶段算出、在下一轮 POST 阶段消费。

## ✅ 自测

<details>
<summary><b>Q1</b>:<code>mhc_fused_post_pre</code> 把 POST 和 PRE 融在一起,省了什么?为什么不能把 attn 也融进去?</summary>

省了一次 `residual`(多流张量)的读写和一次 kernel launch —— POST 和 PRE 都读写同一个 residual,融合后 residual 留在 kernel 内。但 attn 不能融进去:① attn 是另一套完全不同的计算(MLA,带 KV cache、稀疏索引、FlashMLA kernel),和 MHC 混合是异质操作;② attn 需要 cudagraph 的 eager break(`attention.py:426`),而 MHC 混合在捕获图内。所以 attn 必须独立调用,前后用 fused_pre_post 衔接。
</details>

<details>
<summary><b>Q2</b>:如果删掉 Model 末尾的 <code>mhc_post</code>(<code>:1037</code>),直接拿最后一层 ffn 的 <code>x</code> 去 <code>hc_head</code> 折叠,会怎样?</summary>

会算错。最后一层 ffn 的 `x` 是**单流** `layer_input`(还没经 POST 展开),而 `hc_head` 期望输入是 **`(N, hc_mult, hidden)` 的多流** residual。少了 `mhc_post`,就把未展开的单流当成多流折叠,丢失了最后一层的 MHC 混合信息。`mhc_post` 的职责正是「把最后一层 ffn 的单流输出展开回多流 residual」。
</details>

**下一步**:L3-L4 讲清了 MHC。L5 把视角拉回单层,看 DecoderLayer 怎么用 MHC 包夹 attn/ffn(其实 L4 已经触及,L5 做总结);L6 开始进入注意力链。
