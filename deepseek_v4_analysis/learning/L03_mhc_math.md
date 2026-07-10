# L3 · MHC 思想与数学

> 单元②MHC。MHC(Multi-stream Hidden Computation)是 DSV4 最独特的创新。本篇讲它的**核心思想 + 数学**(基于纯 PyTorch 参考实现 `vllm/vllm/model_executor/kernels/mhc/torch.py`)。

## 🎯 学习目标

学完能回答:
1. MHC 为什么要把 1 个 token 展开成 `hc_mult` 流?省了什么、换了什么?
2. `pre_mix` / `post_mix` / `comb_mix` 三组权重各干什么?
3. 为什么 `comb_mix` 要做 Sinkhorn 归一化?
4. attn/ffn 算的是单流还是多流?多流表达力怎么保留?

## 📖 讲解

### 核心思想

普通 Transformer:1 个 token = 1 条 hidden 流,attn/ffn 在这条流上算。
MHC:1 个 token = **`hc_mult` 条流**,但 attn/ffn **只在坍缩后的单流上算**;多流间的信息靠学习到的混合权重在 residual 层面流转。

```
普通:   token → [hidden] ──attn/ffn──> [hidden]         算力 = 1 流

MHC:    token → [stream_0 .. stream_{hc_mult-1}]          表达 = hc_mult 流
                 │ repeat 展开
                 ↓ mhc_pre: pre_mix 加权坍缩
              [layer_input]  (单流) ──attn/ffn──> [x]     算力 = 1 流(省!)
                 │ mhc_post: comb_mix 混合 + post_mix 展开
                 ↓
              [stream_0 .. stream_{hc_mult-1}]            回到多流
```

> 用「便宜的混合」替代「昂贵的多头计算」换取更大的隐式容量。

### 数学(`torch.py:6-91`,`mhc_pre`)

输入:`residual (N, hc_mult, hidden)`、`fn (hc_mult3, hc_mult·hidden)`,其中:

```
hc_mult3 = 2·hc_mult + hc_mult²       # = (2 + hc_mult)·hc_mult,对应 model.py:768 的 mix_hc
```

**第 1 步:投影 + RMSNorm(融合)**

```python
x = residual.view(N, hc_mult * hidden).float()
mixes = x @ fn.T                                  # (N, hc_mult3)
sqrsum = x.square().sum(-1, keepdim=True)
mixes = mixes * rsqrt(sqrsum / (hc_mult*hidden) + rms_eps)   # ← 这就是 attn_norm/ffn_norm!
```

> RMSNorm 被融进了这里(用 x 自己的平方和归一化 mixes)。所以 DecoderLayer 里 `attn_norm`/`ffn_norm` 不单独调。

**第 2 步:三段切分 `mixes (N, hc_mult3)` → 三组权重**

```python
pre_logits  = mixes[:, :hc_mult]              * hc_scale[0] + hc_base[:hc_mult]
post_logits = mixes[:, hc_mult:2*hc_mult]     * hc_scale[1] + hc_base[hc_mult:2*hc_mult]
comb_logits = mixes[:, 2*hc_mult:].view(N, hc_mult, hc_mult) * hc_scale[2] + hc_base[...]

pre_mix  = sigmoid(pre_logits)  + hc_pre_eps          # (N, hc_mult)      ← 坍缩权重
post_mix = sigmoid(post_logits) * hc_post_mult_value  # (=×2.0)           ← 展开缩放
comb_mix = softmax(comb_logits, dim=-1) + hc_sinkhorn_eps  # (N, hc_mult, hc_mult)  ← 流间混合矩阵
```

**第 3 步:Sinkhorn 归一化 `comb_mix`**(交替行列归一化 `sinkhorn_repeat` 轮)

```python
for _ in range(sinkhorn_repeat - 1):
    comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + eps)   # 行归一
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + eps)   # 列归一
```

> Sinkhorn 让 `comb_mix` 趋近**双随机矩阵**(每行、每列和都为 1)。意义:让每条流「输出的总权重」和「被注入的总权重」都守恒,避免某些流被放大/淹没,保证多流间信息均衡流转。

**第 4 步:坍缩成单流 `layer_input`**

```python
layer_input = sum(pre_mix[:,k] * residual[:,k] for k in range(hc_mult)).to(bf16)   # (N, hidden)
```

> `pre_mix` 作为加权系数,把 `hc_mult` 条流加权求和成 1 条 → 喂给 attn/ffn。

### 展开回多流(`torch.py:94-106`,`mhc_post`)

attn/ffn 算出单流 `x (N, hidden)` 后,展开回多流:

```python
mixed_residual = einsum("...ij,...ih->...jh", comb_mix, residual)   # 用双随机矩阵混合旧 residual
post_term      = post_mix * x.unsqueeze(-2)                         # post_mix 缩放新 x,广播到 hc_mult
new_residual   = mixed_residual + post_term                         # (N, hc_mult, hidden)
```

- `comb_mix` 让 `hc_mult` 条旧流互相混合(信息交换)
- `post_mix` 决定新算的 `x` 注入每条流的强度

### 参数结构(每层)

`nvidia/model.py:770-811`:attn 和 ffn 各一组 `hc_*_fn/scale/base`,加上模型级的 `hc_head_*`(`:949`)。全是 `requires_grad=False` 的预训练权重。

## 🔑 小结

> MHC 数学 = **`fn` 投影 + 融合 RMSNorm → 三段切分(pre/post/comb)→ Sinkhorn 让 comb_mix 双随机 → `pre_mix` 加权坍缩成单流喂 attn/ffn → `comb_mix` 混合旧 residual + `post_mix` 缩放新输出,展开回多流**。算单流、表多流。

## ✅ 自测

<details>
<summary><b>Q1</b>:attn/ffn 只在单流 <code>layer_input</code> 上算,那多流的信息怎么没丢?</summary>

没丢,因为多流信息保存在两处:① **旧 residual 本身**(每条流单独的值,在 `mhc_post` 里被 `comb_mix` 混合后传给下一层);② **混合权重 `pre_mix`/`post_mix`/`comb_mix`**(决定怎么坍缩和展开)。attn/ffn 处理的是「`pre_mix` 加权坍缩后的摘要」,处理完再用 `post_mix`/`comb_mix` 展开并和旧多流 residual 融合,所以多流状态持续流转。
</details>

<details>
<summary><b>Q2</b>:如果 <code>comb_mix</code> 不做 Sinkhorn,直接用 softmax 的输出,会有什么问题?</summary>

softmax 只保证每行(每条输出流对各输入流的权重)和为 1,但**不保证列和为 1**。这意味着某些输入流可能被所有输出流都忽略(列和≈0),其信息在混合中丢失;另一些流被过度使用。Sinkhorn 的行列双归一让每条流既「输出守恒」又「输入守恒」,保证信息均衡,不出现流的饿死/爆炸。
</details>

**下一步**:L3 讲了一次 pre + post 的数学;L4 讲这些调用在 forward 里**串起来**的状态机(首层、中间层、收尾怎么衔接)。
