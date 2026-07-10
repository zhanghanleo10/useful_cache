# L2 · Model.forward 主干

> 单元①骨架。这是整个 forward 的**主干流**:`embed → repeat → layers → hc_head → norm`。掌握这一篇,就有了全局骨架。

## 🎯 学习目标

学完能回答:
1. `hidden_states` 的形状在 forward 全程怎么变化?
2. `hc_mult` 这个多流维度何时引入、何时消失?
3. 末 rank 在 `hc_head` 前后各做了什么?为什么要 stash 一份给 MTP?

## 📖 讲解

代码:`nvidia/model.py:1006-1057`。分四段。

### ① 入口 + MHC 展开(`:1013-1024`)

```python
if get_pp_group().is_first_rank:
    hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
    hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)   # ★ 引入 hc_mult 维
else:
    hidden_states = intermediate_tensors["hidden_states"]   # 已含 hc_mult 维

if self.use_mega_moe:
    input_ids = input_ids.to(torch.int64)                    # hash 路由需要 int64
```

形状变化:

```
embed 输出:    (N, hidden)                         # N = token 数
              ↓ unsqueeze(-2).repeat(1, hc_mult, 1)
MHC 展开后:    (N, hc_mult, hidden)                 # ← 这个形状贯穿到 hc_head 前
```

> `hc_mult` 是 MHC 的多流数(见 L3)。1 个 token 的 embedding 被复制成 `hc_mult` 份,后续每层在多流上做混合。

### ② 主循环:逐层(`:1026-1039`)

```python
residual, post_mix, res_mix = None, None, None              # MHC 跨层状态
for layer in islice(self.layers, self.start_layer, self.end_layer):
    hidden_states, residual, post_mix, res_mix = layer(
        hidden_states, positions, input_ids, post_mix, res_mix, residual)
if layer is not None:
    hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)  # 收尾展开
```

> 维护 **4 个跨层状态**:`hidden_states, residual, post_mix, res_mix`。普通模型只传 hidden(+ residual);DSV4 多了 `post_mix`、`res_mix`(MHC 混合权重,上一层算、下一层用,见 L4)。

### ③ PP 中间出口(`:1041-1042`)

```python
if not get_pp_group().is_last_rank:
    return IntermediateTensors({"hidden_states": hidden_states})   # 仍是 (N, hc_mult, hidden)
```

> PP 中间张量带 `hc_mult` 维(`make_empty_intermediate_tensors`,`:986`)—— 多流维度跨 stage 传递。

### ④ 末 rank:stash + 折叠 + norm(`:1044-1057`)

```python
# (a) stash pre-hc_head residual 给 MTP draft
self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))    # (N, hc_mult·hidden)

# (b) hc_head 折叠
hidden_states = hc_head_fused_kernel_tilelang(hidden_states, self.hc_head_fn, ...)   # (N, hc_mult, hidden) → (N, hidden)

# (c) norm
hidden_states = self.norm(hidden_states)
return hidden_states
```

**为什么 stash 在折叠前?** MTP draft(投机解码)需要的是「**多流 residual**」(`hc_mult·hidden` 维),因为它要复用主模型的 MHC 结构。折叠后的单流 hidden 不适合直接喂 draft。所以先 stash 多流版,再折叠。

## 🔑 小结

> Model.forward = **embed → `repeat(hc_mult)` 引入多流维 → 逐层(多流 + MHC 状态跨层传递)→ `mhc_post` 收尾 → stash 多流 residual 给 MTP → `hc_head` 折叠回单流 → norm**。`(N, hc_mult, hidden)` 是贯穿全程的标志形状。

## ✅ 自测

<details>
<summary><b>Q1</b>:画一下 <code>hidden_states</code> 形状从 embed 到返回的全过程。</summary>

```
embed:        (N, hidden)
repeat:       (N, hc_mult, hidden)      ← 多流维引入
layers 全程:  (N, hc_mult, hidden)      ← 保持
mhc_post:     (N, hc_mult, hidden)      ← 收尾展开,仍是多流
stash:        flatten → (N, hc_mult·hidden)   ← 复制一份给 MTP(原 tensor 不变)
hc_head:      (N, hidden)               ← 多流维消失
norm:         (N, hidden)               ← 返回
```
</details>

<details>
<summary><b>Q2</b>:为什么循环里要传 <code>post_mix</code> 和 <code>res_mix</code> 给下一层,而不是在当前层内部用完?</summary>

因为 MHC 的混合权重是在「当前层 attn/ffn 之前的 pre 阶段」算出来的,但其中 `post_mix`/`comb_mix`(即 res_mix)要到「**下一层的 post 阶段**」才被消费(用来把 attn/ffn 的单流输出展开回多流 residual)。所以必须跨层传递。见 L4 状态机。
</details>

**下一步**:L2 看到多流维贯穿全程,但没讲「多流到底怎么混合」——那是 L3(MHC 数学)和 L4(MHC 状态机)。
