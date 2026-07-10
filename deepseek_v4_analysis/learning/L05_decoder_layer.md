# L5 · DecoderLayer 结构

> 单元③单层。L2 从 Model 视角看过主循环,L4 从 MHC 视角看过状态机;本篇**聚焦 DecoderLayer 这个模块本身**:`__init__` 装了什么、它的职责边界、与标准 Transformer 层的差异。

## 🎯 学习目标

学完能回答:
1. `DeepseekV4DecoderLayer.__init__` 装了哪些组件?各自职责?
2. 它和标准 Transformer DecoderLayer(Llama 风格)结构上差在哪?
3. 为什么 `attn_norm`/`ffn_norm` 是 RMSNorm 但 forward 里看不到它被调用?

## 📖 讲解

### 职责边界

`DeepseekV4DecoderLayer`(`nvidia/model.py:740`)做的事:**输入多流 hidden + MHC 状态 → 用 MHC 包夹一次 attn 和一次 ffn → 输出多流 hidden + 新 MHC 状态**。

```
in:  x(上轮 ffn 单流输出), residual, post_mix, res_mix
  │
  ├─ attn 前:MHC 混合 + attn_norm   → layer_input(单流)
  ├─ attn(MLA)                      → attn 单流输出
  ├─ ffn 前:MHC 混合 + ffn_norm    → layer_input(单流)
  ├─ ffn(MoE)                       → ffn 单流输出
  │
out: x, residual, post_mix, res_mix
```

> 详细的调用点与状态流转见 [L4](L04_mhc_state_machine.md)。

### `__init__` 组件清单(`model.py:748-811`)

| 组件 | 类型 | 职责 |
|------|------|------|
| `attn` | `DeepseekV4Attention`(平台子类) | Sparse MLA 注意力(`:754`) |
| `ffn` | `DeepseekV4MoE` | MoE 前向(`:760`) |
| `attn_norm` / `ffn_norm` | `RMSNorm` | 归一化权重容器(`:762-763`)—— **不单独 forward**,融进 `mhc_*` |
| `hc_attn_fn/scale/base` | `Parameter` | attn 的 MHC 混合参数(`:770-783`) |
| `hc_ffn_fn/scale/base` | `Parameter` | ffn 的 MHC 混合参数(`:798-811`) |
| 超参 | `hc_mult, hc_sinkhorn_iters, hc_eps, hc_post_alpha=2.0` | MHC 配置(`:764-767`) |

### forward 整合视角(`model.py:813-885`)

```python
def forward(self, x, positions, input_ids, post_mix, res_mix, residual):
    attn_norm_weight = self.attn_norm.weight.data          # 只取权重,不调 forward
    ...
    if residual is None:
        # 首层:standalone PRE(见 L4)
        residual = x
        post_mix, res_mix, x = mhc_pre_tilelang(x, hc_attn_*, norm_weight=attn_norm_weight, ...)
    else:
        # 中间层:fused POST(prev ffn) + PRE(attn)
        residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(x, residual, post_mix, res_mix,
                                                                      hc_attn_*, norm_weight=attn_norm_weight, ...)

    x = self.attn(positions, x, None)                       # 单流 MLA

    residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(x, residual, post_mix, res_mix,
                                                                  hc_ffn_*, norm_weight=ffn_norm_weight, ...)
    x = self.ffn(x, input_ids)                              # 单流 MoE
    return x, residual, post_mix, res_mix
```

### 与标准 Transformer DecoderLayer 对比

| 维度 | 标准(Llama 风格) | DSV4 |
|------|-------------------|------|
| hidden 形状 | `(N, hidden)` 单流 | `(N, hc_mult, hidden)` 多流 |
| 残差连接 | `x = x + attn(norm(x))` | 由 `mhc_*` 的 POST 展开 handled(comb_mix 混合 + post_mix 缩放) |
| norm 调用 | `self.input_layernorm(x)` 显式 | 融进 `mhc_*` kernel,只传 weight |
| 跨层状态 | 仅 `x`(+ PP 的 residual) | `x, residual, post_mix, res_mix` 四元组 |
| FFN | MLP(SwiGLU) | MoE(Hash/sqrtsoftplus 路由 + MegaMoE) |
| 注意力 | MHA/GQA | Sparse MLA(SWA + 稀疏全局) |

## 🔑 小结

> DecoderLayer = **`attn`(MLA)+ `ffn`(MoE)+ 两组 MHC 混合参数 + 两个 RMSNorm(融合)**,用 MHC 的 PRE/POST 包夹 attn/ffn,输入输出都是多流 + 四元组跨层状态。它是一个「**MHC 流转 + 单流计算**」的单元。

## ✅ 自测

<details>
<summary><b>Q1</b>:<code>attn_norm</code> 是 RMSNorm 实例,但 forward 里没有 <code>self.attn_norm(x)</code> 这样的调用,归一化在哪发生?</summary>

在 `mhc_pre_tilelang` / `mhc_fused_post_pre_tilelang` 内部。forward 只把 `self.attn_norm.weight.data` 作为参数传进去(`:837`),融合 kernel 在算 mixes 时用 `x` 的平方和做 RMS 归一化(见 L3 第 1 步)。这样省一次 kernel launch 和一次 x 读写。`attn_norm` 实例只作为「权重容器」存在。
</details>

<details>
<summary><b>Q2</b>:标准层的残差是 <code>x = x + sublayer(x)</code>,DSV4 的残差连接长什么样?</summary>

不是显式加法,而是融在 `mhc_post` / `mhc_fused_post_pre` 的 POST 阶段:`new_residual = comb_mix·residual + post_mix·x`(L3 的 `mhc_post` 公式)。其中 `comb_mix`(双随机矩阵)混合旧多流 residual,`post_mix` 缩放新算的子层输出 `x` 并广播到多流。所以残差是「**带混合权重的残差**」,比简单加法更灵活。
</details>

**下一步**:L5 把单层结构看完了。L6 开始进入注意力子链 —— 先讲 MLA 的**原理**(为什么这样设计),L7+ 再看代码实现。
