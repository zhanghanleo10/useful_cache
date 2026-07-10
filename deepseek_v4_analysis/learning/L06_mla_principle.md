# L6 · MLA 原理

> 单元④注意力(原理篇)。MLA(Multi-head Latent Attention)是 DeepSeek-V2 提出、V3/V4 延续的核心注意力。本篇讲**为什么和数学**,代码实现见 L7-L13。深入版见 [`../mla.md`](../mla.md)。

## 🎯 学习目标

学完能回答:
1. 长上下文推理的真正瓶颈是什么?MHA/MQA/GQA 各怎么解决?代价?
2. MLA 的核心思想?为什么能「既省显存又保质量」?
3. 「矩阵吸收」是什么?为什么可行?
4. RoPE 为什么破坏吸收?DeepSeek 怎么解决(解耦)?

## 📖 讲解

### 动机:KV cache 瓶颈

推理时每 token 每层要缓存它的 K、V。标准 MHA:

```
KV cache ≈ 2 × n_heads × d_head × seq_len × n_layers
```

128 head、`d_head=128`、64 层、1M 序列 → ≈ 4.4 TB,装不下。

三条已有路线:

| 方案 | 做法 | 代价 |
|------|------|------|
| MQA | 所有 head 共享 1 个 K/V | 省得多,但质量掉 |
| GQA | 几个 head 共享 1 个 K/V | 折中 |
| —— | 都靠**减少 KV 份数**(砍 head) | 不同 head 看到一样的 K/V,表达力降 |

### MLA 核心思想:KV 联合压缩

不砍 head 数,而是把 K、V **联合压成一个低维 latent** `c_KV`,存 latent 而非完整 K/V;需要时上投影恢复多头。

```
存:c_KV (d_c 维, d_c << n_h·d_h)        ← KV cache 只存这个
用:c_KV →(上投影)→ 多头 K, V            ← 各 head 的 K/V 仍不同
```

### 数学:下投影、上投影、吸收

```python
# 下投影(压缩)—— c_KV 存进 KV cache
c_Q  = W_DQ  · h        # query latent
c_KV = W_DKV · h        # KV 联合 latent   ← 存这个

# 上投影(展开多头)
q = W_UQ · c_Q          # 多头 Query
k = W_UK · c_KV         # 多头 Key
v = W_UV · c_KV         # 多头 Value
```

**矩阵吸收**(推理优化灵魂):注意力分数 `q·kᵀ`,代入 `k = W_UK·c_KV`:

```
q·kᵀ = q·(W_UK·c_KV)ᵀ = (q·W_UKᵀ)·c_KVᵀ
                          └─────────┘
                          预先算好,吸收进 query
```

→ 推理时**不需要把 c_KV 上投影成多头 K**!只要 query 多乘一个 `W_UKᵀ`(吸收),直接拿 latent `c_KV` 算分数。KV cache 只存 latent。`W_UV` 对应吸收进 output 投影 `W_O`。

### RoPE 解耦(最精妙)

吸收的死敌:RoPE。K 带 RoPE 是 `k = R·(W_UK·c_KV)`,而旋转 `R` 和投影 `W_UK` **不可交换**:`R·W_UK ≠ W_UK·R` → 吸收失效。

**解法**:把 K 拆两半,NoPE 部分(可吸收)+ RoPE 部分(独立小维度):

```
k = [ k_C  ;  k_R ]
k_C = W_UK · c_KV              # NoPE,走吸收
k_R = RoPE( W_KR · c_KV )     # RoPE,单独算(维度小,如 64)

score = q_C·k_Cᵀ + q_R·k_Rᵀ
```

KV cache 最终存:`c_KV`(d_c)+ `k_R`(d_rope,小)。

### KV cache 省多少(`n_h=128, d_h=128`)

| 方案 | 每 token 每层 | 相对 MHA |
|------|--------------|---------|
| MHA | 32768 | 1× |
| GQA(8 组) | 4096 | 1/8 |
| **MLA**(`d_c≈512, d_rope=64`) | **576** | **~1/57** |

> MLA 用「**降维**」代替「**降 head 数**」,多头表达力完整 → 既省显存又保质量。

### V4 落地(代码锚点,详见 L7)

| MLA 概念 | V4 代码 |
|---------|---------|
| query 下投影 `W_DQ` | `fused_wqa_wkv` 的 `WQ_A` → `q_lora_rank`(`attention.py:194`) |
| KV latent `c_KV` | `fused_wqa_wkv` 的 `WKV` → `head_dim` |
| query 上投影(吸收 `W_UKᵀ`) | `wq_b`(`attention.py:203`) |
| output(吸收 `W_UV`) | `wo_a`/`wo_b`(`attention.py:213/223`) |
| head 维度 | 512 = 448 NoPE + 64 RoPE |

## 🔑 小结

> MLA = **K/V 联合压成低维 latent 存 cache** + **上投影矩阵吸收进 query/output**(推理时 K 不展开,直接用 latent)+ **RoPE 解耦**(位置编码单独切出来不破坏吸收)。降维不降 head,省显存保质量。

## ✅ 自测

<details>
<summary><b>Q1</b>:「矩阵吸收」本质利用了矩阵乘法的什么性质?</summary>

结合律:`q·(W_UK·c_KV)ᵀ = q·W_UKᵀ·c_KVᵀ = (q·W_UKᵀ)·c_KVᵀ`。先把 `q·W_UKᵀ` 算出来(吸收进 query),之后每步 attention 只用 latent `c_KV` 算分数,不需要显式构造多头 K。前提是 `W_UK` 和后续操作可结合 —— RoPE 旋转矩阵打破了这个前提(不可交换),所以要解耦。
</details>

<details>
<summary><b>Q2</b>:为什么 RoPE 部分不能也吸收,而要单独算?</summary>

RoPE 给 K 乘位置相关旋转 `R`:`k_R = R·(...)`。旋转矩阵 `R` 依赖**绝对位置**,且 `R·W ≠ W·R`(不可交换)。如果硬要吸收,query 端无法预计算一个固定的 `W_UKᵀ`(因为每个位置的 R 不同),吸收推导不成立。所以 RoPE 部分只能保留小维度(64),每步单独算 `q_R·k_Rᵀ`,代价可控。
</details>

**下一步**:L6 搞清原理。L7 开始读代码 —— MLA 的投影链(`fused_wqa_wkv`/`wq_b`/`wo_a`/`wo_b`)怎么落地这些概念。
