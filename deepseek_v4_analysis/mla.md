# MLA — Multi-head Latent Attention 机制原理

> 适用:DeepSeek-V2 / V3 / V4 通用机制(**原理篇**)
> V4 实现:`vllm/vllm/models/deepseek_v4/attention.py:98`(`DeepseekV4Attention`)
> 配套:[`sparse_mla_indexer.md`](sparse_mla_indexer.md)(V4 在 MLA 之上的 Compressor / Indexer 工程实现)

本文讲 MLA 机制**本身**:为什么需要、数学原理、为什么省显存。V4 特有的稀疏化扩展见配套文档。

## 一、为什么需要 MLA:KV cache 瓶颈

长上下文推理的真正瓶颈不是计算,而是 **KV cache 的显存**。

标准 **MHA**(Multi-Head Attention),每个 token 在每层要缓存它的 K 和 V,共 `n_heads` 份:

```
KV cache 大小 ≈ 2 × n_heads × d_head × seq_len × n_layers
```

以 128 head、`d_head=128`、64 层的模型为例,单条 **1M 序列**:

```
2 × 128 × 128 × 1,048,576 × 64 × 2 bytes ≈ 4.4 TB   ← 完全装不下
```

业界的几条路线:

| 方案 | 做法 | KV cache | 代价 |
|------|------|---------|------|
| **MHA** | 每 head 独立 K/V | `2·n_h·d_h`(大) | 质量最好,但显存爆 |
| **MQA** | 所有 head 共享 **1** 个 K/V | `2·d_h`(省 n_h 倍) | 质量掉得厉害 |
| **GQA** | 几个 head 共享 1 个 K/V(折中) | `2·(n_h/g)·d_h` | 质量/省显存折中 |

> MQA/GQA 的思路是**「减少 KV 的份数」**(砍 head 数),代价是不同 head 看到的 K/V 一模一样,表达力下降。

**MLA 走了第四条完全不同的路**:不砍 head 数,而是**把 K 和 V 联合压缩成一个低维向量**。

## 二、MLA 核心思想:KV 联合压缩成 latent

关键洞察:K 和 V 都从同一个 hidden state `h` 算出来,高度相关 → 可先压成一个**低维 latent 向量** `c_KV`,需要时再展开。

```
传统:缓存完整多头 K 和 V
  KV cache: [n_heads × d_head] × 2          ← 巨大

MLA:只缓存压缩 latent c_KV
  KV cache: [d_c]   (d_c << n_h · d_h)      ← 极小
  需要时:c_KV →(上投影)→ 多头 K, V
```

`d_c`(latent 维度)远小于 `n_heads × d_head`,KV cache 大幅缩小;而展开后**所有 head 的 K/V 依然各不相同**(由上投影矩阵生成),表达力不丢。

## 三、数学:下投影、上投影、矩阵吸收

### 1. 压缩与展开

对每个 token 的 hidden state `h_t`:

```python
# 下投影(压缩)—— 输出存进 KV cache
c_Q  = W_DQ  · h_t      # query latent (维度 d_cQ)
c_KV = W_DKV · h_t      # KV 联合 latent (维度 d_c)   ← 存这个!

# 上投影(展开成多头)—— 推理时按需做
q = W_UQ · c_Q          # → [n_heads, d_head]   多头 Query
k = W_UK · c_KV         # → [n_heads, d_head]   多头 Key
v = W_UV · c_KV         # → [n_heads, d_head]   多头 Value
```

### 2. 矩阵吸收(absorption)—— 推理优化的灵魂

注意力分数 `q · kᵀ`,把 `k = W_UK · c_KV` 代入:

```
q · kᵀ = q · (W_UK · c_KV)ᵀ = (q · W_UKᵀ) · c_KVᵀ
          └────────────────┘
            可预先算好,「吸收」进 query
```

**结论**:推理时根本不需要把 `c_KV` 上投影成多头 K!只要在算 query 时多乘一个 `W_UKᵀ`(吸收),然后直接拿 latent `c_KV` 算分数即可。**KV cache 里只存 `c_KV`,省掉了上投影出来的巨大 K**。V 对应的 `W_UV` 则被吸收到输出投影 `W_O` 里。

> 这就是 MLA「latent」名字的由来:存的是 latent,不是显式的 K/V。

## 四、RoPE 解耦 —— MLA 最精妙的部分

矩阵吸收有个死敌:**RoPE(旋转位置编码)**。

RoPE 给 K 乘一个位置相关的旋转矩阵 `R`:`k = R · (W_UK · c_KV)`。问题是旋转和投影**不可交换**:

```
R · W_UK  ≠  W_UK · R
```

一旦 K 带 RoPE,就无法把 `W_UK` 吸收到 query 里 —— 上面的吸收推导失效。

### DeepSeek 的解法:解耦(decouple)

把每个 head 的 K 拆成**两半**,各自处理:

```
k = [ k_C  ;  k_R ]
     │         │
     │         └─ RoPE 部分:带位置编码,独立维度,不吸收
     └─ NoPE 部分:可被吸收(走吸收路径)
```

```python
k_C = W_UK · c_KV              # NoPE,可吸收
k_R = RoPE( W_KR · c_KV )     # RoPE,单独算(维度小)
q_C = W_UQ · c_Q              # 配 k_C
q_R = RoPE( W_QR · c_Q )      # 配 k_R

# 最终注意力分数 = 两部分相加
score = q_C·k_Cᵀ  +  q_R·k_Rᵀ
```

- **NoPE 部分**:走吸收路径,KV cache 只存 latent `c_KV`,省显存。
- **RoPE 部分**:维度很小(如 64),无法吸收但占得少,KV cache 额外存一个小的 `k_R`。

**KV cache 最终存**:`c_KV`(d_c 维)+ `k_R`(d_rope 维,很小)。从 `2·n_h·d_h` 降到 `d_c + d_rope`。

## 五、KV cache 省了多少

以 DeepSeek-V3/V4 量级(`n_heads=128, d_head=128`)为例:

| 方案 | 每 token 每层 KV cache | 相对 MHA |
|------|----------------------|---------|
| MHA | `2 × 128 × 128 = 32768` | 1× |
| GQA(8 组) | `2 × 16 × 128 = 4096` | 1/8 |
| **MLA**(`d_c≈512, d_rope=64`) | **`512 + 64 ≈ 576`** | **~1/57** |

> MLA 用「**降维**」而非「**降 head 数**」省显存,且多头表达力完整保留 —— 这就是它能同时做到「省显存 + 高质量」的原因。

## 六、V4 的具体实现(代码锚点)

V4 的 MLA 在 `attention.py:98` 的 `DeepseekV4Attention`,参数与标准 MLA 的对应:

| MLA 概念 | V4 代码 | 说明 |
|---------|---------|------|
| query 下投影 `W_DQ` | `fused_wqa_wkv` 的 `WQ_A` 分支 → `q_lora_rank` 维 | `attention.py:194`(融合矩阵,`disable_tp`) |
| KV latent `c_KV` 生成 `W_DKV` | `fused_wqa_wkv` 的 `WKV` 分支 → `head_dim` 维 | 同上 |
| query 上投影 `W_UQ`(吸收了 `W_UKᵀ`) | `wq_b`:`q_lora_rank → n_heads·head_dim` | `attention.py:203` |
| output 下投影(吸收了 `W_UV`) | `wo_a`:按 `o_groups` 分组 bmm | `attention.py:213` |
| output 上投影 | `wo_b` | `attention.py:223` |
| RoPE head 维度 | `qk_rope_head_dim = 64` | `attention.py:170` |
| head 总维度 | `head_dim = 512` = 448 NoPE + 64 RoPE | `sparse_mla.py:78` |
| GPT-J 风格 RoPE | `is_neox_style=False`(interleaved) | `common/rope.py:35` |

### `forward_mqa` = 吸收后的形式

V4 前向直接调 `self.forward_mqa(q, kv, ...)`(`attention.py:505`)—— 注意 K 传入的是 **`kv`(`head_dim=512` 的 latent)**,不是多头 K。这正是**矩阵吸收后的形态**:`W_UK` 已被吸收进 `wq_b`,所以推理时 K 就是 latent 本身,所有 query head 对这同一个 latent 做注意力(MQA 形态)。

```
q = wq_b(c_Q)                # [n_heads, 512],已吸收 W_UKᵀ
k = kv = c_KV                 # [512] latent,所有 head 共享(MQA)
score = q_C·k_Cᵀ + q_R·k_Rᵀ    # 448 维 NoPE 走吸收,64 维 RoPE 走解耦
```

### V4 相对标准 MLA 的两点变化

1. **直接让 KV latent 维度 = head_dim(512)**:标准 MLA 的 `c_KV` 维度 `d_c` 通常和 `d_head` 不同;V4 既然吸收后 K 就是 latent,干脆把 latent 维度设成 `head_dim`,Q 多头共享这个 512 维 latent。
2. **KV latent 再被 Compressor 二次压缩 + 量化**(`compressor.py`):存进 KV cache 的不是 bf16 latent,而是 UE8M0 量化的 `fp8_ds_mla` 格式(584B/token),进一步省带宽 —— 这是 V4 独有的,见 [`sparse_mla_indexer.md`](sparse_mla_indexer.md)。

## 七、MLA 在 V4 里的位置

V4 把 MLA 包进一个更大的稀疏注意力框架:

```
MLA 主体(latent 压缩 + 吸收 + RoPE 解耦)        ← 本文
   │
   ├─ Compressor:把 latent 量化成 fp8_ds_mla 写进压缩 page   (V4 新增)
   ├─ Lightning Indexer:用 MQA 选 topk token,主 MLA 只算稀疏子集  (V4 新增)
   └─ C4A / C128A 两种压缩模式(逐层可选)
```

标准 MLA 解决「**KV cache 显存**」问题;V4 的 Compressor/Indexer 进一步解决「**超长序列(1M)的注意力计算量**」问题 —— 详见 [`sparse_mla_indexer.md`](sparse_mla_indexer.md)。

## 一句话心智模型

> MLA = **把 K/V 联合压成一个低维 latent 存进 cache**(省显存)+ **上投影矩阵吸收进 query**(推理时 K 不展开,直接用 latent 算)+ **RoPE 解耦**(把位置编码单独切出来,不破坏吸收)。用「降维」代替 MQA/GQA 的「降 head 数」,既省显存又保质量。
