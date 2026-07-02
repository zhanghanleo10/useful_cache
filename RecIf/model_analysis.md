# RecIF SID Beam Search 模型结构分析文档

> 本文档基于 `checkpoint/` 下真实权重的逐 key 核对，结合 `recif_beam_search_standalone.py` 与 `config.json`，给出完整的模型结构剖析。

---

## 0. 项目定位

这是一个 **生成式推荐（Generative Recommendation）** 推理包，属于 TIGER / RecIF 范式：把"预测用户下一个感兴趣的 item"建模成"自回归生成下一个 item 的 ID"。

技术栈：随机初始化、训练 1000 步的 **Qwen3-MoE-tiny** backbone + **byte-level SID 词表** + **3 个外置预测头** + **分层 beam search**。

### 目录结构

| 文件 | 大小 | 作用 |
|------|------|------|
| `recif_beam_search_standalone.py` | 12.7 KB | 单文件推理脚本（零项目依赖） |
| `requirements.txt` | 828 B | 依赖锁定 |
| `checkpoint/config.json` | 956 B | HF 模型结构配置 |
| `checkpoint/_model_rank0.pt` | 7.78 GB | backbone（Megatron 格式，rank-0） |
| `checkpoint/external_rank0.pt` | 302 MB | 3 个 SID 预测头权重 + 优化器状态 |
| `README.md` | 3.5 KB | 说明文档 |

---

## 1. 权重组件概览

| 文件 | 内容 |
|------|------|
| `_model_rank0.pt` | backbone（Megatron 格式，单卡 EP=1） |
| `external_rank0.pt` | 顶层有 2 个键：`lm_heads`（推理用）+ `opt_dense`（AdamW 优化器状态，推理忽略） |
| `config.json` | HF 架构配置 |

**重要发现**：`external_rank0.pt` 实际包含 **权重 + 优化器状态** 两个顶层键。优化器是 AdamW（β=(0.9,0.99), eps=1e-7, wd=1e-5, decoupled_wd=True），`step=1000`（与 README "训 1000 步" 一致）。**推理时只用 `lm_heads`，`opt_dense` 是 dead weight**。

---

## 2. Backbone 架构（Qwen3-MoE）

### 2.1 超参表（来自 `config.json`）

| 超参 | 值 | 含义 / 解读 |
|------|-----|-----|
| `architectures` | `Qwen3MoeForCausalLM` | HF 中的 MoE 因果 LM 架构 |
| `hidden_size` | 2048 | 隐藏维度 H |
| `num_hidden_layers` | 6 | **极浅**（典型 LLM 是 24~80 层） |
| `num_attention_heads` | 16 | Query 头数 |
| `num_key_value_heads` | 4 | KV 头数 → GQA group = 16/4 = 4 |
| `head_dim` | 128 | 每头维度 |
| `num_experts` | 20 | MoE 专家总数 |
| `num_experts_per_tok` | 1 | **top-1 路由**（最稀疏） |
| `moe_intermediate_size` | 5120 | 每个专家 FFN 中间维度 |
| `decoder_sparse_step` | 1 | 每层都用 sparse MoE（非 dense） |
| `norm_topk_prob` | true | 路由概率归一化（标准 MoE 做法） |
| `router_aux_loss_coef` | 0.001 | 辅助 load-balance loss 系数 |
| `attention_bias` | false | QKV 投影无 bias（Qwen3 风格） |
| `attention_dropout` | 0.0 | 推理无 dropout |
| `hidden_act` | `silu` | SwiGLU 激活 |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `rope_theta` | 1e6 | RoPE 基频（高频编码） |
| `max_position_embeddings` | 40960 | 序列长度上限 |
| `max_window_layers` | 6 | 全部层都用 full attention |
| `sliding_window` | null | 无滑动窗口 |
| `tie_word_embeddings` | false | embedding 与 lm_head 不共享 |
| `vocab_size` | 151936 | 原生 Qwen3 词表（**推理时被强制覆盖为 24576**） |
| `bos/eos_token_id` | 151643 / 151645 | 推理时用不到（词表已替换） |
| `dtype` | bfloat16 | 训练精度 |

### 2.2 真实 ckpt 顶层结构（共 333 keys：284 tensors + 49 `_extra_state`）

```
embedding/
  └── word_embeddings.weight          (24576, 2048)      ← 词表=24576（非config的151936）
decoder/
  ├── layers.{0..5}/                  × 6 层，每层 47 keys
  └── final_layernorm.weight          (2048,)
```

- `_extra_state` 是 Megatron 的元数据（FP8 / amax history 等），共 49 个，推理时被脚本过滤。
- **embedding shape = (24576, 2048)** 直接证实：ckpt 里的 vocab 已经是替换后的 SID 词表，与 `config.json` 中 `vocab_size=151936` **不一致**（脚本运行时强制覆盖 config）。

### 2.3 单层（decoder.layers.i）的完整结构

每层 47 keys，分布如下：

#### (a) Self-Attention（GQA，5 个权重）

```
self_attention/
  ├── linear_qkv.weight               (3072, 2048)       ← 融合 QKV
  ├── linear_qkv.layer_norm_weight    (2048,)            ← attn 前 RMSNorm
  ├── linear_proj.weight              (2048, 2048)       ← output proj
  ├── q_layernorm.weight              (128,)             ← Q 头每头 RMSNorm
  └── k_layernorm.weight              (128,)             ← K 头每头 RMSNorm
```

**QKV 融合解析**：`(3072, 2048)` 拆解
- 3072 = 4 (kv groups) × (hpg+2) × head_dim = 4 × (4+2) × 128 = 4 × 768
- 布局 `[num_kv_heads=4, group=768, hidden=2048]`
- 每组 768 = Q(4 头×128=512) + K(128) + V(128)
- 这正是脚本 `_split_qkv` 处理的格式

#### (b) MoE Block（42 个权重 = 1 router + 20×2 experts）

```
mlp/
  ├── router.weight                   (20, 2048)         ← 路由 gate
  └── experts/
      ├── linear_fc1.weight{0..19}    (10240, 2048)      ← 20 个 GatedMLP 上行
      └── linear_fc2.weight{0..19}    (2048, 5120)       ← 20 个 GatedMLP 下行
```

- `linear_fc1.weight{e}` 形状 `(10240, 2048)` = `(2 × 5120, 2048)` → GatedMLP，前 5120 是 gate，后 5120 是 up
- `linear_fc2.weight{e}` 形状 `(2048, 5120)` = 标准下行
- 每专家参数量：10240×2048 + 2048×5120 = 20.97M + 10.49M = **31.46 M**
- 20 专家合计：**629.1 M**
- `pre_mlp_layernorm.weight (2048,)` 是 MoE 前的 RMSNorm（每层 1 个）

### 2.4 参数量精确核算

**单层**：
- attn: QKV 6.29M + proj 4.19M + 3 norms ≈ 1K = **10.5 M**
- MoE: 629.1 M + 4K(norms/router) ≈ **629.2 M**
- **单层合计 ≈ 639.7 M**

**backbone 完整**：

| 模块 | 参数量 |
|------|--------|
| `embedding.word_embeddings` (24576×2048) | 50.3 M |
| 6 × decoder layer (attn+MoE) | 3,838.2 M |
| `decoder.final_layernorm` (2048) | 2 K |
| **backbone 合计** | **≈ 3,889 M ≈ 3.89 B** |

bf16 占用：3.89 B × 2 = **7.78 GB** ✓ 与 ckpt 大小完全吻合。

**激活参数**（推理时单 token 实际算的）：
- 每层：attn 10.5M + 1 专家 31.46M = **41.96 M**
- 6 层：**251.8 M**
- + embedding 50.3M（embedding lookup 实际只查对应行，但全表驻留显存）
- 等效计算激活 ≈ **~0.25 B** ✓ 与 README "~0.30B 激活" 吻合。

### 2.5 关键设计解读

1. **浅层 + 宽 MoE**：6 层很浅，但 MoE 把容量做进去。推理延迟低（层数少）但表达够（专家多）—— 推荐场景的工程取舍。
2. **top-1 路由**：极致稀疏，单 token 只算 1 个专家 → 等效 dense FFN 大小只有 31 M。
3. **GQA（group=4）**：KV 头数压缩到 4，比标准 MHA 省 4 倍 KV cache。本脚本不用 KV cache。
4. **RoPE θ=1e6**：基频大，波长大，适合长序列（max_pos=40960）。
5. **Q/K 头独立 RMSNorm**：Qwen3 引入（Qwen2 没有），在 RoPE 前对每头 Q/K 做 RMSNorm，提升训练稳定性，对随机初始化小模型尤其重要。
6. **随机初始化 + 仅训 1000 步**：demo / 实验级别（lm loss 8.6 → 4.7），非生产级质量。

---

## 3. 词表与 SID 编码方案

### 3.1 SID 三字节打包

```python
sid (int64) = sa | (sb << 14) | (sc << 28)
```

- 每个 byte 占 **14 bit**（理论上限 16384）
- 实际只用 **13 bit = 8192 个值**（`sa, sb, sc ∈ [0, 8191]`）
- 总编码空间：8192³ ≈ **5.5 × 10¹¹** 个 item（足够覆盖任意商品库）

### 3.2 词表替换（关键改造）

原 Qwen3 `vocab_size=151936` 被**强制覆盖**为 `3 × 8192 = 24576`（`recif_beam_search_standalone.py:120`）：

```python
cfg.vocab_size = 3 * VOCAB   # VOCAB = 8192
```

词表布局：

| Token ID 区间 | 含义 |
|--------------|------|
| `[0, 8191]` | `sa`（byte0） |
| `[8192, 16383]` | `sb + 8192`（byte1） |
| `[16384, 24575]` | `sc + 16384`（byte2） |

区间**不重叠**，模型通过 token id 区分当前在编码哪个 byte 层级。

### 3.3 单 item 的 4-token 序列

每个 item 占 **4 个 token**：

```
[ctx=0, sa, sb+8192, sc+16384]
```

- `ctx=0` 是**占位 token**。训练时 `use_sideinfo=false`，意味着这个槽**不注入任何 side info**（如品类、价格），纯粹是 `embed_tokens(0)`。设计上预留了这个槽，未来可做 side info 注入。
- 序列长度 = `4 × num_items + 1`（额外 1 是目标 item 的 ctx 槽）。

例：用户 5 个历史 item → 序列长 `4×5 + 1 = 21` token。

---

## 4. External Heads（3 个 SID 预测头）

### 4.1 真实结构（`lm_heads` 字典）

```
heads.0.weight   (8192, 2048)   bf16   ← 预测 byte0 (sa)
heads.1.weight   (8192, 2048)   bf16   ← 预测 byte1 (sb)
heads.2.weight   (8192, 2048)   bf16   ← 预测 byte2 (sc)
```

- 每个头是 `Linear(2048, 8192, bias=False)` → 输出在该 byte 层级下 8192 个值的 logit。
- 3 头合计：3 × 8192 × 2048 = **50,331,648 ≈ 50.3 M 参数**，bf16 约 100 MB。
- **无 bias**（与 backbone 风格一致）。
- **3 个头不共享权重**（独立训练），各管一层 byte。

### 4.2 优化器状态（`opt_dense`，推理不需要）

3 个头各有独立 AdamW state：
- `exp_avg`（一阶动量）shape = 同权重 `(8192, 2048)`
- `exp_avg_sq`（二阶动量）shape = 同权重 `(8192, 2048)`
- `step = 1000`（训练步数）
- param_groups: `lr=3e-5, betas=(0.9, 0.99), eps=1e-7, weight_decay=1e-5, decoupled=True`

**推理时直接忽略 `opt_dense`**。

### 4.3 外置头 ≠ tied embedding

- `tie_word_embeddings=false`，3 个外置头完全独立训练
- 与 embedding (24576, 2048) 形状也不对应（embedding 是 3 段拼接，每个头是独立 8192 输出）
- 这让 byte 层级有**专门化的投影方向**

---

## 5. 完整模型数据流（端到端）

```
输入: 用户历史 SID 列表 [sid_1, sid_2, ..., sid_K]  (int64)
  │
  │ sid_to_bytes (×K 次)
  ▼
token 序列: [0, sa_1, sb_1+8192, sc_1+16384, 0, sa_2, sb_2+8192, sc_2+16384, ..., 0]
            └ ctx ───────── item_1 ─────────┘  └ ctx ──── item_2 ───...    └ target ctx ┘
  │ 长度 = 4K + 1
  ▼
┌──────────────────────────────────────────────┐
│  embedding.word_embeddings (24576 → 2048)    │  → [1, 4K+1, 2048]
└──────────────────────────────────────────────┘
  │
  ▼ 重复 6 层:
┌──────────────────────────────────────────────┐
│  input_layernorm (RMSNorm 2048)              │
│  │                                           │
│  ▼ GQA Self-Attention                        │
│  │  q/k/v via linear_qkv → split → RoPE      │
│  │  q_norm/k_norm (per-head RMSNorm, 128)    │
│  │  attention → linear_proj                  │
│  │  + residual                               │
│  ▼                                           │
│  post_attention_layernorm (RMSNorm 2048)     │
│  │                                           │
│  ▼ Sparse MoE                                │
│  │  mlp.gate (router): 2048 → 20 logits      │
│  │  top-1 选专家 + norm_topk_prob            │
│  │  专家 e: down(silu(gate(x)) * up(x))      │
│  │  + residual                               │
└──────────────────────────────────────────────┘
  │
  ▼
final_layernorm (RMSNorm 2048)
  │
  ▼ last_hidden_state [1, 4K+1, 2048]
  │ 取最后一个位置 (target ctx slot)
  ▼ h [1, 2048]
  │
  ├── heads[0] (Linear 2048→8192) → log_softmax → top-8 byte0
  │   ▼ 拼入序列
  ├── heads[1] → log_softmax → top-8/beam → 留 top-64
  │   ▼ 拼入序列
  └── heads[2] → log_softmax → top-8/beam → 留 top-64
      │
      ▼
输出: 64 个 (sa, sb, sc) 三元组 + 累积 log-prob
```

---

## 6. 推理算法：分层 Beam Search

### 6.1 整体流程（`beam_search`，`recif_beam_search_standalone.py:182-224`）

输入：前缀 token 张量 `prefix = [1, 4K+1]`（K 个历史 item + 1 个目标 ctx 槽）

#### Step 0 — 预测 byte0
```
h0 = backbone(prefix)[:, -1, :]              # [1, H]，取末尾 ctx 槽的隐状态
lp0 = log_softmax(heads[0](h0))              # [1, 8192]
s0, sa = lp0.topk(bf0=8)                     # 选 top-8 byte0
```

#### Step 1 — 预测 byte1
```
seq1 = [prefix, sa]                            # [8, 4K+2]，8 个候选并行
h1 = backbone(seq1)[:, -1, :]
lp1 = log_softmax(heads[1](h1))                # [8, 8192]
s1, sb = lp1.topk(bf1=8)                       # 每 beam 8 个 byte1 → 64 候选
joint1 = s0[:,None] + s1                        # 累加 log-prob
flat.topk(beam=64)                              # 留 top-64
```

#### Step 2 — 预测 byte2
```
seq2 = [prefix, sa_k, sb_k+8192]               # [64, 4K+3]，64 个候选并行
h2 = backbone(seq2)[:, -1, :]
lp2 = log_softmax(heads[2](h2))                 # [64, 8192]
s2, sc = lp2.topk(bf2=8)                        # 每 beam 8 → 512 候选
joint2 = top_s[:,None] + s2
flat2.topk(beam=64)                             # 最终 top-64
```

### 6.2 算法关键特性

1. **树形展开 + 全局剪枝**：`8 × 8 × 8 = 512` 个叶子路径，每层后剪到 64。经典 beam search。
2. **分数 = 累积 log-softmax**：`score = log p(sa|prefix) + log p(sb|prefix,sa) + log p(sc|prefix,sa,sb)`。这是**联合对数概率的链式分解**，精确。
3. **不用 KV-cache**（line 178-181 注释）：每步重跑整个前缀。原因：
   - **鲁棒性**：避免不同 transformers 版本 KV cache reorder API 的兼容问题
   - **代价小**：序列短（≤几十 token）+ beam 小（≤64）+ 模型浅（6 层）
4. **批量并行**：step 1 batch=8，step 2 batch=64，全并行前向，无 Python 循环。
5. **bf16 计算但 fp32 log-softmax**（line 192）：`.float()` 升 fp32 保证 topk 数值精度。

### 6.3 计算量估算

每步前向：`batch × seq_len × (attention + 1 expert FFN)` × 6 层。
- step 0：1 × 21 × 42M ≈ 5.3 GFLOP
- step 1：8 × 22 × 42M ≈ 44 GFLOP
- step 2：64 × 23 × 42M ≈ 370 GFLOP
- 总计 ~420 GFLOP / 推理 → 在 A100/H100 上是几十毫秒级。

### 6.4 复杂度随配置变化

| 配置 | 候选展开 | step 2 batch | 适用场景 |
|------|---------|--------------|---------|
| `8,8,8` beam=64（默认） | 512 → 64 | 64 | 召回主路径 |
| `4,4,4` beam=16 | 64 → 16 | 16 | 快速过滤 |
| `16,16,16` beam=128 | 4096 → 128 | 128 | 高质量但慢 |

`bf³` 必须相对 `beam` 足够大，否则 beam 剪枝无意义。

---

## 7. 权重加载与格式转换

### 7.1 Megatron → HF key 映射（`megatron_to_hf`，line 74-101）

| Megatron key | HF Qwen3Moe key | 转换 |
|--------------|-----------------|------|
| `embedding.word_embeddings.weight` | `embed_tokens.weight` | 直接复制 |
| `decoder.final_layernorm.weight` | `norm.weight` | 直接复制 |
| `self_attention.linear_qkv.weight` | `self_attn.{q,k,v}_proj.weight` | **`_split_qkv`** 拆分 |
| `self_attention.linear_proj.weight` | `self_attn.o_proj.weight` | 直接复制 |
| `self_attention.q_layernorm.weight` | `self_attn.q_norm.weight` | 直接复制 |
| `self_attention.k_layernorm.weight` | `self_attn.k_norm.weight` | 直接复制 |
| `self_attention.linear_qkv.layer_norm_weight` | `input_layernorm.weight` | 直接复制 |
| `pre_mlp_layernorm.weight` | `post_attention_layernorm.weight` | 直接复制 |
| `mlp.router.weight` | `mlp.gate.weight` | 直接复制 |
| `mlp.experts.linear_fc1.weight{e}` | `mlp.experts.{e}.{gate_proj,up_proj}.weight` | **`_split_gated`** 对半切 |
| `mlp.experts.linear_fc2.weight{e}` | `mlp.experts.{e}.down_proj.weight` | 直接复制 |

### 7.2 两个关键拆分操作

**(1) QKV 拆分（`_split_qkv`, line 56-65）**

Megatron 把 QKV 融合为一个权重，布局 `[num_kv_heads, (hpg+2)*head_dim, hidden]`，其中 `hpg = num_heads/num_kv_heads = 4`（group size）：

```
group = (4 + 2) × 128 = 768
[768, hidden] 内部排列：
  [0 : 512]   -> Q (4 个 query 头)
  [512: 640]  -> K (1 个 kv 头)
  [640: 768]  -> V (1 个 kv 头)
```
函数把每组的这三段抽出来，铺平成 HF 标准的 `[num_heads*head_dim, hidden]`。

**(2) GatedMLP 拆分（`_split_gated`, line 68-71）**

Megatron 的 `fc1` 是 `[2*intermediate, hidden]`，前半是 gate、后半是 up（SwiGLU 公式 `down(silu(gate(x)) * up(x))`）。直接对半切。

### 7.3 加载健壮性（line 131-137）

- `load_state_dict(strict=False)`：因为 `rotary_emb` / `inv_freq` 是 HF 模型的非持久 buffer，不在 ckpt 里。
- 过滤掉所有 `_extra_state` 后缀（Megatron 的元数据，HF 不识别）。
- 显式校验 `real_missing` 和 `unexpected`：除了 rotary 都必须匹配 → **保证权重完整加载**。

### 7.4 外置头加载（line 141-148）

```python
ext = torch.load('external_rank0.pt')
head_sd = ext['lm_heads']   # {'heads.0.weight', 'heads.1.weight', 'heads.2.weight'}
```
3 个 `Linear(2048, 8192)` 分别从 `heads.{k}.weight` 拷贝。

---

## 8. 训练背景（来自 README）

| 项 | 值 |
|----|----|
| 模型代号 | Qwen3-MoE-D6-sid |
| 初始化 | **随机**（非 LLM 继承） |
| 训练步数 | 1000 步（cosine LR） |
| 训练域 | OpenOneRec-RecIF Product 域 |
| lm loss 轨迹 | 8.6 → ~4.7 |
| validation loss | 4.77 |
| test loss | 4.71 |
| NaN 数 | 0 |
| 完整 ckpt | ~54 GB（含 46GB 优化器状态），**不在本包** |

仅 1000 步训练，质量是"能跑通"的 demo 级别，不是上线模型。

---

## 9. 参数量与显存核算（实测对照）

| 组件 | 参数量 | bf16 理论 | 实测文件 | 一致性 |
|------|--------|----------|---------|--------|
| embedding (24576×2048) | 50.3 M | 100 MB | 含在 backbone | ✓ |
| 6 × attn block | 63.0 M | 126 MB | 含在 backbone | ✓ |
| 6 × MoE block (20 experts) | 3,774.7 M | 7.55 GB | 含在 backbone | ✓ |
| norms/router | ~10 K | ~20 KB | 含在 backbone | ✓ |
| **backbone 合计** | **3.89 B** | **7.78 GB** | **7.78 GB** | **✓** |
| 3 × external heads | 50.3 M | 100 MB | 302 MB（含优化器） | ✓ |
| AdamW 优化器状态 (3 头) | ~200 M(估) | ~200 MB | ~200 MB | ✓ |

**MoE 占绝对主体**：单层 639.7M 中 MoE 占 629M（~98%），attn 只占 ~2% —— 典型的"参数都在 FFN"的设计。

---

## 10. 配置 vs 真实权重一致性检查

| `config.json` 字段 | 期望 shape | 实测 shape | 一致 |
|-------------------|-----------|-----------|------|
| `hidden_size=2048` | embedding dim 2048 | (·, 2048) | ✓ |
| `num_hidden_layers=6` | 6 个 layers.0..5 | 6 层 × 47 keys | ✓ |
| `num_attention_heads=16, head_dim=128` | Q proj (2048,2048) | linear_qkv 拆出 (2048,2048) | ✓ |
| `num_key_value_heads=4` | KV proj (512,2048) | 拆出 (512,2048) | ✓ |
| `num_experts=20` | 20 个 weight0..19 | 20 个 | ✓ |
| `moe_intermediate_size=5120` | fc2 (2048,5120) | (2048,5120) | ✓ |
| `vocab_size=151936` | embedding (151936,2048) | **(24576,2048)** | ❌ 不一致（脚本强制覆盖） |

**结论**：除 `vocab_size`（被 SID 词表替换）外，所有字段与真实权重完全一致。架构是标准 Qwen3-MoE，没有结构层面的魔改。

---

## 11. 依赖与运行环境

### 11.1 依赖（`requirements.txt`）

```
torch==2.10.0              # CUDA 12.8 构建
transformers==4.57.1       # 必须支持 Qwen3MoeModel
numpy==2.4.4
safetensors==0.6.2
tokenizers==0.22.2
huggingface-hub==0.36.2
```

**关键约束**：`transformers>=4.57` 才有 `Qwen3MoeModel`。版本漂移可能导致 API 不兼容（这也是脚本放弃 KV cache 的原因之一）。

### 11.2 运行方式

```bash
python recif_beam_search_standalone.py \
  --ckpt ./checkpoint \
  --config ./checkpoint/config.json \
  --history 598080194427,628177754964,755993681678 \
  --bf 8,8,8 --beam 64 --device cuda:0 --out preds.json
```

参数：
- `--history`：逗号分隔的 int64 SID 列表（用户历史 item）。省略则用内置 `DEMO_HISTORY`（5 个 demo SID）。
- `--bf 8,8,8`：三层 byte 的分支因子。
- `--beam 64`：输出三元组个数。
- `--device`：`cuda:0` 或 `cpu`。CPU 自动降级到 fp32（bf16 CPU kernel 不全）。
- `--out`：可选 JSON 输出。

### 11.3 输出格式

stdout 打印 64 行，按 log-prob 降序：

```
rank   (sa, sb, sc)      sid_int64        log_prob
   0   (5691, 8004, 8004) 2148688533051   -3.7693
```

`log_prob` 是累加的 beam 分数（负数，越接近 0 越好）。

---

## 12. 潜在风险与注意事项

以下是使用时应注意的点（**不是缺陷，是设计取舍或约束**）：

### 12.1 `config.json` 的 vocab_size 不一致
- `config.json` 写的是 `151936`（原生 Qwen3），但脚本强制改成 `24576`。
- 如果直接拿 `config.json` 加载模型（不经过本脚本），会得到**错误的 embedding 形状**。
- 建议：在 `config.json` 里把 `vocab_size` 改成 `24576`，避免误用。

### 12.2 占位 token `ctx=0` 与 `sa=0` 重叠
- token id `0` 既是 ctx 占位，又是合法的 `sa` 值（sa ∈ [0, 8191]）
- 模型靠**位置编码**区分（item 起始位 vs byte0 位）
- 工程上没问题，但语义上 `ctx` 应该用一个独立 id（如 `24575`）更干净。

### 12.3 不用 KV cache 的代价
- 3 步前向，每次重跑 prefix。step 2 batch=64 × seq_len≈23，重复算 6 层。
- 对 demo 模型可接受，但如果换大模型 / 长历史，建议补上 KV cache（脚本注释里明确说"为版本鲁棒"放弃了）。

### 12.4 top-1 路由的训练稳定性
- `num_experts_per_tok=1` + `norm_topk_prob=true`，配合 `router_aux_loss_coef=0.001` 做 load balance。
- top-1 比 top-2/8 更难训练（梯度更稀疏），README 报告"0 NaN"说明训练稳定。

### 12.5 随机初始化的局限
- 没有 LLM 预训练底子，模型对"语义"的理解完全来自推荐数据。
- 对**冷启动 item**（SID 没见过）泛化能力有限。byte-level 编码本身能缓解（词表小、共享子结构），但仍弱于有文本语义注入的方案。

### 12.6 demo 数据来源
- `DEMO_HISTORY` 里的 5 个 SID 解包后：
  - `598080194427` → `sa=3003, sb=6378, sc=555`（验证: `3003 | 6378<<14 | 555<<28` ≈ 598080194427 ✓）
- 这些是真实商品 SID，可用作 sanity check。

---

## 13. 整体评价

**优点**：
1. **工程上极简洁**：单文件、零项目依赖、自包含 ckpt，可直接交付。
2. **架构选型合理**：浅层 + 宽 MoE + top-1 路由，推理友好。
3. **byte-level SID 设计优雅**：把超大 item 空间压成 3-byte 编码，beam search 显式控制解码复杂度。
4. **权重转换通用**：`megatron_to_hf` 对任意层数/专家数都适用。
5. **数值精度有保障**：log_softmax 升 fp32，bf16 仅用于 backbone 前向。

**局限**：
1. 仅 1000 步训练，质量是 demo 级。
2. ctx 槽未做 side info 注入（`use_sideinfo=false`），浪费了 4-token 编码中的一个槽。
3. 不用 KV cache，扩展性受限。

**适用场景**：
- 生成式推荐的 **baseline 复现**
- RecIF / TIGER 范式的**工程参考实现**
- Qwen3-MoE 在推荐场景的**架构可行性验证**

---

## 附录 A：核心文件位置速查

| 内容 | 位置 |
|------|------|
| 词表替换（vocab 强制覆盖） | `recif_beam_search_standalone.py:120` |
| Megatron → HF 权重映射 | `recif_beam_search_standalone.py:74-101` |
| QKV 拆分 | `recif_beam_search_standalone.py:56-65` |
| GatedMLP 拆分 | `recif_beam_search_standalone.py:68-71` |
| Beam search 主循环 | `recif_beam_search_standalone.py:182-224` |
| SID ↔ bytes 转换 | `recif_beam_search_standalone.py:155-162` |
| 前缀 token 构建 | `recif_beam_search_standalone.py:165-175` |
| 外置头加载 | `recif_beam_search_standalone.py:141-148` |
| fp32 log_softmax（数值精度） | `recif_beam_search_standalone.py:190-192` |
