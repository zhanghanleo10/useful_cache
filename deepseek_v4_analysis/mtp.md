# MTP — Multi-Token Prediction(投机解码)

> 实现:`vllm/vllm/models/deepseek_v4/nvidia/mtp.py`(AMD/XPU 各有同名 `mtp.py`)
> Draft 结构 + target 协同:`nvidia/model.py` 的 `_mtp_hidden_buffer` / `get_mtp_target_hidden_states`

## 作用

投机解码(speculative decoding):target 模型每步额外预测后续若干 token(draft),验证后一次接受多个 token,降低解码延迟。V4 的 draft 复用主模型的 decoder 层结构 + 自己的输入投影 + 词表头。

## 类层次

```
DeepSeekV4MTP(nn.Module)                                # nvidia/mtp.py:260
└── .model: DeepSeekV4MultiTokenPredictor               # :170
    ├── .layers: ModuleDict[str → DeepSeekV4MultiTokenPredictorLayer]  # :188
    │     (key = num_hidden_layers + i,共 num_nextn_predict_layers 个)
    ├── .embed_tokens: VocabParallelEmbedding           # :202
    └── .logits_processor

DeepSeekV4MultiTokenPredictorLayer(nn.Module)           # :68
├── .enorm / .hnorm: RMSNorm                           # :84-85
├── .e_proj / .h_proj: ReplicatedLinear(hidden→hidden)  # :89-104  ★ V4 分离(V3 是 fused eh_proj)
├── .hc_head_fn / .hc_head_base / .hc_head_scale         # 超压缩词表投影参数(:109-120)
├── .shared_head: SharedHead                            # :122  (复用主模型 SharedHead)
└── .mtp_block: DeepseekV4DecoderLayer                  # :125  ★ 复用主 decoder 层(含 attn+MoE+MHC)
```

> `mtp_start_layer_idx = config.num_hidden_layers`(`mtp.py:174`),draft 层逻辑上接在主模型层数之后。

## 前向数据流

### Draft forward(`mtp.py:132-167`)

```
forward(input_ids, positions, previous_hidden_states, inputs_embeds, spec_step_idx)
│
├─ previous_hidden_states = prev.view(-1, hc_mult, hidden_size)        # :144  恢复多流布局
├─ inputs_embeds, prev = fused_mtp_input_rmsnorm(                      # :148
│       inputs_embeds, positions, prev, enorm_w, hnorm_w, eps, hc_mult)
│     # 融合:position 0 mask + enorm + hnorm
│
├─ hidden = h_proj(prev) + e_proj(inputs_embeds).unsqueeze(-2)         # :157  ★ 双输入投影
│     # e_proj 输出 unsqueeze 到 hc_mult 流维度;h_proj 处理上一 token 的 hidden
│
├─ hidden, residual, post_mix, res_mix = mtp_block(positions, hidden)  # :160  复用 DecoderLayer
├─ hidden = mhc_post_tilelang(hidden, residual, post_mix, res_mix)     # :163  收尾混合
└─ return hidden.flatten(1)                                            # :167  返回 pre-hc_head residual
```

> **返回的是 pre-hc_head 的扁平 residual** `(T, hc_mult * hidden_size)`,而不是 logits。这样多 spec step 时可直接喂给下一 step 的 `previous_hidden_states`(`mtp.py:163-167` 注释)。

### compute_logits(`mtp.py:231-257`)

```
compute_logits(hidden_states, spec_step_idx)
│
├─ hidden = hidden.view(-1, hc_mult, hidden_size)                      # :240
├─ hidden = hc_head_fused_kernel_tilelang(hidden, hc_head_*, eps)      # :243  ★ 超压缩折叠
├─ hidden = mtp_shared_head_rmsnorm(hidden, shared_head.norm)          # :251
└─ logits = logits_processor(shared_head.head, hidden)                 # :256
```

## V4 vs V3 的 MTP 差异

| 维度 | V3 | V4 |
|------|----|----|
| 输入投影 | fused `eh_proj`(单矩阵) | **分离** `e_proj` + `h_proj`(`mtp.py:89-104`),各带 FP8 量化 |
| 词表投影 | 直接 shared head | **`hc_head` 超压缩投影**(`hc_head_fn/scale/base`)再过 shared head |
| Decoder 块 | V3 decoder layer | **复用 `DeepseekV4DecoderLayer`**(含 Sparse MLA + MoE + MHC 混合) |
| 多流 | 无 | hidden 维度带 `hc_mult` 流,需 view/flatten 转换 |

## 与 Target 模型的协同

Target(`DeepseekV4ForCausalLM`)在 forward 末尾 stash 一份 **pre-hc_head residual** 供 draft 使用:

- `_mtp_hidden_buffer`:`(max_num_batched_tokens, hc_mult * hidden_size)`,仅末 PP rank 分配(`nvidia/model.py:974-981`)
- 填充:`model.py:1046` —— `hc_head` 折叠**之前**的 residual `flatten(1)` copy 进去
- 读取:`get_mtp_target_hidden_states()`(`model.py:1327`)返回该 buffer
- 用 cudagraph 时 buffer 地址固定(`model.py:969-973` 注释),跨 capture shape 的 `copy_` 仍正确刷新

## 权重加载(`mtp.py:293-480`)

- **层索引重写**:checkpoint 的 `mtp.{i}.*` → `model.layers.{num_hidden_layers + i}.*`(`mtp.py:366-369`),再经 `get_spec_layer_idx_from_weight_name` 识别
- **`_rewrite_spec_layer_name`**(`mtp.py:486`):区分「层块内权重」(加 `.mtp_block`)vs「顶层共享权重」(`embed_tokens` 等)
- checkpoint 名映射:`.emb.tok_emb.weight`→`.embed_tokens.weight`、`.head.weight`→`.shared_head.head.weight`、`.norm.weight`→`.shared_head.norm.weight`(`mtp.py:296-300`)
- **缺失校验**(`mtp.py:466-477`):任一 MTP 层权重缺失则报错,提示「检查点可能未含量化后的 MTP 权重,或禁用投机解码」
- expert scale 后缀按 `expert_dtype` 选 `.weight_scale`(fp4)或 `.weight_scale_inv`(fp8),与主模型一致(`mtp.py:355-359`)
- e8m0 scale 同样 `.view(uint8)` 保护(`mtp.py:406`)

## 数量与步进

- `num_mtp_layers = config.num_nextn_predict_layers`(`mtp.py:175`)
- `num_speculative_tokens > 1` 时,draft 按步复用层:`current_step_idx = spec_step_idx % num_mtp_layers`(`mtp.py:222`),选对应 `layers[str(start + current_step_idx)]`
