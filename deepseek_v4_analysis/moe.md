# MoE 体系(Mixture of Experts)

> 实现:`vllm/vllm/models/deepseek_v4/nvidia/model.py`(`DeepseekV4MoE` / `DeepseekV4MegaMoEExperts` / `DeepseekV4MLP`)
> 路由:`vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`

V4 的 MoE 有三个看点:**两种后端**(MegaMoE / FusedMoE)、**混合路由**(Hash 浅层 + sqrtsoftplus 深层)、**EPLB 动态负载均衡**。

## `DeepseekV4MoE` 结构(`nvidia/model.py:478`)

```
DeepseekV4MoE
├── .gate: GateLinear(hidden → n_routed_experts, fp32 out)        # :520
├── .shared_experts: DeepseekV4MLP | None                          # :551  共享专家
└── .experts:
     ├── use_mega_moe → DeepseekV4MegaMoEExperts                   # :601  SM100 专用
     └── else         → FusedMoE                                    # :639  通用
```

**关键开关**:`use_mega_moe = (kernel_config.moe_backend == "deep_gemm_mega_moe")`(`model.py:490`)。MegaMoE **必须开 EP**(`model.py:493`),否则报错。

**路由配置**(`model.py:500-518`):
- `scoring_func = "sqrtsoftplus"`(默认,`config.py` 读取;MegaMoE 仅支持此函数)
- `swiglu_limit`:SiLU + clamp 激活限幅
- `routed_scaling_factor`、`norm_topk_prob`(renormalize)
- MegaMoE 强制 `expert_dtype == "fp4"`

## 路由:Hash + sqrtsoftplus 混合

### Hash MoE(浅层,`model.py:530-544`)

前 `num_hash_layers` 层**不走 router 计算**,直接用 `tid2eid` 查找表按 token id 映射专家:

```python
is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers   # :530
if is_hash_moe:
    self.gate.tid2eid = nn.Parameter(
        torch.randint(0, n_routed_experts, (vocab_size, num_experts_per_tok), ...))  # :536
```

> 动机:浅层 token 嵌入区分度低,学一个基于 token id 的固定哈希路由更高效且稳定。`fused_topk_bias` 收到 `hash_indices_table` 时走哈希路径(`model.py:681`)。

### sqrtsoftplus(深层,`fused_topk_bias_router.py:223`)

深层用新的打分函数:

```python
elif scoring_func == "sqrtsoftplus":
    return vllm_topk_softplus_sqrt(..., e_score_correction_bias,
                                   input_tokens, hash_indices_table, routed_scaling_factor)
```

配合 **noaux_tc** topk method + `e_score_correction_bias`(`model.py:545-549`)做辅助负载均衡。配置组合在 `fused_moe/config.py:126` 标注为「Deepseek V4 → sqrtsoftplus + Bias + Normalize」。

## 后端一:MegaMoE(`DeepseekV4MegaMoEExperts`,`model.py:140`)

SM100(Blackwell)专用,用 DeepGEMM 的 **fp8 激活 × fp4 权重**混合精度专家 GEMM:

```python
deep_gemm.fp8_fp4_mega_moe(y, l1_weights, l2_weights, symm_buffer,
                           activation_clamp=..., fast_math=...)   # model.py:464
```

| 要点 | 说明 | 代码 |
|------|------|------|
| **硬件** | 必须 SM100(`get_device_capability()[0]==10`) | `model.py:288-289` |
| **并行** | 必须 EP(`model.py:493`);`n_physical_experts % ep_size == 0` | `model.py:587` |
| **权重布局** | w13/w2 存 uint8,ue8m0 scale;`finalize_weights` 调 `transform_weights_for_mega_moe` 重排 | `model.py:296-334` |
| **ue8m0 解码** | `_ue8m0_uint8_to_float`:`(sf << 23).view(fp32)` | `model.py:282-284` |
| **symm_buffer** | 跨层复用的对称缓冲(按 (group,device,E,T,k,H,I) 缓存) | `model.py:336-363` |
| **限制** | 仅 fp4 专家 + sqrtsoftplus 路由(`model.py:509-518`) | — |

**EPLB 集成**(`model.py:365-409`):`get_expert_weights` 返回 EPLB 视图(`_to_eplb_view` 把权重/scale 重排成 `(num_local_experts, -1)` 连续布局),供 EPLB 重平衡使用。`update_expert_map` 当前是空实现(`model.py:408`,预留)。

## 后端二:FusedMoE(`model.py:639`)

通用后端,支持 MXFP4/NVFP4/FP8 专家(由 `DeepseekV4FP8Config.get_quant_method` 派发,见 [quantization.md](quantization.md))。把 hash table、bias、scoring 等都透传给 `FusedMoE` 构造(`model.py:655-657`)。

## EPLB — Expert-Level Load Balancing

| 概念 | 说明 | 代码 |
|------|------|------|
| logical experts | `n_routed_experts`(模型定义的专家数) | `model.py:585` |
| physical experts | `n_logical + num_redundant_experts`(含副本) | `model.py:586` |
| logical→physical | 运行时映射,把热专家复制到多个物理 slot | `EplbLayerState`(`model.py:171`) |
| `set_eplb_state` | 注入 `(moe_layer_idx, expert_load_view, logical_to_physical_map, logical_replica_count)` | `model.py:365` |
| 运行时映射 | `eplb_map_to_physical_and_record`:把 topk 的逻辑 ID 转物理 ID,并记录负载 | `model.py:440` |

> 动机:不同专家负载不均,EPLB 动态把高负载专家复制(replicate)到冗余物理 slot,EP 下均匀分布计算。`DeepseekV4MixtureOfExperts`(`model.py:1214`)聚合所有 MoE 层的专家元信息,支持运行时调整物理专家数(`update_physical_experts_metadata`,`model.py:1235`)。

## swiglu_limit(SiluAndMulWithClamp)

`DeepseekV4MLP`(`model.py:70`)的激活:`SiluAndMulWithClamp(swiglu_limit)`(`model.py:110`)—— SiLU 门控后对输出做 clamp 限幅。MegaMoE 把 limit 透传为 `activation_clamp`(`model.py:684,469`)。动机:控制激活幅值,稳定 FP4/FP8 低精度下的数值范围。

## Shared Experts(`model.py:551-564`)

`n_shared_experts` 个共享专家(所有 token 都过),合成一个 `DeepseekV4MLP`(`intermediate = moe_intermediate_size * n_shared_experts`)。MegaMoE 路径下 `reduce_results=True`(`model.py:562`),输出在专家侧累加:`final = experts(...) + shared_experts(...)`(`model.py:694-696`)。

## 前向(`model.py:659-698`)

```
forward(hidden_states, input_ids)
├─ 非 mega_moe → _forward_fused_moe(...)              # :665,700  FusedMoE 标准
└─ mega_moe:
   ├─ router_logits = gate(hidden_states)             # :669
   ├─ topk_weights, topk_ids = fused_topk_bias(...)   # :670  sqrtsoftplus/hash 选 topk
   ├─ activation_clamp = float(swiglu_limit)          # :684
   ├─ final = experts(hidden, topk_weights, topk_ids, activation_clamp)  # :687 DeepGEMM
   └─ if shared_experts: final += shared_experts(hidden)  # :694
```

## 一句话心智模型

> V4 MoE = **浅层用 Hash 路由**(免算 router,按 token id 查表)+ **深层用 sqrtsoftplus + bias 路由**(新打分函数 + 负载均衡)+ **MegaMoE(fp8×fp4)/ FusedMoE(MXFP4 等)双后端**(按硬件/量化选)+ **EPLB 动态复制热专家**(EP 下负载均衡)+ **swiglu_limit 稳数值**。一切围绕「路由准、专家算得快、负载匀」。
