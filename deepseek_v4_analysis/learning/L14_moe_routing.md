# L14 · MoE 路由(Hash / sqrtsoftplus / noaux_tc)

> 单元⑤MoE。MegaMoE/FusedMoE 之前,先要决定**每个 token 去哪些专家**。本篇讲 DSV4 的混合路由策略。

## 🎯 学习目标

学完能回答:
1. DSV4 的 MoE 路由分哪两种?分别在哪些层用?
2. Hash MoE 是什么?为什么浅层用它?
3. `sqrtsoftplus` 是什么打分函数?和 sigmoid/softmax 有何不同?
4. `noaux_tc` + `e_score_correction_bias` 起什么作用?

## 📖 讲解

### 路由总览(`nvidia/model.py:478-698`)

`DeepseekV4MoE.forward` 的路由部分:

```python
router_logits, _ = self.gate(hidden_states)                          # :669  GateLinear
topk_weights, topk_ids = fused_topk_bias(                            # :670  选 topk 专家
    hidden_states, router_logits,
    scoring_func=self.scoring_func,        # "sqrtsoftplus"
    e_score_correction_bias=...,            # noaux_tc 偏置
    topk=self.n_activated_experts,
    hash_indices_table=self.gate.tid2eid,   # Hash MoE 查表
    ...)
```

**两种路由,按层切换**:

| 层 | 路由 | 触发条件 |
|----|------|---------|
| 前 `num_hash_layers` 层 | **Hash MoE** | `extract_layer_index(prefix) < num_hash_layers`(`:530`) |
| 其余层 | **sqrtsoftplus + bias** | 默认 |

### Hash MoE(浅层,`:530-544`)

```python
if is_hash_moe:
    self.gate.tid2eid = nn.Parameter(
        torch.randint(0, n_routed_experts, (vocab_size, num_experts_per_tok), ...))
```

- **不走 router 计算**:直接用 `tid2eid` 查找表,按 **token id** 映射到 `num_experts_per_tok` 个专家
- `fused_topk_bias` 收到 `hash_indices_table` 时走哈希路径(`model.py:681`)

> **为什么浅层用 Hash?** 浅层 token 嵌入区分度低,学一个基于 token id 的固定哈希路由,既高效(免算 router GEMM)又稳定(避免浅层路由训练不稳)。深层区分度高了再改用学习的 sqrtsoftplus。

### sqrtsoftplus(深层,`fused_topk_bias_router.py:223`)

```python
elif scoring_func == "sqrtsoftplus":
    return vllm_topk_softplus_sqrt(..., e_score_correction_bias, ...)
```

V4 专属打分函数,配合 bias 选 topk。配置组合在 `fused_moe/config.py:126` 标注为「Deepseek V4 → sqrtsoftplus + Bias + Normalize」。MegaMoE **仅支持** sqrtsoftplus(`model.py:509-512`)。

> 与 sigmoid/softmax 的区别:sqrtsoftplus 是另一族非线性变换(`√(softplus(·))`),配合 bias 和归一化,平衡负载均衡与路由精度。具体公式在 `vllm_topk_softplus_sqrt` kernel 内。

### noaux_tc + e_score_correction_bias(负载均衡,`:545-549`)

```python
elif getattr(config, "topk_method", None) == "noaux_tc":
    self.gate.e_score_correction_bias = nn.Parameter(torch.empty(n_routed_experts, dtype=torch.float32))
```

- `e_score_correction_bias`:每个专家一个偏置,在 topk 选择时调整专家得分
- **noaux_tc**(no auxiliary loss, top-k with correction):不用辅助损失(auxiliary loss)做负载均衡,而是用这个学习的 bias 动态调节 —— 比传统 aux loss 更轻量、训练更稳

### 其它路由配置(`:500-518`)

- `routed_scaling_factor`:路由后缩放因子
- `norm_topk_prob`:topk 权重是否归一化(renormalize)
- `swiglu_limit`:激活 SiLU 后 clamp 限幅(稳数值)

## 🔑 小结

> DSV4 MoE 路由 = **浅层 Hash MoE**(`tid2eid` 按 token id 查表,免算 router)+ **深层 sqrtsoftplus + noaux_tc bias**(`vllm_topk_softplus_sqrt` + `e_score_correction_bias` 学习偏置做无 aux-loss 负载均衡)。两种路由都经 `fused_topk_bias` 选 topk 专家。

## ✅ 自测

<details>
<summary><b>Q1</b>:为什么不是所有层都用 sqrtsoftplus,而浅层特意换成 Hash?</summary>

浅层 token 嵌入刚从 embedding 来,语义区分度低,学习型 router 难训稳、且 router GEMM 在浅层(序列还长)开销占比大。Hash 路由用固定查表(按 token id),既省 router 计算,又避免浅层路由的不稳定。深层语义丰富了,学习型 sqrtsoftplus 才能发挥精度优势。
</details>

<details>
<summary><b>Q2</b>:<code>noaux_tc</code> 相比传统 auxiliary-loss 负载均衡,优势在哪?</summary>

传统 aux loss 在训练目标里加一项「鼓励专家负载均匀」的损失,需要额外超参权衡、且梯度干扰主任务。noaux_tc 用一个**学习的 bias**(`e_score_correction_bias`)在推理/训练的 topk 选择阶段直接调节专家得分 —— 负载偏的专家被 bias 压低、负载轻的被抬高,实现动态均衡,无需额外 loss 项,训练更稳、更轻量。
</details>

**下一步**:L14 选好专家。L15 讲选中的 token 怎么过专家计算(MegaMoE 的 fp8×fp4 GEMM)。
