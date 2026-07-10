# L15 · MegaMoE 专家计算(fp8×fp4 DeepGEMM)

> 单元⑤MoE(专家计算)。L14 选好 topk 专家;本篇讲 token 怎么过这些专家。MegaMoE 是 SM100(Blackwell)专用的高吞吐 fp8 激活 × fp4 权重 MoE kernel。

## 🎯 学习目标

学完能回答:
1. MegaMoE 的「fp8×fp4」指什么?为什么这样混精度?
2. 为什么 MegaMoE 必须 SM100 + EP?
3. `finalize_weights` / `transform_weights_for_mega_moe` 做什么?
4. EPLB(Expert-Level Load Balancing)解决什么问题?

## 📖 讲解

### 定位(`nvidia/model.py:490-498`)

```python
self.use_mega_moe = (vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe")
if self.use_mega_moe and not enable_expert_parallel:
    raise NotImplementedError("DeepSeek V4 MegaMoE currently requires expert parallel.")
```

两种专家后端:
- **MegaMoE**(`DeepseekV4MegaMoEExperts`,`:140`):SM100 专用,fp8×fp4,必须 EP
- **FusedMoE**(`:639`):通用,支持 MXFP4/NVFP4/FP8

### fp8×fp4 混合精度(`model.py:464`)

```python
deep_gemm.fp8_fp4_mega_moe(
    y, l1_weights, l2_weights, symm_buffer,
    activation_clamp=activation_clamp, fast_math=True)
```

- **激活(fp8)**:token hidden 用 FP8(E4M3)传进来
- **权重(fp4)**:专家权重用 MXFP4(4-bit),scale 是 ue8m0(e8m0fnu)

> 为什么混:权重占显存大头(专家多),用 fp4 省一半显存 + 减半访存带宽;激活对精度更敏感,保留 fp8。两者都低精度 → 算力/带宽双省。SM100 有原生 fp4 Tensor Core 支持,这是 MegaMoE 必须 SM100 的根本原因。

### 权重布局与 finalize(`model.py:174-334`)

加载时权重是 uint8 + ue8m0 scale(`:174-218`),运行前要重排:

```python
def finalize_weights(self):
    w13_scale = deep_gemm.transform_sf_into_required_layout(ue8m0→fp32, ...)   # scale 重排
    w2_scale  = deep_gemm.transform_sf_into_required_layout(...)
    (l1, l2) = deep_gemm.transform_weights_for_mega_moe(                       # 权重重排
        (w13.view(int8), w13_scale), (w2.view(int8), w2_scale))
    self._transformed_l1_weights, self._transformed_l2_weights = (l1, l2)
    self.w13_weight = None  # 丢弃原始参数,kernel 只用 transformed 版本
```

> ue8m0 → fp32 还原:`_ue8m0_uint8_to_float(sf) = (sf << 23).view(fp32)`(`:282-284`)。

### symm_buffer(跨层复用,`:336-363`)

```python
symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, max_tokens, topk, H, I)
```

按 `(group, device, num_experts, max_tokens, topk, hidden, intermediate)` 缓存,同配置的层共享 → 省 alloc。

### EPLB(Expert-Level Load Balancing,`:365-409,434-446`)

```python
# 运行时:logical expert id → physical slot 映射
if eplb_state.logical_to_physical_map is not None:
    topk_ids = eplb_map_to_physical_and_record(
        topk_ids, expert_load_view, logical_to_physical_map, logical_replica_count, ...)
```

| 概念 | 说明 |
|------|------|
| logical experts | `n_routed_experts`(模型定义的专家) |
| physical experts | `n_logical + num_redundant_experts`(含副本) |
| logical→physical | 把**热专家复制**到多个物理 slot,EP 下均匀分布 |

> 解决「专家负载不均」:不同专家被选次数差异大,热专家成瓶颈。EPLB 动态把热专家复制到冗余 slot,让 EP 各 rank 负载均衡。

### shared experts 叠加(`:694-696`)

```python
final_hidden_states = self.experts(hidden_states, topk_weights, topk_ids, activation_clamp=...)
if self.shared_experts is not None:
    final_hidden_states += self.shared_experts(hidden_states)   # 共享专家所有 token 都过
```

### 限制(`:509-518`)

MegaMoE 仅支持:**fp4 专家 + sqrtsoftplus 路由**。不满足会报错,提示换 backend。

## 🔑 小结

> MegaMoE(SM100+EP)= **fp8 激活 × fp4(MXFP4)权重**(DeepGEMM `fp8_fp4_mega_moe`)+ **`finalize_weights` 把 ue8m0/uint8 权重重排成 kernel 布局** + **symm_buffer 跨层复用** + **EPLB 动态复制热专家均衡 EP 负载** + **shared experts 叠加**。仅 fp4 + sqrtsoftplus。

## ✅ 自测

<details>
<summary><b>Q1</b>:为什么权重用 fp4 而激活用 fp8,不都用 fp4 或都用 fp8?</summary>

权衡显存/带宽与精度:① **权重**数量大(几百专家 × 大矩阵),用 fp4 省一半显存 + 减半访存,收益巨大;② **激活**对精度更敏感(每层都要累加,误差累积),用 fp8 保精度。混精度在「省权重带宽」和「保激活精度」间取最优。SM100 原生支持 fp4 Tensor Core 才让 fp4 权重成为可能(老架构无 fp4 硬件,只能软件模拟,反而更慢)。
</details>

<details>
<summary><b>Q2</b>:EPLB 把热专家复制到多个物理 slot,这不会让结果出错吗(同一逻辑专家算多遍)?</summary>

不会出错,正是设计意图。「复制」意味着同一个逻辑专家的**权重**出现在多个物理 slot 上,EP 下不同 rank 各持一份。某个 token 被路由到这个逻辑专家时,`logical_to_physical_map` 把它映射到**其中一个**物理副本(选负载最轻的那个),只算一次。多副本是为了**分散访问压力**(不同 token 的请求分到不同副本),不是同一 token 算多遍。结果语义等价于单副本,只是吞吐更高。
</details>

---

🎉 **L1-L15 全部完成**。回到 [`README.md`](README.md) 复习依赖图,或挑感兴趣的单元重读。配套分析文档见 [`../`](../)。
