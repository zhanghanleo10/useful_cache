# 量化体系

> 配置:`vllm/vllm/models/deepseek_v4/quant_config.py`(`DeepseekV4FP8Config`)
> KV cache:`vllm/vllm/models/deepseek_v4/attention.py`(`_resolve_dsv4_kv_cache_dtype`)
> 权重加载:`vllm/vllm/models/deepseek_v4/nvidia/model.py:1059`(主)、`nvidia/mtp.py:293`(MTP)

V4 是**全链路量化**架构:Linear/Attention FP8、MoE 专家 MXFP4/NVFP4/FP8、KV cache UE8M0/bf16/fp8、Indexer cache FP8/MXFP4。

## `DeepseekV4FP8Config`(`quant_config.py:29`)

继承自通用 `Fp8Config`,核心是**按 `expert_dtype` 派发 MoE 量化方法**。关键设计:`expert_dtype` **延迟解析**(lazy resolve)。

### 为什么延迟解析(`quant_config.py:43-78`)

> `DeepseekV4FP8Config` 在 `VllmConfig` setup 阶段就被构造,那时 `set_current_vllm_config` 还没激活。若在 `__init__` 里急切读 `hf_config.expert_dtype`,会永远看到默认值 `"fp4"`,**错误地把 Flash-Base(fp8)检查点当成 fp4 路由**。

实现:`expert_dtype` / `is_scale_e8m0` / `moe_quant_algo` 都是 `@property`,首次读取时从 `get_current_vllm_config().model_config.hf_config` 解析并缓存。

### 两种 expert_dtype

| `expert_dtype` | 示例检查点 | 专家量化 | linear scale dtype | MoE 方法 |
|----------------|-----------|---------|--------------------|---------|
| `"fp4"`(默认) | DeepSeek-V4-Flash | **MXFP4** | **ue8m0(e8m0fnu)** | `Mxfp4MoEMethod`,或 `moe_quant_algo=="NVFP4"` 时 `ModelOptNvFp4FusedMoE` |
| `"fp8"` | DeepSeek-V4-Flash-Base | **FP8 block** | **float32** | `Fp8MoEMethod`(block_quant,父类 `Fp8Config.get_quant_method`) |

派发逻辑(`quant_config.py:134-155`):

```python
def get_quant_method(self, layer, prefix):
    if isinstance(layer, RoutedExperts):
        if is_layer_skipped(...): return UnquantizedFusedMoEMethod(...)
        if self.expert_dtype == "fp4":
            if self.moe_quant_algo == "NVFP4":
                return ModelOptNvFp4FusedMoE(...)          # NVFP4(ModelOpt,group_size=16)
            return Mxfp4MoEMethod(layer.moe_config)        # MXFP4(默认)
        # expert_dtype == "fp8": 落到 Fp8Config → Fp8MoEMethod(block, fp32 scale)
    return super().get_quant_method(layer, prefix)
```

## 量化层级总览

| 层 | 量化 | scale | 参数名 |
|----|------|-------|--------|
| Linear / Attention(非专家) | **FP8 block** | float32 | `weight_scale_inv` |
| Shared experts(DeepseekV4MLP) | FP8 block | float32 | `weight_scale_inv` |
| MoE 专家(fp4) | **MXFP4** | ue8m0(e8m0fnu) | `w{123}_weight_scale` |
| MoE 专家(fp4 + NVFP4) | **NVFP4** | group_size=16 | (ModelOpt) |
| MoE 专家(fp8) | **FP8 block** | float32 | `w{13,2}_weight_scale_inv` |
| KV cache(FlashMLA/ROCm) | `fp8_ds_mla`:**UE8M0** block-scaled fp8 打包 uint8 | — | — |
| KV cache(FlashInfer) | bf16 或 per-tensor fp8(e4m3) | — | — |
| Indexer cache | FP8(132B/token)或 **MXFP4** | — | — |

## KV cache 三格式(`attention.py:64-95`)

`_resolve_dsv4_kv_cache_dtype(use_flashmla_fp8_layout, kv_cache_dtype, cache_config)`:

| `use_flashmla_fp8_layout` | `--kv-cache-dtype` | 结果 dtype | cache 字符串 |
|:---:|---|---|---|
| True(FlashMLA/ROCm) | `fp8*` | `torch.uint8` | `fp8_ds_mla`(自动改写,576B 对齐) |
| False(FlashInfer) | `fp8*` | `torch.float8_e4m3fn` | 原 fp8 串(per-tensor) |
| False(FlashInfer) | auto/bfloat16 | `torch.bfloat16` | 原 bf16 |

- FlashMLA fp8 必须以 `fp8` 开头,否则自动改写 `cache_config.cache_dtype = "fp8_ds_mla"` 并 `info_once` 提示(`attention.py:84-88`)。
- 主 MLA cache 每 token **584B** = 448 NoPE + 128 RoPE + 8 fp8 scale(`sparse_mla.py:102`)。

## Indexer cache(`attention.py:723-734`)

`k_cache_head_dim = head_dim + head_dim//quant_block_size * 4 = 128 + 4 = 132`(128 fp8 + 4B fp32 scale)。FP4 indexer cache 占同样内存但只用前半:

```python
use_fp4_kv = vllm_config.attention_config.use_fp4_indexer_cache   # attention.py:686
# MXFP4_BLOCK_SIZE = 32(sparse_attn_indexer.py:37)
```

## 权重 scale 后缀映射

checkpoint 里所有 scale 都叫 `.scale`,加载时按上下文重命名:

| checkpoint | 模型参数 | 条件 | 代码 |
|-----------|---------|------|------|
| `experts.{i}.w{123}.scale` | `.w{123}_weight_scale` | **fp4 专家**(Mxfp4MoEMethod) | `model.py:1183-1186` |
| `experts.{i}.w{123}.scale` | `.w{13,2}_weight_scale_inv` | **fp8 专家**(Fp8MoEMethod block) | `model.py:1188-1193` |
| 其它 `.scale` | `.weight_scale_inv` | FP8 block linear / shared experts | `model.py:1185-1186` |

MTP 加载同样逻辑(`mtp.py:380-386, 410-413`)。

## e8m0 scale 的特殊处理

ue8m0(e8m0fnu)scale 在 checkpoint 是 `float8_e8m0fnu`,模型参数是 `uint8`。**不能直接 `copy_()`** —— 会做数值转换(如 2^-7 → 0)破坏原始指数字节:

```python
# model.py:1106-1109
if "weight_scale" in name and loaded_weight.dtype == torch.float8_e8m0fnu:
    loaded_weight = loaded_weight.view(torch.uint8)   # 当原始字节处理
```

MegaMoE 运行时再把 uint8 还原成 fp32:`_ue8m0_uint8_to_float(sf) = (sf << 23).view(fp32)`(`model.py:282-284`)。

## NVFP4 路径(`quant_config.py:102-114`)

当 `moe_quant_algo == "NVFP4"`(从 `hf_config.quantization_config.moe_quant_algo` 读)时,fp4 专家走 ModelOpt 的 NVFP4(`ModelOptNvFp4Config`,`group_size=16`,`is_checkpoint_nvfp4_serialized=True`)。这是 fp4 专家的第二条路径(MXFP4 之外的可选)。

## 量化方法注册

- `get_name() → "deepseek_v4_fp8"`(`quant_config.py:117`)
- `override_quantization_method`(`quant_config.py:120`):识别 `model_type == "deepseek_v4"` 或用户显式 `--quantization deepseek_v4_fp8`,把通用 `fp8` 量化方法覆盖为 `deepseek_v4_fp8`(以便走 V4 专属的 expert_dtype 派发)

## 一句话心智模型

> V4 量化 = **Linear/Attn 一律 FP8 block** + **MoE 专家按 `expert_dtype` 选 MXFP4 / NVFP4 / FP8**(ue8m0 vs fp32 scale)+ **KV cache 三格式**(`fp8_ds_mla` UE8M0 / bf16 / per-tensor fp8,按 attention backend 选)+ **Indexer cache FP8/MXFP4**。工程难点在 `expert_dtype` 延迟解析、`.scale` 后缀按专家类型重命名、e8m0 字节级 scale 保护。
