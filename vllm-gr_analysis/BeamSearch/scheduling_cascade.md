# 调度与 Cascade Attention 精读

**版本**: 1.0
**状态**: 基于代码逐行核对(engine_core_patch.py / beam_attn.py)
**配套**: 在线链路见 `online_pipeline.md`,概览见 `beam_search_walkthrough.md`

本文档覆盖子请求进 `input_queue` 之后的**调度与计算**阶段:`beam_prefix_groups` 的诞生与跨层传递 + Cascade Attention 三阶段。这是"共享前缀"思想在**计算层**的落地。

---

## 位置:连接通信链路与计算层

第❶–❽章(见 `online_pipeline.md`)解决了 **KV cache 存储的共享**(template-clone + beam_cache)。但 attention 计算本身还没省——多个 beam 共享前缀,标准 attention 会把前缀**重复算 N 次**。这一章就是为"前缀只算一次"做准备 + 落地。

```
子请求进 input_queue(❽)
  → Scheduler.schedule()         ← 阶段A:算出 beam_prefix_groups
  → GPUModelRunner.execute_model  ← 阶段B:取出 groups 存 self
  → _build_attention_metadata     ← 阶段B:翻译 req_id→idx,塞进 ContextVar
  → BeamAttentionMetadataBuilder  ← 阶段C:读 ContextVar,构建 prefix/suffix metadata
  → cascade_attention             ← 阶段C:前缀算一次,LSE 合并
```

---

## 阶段A · Scheduler 注入

**文件**: `engine_core_patch.py`(`_get_lcp:20-50`、`_compute_beam_prefix_groups:53-103`、`apply_scheduler_patch:348-429`)

### `apply_scheduler_patch` 做两件事

#### 优化1:`get_computed_blocks` 缓存

```python
def patched_schedule(self):
    computed_blocks_cache = {}
    _original = self.kv_cache_manager.get_computed_blocks

    def get_cache_computed_blocks(req):
        cache_th = 256
        last_hash = req.block_hashes[hash_idx] if hash_idx >= 0 else req.request_id
        key = (req.num_tokens, req.skip_reading_prefix_cache, last_hash)    # 缓存 key
        if key == last_key and is_cache_worthy(last_res):  res = last_res   # ① 上一个同 key → 复用
        elif key in computed_blocks_cache:                  res = cache[key] # ② 缓存命中
        else:                                                              # ③ 算并缓存(命中>256 tokens 才存)
            res = _original(req)
            if is_cache_worthy(res): computed_blocks_cache[key] = res

    self.kv_cache_manager.get_computed_blocks = get_cache_computed_blocks   # 临时替换
    try: output = _original_schedule(self)
    finally: self.kv_cache_manager.get_computed_blocks = _original          # 恢复
```

**逻辑**:同 beam 组请求共享前缀、同长度 → `last_hash` 相同 → key 相同 → 第一个算完缓存,后续命中。**阈值 256**:命中超 256 tokens 才缓存(避免缓存污染)。

#### 优化2:计算 `beam_prefix_groups`

```python
req_ids = list(output.num_scheduled_tokens.keys())
output.beam_prefix_groups = _compute_beam_prefix_groups(req_ids, self.requests, self.cache_config.block_size)
```
schedule 算完后,把 `beam_prefix_groups` **动态附加**到 `SchedulerOutput`(原生没有的字段)。

### `_compute_beam_prefix_groups`(`53-103`)

```python
min_group_th = 256 // block_size     # 阈值:至少共享这么多 block 才值得分组

# 1. 按 priority(beam 组身份)分组
beam_groups = {}
for req_id in req_ids:
    g = requests[req_id].priority    # priority = 在线链路的微秒时间戳
    beam_groups.setdefault(g, []).append(req_id)

# 2. 每组算 min_lcp
for g, group_req_ids in beam_groups.items():
    min_lcp = inf
    for i in range(n-1):
        common_len = _get_lcp(req1.block_hashes, req2.block_hashes, hint=last_lcp)
        min_lcp = min(min_lcp, common_len)
        if min_lcp < min_group_th: min_lcp = 0; break    # 太短不分组
    all_groups.append((min_lcp, group_req_ids))
```

**产出**:`[(depth, [req_ids]), ...]`。depth = 这组请求共享的 block 数(取 min_lcp 保证不越界)。

**三关键点**:
1. 按 priority 分组(同一 beam search 的请求同 priority)。
2. min_lcp(组内所有相邻请求对 LCP 取最小 = 整组安全共享深度)。
3. 阈值过滤(min_lcp < 16 → depth=0,不分组)。

### `_get_lcp`(`20-50`)hint 优化

二分搜索找两个 hash list 的最长公共前缀。beam search 相邻步 LCP 变化小,用上一次 LCP 当 hint,先查 hint 附近再二分。还有两处快速路径:`m==0` 早返回、`a[end]==b[end]` 尾端捷径。

### ⚠️ 分组粒度:按 priority,不混算

一次 schedule 的 batch 是**混合的**(可能含多个 beam search + 普通请求),但 `_compute_beam_prefix_groups` 按 priority 自动分开:
- 同一 beam search 的 beam → 同 priority → 同组
- 不同 beam search → 不同 priority → 不同组
- 普通请求 → priority=0 → depth=0(单请求直接 0)

即使同 priority,还要 min_lcp 过阈值才真正 cascade(分叉太早 → depth=0)。

**两套机制分工**:
| 共享场景 | 谁处理 | 层次 |
|---|---|---|
| 同一 beam search 内的 beam | beam_prefix_groups → cascade | 计算层 |
| 不同请求/不同 beam search 碰巧同前缀 | 原生 prefix caching | 存储层 |

---

## 阶段B · Worker 注入(ContextVar 跨层传递)

**文件**: `engine_core_patch.py:apply_worker_patches:432-479`

`beam_prefix_groups` 挂在 `SchedulerOutput` 上,但 attention builder 拿不到(原生调用链不通)。所以 patch GPUModelRunner 两个方法,用 **`self` + ContextVar** 串联:

```python
def patched_execute_model(self, scheduler_output, intermediate_tensors=None):
    beam_groups = getattr(scheduler_output, "beam_prefix_groups", None)
    self._current_beam_prefix_groups_str = beam_groups      # ① 取出存 self
    try: return _original_execute_model(self, scheduler_output, intermediate_tensors)
    finally: self._current_beam_prefix_groups_str = None

def patched_build_attention_metadata(self, *args, **kwargs):
    beam_groups_str = getattr(self, "_current_beam_prefix_groups_str", None)
    if beam_groups_str is not None:
        req_id_to_idx = {r_id: i for i, r_id in enumerate(self.input_batch.req_ids)}
        translated_groups = []
        for depth, group_req_ids in beam_groups_str:
            indices = [req_id_to_idx[r] for r in group_req_ids if r in req_id_to_idx]   # ② req_id→batch idx
            if indices: translated_groups.append((depth, indices))
        token = BEAM_PREFIX_GROUPS_VAR.set(translated_groups)     # ③ 塞 ContextVar
        try: return _original_build_attention_metadata(self, *args, **kwargs)
        finally: BEAM_PREFIX_GROUPS_VAR.reset(token)
```

**为什么 self + ContextVar 两段**:`execute_model` 和 `_build_attention_metadata` 原生签名都不带 beam_groups,只能旁路传。ContextVar 不改任何原生签名。

---

## 阶段C · Cascade Attention

**文件**: `v1/attention/backends/beam_attn.py`

### 核心思想:prefix 算一次,suffix 各算,LSE 合并

```
标准 attention(前缀重复算):
  beam_A: attend([P | SA])   → 算了一遍 P
  beam_B: attend([P | SB])   → 又算了一遍 P   ← P 被算 N 次!

cascade(前缀只算一次):
  Phase 1: attend(P) 对 [beam_A 的 q, beam_B 的 q] 一起算   → P 只算 1 次
  Phase 2: beam_A attend(SA), beam_B attend(SB)            → suffix 各算
  Phase 3: LSE 合并                                         → 还原完整 attention
```

prefix `causal=False`(历史 KV,query 无因果约束);suffix `causal=True`。

### 为什么必须 LSE 合并(softmax 数学)

attention = `softmax(q·K^T)·V`。拆 K=K_prefix+K_suffix 后**不能**直接相加(各自 softmax 分母不同):

```
完整:  out = (Σexp(p_i)·v_p_i + Σexp(s_j)·v_s_j) / Z,  Z = Z_p + Z_s
拆开:  out_p = Σexp(p_i)·v_p_i / Z_p  (lse_p = log Z_p)
        out_s = Σexp(s_j)·v_s_j / Z_s  (lse_s = log Z_s)
合并:  out = (Z_p/Z)·out_p + (Z_s/Z)·out_s
         = [exp(lse_p)/(exp(lse_p)+exp(lse_s))]·out_p + [exp(lse_s)/(...)]·out_s
```

flash_attn 的 `return_softmax_lse=True` 就是为拿每部分 LSE 供合并。

### build 流程(`350-377`)

```python
beam_prefix_groups = BEAM_PREFIX_GROUPS_VAR.get()
if beam_prefix_groups is None:
    return self._build_standard_metadata(...)                    # 无 groups → 标准
shared_groups = [g for g in beam_prefix_groups if g[0] > 0]      # 过滤 depth=0
shared_groups = self._filter_groups_by_query_len(...)            # 过滤 query_len 超长(>2 blocks)
if not shared_groups:
    return self._build_standard_metadata(...)
return self._build_shared_metadata(common_meta, shared_groups, ...)
```

`_filter_groups_by_query_len`(`435-468`):只留 query_len 在 `(0, block_size*2]` 的(beam search 每步 query 很短)。

### 两个核心数据结构

**`_create_prefix_metadata`(`583-681`)** 对每个 group `(depth, [req_indices])`:
- `prefix_block_table` = `block_table[组内第一个请求][:depth]`(`654-656`)——全组共享的前缀,取任一代表。
- `prefix_kv_lens = depth × block_size`(`589`)。
- `prefix_indices`(`612-620`):scatter map,把同组多请求的 query 按组拼一起 attend 共享 prefix。identity 时免 gather。
- `cu_prefix_query_lens`(`636-640`):每组累积 query 长度(varlen 格式)。
- `shift_amounts`(`664-668`):每请求的 prefix block 数(给 suffix 用)。

**`_create_suffix_metadata`(`683-740`)**:
- `suffix_kv_lens = seq_lens - shift_amounts × block_size`(`691-693`)。
- `suffix_block_table`:`torch.gather` + 列偏移(`726-732`),`col_indices = arange + shift_amounts`。**注意是 gather 不是 slice**(原文档勘误点)。

### 三阶段 `cascade_attention`(`912-1020`)

```python
# Phase 1: Prefix(共享 blocks,non-causal)
if has_work:
    prefix_query = query if prefix_indices_is_identity else query[prefix_indices]   # 同组 query 拼一起
    prefix_out, prefix_lse = flash_attn_varlen_func(
        q=prefix_query, cu_seqlens_q=cu_prefix_query_lens, seqused_k=prefix_kv_lens,
        block_table=prefix_block_table, causal=False, return_softmax_lse=True)

# Phase 2: Suffix(各自 blocks,causal,直接写 output)
suffix_out, suffix_lse = flash_attn_varlen_func(
    q=query, cu_seqlens_q=cu_suffix_query_lens, seqused_k=suffix_kv_lens,
    block_table=suffix_block_table, causal=True, return_softmax_lse=True, out=output)

# Phase 3: Merge(LSE 归一化)
if prefix_out is not None:
    if prefix_indices_is_identity:
        merge_attn_states(output, prefix_out, prefix_lse, suffix_out, suffix_lse)        # 原生
    else:
        merge_indexed_attn_states(output, suffix_lse, prefix_out, prefix_lse, prefix_indices)  # 自研 triton
```

### Triton kernel LSE 合并(`768-851`)

`merge_indexed_attn_states`(非 identity 情况)是 LSE 公式的 GPU 实现:

```python
@triton.jit
def _merge_indexed_attn_states_kernel(Out, Lse_s, Prefix_out, Prefix_lse, Indices, ...):
    pid_n = tl.program_id(0); pid_h = tl.program_id(1)
    idx_out = tl.load(Indices + pid_n)              # scatter:prefix 第 n 个 → output 的 idx_out 位
    lse_p = tl.load(Prefix_lse + ...)
    lse_s = tl.load(Lse_s + ... idx_out ...)
    lse_p = tl.where(lse_p == inf, -inf, lse_p)     # FA2/FA3 一致性
    m = tl.maximum(lse_p, lse_s)                    # 数值稳定
    se_p = tl.exp(lse_p - m); se_s = tl.exp(lse_s - m)
    sum_exp = se_p + se_s
    res = (se_p/sum_exp) * val_p + (se_s/sum_exp) * val_s   # = (Z_p/Z)·out_p + (Z_s/Z)·out_s
    tl.store(Out + ... idx_out ..., res)
```

`grid = (num_prefix_tokens, num_heads)`。`Indices` 把 prefix 结果 scatter 回正确 output 位置——这就是 non-identity 时需单独 triton kernel 的原因(原生 `merge_attn_states` 不做 scatter)。

### `forward` 分发(`1022-1187`)

```python
if self.kv_sharing_target_layer_name is None:
    reshape_and_cache_flash(key, value, key_cache, value_cache, attn_metadata.slot_mapping, ...)  # 更新 KV cache

if attn_metadata.prefix_indices is not None or attn_metadata.prefix_indices_is_identity:
    self.cascade_attention(output, query, key_cache, value_cache, ...)     # cascade 路径
else:
    flash_attn_varlen_func(q=query, ..., block_table=attn_metadata.block_table, ...)  # 标准路径
```

**分发条件**:`prefix_indices is not None **or** prefix_indices_is_identity`(原文档漏了 identity 分支)。

### 三个边角(文档可补)

- `shared_k_buffer`/`shared_v_buffer`(`278-279`):dataclass 定义但**从未赋值**(预留/遗留)。
- encoder 路径(`attn_type` 为 ENCODER/ENCODER_ONLY)直接走 `_forward_encoder_attention`(`1066-1075`),绕过 cascade。
- `update_block_table`(`742-762`):CUDA Graph 场景重建 prefix/suffix block table。

---

## 三层"共享前缀"总结

整个 vllm-gr beam search 优化,"共享前缀"在三个层次各实现一次:

| 层次 | 机制 | 章节 | 省 |
|---|---|---|---|
| 存储 | template-clone + beam_cache | ❶–❷ | N 份 KV → 1 份共享 + N 份增量 |
| 调度 | beam_prefix_groups(computed_blocks 缓存) | 阶段A | N 次 hash 遍历 → 1 次 |
| 计算 | cascade attention(prefix 算一次 + LSE 合并) | 阶段C | N 次前缀 attention → 1 次 |

配合 prefill/decode 直觉:ADD_BATCH≈prefill(冷启动建 KV)、BEAM_FORK≈decode(增量复用父 KV)。三层优化都是为让 decode 阶段 N 条 beam 路径尽可能少重复劳动。
