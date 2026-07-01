# Mega 机制难点深化(数值与张量级)

> 配套 [`online_pipeline_mega.md`](online_pipeline_mega.md)。前者是全流程概览走读;本文对其中**最难的三处**——position 重写 + `InputBatchProxy`(单元 4)、cascade 索引(单元 5)、`_remux`(单元 6)——做**逐行 + 张量形状 + 数值推演**级展开。
>
> 全文用一个**贯穿例子**,保证三块连贯。所有数值均按实际代码逻辑推演。

---

## 0. 贯穿例子

### 0.1 参数定义

| 参数 | 值 | 含义 |
|---|---|---|
| `W` = `beam_width` | **3** | 活跃 beam 数 |
| `prefix_len` | **4** | 公共前缀(prompt + sid_begin)长度 |
| `decode_steps` | **2** | 当前步每条 beam 的后缀长度(= 已是第 2 个 mega 步) |
| `cache_len` | **4** | 调度前已算进 KV cache 的 token 数 |
| `K` = `logprobs_num` | **3** | 每行取的 top-K(= beam_width) |

### 0.2 Mega Request 的 token 布局

`_handle_mega_request_step_update` 拼接出(`core.py:128-131`):

```
mega_token_ids = [prefill] + [beam_0 后缀] + [beam_1 后缀] + [beam_2 后缀]
               = [p0,p1,p2,p3, b0t0,b0t1, b1t0,b1t1, b2t0,b2t1]
索引:            0  1  2  3   4   5    6   7    8   9
                                                                 总长 10
```

### 0.3 ⭐ 关键澄清:每步重算全部后缀

主线 **`cache_len = prefix_len`**(本例都 = 4),原因:
- 每步 mega request 是**全新构造**(`core.py:133`),但 `block_hashes` 继承首步 prefill。
- `patched_schedule` 把 `get_computed_blocks` **裁到 `prefix_len`**(`engine_core_patch.py:247-253`),所以前缀命中、**全部 W·decode_steps 个后缀 token 每步都要重算**。
- `cache_len` 在 `patched_schedule` 里 = `num_computed_tokens(调度后) - num_scheduled_tokens` = 调度前已算量 = `prefix_len`。

> 这正是 cascade attention 存在的理由:虽然每步重算 W·N 个后缀,但前缀 KV 只算一份(命中 cache),W 条 beam 共享 → 远比朴素「W 次完整 prefill」便宜。

本例 `num_scheduled_tokens` = 总长 10 - cache_len 4 = **6**(即 W·decode_steps = 3·2),全部是后缀 token,prefix 不在本步 query 里。

---

## 难点 A:position 重写(单元 4 深化)

### A.1 为什么必须重写

vllm 默认给 mega_token_ids 赋**连续递增** position:

| token | p0 | p1 | p2 | p3 | b0t0 | b0t1 | b1t0 | b1t1 | b2t0 | b2t1 |
|---|---|---|---|---|---|---|---|---|---|---|
| **原始 position(错)** | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |

**错在哪**:beam_1 的后缀 b1t0/b1t1 接在 beam_0 之后(position 6,7),意味着它们被当成 beam_0 的延续;attention 时 b1t0 会 attend 到 b0t0/b0t1(跨 beam 串扰),且相对前缀的位置错位。

**正确语义**:W 条 beam 都是「prefix 之后各自独立的续写」,后缀都应从 `prefix_len=4` 起:

| token(仅后缀 query) | b0t0 | b0t1 | b1t0 | b1t1 | b2t0 | b2t1 |
|---|---|---|---|---|---|---|
| **重写后 position** | **4** | **5** | **4** | **5** | **4** | **5** |

→ W 条 beam 的后缀**共享同一段 position 区间 `[4,6)`**。这才是 cascade 复用前缀 KV 的前提。

### A.2 `_compute_beam_bounds` 完整语义(`engine_core_patch.py:300-317`)

```python
def _compute_beam_bounds(b, prefix_len, cache_len, decode_steps, chunk_budget, delta):
    past_suffix_b = max(0, min(decode_steps, delta - b * decode_steps))
    if prefix_len >= cache_len:                                   # 主线成立
        remaining_prefix = prefix_len - cache_len                 # = 0
        return (min(chunk_budget, remaining_prefix + decode_steps) if b == 0
                else min(chunk_budget, decode_steps)), past_suffix_b
    ...
```

**主线推演**(cache_len=4=prefix_len, delta=0):

| b | past_suffix_b | b_uncached | 说明 |
|---|---|---|---|
| 0 | `max(0, min(2, 0-0))=0` | `min(6, 0+2)=2` | beam_0 算 2 token |
| 1 | `max(0, min(2, 0-2))=0` | `min(4, 2)=2` | beam_1 算 2 token |
| 2 | `max(0, min(2, 0-4))=0` | `min(2, 2)=2` | beam_2 算 2 token |

→ 每条 beam `b_uncached=2`,正好覆盖各自全部后缀。

**chunked 边界**(cache_len < prefix_len, delta>0):此时 `remaining_prefix>0`,beam_0 的 `b_uncached` 会包含「未算前缀 + 后缀」(用于补算前缀),其余 beam 只算后缀。这是为 chunked prefill 留的路径,主线不会触发。

### A.3 `_process_mega_decode_request` 逐 beam 推演(`:338-381`)

```python
for b in range(beam_width):              # b=0,1,2
    b_uncached, past_suffix_b = _compute_beam_bounds(...)   # (2,0)
    st = cache_len if (prefix_len>=cache_len and b==0) else prefix_len   # 主线都=4
    _overwrite_position_ids(st, b_uncached, curr_offset, gpu_pos, cpu_pos)
    if past_suffix_b + b_uncached >= decode_steps:          # 0+2>=2 成立
        new_logits_indices.append(curr_offset + b_uncached - 1)
    curr_offset += b_uncached; chunk_budget -= b_uncached
```

逐 beam 填值(`token_offset=0` 起步):

| b | `st` | 写入区间 `[curr_offset, +b_uncached)` | position 写入值 | `new_logits_indices` 追加 |
|---|---|---|---|---|
| 0 | 4 | `[0, 2)` | `arange(4,6)` = **4,5** | `0+2-1` = **1** |
| 1 | 4 | `[2, 4)` | `arange(4,6)` = **4,5** | `2+2-1` = **3** |
| 2 | 4 | `[4, 6)` | `arange(4,6)` = **4,5** | `4+2-1` = **5** |

**结果**:
- `positions[0:6]` = `[4,5,4,5,4,5]`(GPU + CPU 两份都改,`:328-335`)。
- `new_logits_indices` = `[1, 3, 5]` → 每条 beam 取**末尾 token** 的 logits → **3 行 logits**。
- 返回 `(valid_rows=3, logits_row_cnt=3, next_offset=6)`。

### A.4 ⭐ `InputBatchProxy` 如何展开 W 行(`:542-591`)

**问题**:下游 sampler 按「每请求 1 行」写死,不能改。但 mega 请求要变 3 行。

**机制**:用 proxy 包一层,把 1 物理行伪装成 3 逻辑行。本例 `input_batch` 原本只有 1 行(session_id 这条请求):

```
原 input_batch (1 行):
  req_ids          = [session_id]
  num_tokens_no_spec = [6]          # 这条请求 6 个 token
  token_ids_cpu    = [[p0,p1,p2,p3,b0t0,b0t1,b1t0,b1t1,b2t0,b2t1]]  # 实际存的是拼接序列
  sampling_metadata.temperature = [T0]   # 1 个采样参数
```

`_prepare_inputs` 构造的映射(`:678-680`):
```
row_to_req_id    = [session_id, session_id, session_id]   # 3 行同 id
row_to_batch_idx = [0, 0, 0]                              # 3 行都指向 input_batch[0]
logits_to_batch_idx = [0, 0, 0]                           # 3 个 logits 行
```

包成 proxy 后,sampler 看到的:
```
proxy (3 行):
  req_ids          = MegaReqIdsProxy → [session_id, session_id, session_id]   # __getitem__ 查 req_id_map
  num_tokens_no_spec = Mock1DArray → [num_tokens_no_spec[0]]*3 = [6,6,6]     # 查 batch_idx_map
  token_ids_cpu    = Mock2DArray → 每行都返回 input_batch[0] 的 token        # (idx,slice) 查 batch_idx_map
  sampling_metadata.temperature = [T0,T0,T0]   # val[logits_map] 广播 3 份 (:570-573)
```

→ sampler 把它当 **3 个独立请求**采样,产出 **3 行 logits/sampled**。其余属性(`__getattr__` 末尾 `return getattr(obj, name)`,`:588`)透传给原 batch,行为不变。

### A.5 `logits_indices` 对比

| | 内容 | 形状 | 来源 |
|---|---|---|---|
| 原始 `logits_indices` | 按「1 请求」算(可能只指向 1 个位置) | 不定 | `_original_prepare_inputs` |
| **`new_logits_indices`** | `[1, 3, 5]`(每 beam 末尾) | `(3,)` int32 | mega 重写,`:711-713` |

`patched_prepare_inputs` **返回后者替代前者**(`:711`),告诉 sampler「从这 3 个 token 位置取 logits」。

### A.6 张量形状汇总(难点 A)

| 张量 | 形状 | 值 |
|---|---|---|
| `positions`(gpu/cpu) | `(6,)` | `[4,5,4,5,4,5]` |
| `new_logits_indices` | `(3,)` int32 | `[1,3,5]` |
| 采样器看到的 `req_ids` | `(3,)` | `[sid,sid,sid]` |
| 采样器产出的 logits | `(3, K=3)` | 见难点 C |

---

## 难点 B:cascade 索引(单元 5 深化)

### B.1 朴素 vs cascade 成本

朴素做法:W=3 条 beam 各做一次完整 attention(每条 query 2 token,attend prefix 4 + 自己后缀 2 = 6 KV)→ 3 次 varlen。

cascade 做法:prefix 段 1 次(3 条 beam 的 6 个 query 一起 attend 共享前缀 4 KV)+ suffix 段 1 次(6 query attend 各自后缀 2 KV)+ merge。**前缀 KV 只算一份**,prefix 段的 6 个 query 合并成 1 组(B.3)。

### B.2 `_build_mega_mappings` 逐行数值推演(`beam_attn.py:330-426`)

入参:`prefix_len=4, w=3, steps=2, cache_len=4`,`q_starts` 设 mega 请求占 `[0,6)`(q_start=0, q_len=6)。

```python
remaining_prefix = prefix_len - cache_len = 0          # 前缀全在 cache
shared_mapping = np.arange(out_slot=0, 0+0) = []       # 空(无未算前缀)
# 逐 beam:
```

| b | `b_uncached` | `b_suffix_len` | `full_mapping`(remaining_prefix+steps=2 项) | `q_curr` | `chunk_budget` | `out_slot` |
|---|---|---|---|---|---|---|
| 0 | 2 | 2 | `[arange(0,0)=空 , arange(0,2)] = [0,1]` | 0→2 | 6→4 | 0→2 |
| 1 | 2 | 2 | `[空, arange(2,4)] = [2,3]` | 2→4 | 4→2 | 2→4 |
| 2 | 2 | 2 | `[空, arange(4,6)] = [4,5]` | 4→6 | 2→0 | 4→6 |

`prefix_groups` 累积(相邻 beam 的 prefix query 连续等价 → 合并,`:394-397`):
```
b=0: prefix_groups = [[q_curr=0, b_uncached=2, cache_len=4]]
b=1: prefix_groups[-1][1] += 2 → [[0, 4, 4]]
b=2: prefix_groups[-1][1] += 2 → [[0, 6, 4]]      # 最终 1 组,q_start=0 q_len=6
```

产出:
```
mega_suffix_slot_mapping_out_list = [0,1, 2,3, 4,5]      # gather 目标布局
mega_suffix_q_lens        = [2,2,2]
mega_suffix_seq_lens_list = [2,2,2]                       # b_suffix_len
# 收尾:
mega_prefix_indices       = range(0,6) = [0,1,2,3,4,5]    # 所有 mega query
mega_prefix_q_lens        = [6]                            # 合并的 1 组
mega_prefix_seq_lens_list = [4]                           # cache_len=4
mega_prefix_block_tables  = [bt[:max_prefix_blocks_count]]
```

> ⭐ **关键洞察**:3 条 beam 的 prefix query 被**合并成 1 组**(`mega_prefix_q_lens=[6]`),所以 prefix 段只需 1 次 varlen attention 覆盖全部 6 个 query。suffix 段保持 3 段(`mega_suffix_q_lens=[2,2,2]`),各 beam 独立。

### B.3 prefix 段张量形状(`_create_mega_tensors:474-480`)

| 张量 | 形状 | 值 |
|---|---|---|
| `mega_prefix_indices` | `(6,)` | `[0,1,2,3,4,5]` |
| `mega_prefix_cu_seqlens_q` | `(2,)` | `[0, 6]`(cumsum `[6]`) |
| `mega_prefix_seq_lens` | `(1,)` | `[4]`(= cache_len) |
| `mega_prefix_block_table` | `(1, max_prefix_blocks)` | 前缀块表 |
| `mega_prefix_max_q_len` | int | `6` |

prefix 段 attention:`flash_attn_varlen_func(q=6 query, k/v=key_cache, causal=False, block_table=..., seqused_k=[4])`。6 个 query 全部 attend 到 4 个前缀 KV,返回 `out (6, dim)` + `lse (6,)`。

### B.4 suffix 段张量形状 + gather(`:482-504`)

| 张量 | 形状 | 值 |
|---|---|---|
| `mega_suffix_cu_seqlens_q` | `(4,)` | `[0,2,4,6]`(cumsum `[2,2,2]`) |
| `mega_suffix_cu_seqlens_k` | `(4,)` | `[0,2,4,6]`(cumsum `[2,2,2]`) |
| `mega_suffix_seq_lens` | `(3,)` | `[2,2,2]` |
| `mega_suffix_slot_mapping_out` | `(6,)` int32 | `[0,1,2,3,4,5]` |
| `mega_suffix_total_tokens` | int | `6` |

**gather 语义**(`extract_suffix_kv:109`,`paged_kv_to_contig_suffix_kernel:61`):

二级间接索引,逐输出 token:
```
out_token(0) --[SLOT_MAPPING_OUT]--> src_idx=0 --[SLOT_MAPPING]--> slot_idx=s_b0t0 --[paged KV]--> K[s_b0t0],V[s_b0t0]
out_token(1) --> src_idx=1 --> slot_idx=s_b0t1 --> K[s_b0t1],V[s_b0t1]
out_token(2) --> src_idx=? --> slot_idx=s_b1t0 --> ...       # beam_1 后缀
...
```

(本例 `slot_mapping_out=[0,1,2,3,4,5]` 恰好是顺序,实际 slot_idx 由 `attn_metadata.slot_mapping` 给出,是各后缀 token 在 paged cache 的真实槽位。)

产出 `contig_k, contig_v` 形状 `(6, n_kv_heads, head_dim)` —— 6 个后缀 KV 连续排列,按 beam 分段(beam_0:[0,2), beam_1:[2,4), beam_2:[4,6))。

suffix 段 attention:`flash_attn_varlen_func(q=6 query, k=contig_k, v=contig_v, causal=True, block_table=None, cu_seqlens_k=[0,2,4,6])`。每条 beam 的 2 query attend 自己的 2 KV(因果),返回 `out (6,dim)` + `lse (6,)`。

### B.5 两段 attention 输入输出对照

| | query | K/V 源 | causal | 布局参数 | 输出 |
|---|---|---|---|---|---|
| prefix 段 | 6(query[0:6]) | **paged cache**(block_table) | **False** | `cu_seqlens_q=[0,6]`, `seqused_k=[4]` | `out_p (6,dim)`, `lse_p (6,)` |
| suffix 段 | 6(同上) | **contig**(gather 出) | **True** | `cu_seqlens_q=[0,2,4,6]`, `cu_seqlens_k=[0,2,4,6]` | `out_s (6,dim)`, `lse_s (6,)` |
| merge | — | — | — | — | `out (6,dim)` 写回 `output[0:6]` |

### B.6 ⭐ `merge_attn_states` 数学推导

两段分别算的是「query 对部分 KV」的 attention。要等价于「query 对全部 KV(prefix+suffix)」的一次 attention,用 **logsumexp 合并**:

设某 query 对 prefix KV 的 attention 输出为 `o_p`(已归一化)、logsumexp 为 `l_p`;对 suffix 为 `o_s`、`l_s`。则对全 KV 的输出:

```
o = ( exp(l_p)·o_p + exp(l_s)·o_s ) / ( exp(l_p) + exp(l_s) )
```

**推导**:全 KV 的 softmax 分母 = Σ_{i∈prefix} exp(s_i) + Σ_{j∈suffix} exp(s_j) = exp(l_p) + exp(l_s)(因 l = logΣexp)。分子同理加权。`merge_attn_states`(vllm 已有,`vllm.v1.attention.ops.merge_attn_states`)就是实现这个。

> 所以 cascade 数学上**精确等价**于一次全序列 attention,只是拆成「paged 前缀 + 连续后缀」两段以复用前缀 KV、让 W beam 共享。

---

## 难点 C:`_remux`(单元 6 深化)

### C.1 采样器原始输出形状

经难点 A 的 `InputBatchProxy`,sampler 把 mega 请求当 **3 行**采样。`SamplingParams(logprobs=3)` → 每行 K=3 个候选。采样器产出(`engine_core_patch.py:754-757`):

| 张量 | 形状 | 含义 |
|---|---|---|
| `orig_lps`(`lp_tensors.logprobs`) | `(3, 3)` | 3 行 × top-3 logprob |
| `orig_lp_ids`(`lp_tensors.logprob_token_ids`) | `(3, 3)` | 对应 token id |
| `orig_ranks`(`selected_token_ranks`) | `(3,)` 或 `(3,1)` | 选中 token 的 rank |
| `st_ids`(`sampled_token_ids`) | `(3, 1)` | 每行 sampled token |

但 engine 对外要回**一个** RequestOutput(session_id),所以要压回 1 行。

### C.2 `_remux` CASE B 逐行推演(`:439-475`)

`request_row_configs = [(session_id, 0, W_logits=3)]`(本批只有这一个 mega 请求)。

```python
max_chained_dim = max(W * actual_K for ...) = max(3*3) = 9
gpu_row_cursor = 0

# req_id=session_id, W_logits=3 (CASE B, 非 0):
new_st_ids.append(st_ids[0:1, :])                          # 取首行 sampled id → (1,1)

req_lps = orig_lps[0:3, :3].reshape(1, 3*3)                # (3,3) → (1,9)
# W_logits*target_K = 9 == max_chained_dim,无需 pad
new_lps.append(req_lps)                                    # (1,9)

req_lp_ids = orig_lp_ids[0:3, :3].reshape(1, 9)            # (1,9)
new_lp_ids.append(...)

req_ranks = orig_ranks[0:3].reshape(1, 3)                  # (1,3) → pad 到 (1,9) value=0
new_ranks.append(...)

gpu_row_cursor += 3                                        # → 3
```

**reshape 的本质**:把 3 条 beam × 3 候选的 `(3,3)` 矩阵**按行展开成 `(1,9)`**:
```
orig_lps (3,3):         reshape →  final_lps (1,9):
  [lp_b0c0, lp_b0c1, lp_b0c2]      [lp_b0c0, lp_b0c1, lp_b0c2,
  [lp_b1c0, lp_b1c1, lp_b1c2]   →    lp_b1c0, lp_b1c1, lp_b1c2,
  [lp_b2c0, lp_b2c1, lp_b2c2]       lp_b2c0, lp_b2c1, lp_b2c2]
```
→ 每请求**1 行**,packed 了 3 条 beam 的全部 9 个候选。

### C.3 多请求同 batch 时的 `gpu_row_cursor`

假设 batch = [mega 请求(W=3), 普通请求(W=1, last_chunk_prefill)]:

```
request_row_configs = [(sid, 0, 3), (normal_id, 3, 1)]
采样器输出按 input_batch 顺序:mega 占 3 行,普通占 1 行 → orig_lps (4, 3)

max_chained_dim = max(3*3, 1*3) = 9

# 第 1 请求(mega):
gpu_row_cursor=0: lps[0:3,:3].reshape(1,9) → pad? 9==9 不用 → (1,9)
gpu_row_cursor → 3
# 第 2 请求(普通):
gpu_row_cursor=3: lps[3:4,:3].reshape(1,3) → pad 到 (1,9) value=-inf → (1,9)
gpu_row_cursor → 4

final_lps = cat → (2, 9)      # 2 个逻辑请求,各 1 行,行宽统一 9
```

> `gpu_row_cursor` 按 `W_logits` 推进,精确对齐每个请求在采样器输出里占的行块。CASE A(`W_logits==0`,中间 chunked prefill)不推进 cursor(它在采样器里本就没占行),只填 dummy 行占位。

### C.4 原位 `resize_().copy_()` 的必要性(`:772-786`)

```python
st_ids.resize_(final_st_ids.shape).copy_(final_st_ids)
lp_tensors.logprobs.resize_(final_lps.shape).copy_(final_lps)
...
```

- **不换对象**(用 `resize_`+`copy_` 而非赋值新张量):`sampler_output` 是 vllm 内部传递的对象,下游(OutputProcessor)持有它的引用。换对象会导致下游拿到的还是旧张量。**原位改写**保证下游看到 remux 后的内容。
- `resize_` 改 shape(如 `(3,3)`→`(2,9)`),`copy_` 填内容。

### C.5 serving 侧 `stride_k` 切分(逆运算)(`serving_engine.py:470-512`)

engine 回的 `mega_result.outputs[0].logprobs` 是 remux 后的 flat(本例 1 行 9 候选):

```python
raw_token_ids = flat.token_ids       # 长度 9
total_logprobs = 9
stride_k = total_logprobs // len(fork_info) = 9 // 3 = 3

for b_idx in range(3):                # beam 0,1,2
    start_offset = b_idx * 3
    end_offset = start_offset + 3
    # b_idx=0: [0:3]  → [lp_b0c0..c2]   正好是 beam_0 的 3 候选
    # b_idx=1: [3:6]  → beam_1 的 3 候选
    # b_idx=2: [6:9]  → beam_2 的 3 候选
    token_ids_pos, logprobs_pos, ... = _extract_and_dedup_segment(raw_*, start_offset, end_offset, logprobs_num=3)
```

→ `stride_k` 切段**正好逆** remux 的 reshape:remux 把 `(3,3)` 展平成 `(1,9)`,serving 按 `stride_k=3` 切回 3 段。

### C.6 pad 到 `max_chained_dim` 的副作用

若 batch 内有**不同 W** 的多个 mega 请求(多 session),`max_chained_dim` 取最大。较小 W 的请求 remux 后行宽 < max,pad `-inf`。serving 侧 `stride_k = total // W` 会**偏大**(含 pad 段),切出的候选里混入 `-inf` logprob → 但 `-inf` 在后续 `np.argpartition` top-K 时自然被淘汰,**结果正确**,只是多搬运了一些 padding。

> 单 session(beam_search 常见)无此问题:`max_chained_dim = W·K`,`stride_k = K` 精确。

---

## 贯穿示例完整流转表

| 阶段 | 对象 | 形状/值 |
|---|---|---|
| Mega Request prompt | `mega_token_ids` | `[p0..p3, b0t0,b0t1, b1t0,b1t1, b2t0,b2t1]`(10) |
| 本步 query | 后缀 token | 6 个(`num_scheduled_tokens`) |
| **position 重写** | `positions` | `[4,5,4,5,4,5]`(6) |
| | `new_logits_indices` | `[1,3,5]`(3) |
| **InputBatchProxy** | sampler 见到的行 | 3 行(都 = session_id) |
| **cascade prefix 段** | query/KV | 6 query attend 4 前缀 KV,非因果 |
| **cascade suffix 段** | query/KV | 6 query attend contig 6 KV(3×2),因果 |
| **merge** | | lse 合并 → `output[0:6]` |
| **采样** | `orig_lps` | `(3, 3)` |
| **_remux** | `final_lps` | `(1, 9)` |
| 回写 | `lp_tensors.logprobs` | `(1, 9)` 原位 |
| **serving 切分** | `stride_k` | `9//3 = 3`,切 `[0:3][3:6][6:9]` |

---

## 易踩坑速查(深化发现)

1. **`cache_len` 不是「前缀已算」而是「调度前已算」**:主线 = `prefix_len`,导致**每步重算全部后缀**。误以为是「只算新增 1 token」会错估计算量。
2. **position 重写让 W beam 后缀 position 重叠**:`[4,5,4,5,4,5]` 不是笔误,是 cascade 复用前缀的前提。
3. **`InputBatchProxy` 是「1 物理 → W 逻辑」的伪装**:不改下游代码,靠 `MegaReqIdsProxy`/`Mock1DArray`/`Mock2DArray` 重映射少数属性,其余透传。
4. **prefix 段 query 合并**:相邻 beam 的 prefix query 因「前缀全可见」等价,合并成 1 组(`prefix_groups`),prefix 段只 1 次 varlen。
5. **`merge_attn_states` 精确等价全序列 attention**(lse 合并数学),非近似。
6. **remux 用 `resize_+copy_` 原位改写**,不能换对象(下游持有引用)。
7. **remux reshape 与 serving stride_k 切段互逆**:remux `(3,3)→(1,9)`,serving `9//3=3` 切回。
8. **多 session 不同 W 时 pad 有副作用**:小 W 请求行被 pad,`stride_k` 偏大,但 `-inf` 被 top-K 淘汰,结果仍正确。
9. **`mega_prefix_indices` 命名误导**:是所有 mega query 索引(prefix/suffix 段共用),非仅前缀 query。
10. **`new_logits_indices` 每条 beam 只取末尾 token**:W beam → W 行 logits,不是 W·decode_steps 行。
