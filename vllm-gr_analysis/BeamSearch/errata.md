# 原文档勘误表(beam_search_walkthrough.md)

**状态**: 基于实际代码逐行核对
**结论**: 原文档(Draft v1.0)主框架准确——整条 ADD_BATCH/BEAM_FORK 链路、template-clone、cascade attention、flat logprobs 延迟重建都真实存在且方向正确。但**行号大多对得上(±5 行),却有若干 API 名/签名错误、伪代码简化失真,且整条 GR 类型扩展链路完全没写**。

把它当"地图"可以,当代码真相不行——以代码为准。本表按严重程度分 A/B/C 三类。

---

## A. 事实性错误(会误导)

| # | 原文档位置 | 文档说法 | 实际代码 |
|---|---|---|---|
| 1 | §6.2 `add_requests_async` | 参数 `use_batch_message` | 参数名是 **`force_batch`**(`core_client.py:14`) |
| 2 | §6.1 `_add_requests_batch_fn` | `add_requests_async(requests, use_batch_message=True)` | `force_batch=use_batch_message`(`async_llm.py:130`) |
| 3 | §6.2 `beam_fork_async` | 用 `get_core_engine_for_request()` 路由 | 直接 `self.core_engines[data_parallel_rank]`(`core_client.py:71`) |
| 4 | §4 离线初始化 | "初始化 beam_width 个 BeamSearchSequence" | 初始化**只有 1 个 beam**,beam_width 个候选由循环内 Top-K 产生(`gr.py` 用 `BeamSearchInstance`) |
| 5 | §4 主循环步骤 | 含"长度惩罚排序" | `gr.py`/`serving_engine.py` **根本没实现 length_penalty**,只有 TODO;排序用裸 `cum_logprob` |
| 6 | §4 Catalog 过滤 | `catalog.get_valid_tokens()` | 方法名是 **`catalog.valid(...)`**(`gr.py:482`/`serving_engine.py:346`) |
| 7 | §5.2 DP 路由 | `self._dp_counter` / `self._dp_lock` | **模块级全局** `_dp_rank_counter` / `_dp_rank_lock`(`serving_engine.py:27-28`),非实例属性 |
| 8 | §11.7 forward 分发 | `if prefix_indices is not None` | `prefix_indices is not None **or prefix_indices_is_identity**`(`beam_attn.py:1101`),漏判 identity 分支 |
| 9 | §12.3 `reconstruct_beam_logprobs` | 签名 `(beam, sid_end_token_id=None)` | 实际 **`(beam, initial_logprobs, sid_end_token_id=None)`**(`logprobs.py:56`),漏了 `initial_logprobs` |
| 10 | §11.6 Triton 公式 | `lse_merged = log(exp(lse_p)+exp(lse_s))` 显式算 | kernel **不计算/存储 lse_merged**,用 `m=max(...)` 稳定化后做 logsumexp 加权(`beam_attn.py:803-820`) |
| 11 | §3.2 patch 编排 | "6 个独立 patch" | 6 个 + **可选 LMCache patch**;且 `_apply_patches` 还含 `_apply_gr_core_patches`/`_apply_yuanrong_npu_patch`/`_apply_kv_merge_patch`(文档完全没提) |

---

## B. 简化失真(方向对但实现不同)

| # | 原文档位置 | 文档说法 | 实际代码 |
|---|---|---|---|
| 1 | §9.1 `_get_lcp` | 二分搜索 + hint | 还漏两处快速路径:`m==0` 早返回、`if start<=end and a[end]==b[end]: return end+1` 尾端捷径(`engine_core_patch.py:23,38-39`) |
| 2 | §11.4 `_create_suffix_metadata` | column shift 取 `[:, prefix_depth:]`(slice) | 实际 **`torch.gather` + 动态列索引** `col_indices = arange + shift_amounts`(`beam_attn.py:726-732`) |
| 3 | §11.3 `_create_prefix_metadata` | "取第一个 req 的 `[0:prefix_depth]`" | 实际 `block_table[group_first_reqs][:, :max_prefix_blocks]`,`max_prefix_blocks=ceil(prefix_kv_max_len/block_size)`(`beam_attn.py:593,654-655`) |
| 4 | §11.2 `BeamAttentionMetadata` 字段 | 只列 4 个 cascade 字段 | 实际还有 `cu_prefix_kv_lens`、`common_prefix_lengths`、`max_num_splits`、`shared_k/v_buffer`(定义但**未使用**)、`prefix_indices_is_identity`、`group_first_reqs`、`suffix_col_indices` 等(`beam_attn.py:250-286`) |
| 5 | §7.2 `_handle_beam_fork` | 伪代码简化 | 实际第一个子走完整构造 + 快照 `shared_fields`;block_hashes 增量计算用 `partial(self.request_block_hasher, req)`;`input_queue.put_nowait((ADD, (req, fork_req.current_wave)))` 带 current_wave(`core.py:64-182`) |
| 6 | §9.3 `apply_scheduler_patch` | 缓存 key 简化 | 实际 key=`(num_tokens, skip_reading_prefix_cache, last_hash)`,last_hash 经 `max_num_blocks`/`hash_idx` 计算得来;还有 `served_from_sw_cache` 统计逻辑(`engine_core_patch.py:368-410`) |

---

## C. 概念性遗漏(文档没提但重要)

### 1. `FlatLogprobs` 是 vLLM 原生类
原文档 §12 把 FlatLogprobs 说成 vllm-gr 的方案。实际 **`FlatLogprobs` 定义在 `vllm/vllm/logprobs.py:31`**(原生 vLLM)。vllm-gr 的贡献是:
- `logprobs.py` 的 `append_fast`(用 `extend` 替代逐元素 `append`)—— 通过 `logprobs_patch.py` 的 `patch_flat_logprobs` 绑到原生类
- `extract_and_dedup_flat_logprobs` / `reconstruct_beam_logprobs` 工具函数
- parent-pointer **使用模式**

### 2. GR 类型扩展链路(最该补的地基)
patch.py 第一个、最重要的 patch 是 `_apply_gr_core_patches`,把 vLLM 三个核心类型全局替换为 GR 版本,让"生成式推荐的特征"穿过整个管线:

```
GREngineCoreRequest (engine/__init__.py:28)   新增 gr_features: bytes(序列化末位)
      ↓ from_engine_core_request
GRRequest (request.py:17)                     携带 gr_features
      ↓ Scheduler
GRNewRequestData (core/sched/output.py)       下发到 ModelRunner
      ↓
gr_features_codec.py                          adapter 各自定义 payload 编解码
```

详见 `gr_specifics.md`。原文档完全没写这条链路,导致 `priority`、`gr_features` 这些字段的来源无法理解。

### 3. patch.py 的完整 patch 体系
原文档只讲 beam patches。实际 `_apply_patches`(`patch.py:707-713`)调用四类:
- `_apply_gr_core_patches` — 核心类型替换(上文)
- `_apply_yuanrong_npu_patch` — 昇腾 NPU 的 yuanrong KV transfer backend
- `_apply_kv_merge_patch` — HSTU 风格 K+V 合并 KV cache(`kv_cache_per_layer=1`)
- `_apply_gr_beam_patches` — beam search 的 6 个 patch + 可选 LMCache

### 4. 其他未覆盖模块
- **HSTU / FuXi 模型注册**(`register_gr_model_architectures`)
- **adapters 自动发现**(`_register_adapter_families`,扫描 `vllm_gr/adapters/*/`)
- **LMCache 集成**(可选,环境变量 `VLLM_GR_LMCACHE_PATCH` 触发)
- **vllm-ascend 配合**(目录里的 `vllm-ascend/` 是配合对象)

---

## D. 行号核对

原文档绝大多数行号 **±5 行以内准确**(主体结构未变)。关键确认:
- `plugin.py`: `register()` 146-149 ✓ / `initialize_runtime()` 108-143 ✓
- `patch.py`: `_apply_gr_beam_patches` 317-437 ✓
- `types.py`: `BeamForkRequest` 21-45 ✓
- `serving_engine.py`: `beam_search` 156-580 ✓ / `_beam_fork_step` 55-115 ✓ / `_add_batch_step` 118-153 ✓ / `_next_data_parallel_rank` 32-38 ✓
- `async_llm.py`: `prepare_request_fn` 22-116 ✓ / `_add_requests_batch_fn` 119-130 ✓ / `register_beam_output_fn` 133-172 ✓ / `beam_fork_fn` 175-177 ✓
- `core_client.py`: `add_requests_async` 14-51 ✓ / `beam_fork_async` 59-86 ✓
- `engine_core_patch.py`: `_get_lcp` 20-50 ✓ / `_compute_beam_prefix_groups` 53-103 ✓ / `apply_scheduler_patch` 348-429 ✓ / `apply_worker_patches` 432-479 ✓ / `run_engine_core` 268-345 ✓
- `core.py`: `_cache_beam_request` 28-41 ✓ / `_PER_CHILD_FIELDS` 47-61 ✓ / `_handle_beam_fork` 64-182 ✓ / `process_input_sockets` 190-286 ✓
- `beam_attn.py`: `BEAM_PREFIX_GROUPS_VAR` 64 ✓ / `build` 350-377 ✓ / `_create_prefix_metadata` 583-681 ✓ / `_create_suffix_metadata` 683-740 ✓ / triton kernel 768-823 ✓ / `cascade_attention` 912-1020 ✓ / `forward` 1022-1187 ✓

行号可信,但**伪代码和 API 名以本表 A/B 类修正为准**。

---

## 勘误使用建议

- **读原文档**:用它的架构图和章节框架建立全局认知(§1-2 概述、§13 对比表都很好)。
- **读代码细节**:以本表 A 类(事实错误)和 `online_pipeline.md`/`scheduling_cascade.md` 为准。
- **理解 GR 全貌**:补 `gr_specifics.md`(原文档完全没覆盖 GR 类型扩展)。
