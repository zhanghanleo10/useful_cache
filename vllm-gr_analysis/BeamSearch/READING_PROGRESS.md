# BeamSearch 在线推理走读 · 阅读进度

> 走读目标:vllm-gr **在线 BeamSearch 推理链路**(`OpenAIServing.beam_search`,HTTP → 结果返回完整闭环)。
> 机制基线:**Mega**(`MegaRequestStepUpdate` + `ADD_BATCH` + `MEGA_REQUEST_STEP_UPDATE`),对应 vllm==0.14.1。
> ⚠️ 旧 `BEAM_FORK` / template-clone 机制已过时,仅作演进对照(阶段四)。
> 文档集:`useful_cache/vllm-gr_analysis/BeamSearch/` · 阅读路径遵循 `README.md` 推荐。

---

## 🔁 如何复用进度(每次会话)

1. **新会话开始**时,对 Claude 说一句:**「继续走读 BeamSearch」** 或 **「看一下我的 beam search 阅读进度」**。
2. Claude 会读本文件(记忆里也有指针),从下方「👉 下一步」处接着带读,无需重新解释背景。
3. **每读完一个单元**,状态流转 `⬜ → 🟨 → ✅`,顶部进度条 / 百分比 / 日志同步更新。
4. 走读中产生的疑问/笔记,直接追加到对应单元的 `> 笔记:` 下,或写进底部「阅读日志」。

> **维护约定**:本文件是进度的**唯一事实源**;Claude 记忆只存指针 + 关键背景,不重复存实时进度(避免失真)。每次更新进度,只改本文件。

---

## 🧠 核心心智模型(打开即看)

**Mega 一句话**:在线 beam_search 是个 async generator,把同一 prompt 下 W 条 beam 的解码压成「**首步一条 ADD_BATCH + 后续每步一条 Mega Request**」,让 Engine 复用 prefill KV、cascade attention 一次算完 W 条 beam,最后只回一个合并的 RequestOutput。

**贯穿全流程的 3 条主线**:
1. **session_id 单物理请求**:W 条 beam 始终由 `session_id` 一条 Request 承载;parent/child/pruned id 都是逻辑路由。
2. **W 行 ↔ 1 行的可逆变换**:worker 内 `InputBatchProxy` 展开 W 行(采样),`_remux` 压回 1 行(输出);serving 用 `stride_k` 再切回。
3. **KV 复用三重保障**:继承 `block_hashes`(mega 合成)+ scheduler 裁 `prefix_len`(只复用前缀)+ cascade prefix 段(前缀 KV 只算一份)。

**两代机制速记**:
| 维度 | 旧 BEAM_FORK / template-clone | 当前 Mega |
|---|---|---|
| 解码步请求 | 每条 beam 一个 BEAM_FORK | **单条 Mega Request** 装全部 W beam |
| KV 共享 | template-clone(W 份共享) | 继承 `block_hashes` + cascade prefix 段 |
| attention | 标准 | **cascade**(prefix 非因果 + suffix 因果 + lse merge) |

---

## 📊 整体进度

```
[█████░░░░░░░░░░░░░░░] 4/16 单元 (25%)
```

- **已完成**:4(单元 0、1、2、3) · **进行中**:1(单元 4) · **待读**:11
- **当前停留**:阶段一 · 单元 4(调度器 + Worker 执行)
- **👉 下一步**:带读 **单元 4**(`patched_schedule` 裁 prefix_len + 透传 mega_data / position 重写三件套让 W beam 共享前缀段 / `InputBatchProxy` 把单条 Request 虚拟展开成 W 行)
- **上次更新**:2026-06-23

---

## 图例

| 标记 | 含义 |
|---|---|
| ⬜ | 未读 |
| 🟨 | 进行中 |
| ✅ | 已读完(已逐节理解 + 代码核对) |
| ⭐ | 重点/难点单元 |

---

## 🗺️ 阅读地图(5 阶段 · 16 单元)

### 阶段一 · ⭐ 当前链路走读(Mega 机制,必读)
> 文档:`online_pipeline_mega.md` · 6 单元串起 HTTP → yield 全闭环

- ✅ **单元 0 · 总览**(`§0`, L11-75)
  - [x] 0.1 为什么改 BeamSearch(GR 场景:beam_width 大 / 合法路径约束 / SID 起止符 / W beam 融合)
  - [x] 0.2 一句话定位(async generator + Mega 压缩)
  - [x] 0.3 ⭐ 全链路总览图(首步 ADD_BATCH 分支 / 后续 MEGA 分支)
- ✅ **单元 1 · 装配与入口**(`§1`, L78-201)
  - [x] 1.1 patch 怎么挂上去(`beam_search_patch.py`:两个赋值接线)
  - [x] 1.2 参数转换 `to_beam_search_params`(SID 走 `vllm_xargs` 偷渡)
  - [x] 1.3 入口签名 + DP rank(微秒时间戳优先级 / 同 beam search 必落同一 rank)
  - [x] 1.4 tokenizer 与 SID token(在线找不到硬报错)
  - [x] 1.5 ⭐ 初始 beam 构造 + mega 探测(`use_mega_request` 运行时探测 / 延迟 logprobs)
- ✅ **单元 2 · 主循环骨架 + 首步 ADD_BATCH**(`§2`, L204-359)
  - [x] 2.1 主循环骨架(`fork_info` 单向开关 / catalog 与 engine 并发)
  - [x] 2.2 首步发送链路(`prepare_request` 不发包 / `force_batch=True` 绕过短路)
  - [x] 2.3 ⭐ 子进程 ADD_BATCH → `_cache_beam_request` 种 session 缓存
  - [x] 2.4 首步结果收集(baseline 解析分支)
  - [x] 2.5 ⭐ EOS + top-K 展开(1 → beam_width,`fork_info` 首次翻转)
- ✅ **单元 3 · Mega 步客户端 + 子进程合成**(`§3`, L363-494)
  - [ ] 3.1 `_mega_request_step`(fork_info → parent/child/tokens 三元映射)
  - [ ] 3.2 `register_beam_output`(只注册队列不发包的"幽灵"请求)
  - [ ] 3.3 发送路径(强制沿用首步 DP rank)
  - [ ] 3.4 子进程 MEGA 分支
  - [ ] 3.5 ⭐ 核心合成 `_handle_mega_request_step_update`(拼 prefill+各beam后缀 / 继承 block_hashes / 打 mega 标记)
- 🟨 **单元 4 · 调度器 + Worker 执行**(`§4`, L498-618) ← 当前
  - [ ] 4.1 解决什么问题(position 默认连续递增是错的)
  - [ ] 4.2 `patched_schedule`(裁 prefix_len + 透传 `mega_data`)
  - [ ] 4.3 `patched_execute_model`(注入 `MEGA_DATA_VAR`)
  - [ ] 4.4 ⭐ position 重写三件套(`_compute_beam_bounds` / `_overwrite_position_ids` / `_process_mega_decode_request`)
  - [ ] 4.5 ⭐ `patched_prepare_inputs`(主循环 / 扩展 4 个映射数组)
  - [ ] 4.6 `InputBatchProxy`(1 物理 → W 逻辑伪装层)
- ⬜ **单元 5 · Cascade Attention**(`§5`, L622-708)
  - [ ] 5.1 解决什么问题(prefix 段 + suffix 段 + merge)
  - [ ] 5.2 整体结构(build / cascade_attention 三段)
  - [ ] 5.3 `_segment_requests`(proxy 与 segment 在此接上)
  - [ ] 5.4 ⭐ `_build_mega_mappings`(prefix/suffix 两套布局)
  - [ ] 5.5 `_create_*_tensors`(prefix paged / suffix 连续)
  - [ ] 5.6 ⭐ `cascade_attention` 三段执行(std / mega-prefix 非因果 / mega-suffix 因果 + merge)
  - [ ] 5.7 triton gather kernel(二级间接索引)
  - [ ] 5.8 `forward`(先写新 KV 再 cascade)
- ⬜ **单元 6 · 采样后合并 + 结果回传(闭环)**(`§6`, L712-801)
  - [ ] 6.1 解决什么问题(W 行 logprobs → 一个 RequestOutput)
  - [ ] 6.2 `patched_bookkeeping_sync`(proxy 视图跑原 bookkeeping + 恢复)
  - [ ] 6.3 ⭐ `_remux_logprobs_tensors`(W 行×K → 1 行×(W·K))
  - [ ] 6.4 回写 sampler_output + 重组返回值
  - [ ] 6.5 ⭐ 回 serving 侧 stride_k 切回 W beam(与 _remux 互逆)
  - [ ] 6.6 收尾(cleanup B=0 / `reconstruct_beam_logprobs` 父链回溯 / yield)

### 阶段二 · 难点深化(数值与张量级,配套阶段一)
> 文档:`mega_deep_dive.md` · 用贯穿例子 W=3/prefix_len=4/decode_steps=2 填实张量形状与索引

- ⬜ **单元 7 · 贯穿例子**(`§0`, L9-44)
  - [ ] 0.1 参数定义 / 0.2 Mega Request token 布局 / 0.3 ⭐ 每步重算全部后缀
- ⬜ **单元 8 · ⭐ 难点 A:position 重写**(`难点A`, L45-165)
  - [ ] A.1 为什么必须重写 / A.2 `_compute_beam_bounds` 语义 / A.3 ⭐ `_process_mega_decode_request` 逐 beam 推演 / A.4 ⭐ `InputBatchProxy` 展开 W 行 / A.5 logits_indices 对比 / A.6 张量形状汇总
- ⬜ **单元 9 · ⭐ 难点 B:cascade 索引**(`难点B`, L166-272)
  - [ ] B.1 朴素 vs cascade 成本 / B.2 ⭐ `_build_mega_mappings` 逐行数值推演 / B.3 prefix 段张量形状 / B.4 suffix 段 + gather / B.5 两段 attention 对照 / B.6 ⭐ `merge_attn_states` 数学推导
- ⬜ **单元 10 · ⭐ 难点 C:`_remux`**(`难点C`, L273-412)
  - [ ] C.1 采样器原始输出形状 / C.2 ⭐ `_remux` CASE B 逐行推演 / C.3 多请求 `gpu_row_cursor` / C.4 原位 resize_copy 必要性 / C.5 serving 侧 stride_k 逆运算 / C.6 pad 到 max_chained_dim 副作用
  - [ ] 附:贯穿示例完整流转表 + 易踩坑速查

### 阶段三 · 建立全局认知
> 文档:`beam_search_walkthrough.md §1-2` + `errata.md` ⚠️ walkthrough 部分概念为旧机制,先读 errata 再读旧文

- ⬜ **单元 11 · `errata.md` 勘误表**(L1-100)— 读旧文档前先看
  - [ ] A 事实错误(11)/ B 简化失真(6)/ C 概念遗漏(4)/ D 行号核对
- ⬜ **单元 12 · `beam_search_walkthrough.md §1-2`**(L1-168)
  - [ ] §1 概述 + 核心数据流 / §2 ⭐ 整体架构 + 文件拓扑 + patch 注入时序

### 阶段四 · 旧版本对照(演进参照)
> ⚠️ 机制已过时,仅理解演进脉络;cascade 三阶段 + LSE 合并数学在 `scheduling_cascade.md` 仍适用

- ⬜ **单元 13 · `online_pipeline.md`**(旧 BEAM_FORK / template-clone 全链路)
- ⬜ **单元 14 · `scheduling_cascade.md`**(调度 + cascade attention,cascade 数学仍适用)

### 阶段五 · GR 专章(推荐场景)
> 文档:`gr_specifics.md` · 做生成式推荐必读

- ⬜ **单元 15 · `gr_specifics.md`**(L1-199)
  - [ ] Catalog / Trie(前缀树做合法下一步约束)/ begin/end_token(SID)/ gr_features 类型扩展全链路透传

---

## 📝 阅读日志

> 每次会话追加一行:`日期 | 进度变化 | 本次覆盖 | 笔记/疑问`

- **2026-06-22** | 0/16 → 0/16(进行中:单元0) | 建立进度跟踪系统(本文件 + Claude 记忆双保险);通读 `online_pipeline_mega.md` 全文校准;下一步带读单元 0 + 单元 1。
- **2026-06-22** | 0/16 → 2/16(完成:单元0、1) | 带读单元 0(总览·全链路图+3主线)+ 单元 1(装配与入口:patch 接线/SID 偷渡/初始1条beam+延迟logprobs/`use_mega_request` 探测);已核对 `beam_search_patch.py`/`protocol.py`/`sampling_params.py`/`serving_engine.py:220-352` 实际代码。留下理解检查:首步为何走 ADD_BATCH。下一步单元 2。
- **2026-06-22** | 2/16 → 3/16(完成:单元2) | 带读单元 2(主循环骨架/`fork_info` 单向开关/catalog 并发/首步 ADD_BATCH 发送链路/`force_batch=True` 绕过短路/`_cache_beam_request` 种缓存/`np.argpartition` 1→W 展开/`fork_info` 翻转);揭晓上轮理解检查(首步为何走 ADD_BATCH)。已核对 `serving_engine.py:40-179,354-619`/`core_client.py:14-51`/`core.py:28-43,251-266`。下一步单元 3。
- **2026-06-23** | 3/16(不变) | 应要求**详细展开单元 2**:贯穿例子 beam_width=128 数字直觉;逐行讲主循环/`prepare_request_fn`(同步拆分准备+发送)/`_add_requests_batch`/`add_requests_async` 短路/ADD_BATCH 子进程分发/`_cache_beam_request` 9 字段/`beam_cache` 在 `patched_init` 初始化/枚举动态注入/FlatLogprobs 6 数组+去重/catalog 副本过滤/argpartition O(n)/`idx//logprobs_num` 映射;附状态变量演变表 + 7 条易踩坑。已核对 `async_llm.py:22-130`/`logprobs.py:1-85`/`serving_engine.py:191-217`/`engine_core_patch.py:22-31`。
- **2026-06-23** | 3/16 → 4/16(完成:单元3) | 应要求**详细展开单元 3**(同款深度):贯穿例子第2步数字直觉;`MegaRequestStepUpdate` 7+ 字段(msgspec array_like/omit_defaults/gc 性能)/物理vs逻辑三类数组;`_mega_request_step` 逐段(fork_info→三元映射/n=W/register session_id 队列/pruned 检测/第3步起才有pruned);`register_beam_output` 幽灵请求(跳过 assign_request_id);`mega_request_step_update_async` 强制沿用首步 DP rank + 维护 reqs_in_flight;`_handle_mega_request_step_update` 8 步(B==0清理/校验/取种子/拼 mega_token_ids 完整后缀非增量/构造单条 Request/继承 block_hashes KV根基/打5动态属性/入队+abort pruned);附物理vs逻辑路由表 + 闭环图 + 9 条易踩坑。已核对 `core.py:46-171`/`async_llm.py:133-177`/`types.py:1-45`/`core_client.py:59-82`。下一步单元 4。

---

## 📂 文档索引

| 阶段 | 文档 | 行数 | 优先级 |
|---|---|---|---|
| 一 | `online_pipeline_mega.md` | 894 | ⭐⭐⭐ |
| 二 | `mega_deep_dive.md` | 412 | ⭐⭐ |
| 三 | `errata.md` / `beam_search_walkthrough.md` | 100 / 1311 | ⭐ / ⭐ |
| 四 | `online_pipeline.md` / `scheduling_cascade.md` | 421 / 280 | ⭐ / ⭐ |
| 五 | `gr_specifics.md` | 199 | ⭐(推荐场景) |
