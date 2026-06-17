# vllm-gr 在线 BeamSearch 链路精读

**版本**: 1.0
**状态**: 基于代码逐行核对(serving_engine / async_llm / core_client / core / engine_core_patch)
**配套**: 概览见 `beam_search_walkthrough.md`,勘误见 `errata.md`

本文档是"从 HTTP 请求到 EngineCore 入队"的**在线服务通信链路**逐层精读,共 ❶–❽ 章。调度与计算见 `scheduling_cascade.md`。

---

## 调用链总览

```
HTTP 请求
  ❶ patch 注入:OpenAIServing.beam_search / AsyncLLM+AsyncMPClient 方法挂上
  ❷ to_beam_search_params → BeamSearchParams
  ❸ (准备)BeamForkRequest 结构
       │
  ❹ 主循环:每步生成 fork_info = [(直接父idx, token)...]
       │   ├─ Step 0: _add_batch_step   (fork_info=None)
       │   └─ Step 1+: _beam_fork_step  (fork_info 非空) → 组装 BeamForkRequest
       │
  ❺ AsyncLLM:prepare_request / register_beam_output(只注册 queue 不发)
       │
  ❻ AsyncMPClient:add_requests_async / beam_fork_async(路由 + _send_input 过 ZMQ)
       │
  ═══════════ 跨进程 ZMQ ═══════════
       │
  ❼ 子进程:run_engine_core wrapper 重新 patch → process_input_sockets 分发
       │   ├─ ADD_BATCH → 逐个 preprocess + _cache_beam_request + 进 input_queue
       │   └─ BEAM_FORK → _handle_beam_fork
       │
  ❽ EngineCore:_handle_beam_fork template-clone 创建子请求 → 进 input_queue
       │
       └─► (scheduler → model runner → cascade attention) 见 scheduling_cascade.md
```

---

## ❶ Patch 注入 — 方法是怎么挂上去的

**文件**: `plugin.py` / `patch.py` / `entrypoints/openai/beam_search_patch.py` / `v1/engine/core_client_patch.py`

### patch 全景表

| vLLM 原生类 | 被挂上的东西 | 来源 | 运行进程 |
|---|---|---|---|
| `OpenAIServing` | `.beam_search`(替换) | `vllm_gr/.../serving_engine.py` | 父进程 |
| `ChatCompletionRequest` | `.to_beam_search_params`(新增) | `vllm_gr/.../protocol.py` | 父进程 |
| `AsyncLLM` | `.prepare_request`/`._add_requests_batch`/`.register_beam_output`/`.beam_fork` | `vllm_gr/.../async_llm.py` | 父进程 |
| `AsyncMPClient` | `.add_requests_async`/`.beam_fork_async` | `vllm_gr/.../core_client.py` | 父进程 |
| `EngineCoreRequestType` | `+ADD_BATCH(b"\x05")` `+BEAM_FORK(b"\x06")` | `_add_enum_member` | **父子都要** |
| `EngineCoreProc` | `.run_engine_core`(替换为 wrapper) | `engine_core_patch.run_engine_core` | spawn 时触发 |
| `EngineCore`/`EngineCoreProc` | `._cache_beam_request`/`._handle_beam_fork`/`.process_input_sockets` | `apply_engine_core_child_patches` | **子进程** |

### 编排入口

`plugin.py:register()`(146-149)→ `initialize_runtime()`(108-143)→ `re_apply_patches()` → `_apply_patches()`(`patch.py:707-713`)依次调用:
- `_apply_gr_core_patches` — 替换核心类型(EngineCoreRequest/Request/NewRequestData 为 GR 版本,见 `gr_specifics.md`)
- `_apply_yuanrong_npu_patch` / `_apply_kv_merge_patch` — NPU/昇腾(可选)
- `_apply_gr_beam_patches`(`patch.py:317-437`)— 6 个 beam patch + 可选 LMCache

### `patch_batch_and_fork`(`beam_search_patch.py:19-40`)

```python
def patch_batch_and_fork():
    apply_batch_fork_patches()                              # 父进程绑客户端方法 + enum
    current = EngineCoreProc.run_engine_core
    if current is run_engine_core: return                  # ① 幂等
    original = getattr(run_engine_core, "_original_run_engine_core", current)  # ② 取原版
    if original is run_engine_core: original = current     # ③ 兜底
    run_engine_core._original_run_engine_core = original   # ④ 原版挂到函数对象属性(可 pickle 进子进程)
    EngineCoreProc.run_engine_core = run_engine_core       # ⑤ 替换为 wrapper
```

**核心**:把入口换成 wrapper,原版存到函数属性。`multiprocessing spawn` 起全新解释器(不继承 monkey-patch),所以子进程靠 wrapper 重新 patch 再启动原版。详见第 ❼ 章。

### `apply_batch_fork_patches`(`core_client_patch.py:30-52`)

```python
_add_enum_member("ADD_BATCH", b"\x05")     # enum 动态扩展(object.__new__ 绕过 enum 限制)
_add_enum_member("BEAM_FORK", b"\x06")
AsyncMPClient.add_requests_async = add_requests_async
AsyncMPClient.beam_fork_async = beam_fork_async
AsyncLLM.prepare_request = prepare_request_fn
AsyncLLM._add_requests_batch = _add_requests_batch_fn
AsyncLLM.register_beam_output = register_beam_output_fn
AsyncLLM.beam_fork = beam_fork_fn
```

**为什么 enum 父子都绑**:父进程**发**消息要构造枚举值;子进程**收**消息要 `bytes→enum` 反查。两进程的 `EngineCoreRequestType` 是各自 import 的独立类(spawn),所以各绑各的。

---

## ❷ 请求参数 — 从 HTTP 到 BeamSearchParams

**文件**: `entrypoints/openai/protocol.py:10-26` + `sampling_params.py:6-15`

### vllm-gr 对参数结构的扩展:仅两个字段

原生 `BeamSearchParams`(`vllm/vllm/sampling_params.py:1037`)有 beam_width/max_tokens/ignore_eos/temperature/length_penalty/include_stop_str_in_output/structured_outputs。vllm-gr 子类只加:

```python
class BeamSearchParams(BaseBeamSearchParams, omit_defaults=True, dict=True):
    begin_token: str | None = None    # session 起始 token
    end_token: str | None = None      # session 结束 token
```

### `to_beam_search_params`(`protocol.py:10-26`)

```python
def to_beam_search_params(self, max_tokens, default_sampling_params) -> BeamSearchParams:
    extra_args = self.vllm_xargs or {}                       # 非标准扩展容器
    n = self.n if self.n is not None else 1                  # ⭐ beam_width = OpenAI 的 n
    temperature = self.temperature or default_sampling_params.get("temperature", ...)  # 三级回退
    return BeamSearchParams(beam_width=n, max_tokens=max_tokens, ...,
                            begin_token=extra_args.get("begin_token"),
                            end_token=extra_args.get("end_token"))
```

**关键**:
- `beam_width` 复用 OpenAI 的 `n` 参数(`n=6` 即 6 路 beam)
- `begin_token`/`end_token` 走 `vllm_xargs`(非标准 OpenAI 字段)
- `max_tokens` 由调用方传入(要预留 begin/end token 位置)

### ⚠️ length_penalty 是"传了但没用"的字段

`to_beam_search_params` 提取 `length_penalty` 存进 `BeamSearchParams`(`:22`),但主循环 `serving_engine.py` **全文无 `params.length_penalty` 读取**,排序只用裸 `cum_logprob`(`:535`)。离线 `gr.py` 同样没用(只有 TODO)。当前 vllm-gr 的 beam search **不做长度归一化**。

---

## ❸ BeamForkRequest 数据结构

**文件**: `v1/engine/types.py:21-45`

```python
class BeamForkRequest(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    parent_ids: list[str]     # ─┐
    child_ids: list[str]      #  ├ 并行四元组
    token_ids: list[int]      #  │  parent_ids[i] + token_ids[i] → child_ids[i]
    abort_ids: list[str]      # ─┘ 独立:本步要中止的旧 beam
    sampling_params: SamplingParams    # 注意:是 SamplingParams(每步 max_tokens=1),不是 BeamSearchParams
    eos_token_id: int | None = None
    priority: int = 0                  # ⭐ beam 组身份(全 beam 共享,scheduler 据此分组)
    data_parallel_rank: int | None = None  # None=非DP; 0=DP rank0
    ...
```

**三个 msgspec 选项**:`array_like=True`(紧凑数组编码)/ `omit_defaults=True`(默认值字段不编码)/ `gc=False`(不进 GC 跟踪)。

**并行数组语义**:`parent_ids`/`child_ids`/`token_ids` 等长,第 i 位描述一次 fork;同一父可重复(一父多子)。`abort_ids` = 上轮所有 beam ID − 本轮出现的 parent。

---

## ❹ 主循环 — `serving_engine.beam_search`(核心)

**文件**: `entrypoints/openai/serving_engine.py`,`beam_search:156-580`

### 三个核心变量(必须分清)

| 变量 | 装什么 | 类比 |
|---|---|---|
| `all_beams` | 当前活着的路径(逻辑对象,含 tokens/cum_logprob/parent pointer) | 现在活着的这代人 |
| `prev_beam_internal_ids` | 这些路径在引擎里的请求 ID(字符串) | 它们的身份证号 |
| `fork_info` | 下一步的 fork 指令 `[(父在 all_beams 的 idx, token)]` | 下一代的生育计划 |

- `all_beams` 是算法层状态(引擎不认);`prev_beam_internal_ids` 是给引擎认人的身份证;`fork_info` 描述下一步动作。
- 三者不可互缺:只有 all_beams 引擎不认人;只有身份证不知道分数;只有 fork_info 不知道父母身份证。

### 初始化(`164-317`)

```python
logprobs_num = beam_width                                    # :252
beam_search_params = SamplingParams(logprobs=logprobs_num, max_tokens=1, detokenize=False, flat_logprobs=True)  # :253-259
initial_logprobs = FlatLogprobs()                           # :267 所有 beam 共享只读基底
all_beams = [BeamSearchSequence(tokens=initial_tokens, cum_logprob=0, logprobs=initial_logprobs)]  # :272 单 beam 起步
all_beams[0]._lp_parent = None; all_beams[0]._lp_step_data = None  # :288-289 parent chain 根
use_batch = hasattr(engine_client, "prepare_request")       # :304
fork_info = None                                             # :317 初始 None → 第一轮走 ADD_BATCH
```

### 一个 token 步的生命周期

```
① 扩展:给 N 条路径每条算 top-K 候选(引擎返回)
② 展平成候选池 all_beams_token_id / all_beams_logprob(N×K 个)
③ 打分:每个候选得分 = 父 cum_logprob + token logprob
④ 挑选:Top-K argpartition 选 beam_width 个
⑤ 记成 fork_info 喂给下一步
```

### Step 分发(`357-390`)— fork_info 是唯一开关

```python
if use_beam_fork and fork_info is not None:    # Step 1+: fork_info 非空
    output, prev_beam_internal_ids = await _beam_fork_step(... fork_info ...)
elif use_batch:                                # Step 0: fork_info 还是 None
    output, new_ids = await _add_batch_step(...)
```

- **第 0 步必走 ADD_BATCH**:没有父可 fork,每个初始 beam 是独立完整 prompt,要 prefill。
- **第 1 步起走 BEAM_FORK**:复用父 KV cache,只算新增 1 token(省 ~99% 计算)。
- **`fork_info` 是两步之间的状态桥**:本轮末尾生成(`:514`),下轮开头消费。

### 候选池展平(`397-459`)

```python
all_beams_token_id.extend(token_ids_pos)                         # 展平:每 beam 贡献 beam_width 个
all_beams_logprob.extend(current_beam.cum_logprob + lp for lp in logprobs_pos)  # 累加父 cum
```
候选池是一维数组,长度 N×beam_width。`idx // beam_width` 反推属于哪个父 beam。

### Top-K + new_beams + fork_info(`490-516`)

```python
topn_idx = np.argpartition(np.negative(all_beams_logprob), beam_width)[:beam_width]   # :491 O(n) 选 top-K
for idx in topn_idx:
    current_beam = all_beams[idx // logprobs_num]     # :496 反推父
    new_beam = BeamSearchSequence(tokens=current_beam.tokens + [token_id], cum_logprob=..., ...)
    new_beam._lp_parent = current_beam                # :508 挂父链
    new_beam._lp_step_data = beam_flat_cache[idx // logprobs_num]  # :509 存本步数据
fork_info = [(idx // logprobs_num, int(all_beams_token_id[idx])) for idx in topn_idx]  # :514 下一步指令
```

### 直接父 vs 祖宗(关键澄清)

- **`fork_info` 填直接父**(上一轮的 beam),不是祖宗。因为 BEAM_FORK 要做 KV 增量复用:子 = 直接父的 KV + 新算 1 token。
- **祖宗信息在 `_lp_parent` 链上**:每 beam 挂 `_lp_parent`(直接父)+ `_lp_step_data`(本步数据)。最后 `reconstruct_beam_logprobs`(`:541`)沿链回溯到根,重建完整 logprobs。中间步从不复制 → O(beams×steps) 而非 O(beams×steps²)。
- 引擎侧也是直接父链:每 req 的 `prompt_token_ids` 是**累积完整序列**(直接父的完整 prompt + 1 token),但 KV 是增量算出来的。

### 收尾(`519-580`)

```python
# cleanup:用空 fork(parent_ids=[])abort 残留 beam_cache
await engine_client.beam_fork(BeamForkRequest(..., abort_ids=prev_beam_internal_ids, ...))   # :519-529
sorted_completed = sorted(completed, key=lambda x: x.cum_logprob, reverse=True)   # :535 ⚠️ 无 length_penalty
best_beams = sorted_completed[:beam_width]
for beam in best_beams: reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id)  # :541-542
# decode text + metrics + yield RequestOutput
```

**error 处理**(`:411-442`):任一 beam 返回 error → 发空 fork abort 残留 cache → yield error → return 中止整个 beam search。**不变量:子进程 beam_cache 必须显式清理**。

---

## ❺ AsyncLLM 层 — 准备与发送

**文件**: `v1/engine/async_llm.py`

```
AsyncLLM (engine_client)
  ├─ input_processor      输入处理(tokenize)
  ├─ output_processor     输出处理器(管理 output queue)
  └─ engine_core = AsyncMPClient   真正发包的(第❻章)
```

| 方法 | 行号 | 发 ZMQ? | 作用 |
|---|---|---|---|
| `prepare_request_fn` | 22-116 | **不发** | tokenize + 建 Request + 注册 queue(故意同步,避免 event-loop 开销) |
| `_add_requests_batch_fn` | 119-130 | **发(ADD_BATCH)** | `engine_core.add_requests_async(reqs, force_batch=use_batch_message)` |
| `register_beam_output_fn` | 133-172 | **不发** | 为 fork child 建壳 Request + 注册 queue(child 由引擎侧创建) |
| `beam_fork_fn` | 175-177 | **发(BEAM_FORK)** | `engine_core.beam_fork_async(fork_request)` |

**两两配对**:
- ADD_BATCH 路径(Step 0):`prepare_request × N` → `_add_requests_batch × 1`
- BEAM_FORK 路径(Step 1+):`register_beam_output × N` → `beam_fork × 1`

**为什么 prepare/register 不发**:
- `prepare_request` 不发是为**批量**:同步攒齐 N 个再一次 ADD_BATCH,引擎一次处理。
- `register_beam_output` 不发是因为**根本不该它发**:child 由引擎侧 BEAM_FORK 创建,父进程只预留 queue 接输出。

---

## ❻ AsyncMPClient 层 — ZMQ 发送与 DP 路由

**文件**: `v1/engine/core_client.py`

| 方法 | 行号 | DP 路由方式 |
|---|---|---|
| `add_requests_async` | 14-51 | `get_core_engine_for_request(requests[0])` |
| `beam_fork_async` | 59-86 | `core_engines[fork_request.data_parallel_rank]` |

**两者都路由到 beam 开始时选定的同一 rank,但方式不同**:
- ADD_BATCH 用 `get_core_engine_for_request`(读 `requests[0].data_parallel_rank`,负载均衡 + 自动注册第一个到 reqs_in_flight)。
- BEAM_FORK **不能用**它:调用时父请求已输出完毕、从 `reqs_in_flight` 移除(`process_engine_outputs` 清的),且必须精确路由到父所在 rank。所以直接 `core_engines[data_parallel_rank]`。这就是 `BeamForkRequest` 要自带 `data_parallel_rank` 的原因。

**`reqs_in_flight` 的双重职责**:输出路由(输出回来时认 engine)+ abort 路由。BEAM_FORK 手动注册 `child_ids`、pop `abort_ids`。

**`force_batch=True` 的意义**:即使单个请求也走 ADD_BATCH,让 EngineCore 缓存进 `beam_cache`(普通 ADD 不缓存)。

**单 engine 场景**(`data_parallel_size=1`):无 `get_core_engine_for_request`,走最简分支 `_send_input(msg_type, payload)`,DP 逻辑全跳过。

---

## ❼ 子进程入口 — wrapper + 消息分发

**文件**: `engine_core_patch.py`(`run_engine_core:268-345`、`apply_engine_core_child_patches:206-261`)+ `core.py:process_input_sockets:190-286`

### `run_engine_core` wrapper

```python
def run_engine_core(*args, **kwargs):
    if _run_engine_core_active: raise RuntimeError("re-entry")        # 重入 guard
    original_run = getattr(run_engine_core, "_original_run_engine_core", None)  # 从函数属性取原版
    # ... 一连串校验确保不是自己(防递归)
    _run_engine_core_active = True
    try:
        apply_engine_core_child_patches(kwargs.get("vllm_config"))    # 子进程重新 patch
        original_run(*args, **kwargs)                                 # 启动原版引擎循环
    finally:
        _run_engine_core_active = False
```

**为什么 spawn 子进程要重新 patch**:multiprocessing spawn 起全新 Python 解释器,父进程的 monkey-patch 全没了。第❶章那套 `_original_run_engine_core` 保存 + 递归保护,就是为这一刻。

### `apply_engine_core_child_patches`(`206-261`)

子进程绑:enum 重新加 / `EngineCore.__init__` wrap(加 `beam_cache`+`beam_cache_lock`,`make_patched_engine_core_init:111-120`)/ 绑 `_cache_beam_request`+`_handle_beam_fork`+`process_input_sockets` / `apply_scheduler_patch`+`apply_worker_patches`(见 scheduling_cascade.md)/ 可选 LMCache。

### `process_input_sockets`(`core.py:190-286`)消息分发

| 消息 | 解码为 | 处理 | 进 input_queue? | 缓存 beam_cache? |
|---|---|---|---|---|
| ADD | 单 `EngineCoreRequest` | preprocess | ✓(标 ADD) | ✗ |
| ADD_BATCH | `list[EngineCoreRequest]` | 逐个 preprocess | ✓(每个标 ADD) | ✓(**每个都缓存**) |
| BEAM_FORK | `BeamForkRequest` | `_handle_beam_fork` | ✗(handler 内部塞) | handler 决定 |

- **ADD_BATCH vs ADD**:多一步 `_cache_beam_request`,为后续 BEAM_FORK 准备父状态。
- **BEAM_FORK 不走 input_queue**:直接调 `_handle_beam_fork`(立即创建子请求的元操作),子请求在 handler 内部才进 input_queue。

---

## ❽ EngineCore 处理 — template-clone

**文件**: `v1/engine/core.py`(`_cache_beam_request:28-41`、`_PER_CHILD_FIELDS:47-61`、`_handle_beam_fork:64-182`)

### `_cache_beam_request`(`28-41`)

```python
self.beam_cache[request.request_id] = (request, list(request.block_hashes))   # block_hashes 必须 snapshot
```
**为什么 snapshot**:scheduler 处理请求时会原地 append 新 hash,存引用会被污染。

### `_PER_CHILD_FIELDS`(`47-61`)

```python
_PER_CHILD_FIELDS = frozenset({
    "request_id", "prompt_token_ids", "num_prompt_tokens",
    "_all_token_ids", "_output_token_ids", "output_token_ids", "all_token_ids",
    "events", "spec_token_ids", "block_hashes", "get_hash_new_full_blocks",
})
```
集合里的字段**每个子各不同**(必须单独设);不在里面的字段**所有子相同**(可共享)。

⚠️ **脆弱点**:必须和上游 `vllm.v1.request.Request` 的可变字段同步。上游新增可变字段若漏加,所有子共享该对象 → 串扰。升级 vLLM 时这里是潜在 bug 源。

### `_handle_beam_fork`(`64-182`)template-clone

```python
for parent_id, child_id, token_id in zip(fork_req.parent_ids, fork_req.child_ids, fork_req.token_ids):
    cached_req, cached_block_hashes = self.beam_cache.get(parent_id, ...)
    child_token_ids = cached_req.prompt_token_ids + [token_id]     # 子 = 父完整序列 + 1

    if shared_fields is None:                                       # 第 1 个子:完整构造
        req = Request(request_id=child_id, prompt_token_ids=child_token_ids, ...)
        shared_fields = {k: v for k, v in req.__dict__.items() if k not in _PER_CHILD_FIELDS}  # 快照共享字段
    else:                                                           # 第 2~N 个子:lean clone
        req = object.__new__(Request)                              # 跳过 __init__!
        child_dict = shared_fields.copy()                          # 浅拷贝共享字段
        child_dict["request_id"] = child_id                        # 填独有字段
        child_dict["prompt_token_ids"] = child_token_ids
        ...
        req.__dict__ = child_dict

    req.block_hashes = list(cached_block_hashes)                   # 从父快照克隆
    if has_block_hasher:
        req.get_hash_new_full_blocks = partial(self.request_block_hasher, req)
        req.block_hashes.extend(req.get_hash_new_full_blocks())    # 增量算新 hash

    self._cache_beam_request(req)                                  # 子也缓存(为下一步 fork)
    self.input_queue.put_nowait((ADD, (req, fork_req.current_wave)))  # 进队等调度

# 清理父 + abort
for pid in set(fork_req.parent_ids) | set(fork_req.abort_ids):
    self.beam_cache.pop(pid, None)
```

**为什么省**:K 个子只付 1 次构造函数成本("第一个付全价,后续批发价")。beam_width=4~8 时省 75%~87%。

**lean clone 安全前提**:同 BEAM_FORK 所有子来自相同 prompt(共享字段值相同)+ scheduler 不会原地修改共享对象。若不变量破坏,需改 deep copy。

---

## 在线链路完整闭环

```
HTTP → ❶patch → ❷params → ❸BeamForkRequest
  → ❹主循环生成 fork_info
  → ❺AsyncLLM(准备不发) → ❻AsyncMPClient(ZMQ 发)
  → ❼子进程分发 → ❽template-clone 创建子请求 → input_queue
  → (scheduler → cascade attention) → 输出回流 → ❹ 收集 → 选 top-K → 下一步 fork_info
```

**两个递归点**形成闭环:
- **算法层**:每步 `fork_info` 喂下一步(❹ 内循环)
- **缓存层**:每步子请求被 `_cache_beam_request`,成下一步 fork 的父(❽→❼→❽)——KV cache 沿这条链逐层复用

---

## 附录:ADD_BATCH / BEAM_FORK ≈ Prefill / Decode

| | Prefill | Decode | → beam search |
|---|---|---|---|
| 触发 | 请求首次到达 | 后续每步 | ADD_BATCH(Step0)/ BEAM_FORK(Step1+) |
| 算什么 | 完整 prompt KV | 1 个新 token KV | 完整 beam prompt / 父+1token |
| KV 来源 | 现算 | 复用本请求已有 KV | ADD_BATCH 现算 / BEAM_FORK 复用父 KV |

三处不能过度类比:
1. Decode 是单请求增量;BEAM_FORK 是**跨请求**复用(子复用父 KV,靠 template-clone)。
2. Decode 线性;BEAM_FORK **树状分叉**(一父多子)。
3. beam search 每步 `max_tokens=1`,是"步进式 decode"(decode 1 个 → 选 beam → fork → 再 decode 1 个)。
