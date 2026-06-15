# vllm-gr BeamSearch 代码走读

**Version**: 1.0  
**Status**: Draft

---

## 目录

- [1. 概述](#1-概述)
- [2. 整体架构](#2-整体架构)
- [3. 插件注入与 Patch 机制](#3-插件注入与-patch-机制)
- [4. 入口层：离线 BeamSearch (gr.py)](#4-入口层离线-beamsearch-grpy)
- [5. 入口层：在线 BeamSearch (serving_engine.py)](#5-入口层在线-beamsearch-serving_enginepy)
- [6. 引擎通信层 (async_llm / core_client)](#6-引擎通信层-async_llm--core_client)
- [7. EngineCore 子进程处理 (core.py)](#7-enginecore-子进程处理-corepy)
- [8. BeamForkRequest 类型定义 (types.py)](#8-beamforkrequest-类型定义-typespy)
- [9. Scheduler 调度优化 (engine_core_patch.py)](#9-scheduler-调度优化-engine_core_patchpy)
- [10. GPUModelRunner 注入 (engine_core_patch.py)](#10-gpumodelrunner-注入-engine_core_patchpy)
- [11. Attention Backend (beam_attn.py)](#11-attention-backend-beam_attnpy)
- [12. FlatLogprobs 与延迟重建 (logprobs.py)](#12-flatlogprobs-与延迟重建-logprobspy)
- [13. 与原生 vLLM BeamSearch 的对比](#13-与原生-vllm-beamsearch-的对比)
- [14. 关键优化总结](#14-关键优化总结)

---

## 1. 概述

vllm-gr 对 vLLM 原生的 BeamSearch 进行了深度优化，主要围绕以下几个方面：

| 优化维度 | 原生 vLLM | vllm-gr |
|---------|----------|---------|
| Logprobs 存储 | `list[dict[int, Logprob]]` 每步复制 | FlatLogprobs + parent pointer 延迟重建 |
| 请求提交 | 每个 beam 独立 `generate()` 调用 | ADD_BATCH 批量提交 + BEAM_FORK 分叉 |
| KV Cache | 每个 beam 独立 prefill | 父请求 KV cache 复用，只计算新增 token |
| Attention | 标准 flash_attn_varlen_func | Cascade Attention（prefix/suffix 分离） |
| 候选过滤 | 无 | Catalog 过滤（推荐场景） |
| 数据并行 | 无 | Round-robin DP 路由 |
| Session 标记 | 无 | begin_token / end_token (SID) |

### 1.1 核心数据流

```text
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    vllm-gr BeamSearch 全链路                 │
                    └─────────────────────────────────────────────────────────────┘

HTTP/API 请求
    │
    ▼
┌──────────────────┐
│ 入口层            │  gr.py (离线) / serving_engine.py (在线)
│ 参数解析          │  BeamSearchParams (含 begin/end_token)
│ Catalog 过滤      │  FlatLogprobs + parent pointer 初始化
└────────┬─────────┘
         │
    ▼─────────────────────────────┐
    │  Step 0: ADD_BATCH          │  批量提交所有初始 beam
    │  Step 1+: BEAM_FORK         │  分叉父请求，复用 KV cache
    ▼                             │
┌──────────────────┐              │
│ 引擎通信层        │              │  async_llm.py → core_client.py
│ ZMQ 消息传递      │◄─────────────┘  ADD_BATCH / BEAM_FORK 消息类型
└────────┬─────────┘
         │
    ▼─────────────────────────────┐
    │  ZMQ DEALER socket          │
    ▼                             │
┌──────────────────┐              │
│ EngineCore 子进程  │              │  core.py: process_input_sockets
│ beam_cache 管理    │◄────────────┘  _handle_beam_fork: template-clone 优化
│ _cache_beam_request│              │  父请求状态缓存
└────────┬─────────┘
         │
    ▼─────────────────────────────┐
    │  Scheduler.schedule()        │
    ▼                             │
┌──────────────────┐              │
│ Scheduler 调度    │              │  engine_core_patch.py
│ computed_blocks   │              │  get_computed_blocks 缓存优化
│ beam_prefix_groups│◄────────────┘  按 priority 分组，计算 LCP
└────────┬─────────┘
         │
    ▼─────────────────────────────┐
    │  ContextVar 传递             │
    ▼                             │
┌──────────────────┐              │
│ Attention Backend │              │  beam_attn.py
│ Cascade Attention │              │  prefix pass (non-causal)
│ LSE 合并          │◄────────────┘  suffix pass (causal)
│ Triton Kernel     │              │  merge_indexed_attn_states
└────────┬─────────┘
         │
    ▼
最终结果组装
    │  reconstruct_beam_logprobs()
    ▼
返回 BeamSearchOutput
```

---

## 2. 整体架构

### 2.1 文件拓扑

```text
vllm_gr/
├── __init__.py                          # 包入口，lazy import
├── plugin.py                            # vLLM plugin 注册入口
├── patch.py                             # 中央 monkey-patch 编排器
├── sampling_params.py                   # 扩展 BeamSearchParams
├── logprobs.py                          # FlatLogprobs 工具函数
├── entrypoints/
│   ├── gr.py                            # 离线 GRLLM.beam_search
│   ├── beam_search_patch.py             # 离线 patch
│   └── openai/
│       ├── serving_engine.py            # 在线 beam_search
│       ├── protocol.py                  # to_beam_search_params
│       └── beam_search_patch.py         # 在线 patch
└── v1/
    ├── engine/
    │   ├── async_llm.py                 # AsyncLLM 注入方法
    │   ├── core_client.py               # AsyncMPClient 注入方法
    │   ├── core_client_patch.py         # 方法绑定
    │   ├── core.py                      # EngineCore 侧处理
    │   ├── types.py                     # BeamForkRequest
    │   └── engine_core_patch.py         # Scheduler/Worker patches
    └── attention/
        └── backends/
            └── beam_attn.py             # 自定义 Attention Backend
```

### 2.2 Patch 注入时序

```text
vLLM 启动
    │
    ▼
load_general_plugins()
    │
    ▼
plugin.py: register()
    │
    ▼
initialize_runtime()
    │
    ├── _apply_gr_core_patches()     # EngineCoreRequest/Request 替换
    ├── _apply_gr_beam_patches()     # 6 个独立 patch
    │   ├── patch_flat_logprobs()
    │   ├── patch_beam_search()
    │   ├── patch_sampling()
    │   ├── patch_add_cli_args()
    │   ├── patch_OpenAIServingModels_init()
    │   └── patch_batch_and_fork()   # ADD_BATCH/BEAM_FORK 注入
    │       ├── apply_batch_fork_patches()
    │       │   ├── AsyncMPClient.add_requests_async
    │       │   ├── AsyncMPClient.beam_fork_async
    │       │   ├── AsyncLLM.prepare_request
    │       │   ├── AsyncLLM._add_requests_batch
    │       │   ├── AsyncLLM.register_beam_output
    │       │   └── AsyncLLM.beam_fork
    │       └── EngineCoreProc.run_engine_core → run_engine_core wrapper
    └── register configs / models / adapters
```

---

## 3. 插件注入与 Patch 机制

### 3.1 插件注册入口

**文件**: `vllm_gr/plugin.py` (lines 146-149)

```python
def register():
    """Entry point called by vLLM's load_general_plugins()."""
    initialize_runtime()
```

`initialize_runtime()` (lines 108-143) 是核心初始化函数，按顺序执行：

1. `re_apply_patches()` — 扫描 `sys.modules` 并应用所有 monkey-patches
2. `register_gr_configs()` — 注册 HSTU 等模型配置
3. `register_gr_model_architectures()` — 注册 GR 模型架构
4. `_register_adapter_families()` — 自动发现 adapter 包

### 3.2 Beam Search Patch 编排

**文件**: `vllm_gr/patch.py` (lines 317-437)

```python
def _apply_gr_beam_patches():
    """Apply 6 independent beam-search patches."""
    patches = [
        ("flat_logprobs", patch_flat_logprobs),
        ("beam_search", patch_beam_search),
        ("sampling", patch_sampling),
        ("cli_args", patch_add_cli_args),
        ("openai_models", patch_OpenAIServingModels_init),
        ("batch_and_fork", patch_batch_and_fork),
    ]
    for name, patch_fn in patches:
        try:
            patch_fn()
        except Exception as exc:
            logger.warning("Failed to apply %s patch: %s", name, exc)
```

每个 patch 独立应用，互不影响。关键 patch 说明：

| Patch 名称 | 作用 | 替换目标 |
|-----------|------|---------|
| `patch_flat_logprobs` | 注入 FlatLogprobs 支持 | vLLM 的 logprobs 处理 |
| `patch_beam_search` | 替换在线 beam_search | `OpenAIServing.beam_search` |
| `patch_sampling` | 注入 `to_beam_search_params` | `ChatCompletionRequest` |
| `patch_batch_and_fork` | 注入 ADD_BATCH/BEAM_FORK | `AsyncMPClient` + `EngineCoreProc` |

### 3.3 batch_and_fork Patch 详解

**文件**: `vllm_gr/entrypoints/openai/beam_search_patch.py` (lines 19-40)

```python
def patch_batch_and_fork():
    """Wire ADD_BATCH + BEAM_FORK into the engine pipeline."""
    from vllm.v1.engine.core import EngineCoreProc
    from vllm_gr.v1.engine.core_client_patch import apply_batch_fork_patches

    # Save original for recursion guard
    original_run = EngineCoreProc.run_engine_core
    run_engine_core._original_run_engine_core = original_run

    # Replace run_engine_core with GR wrapper
    EngineCoreProc.run_engine_core = run_engine_core

    # Inject AsyncMPClient + AsyncLLM methods
    apply_batch_fork_patches()
```

**文件**: `vllm_gr/v1/engine/core_client_patch.py` (lines 30-52)

```python
def apply_batch_fork_patches():
    """Add ADD_BATCH/BEAM_FORK enum members and inject methods."""
    # Add enum members
    _add_enum_member("ADD_BATCH", b"\x05")
    _add_enum_member("BEAM_FORK", b"\x06")

    # Patch AsyncMPClient
    AsyncMPClient.add_requests_async = add_requests_async
    AsyncMPClient.beam_fork_async = beam_fork_async

    # Patch AsyncLLM
    AsyncLLM.prepare_request = prepare_request_fn
    AsyncLLM._add_requests_batch = _add_requests_batch_fn
    AsyncLLM.register_beam_output = register_beam_output_fn
    AsyncLLM.beam_fork = beam_fork_fn
```

---

## 4. 入口层：离线 BeamSearch (gr.py)

### 4.1 GRLLM 类

**文件**: `vllm_gr/entrypoints/gr.py` (lines 39-611)

```python
class GRLLM(LLM):
    """GR offline interface, extends vLLM's LLM class."""
    def __init__(self, model, catalog_path=None, ...):
        # Resolve model_type from config.json or HuggingFace
        # Initialize catalog for beam search filtering
        self.catalog = Catalog(catalog_path) if catalog_path else None
        super().__init__(model, ...)
```

### 4.2 beam_search 主循环

**文件**: `vllm_gr/entrypoints/gr.py` (lines 280-593)

核心流程：

```text
初始化 beam_width 个 BeamSearchSequence
    │
    ├── 创建 FlatLogprobs (共享引用)
    ├── 设置 parent pointer (_lp_parent, _lp_step_data)
    ├── 如果有 begin_token，用 append_fast 追加
    │
    ▼
循环直到所有 beam 完成或达到 max_tokens:
    │
    ├── 1. 构造 SamplingParams(flat_logprobs=True, max_tokens=1)
    ├── 2. 调用 self.generate() 执行一步解码
    ├── 3. 提取 logprobs:
    │       ├── extract_and_dedup_flat_logprobs() — 去重
    │       └── Catalog 过滤 — 非法 token 设为 -inf
    ├── 4. 处理 EOS token:
    │       ├── 创建完成序列，设置 parent pointer
    │       └── EOS token logprob 设为 -inf
    ├── 5. Top-K beam 选择:
    │       ├── np.argpartition 高效选择
    │       └── 创建新 beam，链接 parent pointer
    └── 6. 长度惩罚排序
    │
    ▼
最终:
    ├── reconstruct_beam_logprobs() — 沿 parent chain 重建
    └── 返回 BeamSearchOutput
```

**关键代码片段**:

**FlatLogprobs 初始化** (lines 370-386):
```python
sampling_params = SamplingParams(
    flat_logprobs=True,
    logprobs=beam_width,
    max_tokens=1,
    ...
)
initial_logprobs = FlatLogprobs()
if begin_token_id is not None:
    append_fast(initial_logprobs, [begin_token_id], ...)
```

**Parent Pointer 设置** (lines 415-423):
```python
beam._lp_parent = parent_beam  # 指向父 beam
beam._lp_step_data = (token_id, step_logprobs)  # 当前步数据
```

**Catalog 过滤** (lines 476-505):
```python
if self.catalog is not None:
    valid_tokens = self.catalog.get_valid_tokens(current_item_ids)
    for token_id in logprobs_dict:
        if token_id not in valid_tokens:
            logprobs_dict[token_id].logprob = -float("inf")
```

**Top-K 选择** (lines 549-573):
```python
top_indices = np.argpartition(scores, -beam_width)[-beam_width:]
for idx in top_indices:
    new_beam = BeamSearchSequence(...)
    new_beam._lp_parent = parent_beams[idx]
    new_beam._lp_step_data = (selected_token, step_logprobs)
```

---

## 5. 入口层：在线 BeamSearch (serving_engine.py)

### 5.1 核心差异（对比离线版本）

| 维度 | 离线 (gr.py) | 在线 (serving_engine.py) |
|------|-------------|------------------------|
| 执行模式 | 同步 `generate()` | 异步 `async for` |
| 请求提交 | `self.generate()` | ADD_BATCH + BEAM_FORK |
| 请求类型 | `PromptInputs` | `EngineCoreRequest` |
| DP 路由 | 不支持 | Round-robin 分配 |
| 输出方式 | 直接返回 | `async yield RequestOutput` |

### 5.2 DP 路由

**文件**: `vllm_gr/entrypoints/openai/serving_engine.py` (lines 32-38)

```python
async def _next_data_parallel_rank(self) -> int:
    """Round-robin DP rank assignment."""
    async with self._dp_lock:
        rank = self._dp_counter % self.data_parallel_size
        self._dp_counter += 1
    return rank
```

所有属于同一次 beam search 的请求共享相同的 `priority`（微秒时间戳），确保 scheduler 将它们视为同一优先级组。

### 5.3 ADD_BATCH 路径 (Step 0)

**文件**: `vllm_gr/entrypoints/openai/serving_engine.py` (lines 118-153)

```python
async def _add_batch_step(self, prompts, sampling_params, ...):
    """Step 0: batch-submit all initial beams via ADD_BATCH."""
    prepared = []
    for prompt in prompts:
        output_queue, request = await engine_client.prepare_request(prompt, ...)
        prepared.append((output_queue, request))

    # Batch send — triggers ADD_BATCH message type
    await engine_client._add_requests_batch(prepared)
    return output_list, internal_ids
```

### 5.4 BEAM_FORK 路径 (Step 1+)

**文件**: `vllm_gr/entrypoints/openai/serving_engine.py` (lines 55-115)

```python
async def _beam_fork_step(self, fork_info, abort_ids, ...):
    """Step 1+: fork parent beams via BEAM_FORK."""
    for parent_idx, token_id in fork_info:
        child_id = f"{parent_id}_beam{step}_{beam_idx}"
        # Register output queue for child
        await engine_client.register_beam_output(child_id, ...)

    # Send BEAM_FORK message
    await engine_client.beam_fork(BeamForkRequest(
        parent_ids=[...],
        child_ids=[...],
        token_ids=[...],
        abort_ids=abort_ids,
        sampling_params=sampling_params,
        priority=priority,
    ))
```

### 5.5 主循环流程

**文件**: `vllm_gr/entrypoints/openai/serving_engine.py` (lines 156-581)

```text
async def beam_search(self, request, ...):
    │
    ├── DP 路由: 分配 data_parallel_rank
    ├── Priority 分配: int(time.perf_counter() * 1_000_000)
    ├── 初始化 FlatLogprobs + parent pointers
    │
    ▼
    循环 (step = 0, 1, 2, ...):
        │
        ├── Step 0: _add_batch_step()
        │       └── 所有初始 beam 批量提交
        │
        ├── Step 1+: _beam_fork_step()
        │       └── 分叉父 beam，复用 KV cache
        │
        ├── 收集所有 beam 的输出 (async gather)
        │
        ├── Catalog 过滤 (asyncio.to_thread 并行)
        │
        ├── EOS 处理 + parent pointer 链接
        │
        ├── Top-K beam 选择 (np.argpartition)
        │
        └── 更新 fork_info 供下一步使用
    │
    ▼
    最终:
        ├── reconstruct_beam_logprobs()
        ├── 记录 metrics (overhead, decode_time, num_tokens)
        └── yield RequestOutput
```

---

## 6. 引擎通信层 (async_llm / core_client)

### 6.1 AsyncLLM 注入方法

**文件**: `vllm_gr/v1/engine/async_llm.py`

| 方法 | Lines | 说明 |
|------|-------|------|
| `prepare_request_fn` | 22-116 | 处理输入并注册输出队列，但 **不发送** 到 EngineCore。返回 `(output_queue, request)` 供后续批量发送 |
| `_add_requests_batch_fn` | 119-130 | 批量发送多个已准备的请求到 EngineCore，强制使用 ADD_BATCH 消息类型 |
| `register_beam_output_fn` | 133-172 | 为 beam fork 子请求注册输出队列，不发送到 EngineCore（子请求由 BEAM_FORK 处理器创建） |
| `beam_fork_fn` | 175-177 | 发送 `BeamForkRequest` 到 EngineCore |

**`prepare_request_fn` 关键逻辑** (lines 22-116):
```python
def prepare_request_fn(self, request, sampling_params, ...):
    """Prepare request but DO NOT send to EngineCore.
    Returns (output_queue, engine_core_request) for batch send."""
    # Create output queue
    output_queue = asyncio.Queue()
    self.output_queues[request_id] = output_queue

    # Build EngineCoreRequest (without sending)
    request = EngineCoreRequest(request_id=request_id, ...)
    return output_queue, request
```

**`_add_requests_batch_fn` 关键逻辑** (lines 119-130):
```python
async def _add_requests_batch_fn(self, prepared_requests):
    """Batch-send prepared requests via ADD_BATCH."""
    requests = [req for _, req in prepared_requests]
    await self.engine_client.add_requests_async(
        requests, use_batch_message=True  # Force ADD_BATCH
    )
```

### 6.2 AsyncMPClient 注入方法

**文件**: `vllm_gr/v1/engine/core_client.py`

**`add_requests_async`** (lines 14-51):
```python
async def add_requests_async(self, requests, use_batch_message=False):
    """Send multiple requests as ADD_BATCH or individual ADD."""
    if use_batch_message:
        # Single ADD_BATCH message containing all requests
        self._send(EngineCoreRequestType.ADD_BATCH, requests)
    else:
        for req in requests:
            self._send(EngineCoreRequestType.ADD, req)
```

**`beam_fork_async`** (lines 59-86):
```python
async def beam_fork_async(self, fork_request):
    """Send BEAM_FORK to EngineCore with DP routing."""
    # Route to correct DP engine
    engine = self.get_core_engine_for_request(
        fork_request.data_parallel_rank
    )
    # Register child IDs for abort routing
    for child_id in fork_request.child_ids:
        self.reqs_in_flight.add(child_id)
    # Remove aborted parent IDs
    for abort_id in fork_request.abort_ids:
        self.reqs_in_flight.discard(abort_id)

    engine._send(EngineCoreRequestType.BEAM_FORK, fork_request)
```

---

## 7. EngineCore 子进程处理 (core.py)

### 7.1 beam_cache 缓存

**文件**: `vllm_gr/v1/engine/core.py` (lines 28-41)

```python
def _cache_beam_request(self, request) -> None:
    """Cache Request state for later beam forking.
    Stores (request, block_hashes_snapshot)."""
    with self.beam_cache_lock:
        self.beam_cache[request.request_id] = (
            request,
            list(request.block_hashes),  # snapshot
        )
```

### 7.2 _handle_beam_fork — 模板克隆优化

**文件**: `vllm_gr/v1/engine/core.py` (lines 64-182)

核心优化：**template-clone**

```python
_PER_CHILD_FIELDS = frozenset({
    "request_id", "prompt_token_ids", "num_prompt_tokens",
    "_all_token_ids", "_output_token_ids", "output_token_ids",
    "all_token_ids", "events", "spec_token_ids",
    "block_hashes", "get_hash_new_full_blocks",
})

def _handle_beam_fork(self, fork_req) -> None:
    shared_fields = None

    for parent_id, child_id, token_id in zip(...):
        cached_req, cached_block_hashes = self.beam_cache[parent_id]
        child_token_ids = cached_req.prompt_token_ids + [token_id]

        if shared_fields is None:
            # First child — full constructor + snapshot shared fields
            req = Request(request_id=child_id, ...)
            shared_fields = {
                k: v for k, v in req.__dict__.items()
                if k not in _PER_CHILD_FIELDS
            }
        else:
            # Children 2..N — lean clone
            req = object.__new__(Request)
            child_dict = shared_fields.copy()
            child_dict["request_id"] = child_id
            child_dict["prompt_token_ids"] = child_token_ids
            # ... set per-child fields
            req.__dict__ = child_dict

        # Clone block hashes + compute new ones
        req.block_hashes = list(cached_block_hashes)
        req.block_hashes.extend(req.get_hash_new_full_blocks())

        self._cache_beam_request(req)
        self.input_queue.put_nowait((EngineCoreRequestType.ADD, req))
```

**性能分析**：

| 子请求 | 创建方式 | 开销 |
|--------|---------|------|
| 第 1 个 | 完整 `Request()` 构造函数 | 高（验证、初始化） |
| 第 2~N 个 | `object.__new__` + dict copy | 低（跳过构造函数） |

### 7.3 process_input_sockets — ZMQ 消息分发

**文件**: `vllm_gr/v1/engine/core.py` (lines 190-286)

```python
def process_input_sockets(self, input_addresses, ...):
    """Replacement for EngineCoreProc.process_input_sockets."""
    add_request_decoder = MsgpackDecoder(EngineCoreRequest)
    add_batch_decoder = MsgpackDecoder(list[EngineCoreRequest])
    beam_fork_decoder = MsgpackDecoder(BeamForkRequest)

    while True:
        for socket, _ in poller.poll():
            type_frame, *data_frames = socket.recv_multipart()
            request_type = EngineCoreRequestType(bytes(type_frame.buffer))

            if request_type == EngineCoreRequestType.ADD:
                req = add_request_decoder.decode(data_frames)
                request = self.preprocess_add_request(req)

            elif request_type == EngineCoreRequestType.ADD_BATCH:
                batch = add_batch_decoder.decode(data_frames)
                for req in batch:
                    request = self.preprocess_add_request(req)
                    self._cache_beam_request(request[0])
                    self.input_queue.put_nowait((ADD, request))

            elif request_type == EngineCoreRequestType.BEAM_FORK:
                fork_req = beam_fork_decoder.decode(data_frames)
                self._handle_beam_fork(fork_req)
```

---

## 8. BeamForkRequest 类型定义 (types.py)

**文件**: `vllm_gr/v1/engine/types.py` (lines 21-45)

```python
class BeamForkRequest(
    msgspec.Struct,
    array_like=True,    # 数组序列化，节省带宽
    omit_defaults=True, # 省略默认值字段
    gc=False,           # 无 GC，提升性能
):
    """Lightweight beam search fork request."""

    parent_ids: list[str]      # 父请求 ID 列表
    child_ids: list[str]       # 子请求 ID 列表
    token_ids: list[int]       # 追加的 token ID 列表
    abort_ids: list[str]       # 需要中止的请求 ID
    sampling_params: SamplingParams
    eos_token_id: int | None = None
    client_index: int = 0
    current_wave: int = 0
    priority: int = 0          # beam 组优先级（所有 beam 共享）
    data_parallel_rank: int | None = None
    lora_request: LoRARequest | None = None
    cache_salt: str | None = None
    trace_headers: Mapping[str, str] | None = None
```

**设计要点**：

- **并行数组**: `parent_ids[i]` → `child_ids[i]` + `token_ids[i]` 表示一个 fork 操作
- **msgspec 序列化**: `array_like=True` 使用数组而非字典，减少 ZMQ 传输开销
- **priority 字段**: 同一 beam search 的所有请求共享相同的 priority，用于 scheduler 分组

---

## 9. Scheduler 调度优化 (engine_core_patch.py)

### 9.1 _get_lcp — 最长公共前缀

**文件**: `vllm_gr/v1/engine/engine_core_patch.py` (lines 20-50)

```python
def _get_lcp(a: list, b: list, hint: int = 0) -> int:
    """Find longest common prefix between two hash lists.
    Uses binary search with hint optimization."""
    m = min(len(a), len(b))

    # Hint optimization: check if previous LCP is still valid
    if hint > 0:
        h = min(hint, m)
        if a[h - 1] == b[h - 1]:
            if h == m or a[h] != b[h]:
                return h  # LCP unchanged
            start = h     # LCP extended
        else:
            end = h - 2   # LCP shrunk

    # Binary search for LCP boundary
    low, high = start, end
    while low <= high:
        mid = (low + high) // 2
        if a[mid] == b[mid]:
            ans = mid + 1
            low = mid + 1
        else:
            high = mid - 1
    return ans
```

**hint 优化原理**：利用上一次计算的 LCP 值作为 hint，避免从头开始比较。在 beam search 中，相邻步的 LCP 通常变化不大。

### 9.2 _compute_beam_prefix_groups

**文件**: `vllm_gr/v1/engine/engine_core_patch.py` (lines 53-103)

```python
def _compute_beam_prefix_groups(req_ids, requests, block_size):
    """Group requests by beam priority and find common prefix depth."""
    min_group_th = 256 // block_size  # 最少 256 tokens 共享才值得分组

    # 按 priority (beam 组 ID) 分组
    beam_groups = {}
    for req_id in req_ids:
        g = requests[req_id].priority
        beam_groups.setdefault(g, []).append(req_id)

    all_groups = []
    for g, group_req_ids in beam_groups.items():
        # 计算组内相邻请求的最小 LCP
        min_lcp = float("inf")
        last_lcp = 0
        for i in range(n - 1):
            common_len = _get_lcp(
                req1.block_hashes, req2.block_hashes, hint=last_lcp
            )
            min_lcp = min(min_lcp, common_len)
            last_lcp = common_len

        # 低于阈值的组不分组
        if min_lcp < min_group_th:
            min_lcp = 0

        all_groups.append((min_lcp, group_req_ids))
    return all_groups
```

### 9.3 apply_scheduler_patch — computed_blocks 缓存

**文件**: `vllm_gr/v1/engine/engine_core_patch.py` (lines 348-429)

```python
def apply_scheduler_patch():
    """Patch Scheduler.schedule with cached get_computed_blocks."""

    def patched_schedule(self):
        computed_blocks_cache = {}

        def get_cache_computed_blocks(req):
            cache_th = 256  # 缓存阈值
            key = (req.num_tokens, req.skip_reading_prefix_cache, last_hash)

            if key in computed_blocks_cache:
                return computed_blocks_cache[key]  # 缓存命中

            res = _original_get_computed_blocks(req)
            if res[1] > cache_th:  # 超过阈值才缓存
                computed_blocks_cache[key] = res
            return res

        self.kv_cache_manager.get_computed_blocks = get_cache_computed_blocks
        output = _original_schedule(self)

        # 计算 beam_prefix_groups 并附加到输出
        req_ids = list(output.num_scheduled_tokens.keys())
        output.beam_prefix_groups = _compute_beam_prefix_groups(
            req_ids, self.requests, self.cache_config.block_size
        )
        return output
```

**缓存策略**：

- 同一 beam 组中，第一个请求调用 `get_computed_blocks` 后结果被缓存
- 后续请求如果 key 相同，直接复用缓存结果
- 只有共享 block 数 > 256 的结果才值得缓存（避免缓存污染）

---

## 10. GPUModelRunner 注入 (engine_core_patch.py)

**文件**: `vllm_gr/v1/engine/engine_core_patch.py` (lines 432-479)

### 10.1 execute_model Patch

```python
def patched_execute_model(self, scheduler_output, ...):
    """Extract beam_prefix_groups from scheduler_output."""
    beam_groups = getattr(scheduler_output, "beam_prefix_groups", None)
    self._current_beam_prefix_groups_str = beam_groups
    try:
        return _original_execute_model(self, scheduler_output, ...)
    finally:
        self._current_beam_prefix_groups_str = None
```

### 10.2 _build_attention_metadata Patch

```python
def patched_build_attention_metadata(self, *args, **kwargs):
    """Translate beam groups and set ContextVar."""
    beam_groups_str = getattr(self, "_current_beam_prefix_groups_str", None)
    if beam_groups_str is not None:
        # Translate request IDs to batch indices
        req_id_to_idx = {
            r_id: i for i, r_id in enumerate(self.input_batch.req_ids)
        }
        translated_groups = []
        for depth, group_req_ids in beam_groups_str:
            indices = [req_id_to_idx[r] for r in group_req_ids if r in req_id_to_idx]
            if indices:
                translated_groups.append((depth, indices))

        token = BEAM_PREFIX_GROUPS_VAR.set(translated_groups)
        try:
            return _original_build_attention_metadata(self, *args, **kwargs)
        finally:
            BEAM_PREFIX_GROUPS_VAR.reset(token)
```

**ContextVar 传递机制**：

```text
Scheduler.schedule()
    │ 计算 beam_prefix_groups: [(depth, [req_ids])]
    ▼
SchedulerOutput.beam_prefix_groups
    │
    ▼
GPUModelRunner.execute_model()
    │ 保存到 self._current_beam_prefix_groups_str
    ▼
GPUModelRunner._build_attention_metadata()
    │ 转换 req_ids → batch indices
    │ 设置 BEAM_PREFIX_GROUPS_VAR (ContextVar)
    ▼
BeamAttentionMetadataBuilder.build()
    │ 读取 BEAM_PREFIX_GROUPS_VAR
    │ 决定使用标准 metadata 还是 shared metadata
    ▼
cascade_attention / flash_attn_varlen_func
```

---

## 11. Attention Backend (beam_attn.py)

### 11.1 整体结构

**文件**: `vllm_gr/v1/attention/backends/beam_attn.py` (lines 1-1198)

```text
BeamAttentionBackend (注册为 CUSTOM backend)
    │
    ├── BeamAttentionMetadata (dataclass)
    │   ├── 标准字段 (query_start_loc, block_table, ...)
    │   ├── prefix_block_table     # 共享前缀的 block 表
    │   ├── suffix_block_table     # 非共享后缀的 block 表
    │   ├── prefix_indices         # scatter 映射索引
    │   └── use_cascade_attention  # 是否使用级联 attention
    │
    ├── BeamAttentionMetadataBuilder
    │   ├── build()                # 入口：检查 ContextVar
    │   ├── _build_shared_metadata # 构建 prefix + suffix metadata
    │   ├── _create_prefix_metadata# 构建前缀 block table + 查询索引
    │   └── _create_suffix_metadata# 构建后缀 block table (列偏移)
    │
    └── BeamAttentionImpl
        ├── cascade_attention()    # 三阶段级联 attention
        └── forward()              # 主 forward 分发
```

### 11.2 BeamAttentionMetadataBuilder.build()

```python
def build(self, common_prefix_len, ...):
    """Build attention metadata.
    Checks BEAM_PREFIX_GROUPS_VAR for beam groups."""
    beam_groups = BEAM_PREFIX_GROUPS_VAR.get(None)

    if beam_groups is not None and len(beam_groups) > 0:
        # Beam groups exist — build shared metadata
        return self._build_shared_metadata(beam_groups, ...)
    else:
        # Standard metadata (no beam optimization)
        return self._build_standard_metadata(...)
```

### 11.3 _create_prefix_metadata — 前缀构建

**文件**: `vllm_gr/v1/attention/backends/beam_attn.py` (lines 583-681)

```python
def _create_prefix_metadata(self, beam_groups, block_table, ...):
    """Build prefix block table from shared blocks.

    For each beam group:
    1. Take the first request's block table as reference
    2. Extract blocks [0:prefix_depth] as shared prefix blocks
    3. Build prefix_indices: scatter map for gathering queries
       (multiple beams share the same prefix query)
    4. Build cu_prefix_query_lens: cumulative query lengths
    """
    prefix_block_table = torch.zeros(...)
    prefix_indices = torch.zeros(...)

    for depth, req_indices in beam_groups:
        # First request in group is the reference
        first_req = req_indices[0]
        # Copy shared blocks from reference
        prefix_block_table[group_start:group_end, :depth] = \
            block_table[first_req, :depth]
        # Build scatter indices
        for i, req_idx in enumerate(req_indices):
            prefix_indices[scatter_offset + i] = req_idx
```

### 11.4 _create_suffix_metadata — 后缀构建

**文件**: `vllm_gr/v1/attention/backends/beam_attn.py` (lines 683-740)

```python
def _create_suffix_metadata(self, beam_groups, block_table, ...):
    """Build suffix block table by column-shifting.

    For each beam group:
    1. suffix_kv_len = total_kv_len - prefix_kv_len
    2. Shift block table columns: block_table[:, prefix_depth:]
       becomes the suffix block table
    3. Adjust query start positions for suffix pass
    """
    suffix_block_table = torch.zeros(...)
    for depth, req_indices in beam_groups:
        # Column shift: take blocks after prefix depth
        suffix_block_table[group_rows, :] = \
            block_table[req_indices, depth:]
```

### 11.5 Cascade Attention — 三阶段

**文件**: `vllm_gr/v1/attention/backends/beam_attn.py` (lines 912-1020)

```python
@staticmethod
def cascade_attention(
    output, query, key_cache, value_cache,
    cu_prefix_query_lens, cu_suffix_query_lens,
    prefix_kv_lens, suffix_kv_lens,
    prefix_block_table, suffix_block_table,
    prefix_indices, prefix_indices_is_identity,
    ...
):
    """Three-phase cascade attention.

    Phase 1: Prefix attention on shared blocks (causal=False)
    Phase 2: Suffix attention on per-request blocks (causal=True)
    Phase 3: Merge via LSE normalization
    """

    # Phase 1: Prefix pass
    prefix_out, prefix_lse = flash_attn_varlen_func(
        q=query[prefix_indices],  # scattered queries
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        causal=False,             # non-causal for prefix
        block_table=prefix_block_table,
        return_softmax_lse=True,  # need LSE for merge
    )

    # Phase 2: Suffix pass
    suffix_out, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_suffix_query_lens,
        seqused_k=suffix_kv_lens,
        causal=True,              # causal for suffix
        block_table=suffix_block_table,
        return_softmax_lse=True,
    )

    # Phase 3: Merge
    if prefix_indices_is_identity:
        merge_attn_states(output, prefix_out, prefix_lse, suffix_out, suffix_lse)
    else:
        merge_indexed_attn_states(output, suffix_lse, prefix_out, prefix_lse, prefix_indices)
```

### 11.6 LSE 合并 — Triton Kernel

**文件**: `vllm_gr/v1/attention/backends/beam_attn.py` (lines 768-823)

```python
@triton.jit
def _merge_indexed_attn_states_kernel(
    output_ptr, prefix_out_ptr, prefix_lse_ptr,
    suffix_out_ptr, suffix_lse_ptr, prefix_indices_ptr,
    num_tokens, num_heads, head_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Merge prefix and suffix attention states using LSE normalization.

    Formula:
        lse_merged = log(exp(lse_prefix) + exp(lse_suffix))
        weight_prefix = exp(lse_prefix - lse_merged)
        weight_suffix = exp(lse_suffix - lse_merged)
        output = weight_prefix * prefix_out + weight_suffix * suffix_out
    """
    pid = tl.program_id(0)
    # ... gather prefix_out using prefix_indices scatter
    # ... compute LSE weights
    # ... weighted sum
```

### 11.7 forward 分发

**文件**: `vllm_gr/v1/attention/backends/beam_attn.py` (lines 1022-1187)

```python
def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, ...):
    """Main forward pass."""
    # Update KV cache
    reshape_and_cache_flash(key, value, key_cache, value_cache, ...)

    if attn_metadata.prefix_indices is not None:
        # Beam groups present — use cascade attention
        self.cascade_attention(
            output=output[:num_actual_tokens],
            query=query[:num_actual_tokens],
            prefix_block_table=attn_metadata.prefix_block_table,
            suffix_block_table=attn_metadata.suffix_block_table,
            prefix_indices=attn_metadata.prefix_indices,
            ...
        )
    else:
        # Standard attention (fallback)
        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=attn_metadata.query_start_loc,
            block_table=attn_metadata.block_table,
            ...
        )
```

---

## 12. FlatLogprobs 与延迟重建 (logprobs.py)

### 12.1 问题背景

原生 vLLM 的 beam search 在每一步都为每个 beam 存储完整的 `list[dict[int, Logprob]]`：

```text
# 原生 vLLM — O(beams * steps^2) 内存
step 0: beam_0.logprobs = [{token_a: Logprob, token_b: Logprob, ...}]
step 1: beam_0.logprobs = [{...}, {token_c: Logprob, ...}]  # 复制 + 追加
step 2: beam_0.logprobs = [{...}, {...}, {token_d: Logprob, ...}]
```

每次复制历史 logprobs 列表，导致 O(steps^2) 的内存和时间开销。

### 12.2 FlatLogprobs 方案

vllm-gr 使用 `FlatLogprobs` + parent pointer 实现延迟重建：

```text
# vllm-gr — O(beams * steps) 内存
beam_0._lp_parent = None
beam_0._lp_step_data = (token_a, FlatLogprobs([token_a, top1, top2]))

beam_1._lp_parent = beam_0
beam_1._lp_step_data = (token_c, FlatLogprobs([token_c, top1, top2]))

beam_2._lp_parent = beam_1
beam_2._lp_step_data = (token_d, FlatLogprobs([token_d, top1, top2]))

# 只在最终结果时重建:
reconstruct_beam_logprobs(beam_2)
# → 沿 parent chain 回溯: beam_2 → beam_1 → beam_0
# → 收集所有 step_data
# → 构建完整 FlatLogprobs
```

### 12.3 核心函数

**文件**: `vllm_gr/logprobs.py`

**`append_fast`** (lines 10-31):
```python
def append_fast(flat_logprobs, token_ids, logprobs, ranks, decoded_tokens):
    """Batch-append to FlatLogprobs using extend() for C-level speed."""
    flat_logprobs.token_ids.extend(token_ids)
    flat_logprobs.logprobs.extend(logprobs)
    if isinstance(ranks, (list, tuple)):
        flat_logprobs.ranks.extend(ranks)
    else:
        flat_logprobs.ranks.extend(itertools.islice(ranks, len(token_ids)))
```

**`extract_and_dedup_flat_logprobs`** (lines 34-53):
```python
def extract_and_dedup_flat_logprobs(flat_logprobs, logprobs_num):
    """Extract position-0 logprobs, deduplicating sampled token.

    FlatLogprobs stores: [sampled_token, top1, top2, ..., topK]
    If sampled_token also appears in top-K, skip position 0.
    Returns exactly logprobs_num entries.
    """
    all_token_ids = flat_logprobs.token_ids
    sampled = all_token_ids[0]

    # Check if sampled token appears in top-K (positions 1..K)
    for i in range(1, logprobs_num + 1):
        if all_token_ids[i] == sampled:
            # Deduplicate: skip position 0, take positions 1..K
            return _slice(flat_logprobs, 1, logprobs_num + 1)

    # No duplicate: take positions 0..K-1
    return _slice(flat_logprobs, 0, logprobs_num)
```

**`reconstruct_beam_logprobs`** (lines 56-85):
```python
def reconstruct_beam_logprobs(beam, sid_end_token_id=None):
    """Walk parent chain, collect step data, build FlatLogprobs."""
    steps = []
    current = beam
    while current is not None:
        if current._lp_step_data is not None:
            steps.append(current._lp_step_data)
        current = current._lp_parent

    steps.reverse()  # root → leaf order

    result = FlatLogprobs()
    for token_id, step_logprobs in steps:
        append_fast(result, [token_id], ...)

    # Optionally append SID end token
    if sid_end_token_id is not None:
        append_fast(result, [sid_end_token_id], ...)

    beam.logprobs = result  # Replace in-place
```

---

## 13. 与原生 vLLM BeamSearch 的对比

### 13.1 离线对比

| 维度 | 原生 vLLM (`offline.py`) | vllm-gr (`gr.py`) |
|------|------------------------|-------------------|
| logprobs 数量 | `2 * beam_width` | `beam_width` |
| logprobs 格式 | `list[dict[int, Logprob]]` | `FlatLogprobs` |
| 每步内存 | O(beams * steps) 复制 | O(1) parent pointer |
| 候选过滤 | 无 | Catalog filtering |
| 结构化输出 | Grammar bitmask | 不支持 |
| 长度惩罚 | `sort_beams_key` | 相同 |
| Session 标记 | 无 | begin_token / end_token |

### 13.2 在线对比

| 维度 | 原生 vLLM (`online.py`) | vllm-gr (`serving_engine.py`) |
|------|------------------------|------------------------------|
| 请求提交 | 每 beam 独立 `generate()` | ADD_BATCH + BEAM_FORK |
| KV Cache | 每 beam 独立 prefill | 父请求 KV cache 复用 |
| Attention | 标准 flash_attn | Cascade attention |
| 数据并行 | 不支持 | Round-robin DP 路由 |
| Priority 分组 | 无 | 微秒时间戳 priority |
| 输出方式 | `list[RequestOutput]` | `async yield RequestOutput` |
| Metrics | 无 | overhead / decode_time / num_tokens |

### 13.3 原生 vLLM 在线 beam search 流程

```text
# 原生 vLLM — 每步为每个 beam 发起独立请求
for step in range(max_tokens):
    tasks = []
    for beam in beams:
        task = engine_client.generate(beam.prompt, sampling_params)
        tasks.append(task)
    results = await asyncio.gather(*tasks)

    # 选择 top-K
    new_beams = select_top_k(beams, results, beam_width)
    beams = new_beams
```

### 13.4 vllm-gr 在线 beam search 流程

```text
# vllm-gr — Step 0 批量提交，后续 fork
step 0:
    prepared = [prepare_request(p) for p in prompts]
    await _add_requests_batch(prepared)  # ADD_BATCH

step 1+:
    fork_info = [(parent_idx, token_id), ...]
    await beam_fork(BeamForkRequest(...))  # BEAM_FORK

# EngineCore 侧:
# BEAM_FORK → _handle_beam_fork → template-clone → 复用 KV cache
# Scheduler → beam_prefix_groups → cascade attention
```

---

## 14. 关键优化总结

### 14.1 FlatLogprobs + Parent Pointer 延迟重建

```text
传统方式 (vLLM 原生):
  Step 0: beam_0.logprobs = [step0_data]          ← 分配
  Step 1: beam_0.logprobs = [step0_data, step1_data] ← 复制 + 追加
  Step 2: beam_0.logprobs = [step0, step1, step2]    ← 复制 + 追加
  总内存: O(n + 2n + 3n + ... + kn) = O(k^2 * n)

vllm-gr 方式:
  Step 0: beam_0._lp_step_data = step0_data       ← O(1)
  Step 1: beam_1._lp_step_data = step1_data       ← O(1)
  Step 2: beam_2._lp_step_data = step2_data       ← O(1)
  Final:  reconstruct → O(k * n) 一次性重建
  总内存: O(k * n)
```

### 14.2 BEAM_FORK KV Cache 复用

```text
传统方式:
  beam_0: [prefix_1000_tokens] + [token_A]  → 计算 1001 tokens 的 KV
  beam_1: [prefix_1000_tokens] + [token_B]  → 计算 1001 tokens 的 KV
  beam_2: [prefix_1000_tokens] + [token_C]  → 计算 1001 tokens 的 KV
  总计: 3003 tokens KV 计算

vllm-gr BEAM_FORK:
  parent:  [prefix_1000_tokens] → 已缓存 KV
  beam_0:  parent.fork() + [token_A] → 只计算 1 token 的 KV
  beam_1:  parent.fork() + [token_B] → 只计算 1 token 的 KV
  beam_2:  parent.fork() + [token_C] → 只计算 1 token 的 KV
  总计: 3 tokens KV 计算 (节省 99.9%)
```

### 14.3 Cascade Attention

```text
传统方式:
  beam_0: attention([prefix | suffix_A]) → 计算 prefix + suffix_A
  beam_1: attention([prefix | suffix_B]) → 计算 prefix + suffix_B
  beam_2: attention([prefix | suffix_C]) → 计算 prefix + suffix_C
  prefix 被重复计算 3 次

vllm-gr Cascade:
  Phase 1: attention([prefix]) → 计算 1 次 (non-causal)
  Phase 2: attention([suffix_A/B/C]) → 各自计算 (causal)
  Phase 3: LSE merge → 合并 prefix 和 suffix 结果
  prefix 只计算 1 次 (节省 66%)
```

### 14.4 Scheduler computed_blocks 缓存

```text
无缓存:
  beam_0 → get_computed_blocks() → 遍历 hash 链 → O(n)
  beam_1 → get_computed_blocks() → 遍历 hash 链 → O(n)
  beam_2 → get_computed_blocks() → 遍历 hash 链 → O(n)

有缓存:
  beam_0 → get_computed_blocks() → 遍历 hash 链 → O(n) → 缓存结果
  beam_1 → get_computed_blocks() → 缓存命中 → O(1)
  beam_2 → get_computed_blocks() → 缓存命中 → O(1)
```

### 14.5 Template-Clone 优化

```text
传统方式:
  child_0 = Request(...)        # 完整构造函数
  child_1 = Request(...)        # 完整构造函数
  child_2 = Request(...)        # 完整构造函数
  每个子请求都要执行完整的参数验证和初始化

vllm-gr Template-Clone:
  child_0 = Request(...)        # 完整构造函数 + 快照 shared_fields
  child_1 = object.__new__()    # 跳过构造函数 + dict copy
  child_2 = object.__new__()    # 跳过构造函数 + dict copy
  只有第一个子请求执行完整构造函数
```

---

## 附录：关键常量与阈值

| 常量 | 值 | 位置 | 说明 |
|------|-----|------|------|
| `min_group_th` | `256 // block_size` | `engine_core_patch.py:65` | beam prefix 分组最小共享 block 数 |
| `cache_th` | `256` | `engine_core_patch.py:372` | computed_blocks 缓存阈值 (tokens) |
| `block_size` | 16 的倍数 | `beam_attn.py` | KV cache block 大小要求 |
| `ADD_BATCH` | `b"\x05"` | `types.py` / `core_client_patch.py` | 批量添加请求类型 |
| `BEAM_FORK` | `b"\x06"` | `types.py` / `core_client_patch.py` | beam 分叉请求类型 |
| `beam_width` | 用户指定 | `BeamSearchParams` | beam search 宽度 |
