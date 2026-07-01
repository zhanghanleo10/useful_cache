# vllm-gr 在线 BeamSearch 走读(Mega 机制版)

> **版本基线**:本文基于 `vllm-gr/` 当前代码走读,对应 vllm==0.14.1。
>
> **与旧文档关系**:本目录的 `online_pipeline.md` 基于**旧版本**(`BeamForkRequest` + `template-clone` 机制),该机制已被重构取代。当前代码用的是 **`MegaRequestStepUpdate` + `ADD_BATCH` + `MEGA_REQUEST_STEP_UPDATE`** 机制——即本文内容。两代机制的核心区别见文末「与旧 BEAM_FORK 机制对照」。
>
> **走读范围**:在线服务链路(`OpenAIServing.beam_search`),从 HTTP 请求到结果返回的完整闭环,分 6 个单元。所有 `file:line` 均逐行核对自实际代码。

---

## 0. 总览

### 0.1 为什么改 BeamSearch

vllm-gr 面向**生成式推荐(GR)**场景,BeamSearch 用作推荐解码,特点与上游 NLP 场景不同:

- **beam_width 极大**(示例默认 128,可更大),上游逐 beam 单独发请求 + dict logprobs 的实现成为瓶颈。
- 需要**合法路径约束**:候选 token 必须在商品目录(catalog)内。
- 需要 **session 起止符**(`<|sid_begin|>`/`<|sid_end|>`)。
- 需要把同一 prompt 下 W 条 beam 的解码**融合成一次 Mega 请求**复用 prefill KV + cascade attention。

### 0.2 一句话定位

在线 beam_search 是一个 **async generator**(`OpenAIServing.beam_search`),在 `max_tokens` 步循环里把同一 prompt 下 W 条 beam 的解码压成「**首步一条 ADD_BATCH + 后续每步一条 Mega Request**」,让 Engine 复用 prefill KV、用 cascade attention 一次性算完 W 条 beam,最后只回一个合并后的 `RequestOutput`。

### 0.3 全链路总览图

```
HTTP (use_beam_search=True)
  │  completion/serving.py:197  /  chat_completion/serving.py:312
  ▼
to_beam_search_params   ← patch_sampling 注入 begin_token/end_token (protocol.py:10)
  │
  ▼
OpenAIServing.beam_search          ← patch_beam_search 替换 (serving_engine.py:220)
  │  use_mega_request = hasattr(engine_client, "mega_request_step_update")  ← 依赖 patch_batch_and_fork
  │
  │  for token in range(max_tokens):
  │
  ├─[首步 / fork_info is None]──► _add_batch_step ─► prepare_request ─► _add_requests_batch
  │      (serving_engine.py:54)     (async_llm.py:22)    (async_llm.py:119)
  │            │ force_batch=True(use_mega) → ADD_BATCH
  │            ▼  ZMQ
  │     process_input_sockets ─► _cache_beam_request(建 session 缓存) + ADD  (core.py:251/28)
  │            │ EngineCore 正常 prefill 调度
  │            ▼
  │     各 beam output_queue ─► _gather_beam_results
  │
  └─[后续步 / fork_info 非 None]──► _mega_request_step            (serving_engine.py:89)
         │  register_beam_output(只注册队列不发包)  (async_llm.py:133)
         │  mega_request_step_update(MegaRequestStepUpdate)        (async_llm.py:175)
         │            │ _send_input(MEGA_REQUEST_STEP_UPDATE)      (core_client.py:59)
         │            ▼  ZMQ
         │  process_input_sockets ─► _handle_mega_request_step_update (core.py:263/46)
         │            │ 拼 prefill+各beam后缀 → 单条 Mega Request(session_id) ADD
         │            ▼
         │  Scheduler.schedule(patched) 裁 prefix_len + 透传 mega_data (engine_core_patch.py:241)
         │            ▼
         │  GPUModelRunner.execute_model(patched) 设 MEGA_DATA_VAR  (engine_core_patch.py:617)
         │  GPUModelRunner._prepare_inputs(patched) 重写 position + InputBatchProxy 展开 W 行 (629)
         │            ▼
         │  BeamAttention.cascade_attention  std + mega(prefix/suffix) (beam_attn.py:731)
         │            ▼
         │  采样
         │            ▼
         │  _bookkeeping_sync(patched) _remux_logprobs_tensors 合并成单行 (715)
         │            ▼
         │  session_id 的 output_queue ─► _collect_beam_result(单个合并 RequestOutput)
         │
         ▼
   解析 flat logprobs(stride_k 切分) → catalog 过滤 → numpy top-K → 新 beam + fork_info
   │
循环结束 ─► _mega_request_cleanup(B=0) ─► reconstruct_beam_logprobs ─► yield 最终 RequestOutput
```

---

## 单元 1:装配与入口

涵盖:`beam_search_patch.py`(patch 装配)、`protocol.py`(参数转换)、`serving_engine.py:220-352`(入口头 + 参数解析)。

### 1.1 patch 怎么挂上去(`beam_search_patch.py`)

```python
# beam_search_patch.py:4-8
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.serving_engine import OpenAIServing
from vllm_gr.entrypoints.openai.protocol import to_beam_search_params
from vllm_gr.entrypoints.openai.serving_engine import beam_search

# beam_search_patch.py:11-12
def patch_beam_search():
    OpenAIServing.beam_search = beam_search       # 模块级函数 → 方法

# beam_search_patch.py:15-16
def patch_sampling():
    ChatCompletionRequest.to_beam_search_params = to_beam_search_params

# beam_search_patch.py:19-40
def patch_batch_and_fork():
    ...  # ADD_BATCH/MEGA 接线,属单元 3,这里先跳过
```

- 入口接线极薄:两个赋值。`beam_search` 是模块级 `async def`(`serving_engine.py:220`),靠赋值变成 `OpenAIServing` 的方法。上游 `completion/serving.py:197` / `chat_completion/serving.py:312` 在 `use_beam_search=True` 时调的 `self.beam_search(...)` 就落到这里。
- ⚠️ 前提:`self` 是 `OpenAIServing` 实例,`self.engine_client` 是打了补丁的 `AsyncLLM`(否则 `use_mega_request` 探测为假,退化)。
- `patch_batch_and_fork` 也在这个文件,负责 ADD_BATCH/MEGA 接线,属单元 3。

调用处 `patch.py:339-358`:`_apply_gr_beam_patches` 里每个 patch 套 `try/except`(失败只 debug 不抛),经 vllm 插件入口在**主/EngineCore 子/Worker 子**三个进程都跑。

### 1.2 参数转换:`to_beam_search_params`(`protocol.py:10-26`)

```python
from vllm_gr.sampling_params import BeamSearchParams   # GR 子类(多 begin_token/end_token)

def to_beam_search_params(self, max_tokens, default_sampling_params) -> BeamSearchParams:
    extra_args = self.vllm_xargs or {}
    n = self.n if self.n is not None else 1
    if (temperature := self.temperature) is None:
        temperature = default_sampling_params.get("temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"])
    return BeamSearchParams(
        beam_width=n, max_tokens=max_tokens, ignore_eos=self.ignore_eos,
        temperature=temperature, length_penalty=self.length_penalty,
        include_stop_str_in_output=self.include_stop_str_in_output,
        begin_token=extra_args.get("begin_token"),   # ← GR 新增
        end_token=extra_args.get("end_token"),       # ← GR 新增
    )
```

- **SID token 走 `vllm_xargs` 偷渡**:`vllm_xargs` 是上游 `ChatCompletionRequest` 字段(`chat_completion/protocol.py:405`),透传 OpenAI 规范外的参数。`begin_token`/`end_token` 藏在这里注入,不污染 schema。
- 与上游差异(`chat_completion/protocol.py:535-542`):上游返回的 `BeamSearchParams` 不含 begin/end;GR 版多这两个字段(子类 `sampling_params.py:6`)。
- SID 可选:不传 `vllm_xargs` 时两者都是 `None`,退化为普通 beam search。

### 1.3 入口签名 + DP rank(`serving_engine.py:220-250`)

```python
async def beam_search(self, prompt, request_id, params: BeamSearchParams,
                      lora_request=None, trace_headers=None) -> AsyncGenerator[RequestOutput, None]:
    ...
    beam_width = params.beam_width; max_tokens = params.max_tokens
    ignore_eos = params.ignore_eos; temperature = params.temperature
    begin_token = params.begin_token; end_token = params.end_token
    MICROSECONDS = 1000000
    priority = int(time.perf_counter() * MICROSECONDS)        # 微秒时间戳作优先级

    rank = None
    if (... self.engine_client.vllm_config.parallel_config is not None):
        data_parallel_size = ...data_parallel_size
        if data_parallel_size > 1:
            rank = await _next_data_parallel_rank(data_parallel_size)  # 轮询分 rank
```

- 返回 `AsyncGenerator`,上游 `serving.py` 会 `async for`,最后的 `yield RequestOutput(...)` 是给用户的最终结果。
- ⚠️ 类型隐患:`serving_engine.py:17` import 的是**上游** `BeamSearchParams`,而实参是 GR 子类;`params.begin_token` 能访问只因子类有该字段。
- `priority` = 微秒级时间戳,透传进每个 EngineCore 请求。
- DP>1 时 `_next_data_parallel_rank`(`:31`)用全局计数器轮询分 rank——**同一次 beam search 所有步必须落同一 rank**(KV 局部性),这个 rank 全程不变。

### 1.4 tokenizer 与 SID token(`serving_engine.py:252-292`)

```python
eos_token_id = tokenizer.eos_token_id
sid_begin_token_id = tokenizer.convert_tokens_to_ids(begin_token) if begin_token else None  # 非法则报错
sid_end_token_id = tokenizer.convert_tokens_to_ids(end_token) if end_token else None
# 不支持 enc_dec;拆 prompt_text/prompt_token_ids/multi_modal_data
```

- SID token 字符串 → id,找不到**硬报错**(对比离线版 `gr.py:344` 只是 warning——在线更严格)。
- ⚠️ `prompt_token_ids` 纯文本场景下可能为空 list,影响后面 `tokenized_length` 和最终截断。

### 1.5 初始 beam 构造 + mega 探测(`serving_engine.py:294-352`)

```python
logprobs_num = beam_width          # ← 非上游的 2×beam_width
beam_search_params = SamplingParams(
    logprobs=logprobs_num, max_tokens=1, temperature=temperature,
    detokenize=False, flat_logprobs=True)        # flat_logprobs 配合 numpy 处理

initial_tokens = list(prompt_token_ids)
initial_logprobs = FlatLogprobs()                # 共享只读,勿 mutate
if sid_begin_token_id is not None:
    initial_tokens.append(sid_begin_token_id)
    initial_logprobs.append_fast([sid_begin_token_id], [0.0], [None], [None])

all_beams = [BeamSearchSequence(tokens=initial_tokens, cum_logprob=0, logprobs=initial_logprobs, ...)]  # 仅 1 条
all_beams[0]._lp_parent = None                    # 延迟 logprobs 父链起点
all_beams[0]._lp_step_data = None

pre_calc = (1 if sid_begin else 0) + (1 if sid_end else 0)   # SID 占位步数
use_mega_request = hasattr(self.engine_client, "mega_request_step_update")  # ⭐ 总开关

prev_beam_internal_ids = []
fork_info = None                  # None 表示未进过 mega 路径(首步必走 ADD_BATCH)
prefill_id = None                 # 首步后 = session_id
```

- **`logprobs_num = beam_width`**(GR 场景 beam_width 大,只取 top-beam_width 够用)。
- **初始只有 1 条 beam**,靠 `_lp_parent`/`_lp_step_data` 延迟 logprobs(把 O(beams×steps²) 降到 O(beams×steps))。⚠️ 这俩属性挂在无 `__slots__` 的 `@dataclass` 上,上游若加 `__slots__` 会崩。
- **`use_mega_request`** 是运行时探测(非配置开关),依赖 `patch_batch_and_fork` 是否挂上 `mega_request_step_update`。
- 三个状态变量(`fork_info`/`prev_beam_internal_ids`/`prefill_id`)初值为空——主循环起点。

**单元 1 小结**:接线极薄;SID 走 vllm_xargs;每步采样 `logprobs=beam_width`;初始 1 条 beam + 延迟 logprobs;`use_mega_request` 在 `:348` 探测。

---

## 单元 2:主循环骨架 + 首步 ADD_BATCH

前置:上一单元结束时 `all_beams`=[1 条],`fork_info`=None。

### 2.1 主循环骨架(`serving_engine.py:354-432`)

```python
for token in range(max_tokens - pre_calc):
    if token == 1: beam_search_start = time.perf_counter()   # 跳过首步 prefill 计时

    prompts_batch, lora_req_batch = zip(*[(TokensPrompt(prompt_token_ids=beam.tokens, ...), beam.lora_request)
                                          for beam in all_beams])   # 每步整段重发
    if token > 0: num_generation_tokens += len(all_beams)
    request_id_batch = f"{request_id}-{random_uuid()}"

    catalog_task = None
    if self.models.catalog is not None:
        # 对每条 beam 预查合法 token 集合,asyncio.to_thread 丢线程池
        catalog_task = asyncio.create_task(...)                # 先启动不 await,与 engine step 并发

    gen_start = time.perf_counter()
    if use_mega_request and fork_info is not None:             # (A) 后续解码步
        output, prev_beam_internal_ids = await _mega_request_step(..., session_id=prefill_id, fork_info=fork_info, ...)
    else:                                                       # (B) 首步 / 退化模式
        output, new_ids = await _add_batch_step(..., use_mega_request, ...)
        if new_ids:
            prev_beam_internal_ids = new_ids
            if prefill_id is None: prefill_id = new_ids[0]     # ← session_id 首次赋值

    valid_tokens_sets = await catalog_task if catalog_task else None
    if token > 0: generation_time += time.perf_counter() - gen_start
```

- ⭐ **分叉判断** `use_mega_request and fork_info is not None`:首步 `fork_info=None` 必走 (B);`(B)` 里把 `use_mega_request` 作 `use_beam_fork` 透传。
- catalog 查询与 engine step **并发**(create_task 先启,engine 跑完才 await)。
- `prefill_id` 只在首次(`None` 时)赋值 = `new_ids[0]` = **第一个 beam id = 后续 session_id**。
- `fork_info` 是 None→非 None 的**单向开关**,末尾(`:609`)赋值后后续都走 (A)。

### 2.2 首步发送链路(客户端侧)

`_add_batch_step`(`serving_engine.py:54-86`) → `prepare_request` → `_add_requests_batch` → `add_requests_async` → ZMQ ADD_BATCH。

```python
# _add_batch_step (serving_engine.py:54-86)
for i, (prompt, lora_req) in enumerate(zip(prompts_batch, lora_req_batch)):
    request_id_item = f"{request_id_batch}-beam-{i}"
    q, ec_req = engine_client.prepare_request(request_id_item, prompt, beam_search_params, ...)
    queues.append(q); engine_core_requests.append(ec_req)
await engine_client._add_requests_batch(engine_core_requests, use_batch_message=use_beam_fork)
internal_ids = [r.request_id for r in engine_core_requests] if use_beam_fork else []
output = await _gather_beam_results(queues)
```

- 首步 `all_beams` 只有 1 条,所以 `engine_core_requests` 长度 = 1。
- **`prepare_request`**(`async_llm.py:22-116`):处理输入 + 注册输出队列,**但不发 ZMQ**,返回 `(queue, request)`。拆分了上游"准备+发送",专为 beam search 紧循环避免事件循环开销。
- **`_add_requests_batch`**(`async_llm.py:119-130`)→ `add_requests_async(requests, force_batch=use_mega)`。

```python
# add_requests_async (core_client.py:14-51)
if len(requests) == 1 and not force_batch:
    return await self.add_request_async(requests[0])     # 单条且不强制 → 普通 ADD
# 否则 ADD_BATCH
... request.client_index = self.client_index; request.current_wave = ...
chosen_engine = self.get_core_engine_for_request(requests[0])  # 按 data_parallel_rank 选引擎
await self._send_input(EngineCoreRequestType.ADD_BATCH, requests, chosen_engine)
```

- ⭐ **核心**:`:22` 的 `len==1 and not force_batch` 短路——**但首步 mega 模式 `force_batch=True`,绕过短路**,所以单条 beam 也走 ADD_BATCH。这是为了让子进程 `_cache_beam_request` 建立 session 缓存。
- `_send_input`/`get_core_engine_for_request`/`add_request_async` 都是上游 AsyncMPClient 方法;GR 只新增 `ADD_BATCH` 枚举值 + `add_requests_async` 入口。

### 2.3 子进程接收:ADD_BATCH → 缓存 → ADD(`core.py`)

```python
# core.py:251-261  ADD_BATCH 分支
elif request_type == EngineCoreRequestType.ADD_BATCH:
    batch = add_batch_decoder.decode(data_frames)
    for req in batch:
        request = self.preprocess_add_request(req)
        self._cache_beam_request(request[0])                       # ← 缓存 prefill 状态
        self.input_queue.put_nowait((EngineCoreRequestType.ADD, request))  # 当普通 ADD 入队
    continue

# core.py:28-43  _cache_beam_request
def _cache_beam_request(self, request):
    with self.beam_cache_lock:
        self.beam_cache[request.request_id] = {
            "all_token_ids": ..., "block_hashes": ..., "eos_token_id": ...,
            "lora_request": ..., "cache_salt": ..., "mm_features": ...,
            "prompt_embeds": ..., "num_prompt_tokens": ...,
        }
```

- ADD_BATCH 相比普通 ADD **多做的唯一一件事**:`_cache_beam_request`(按 request_id 存 prefill 全状态),供后续 mega 步复用。然后当普通 ADD 入队走标准 prefill。
- `self.beam_cache` + `self.beam_cache_lock` 在 `EngineCore.__init__` 包裹器里加(`engine_core_patch.py:26-29`)。
- ⚠️ 缓存是 mega 机制的"种子",没有它后续 `_handle_mega_request_step_update` 会报 "session not in cache"(`core.py:95-114`)。

### 2.4 首步结果收集与解析

`_gather_beam_results`(`serving_engine.py:48-51`):每条 beam 一个 task,drain 队列直到 finished。首步 1 个 queue。

**结果解析走 baseline 分支**(`serving_engine.py:516-555`)——首步 `fork_info` 还是 None,即使 `use_mega_request=True` 也走 else:

```python
for i, result in enumerate(output):      # output 与 all_beams 一一对应
    current_beam = all_beams[i]
    if result.outputs[0].finish_reason == "error": yield ...error...; return
    if result.outputs[0].logprobs is not None:
        flat = result.outputs[0].logprobs
        token_ids_pos, logprobs_pos, ranks_pos, decoded_pos = extract_and_dedup_flat_logprobs(flat, logprobs_num)
        if valid_tokens_sets is not None:                       # catalog 合法约束
            logprobs_pos = [lp if tid in valid_tokens_set else -inf for tid, lp in ...]
        beam_flat_cache.append((...)); all_beams_token_id.extend(...); all_beams_logprob.extend(cum + lp for lp in ...)
```

- baseline:每条 beam 独立处理,无 stride 切分。首步 1 条 beam 产生 beam_width 候选。

### 2.5 EOS + top-K 展开(1 → beam_width)(`serving_engine.py:557-612`)

```python
all_beams_token_id = np.array(all_beams_token_id); all_beams_logprob = np.array(all_beams_logprob)
if not ignore_eos:
    eos_idx = np.where(all_beams_token_id == eos_token_id)[0]
    for idx in eos_idx:
        ...eos_beam(挂 _lp_parent/_lp_step_data)... completed.append(eos_beam)
    all_beams_logprob[eos_idx] = -np.inf

if all_beams_logprob.size > beam_width:
    topn_idx = np.argpartition(np.negative(all_beams_logprob), beam_width)[:beam_width]   # O(n) top-K
else:
    topn_idx = np.arange(all_beams_logprob.size)

for idx in topn_idx:
    new_beam = BeamSearchSequence(tokens=current_beam.tokens + [token_id], logprobs=initial_logprobs,
                                  cum_logprob=..., ...)
    new_beam._lp_parent = current_beam; new_beam._lp_step_data = cached
    new_beams.append(new_beam)

if use_mega_request:
    fork_info = [(idx // logprobs_num, int(all_beams_token_id[idx])) for idx in topn_idx]   # ⭐ 开关翻转
all_beams = new_beams
```

- ⭐ **首步展开发生在这里**:1 条 beam × beam_width 候选 → top-K 选满 beam_width → `new_beams` 长度 = beam_width。
- `fork_info` 首次赋值(从 None 变非 None),下一轮进入 mega 路径 (A)。首步所有候选父都是 beam 0(`idx//logprobs_num==0`)。
- `idx // logprobs_num` 把候选索引映射回父 beam 索引。

**单元 2 数据流**:
```
首步 token==0, all_beams=[1条], fork_info=None
  ├─ _add_batch_step(force_batch=True) → ADD_BATCH(1条)
  │    子进程 _cache_beam_request(种缓存) + ADD → 标准 prefill → top-beam_width
  ├─ baseline 解析: 1条beam × beam_width候选
  ├─ np.argpartition → new_beams[beam_width条]
  └─ 设 prev_beam_internal_ids, prefill_id=session_id, fork_info(开关翻转)
下一步 → 进入 mega 路径 (A)
```

---

## 单元 3:Mega 步客户端 + 子进程合成

前置:`all_beams`=[beam_width 条],`fork_info` 非 None,`prefill_id`=session_id 已设,子进程 `beam_cache[session_id]` 有缓存。

### 3.1 客户端编排:`_mega_request_step`(`serving_engine.py:89-166`)

```python
parent_beam_ids = []; child_beam_ids = []; beam_tokens = []
W = len(fork_info)
for new_idx, (parent_beam_idx, tok) in enumerate(fork_info):
    parent_beam_ids.append(prev_beam_internal_ids[parent_beam_idx])
    child_beam_ids.append(f"{request_id_batch}-beam-{new_idx}")
    beam_tokens.append(all_beams[new_idx].tokens[prefix_len:])     # 去公共前缀的后缀

decode_sampling_params = copy.copy(beam_search_params)
decode_sampling_params.n = W                                       # 告诉 sampler 有 W beam

q = engine_client.register_beam_output(session_id, all_beams[0].tokens, decode_sampling_params, ...)

used_parents = set(parent_beam_ids)
pruned_ids = [pid for pid in prev_beam_internal_ids if pid not in used_parents]   # 被剪掉的父

await engine_client.mega_request_step_update(MegaRequestStepUpdate(
    session_id=session_id, parent_beam_ids=..., child_beam_ids=..., beam_tokens=...,
    pruned_ids=pruned_ids, prefix_len=prefix_len, beam_width=W,
    sampling_params=decode_sampling_params, ...))

single_output = await _collect_beam_result(q)                       # 收单个合并结果
return [single_output], child_beam_ids
```

- `fork_info` → 三元映射:parent/child id(逻辑路由)+ beam_tokens(物理拼接用)。⚠️ `fork_info` 里的 `tok` 没被用(已 append 到 tokens)。
- `decode_sampling_params.n = W`——sampler 按 W beam 组织输出(故子进程要 `_verify_greedy_sampling` 置空)。
- **`register_beam_output` 注册 session_id 队列**(不是 child id)——物理请求是 session_id 那条。
- `pruned_ids` = 上一步未被选为父的 beam。

### 3.2 `register_beam_output_fn`:只注册不发包(`async_llm.py:133-172`)

```python
ec_request = EngineCoreRequest(request_id=request_id, prompt_token_ids=..., sampling_params=..., ...)
ec_request.external_req_id = request_id            # ← 跳过 assign_request_id,直接设
self._run_output_handler()
queue = RequestOutputCollector(sampling_params.output_kind, request_id)
self.output_processor.add_request(ec_request, None, None, 0, queue)   # 注册但不发 ZMQ
return queue
```

- 在主进程 OutputProcessor 占位,让 session_id 输出有地方落。**不发 ZMQ**,物理请求子进程造。
- 与 `prepare_request`(首步)区别:首步真请求走 prefill 要 `process_inputs`;mega 是"幽灵"只造壳 + 注册队列。

### 3.3 发送路径(`async_llm.py:175` / `core_client.py:59-82`)

```python
# core_client.py:59-82  mega_request_step_update_async
update.client_index = self.client_index; update.current_wave = ...
engine = None
if hasattr(self, "get_core_engine_for_request") and update.data_parallel_rank is not None:
    engine = self.core_engines[update.data_parallel_rank]           # 沿用首步 rank(KV 局部性)
    for child_id in update.child_beam_ids: self.reqs_in_flight[child_id] = engine   # 注册逻辑路由
    for pruned_id in update.pruned_ids: self.reqs_in_flight.pop(pruned_id, None)    # 清剪掉的
to_await = self._send_input(EngineCoreRequestType.MEGA_REQUEST_STEP_UPDATE, update, engine)
...
```

- **强制沿用首步 DP rank**;维护 child/pruned 逻辑路由表。
- ⚠️ 维护的是 child/pruned 逻辑路由,session_id 物理路由首步已建,不动。

### 3.4 子进程 MEGA 分支(`core.py:263-266`)

```python
elif request_type == EngineCoreRequestType.MEGA_REQUEST_STEP_UPDATE:
    mega_update = mega_request_decoder.decode(data_frames)
    self._handle_mega_request_step_update(mega_update)
    continue
```

### 3.5 ⭐ 核心合成:`_handle_mega_request_step_update`(`core.py:46-171`)

**(a) B==0 清理分支(`:50-63`)**:
```python
if B == 0:
    with self.beam_cache_lock: self.beam_cache.pop(session_id, None)
    self.aborts_queue.put_nowait(session_id)        # abort session 本身
    for abort_id in update.pruned_ids: self.aborts_queue.put_nowait(abort_id)
    return
```
- `beam_width=0` 是清理信号(由 `_mega_request_cleanup` 发出),清缓存 + abort。

**(b) 校验并行数组等长(`:65-89`)**:不等长给每个 child 发 ERROR。

**(c) 取 prefill 缓存(`:91-114`)**:取不到就 ERROR(这就是首步必须 ADD_BATCH 建 cache 的原因)。

**(d) ⭐ 拼接 Mega token 序列(`:116-131`)**:
```python
prefill_tokens = session_cached["all_token_ids"]
decode_steps = len(update.beam_tokens[0])           # 所有 beam 后缀等长(同步推进)
mega_token_ids = list(prefill_tokens)               # 公共前缀
for gen_tokens in update.beam_tokens:
    mega_token_ids.extend(gen_tokens)               # 各 beam 后缀依次拼接
```
- 拼接:`[prefill] + [beam_0后缀] + ... + [beam_{W-1}后缀]`,总长 = `prefix_len + W*decode_steps`。**一条 Request 装下 W 条 beam 全部 token**。

**(e) 构造单条 Request(`:133-148`)**:
```python
req = Request(request_id=session_id,                # ← 单一 id 代表全部 beam
              prompt_token_ids=mega_token_ids, sampling_params=update.sampling_params, ...)
```

**(f) 继承 block_hashes(KV 复用关键)(`:150-154`)**:
```python
req.block_hashes = prefill_block_hashes             # ← 继承,prefix cache 命中
if self.request_block_hasher is not None:
    req.get_hash_new_full_blocks = partial(self.request_block_hasher, req)
    req.block_hashes.extend(req.get_hash_new_full_blocks())   # 增量算新块
```
- ⭐ **继承 block_hashes 是 KV 复用的根基**,不继承则 prefill 白算。

**(g) 打 mega 标记(`:156-161`)**:
```python
req.is_mega_beam = True; req.mega_beam_width = B
req.is_mega_decode = True; req.mega_decode_steps = decode_steps; req.prefix_len = update.prefix_len
```
- 五个动态属性是下游(scheduler/worker/attention)识别处理 mega 请求的全部信号。

**(h) 入队 + pruned abort(`:163-171`)**:
```python
inputs_to_push.append((EngineCoreRequestType.ADD, (req, update.current_wave)))
self.input_queue.put_nowait(...)
for abort_id in update.pruned_ids: self.aborts_queue.put_nowait(abort_id)
```

**单元 3 小结**:一条 MegaRequestStepUpdate 代表一整步全部 W 条 beam;parent/child id 是逻辑路由,物理只有 session_id 一条 Request。核心合成 = 拼 `prefill+各beam后缀` 成一条长 Request,继承 block_hashes,n=W。

---

## 单元 4:调度器 + Worker 执行

### 4.1 这块解决什么问题

合成的 Mega Request 的 token 默认被赋连续递增 position id,**这是错的**——W 条 beam 后缀是接在同一 prefix 之后的 W 条独立续写,各自都应从 `prefix_len` 起。本单元 = **重写 position id** + 让采样器把它当 **W 行**。

### 4.2 `patched_schedule`:KV 裁剪 + 透传(`engine_core_patch.py:229-280`)

```python
def patch_get_computed_blocks(req):
    bn, tn = _original_get_computed_blocks(req)
    if req and getattr(req, "is_mega_decode", False):
        pl = getattr(req, "prefix_len", 0) // block_size * block_size
        if tn > pl:                                  # 裁命中块数到 prefix_len
            sliced_blocks = tuple(group[: pl // block_size] for group in bn.blocks)
            return self.kv_cache_manager.create_kv_cache_blocks(sliced_blocks), pl
    return bn, tn
# ... 跑原 schedule,然后:
mega_data[req_id] = {
    "is_mega_decode": True, "mega_beam_width": ..., "mega_decode_steps": ...,
    "prefix_len": ..., "cache_len": req.num_computed_tokens - output.num_scheduled_tokens[req_id],
}
output.mega_data = mega_data
```

- **裁 prefix cache 到 `prefix_len`**——只让公共前缀复用 KV,后缀新算。
- **透传 mega 元数据到 `SchedulerOutput.mega_data`**(含 `cache_len` = 调度前已算 token 数)。

### 4.3 `patched_execute_model`:注入 contextvar(`engine_core_patch.py:617-627`)

```python
self._current_mega_data = getattr(scheduler_output, "mega_data", {})
mega_info = {"mega_data": self._current_mega_data, "req_ids": []}
token = MEGA_DATA_VAR.set(mega_info)                # ← attention 读取通道
try: return _original_execute_model(self, scheduler_output, *args, **kwargs)
finally: ...; MEGA_DATA_VAR.reset(token)
```

### 4.4 ⭐ position 重写三件套(`engine_core_patch.py:288-381`)

```python
def _compute_beam_bounds(b, prefix_len, cache_len, decode_steps, chunk_budget, delta):
    past_suffix_b = max(0, min(decode_steps, delta - b * decode_steps))
    if prefix_len >= cache_len:
        remaining_prefix = prefix_len - cache_len
        return (min(chunk_budget, remaining_prefix + decode_steps) if b == 0
                else min(chunk_budget, decode_steps)), past_suffix_b
    remaining_suffix_b = decode_steps - past_suffix_b
    return min(chunk_budget, remaining_suffix_b), past_suffix_b
# 返回 (b_uncached, past_suffix_b): 本步该 beam 算多少 / 后缀已算多少

def _overwrite_position_ids(st, b_uncached, curr_offset, gpu_pos, cpu_pos):
    gpu_pos[curr_offset:curr_offset+b_uncached] = torch.arange(st, st+b_uncached, ...)
    cpu_pos[...] = ...                                # GPU/CPU 都改

def _process_mega_decode_request(mdata, seq_len, token_offset, gpu_pos, cpu_pos, new_logits_indices):
    # 逐 beam:算 b_uncached → 算 st → 重写 position → 取末尾 token logits
    for b in range(beam_width):
        b_uncached, past_suffix_b = _compute_beam_bounds(...)
        st = cache_len if (prefix_len>=cache_len and b==0) else prefix_len   # b>0 从 prefix_len 起
        _overwrite_position_ids(st, b_uncached, curr_offset, gpu_pos, cpu_pos)
        if past_suffix_b + b_uncached >= decode_steps:
            new_logits_indices.append(curr_offset + b_uncached - 1)           # 每条 beam 末尾 logits
        ...
    return valid_rows_for_req, logits_row_cnt, curr_offset      # 主线 (W, W, offset+W*steps)
```

- ⭐ **position 重写让 W 条 beam 后缀共享同一段 `[prefix_len, prefix_len+decode_steps)`**——cascade 复用前缀 KV 的前提。
- 每条 beam 只取**末尾 token** logits → W 条 beam → W 行 logits。
- ⚠️ `_compute_beam_bounds` 主要为 chunked prefill 边界设计;主线(cache_len≥prefix_len)每条 beam 算 decode_steps。

### 4.5 ⭐ `patched_prepare_inputs`:主循环(`engine_core_patch.py:629-713`)

```python
logits_indices, spec = _original_prepare_inputs(self, scheduler_output, num_scheduled_tokens)
self._current_mega_info["req_ids"] = self.input_batch.req_ids
mega_data = getattr(scheduler_output, "mega_data", {})
if not mega_data: return logits_indices, spec        # 无 mega,原样返回(零开销)

for batch_idx, req_id in enumerate(orig_req_ids):
    seq_len = num_scheduled_tokens[batch_idx]
    mdata = mega_data.get(req_id)
    if mdata and mdata.get("is_mega_decode", False):
        W, W_logits, _ = _process_mega_decode_request(mdata, seq_len, token_offset, ...)
        row_to_req_id.extend([req_id] * W); row_to_batch_idx.extend([batch_idx] * W)
        logits_to_batch_idx.extend([batch_idx] * W_logits)
        request_row_configs.append((req_id, input_row_idx, W_logits))
        input_row_idx += W
    else:                                            # 普通请求 W=1
        W = 1
        if is_last_chunk_prefill: new_logits_indices.append(...); W_logits_count = 1
        request_row_configs.append((req_id, input_row_idx, W_logits_count)); input_row_idx += 1
    token_offset += seq_len

self.input_batch = InputBatchProxy(self.input_batch, row_to_req_id, row_to_batch_idx, logits_to_batch_idx)
scheduler_output._gr_request_row_configs = request_row_configs
return torch.tensor(new_logits_indices, ...), spec
```

- 对 mega 请求调 `_process_mega_decode_request` 得 W/W_logits,扩展 4 个映射数组(W 行同 session_id)。
- `request_row_configs` 覆盖所有请求(mega + 普通),供 bookkeeping remux 用。
- **包 `InputBatchProxy`**(W 行伪装)+ 返回重写的 `new_logits_indices`。

### 4.6 `InputBatchProxy` 及代理:虚拟展开(`engine_core_patch.py:490-591`)

```python
class MegaReqIdsProxy(list):       # req_ids[i] → row_to_req_id[i](W 行同 id)
class Mock1DArray:                 # 1D(num_tokens_no_spec) 按 row_to_batch_idx 重映射
class Mock2DArray:                 # 2D(token_ids_cpu/is_token_ids) 按 (idx,slice) 重映射
class InputBatchProxy:             # 属性拦截器
    def __getattr__(self, name):
        if name == "req_ids": return MegaReqIdsProxy(obj.req_ids, req_map)
        elif name == "num_tokens_no_spec": return Mock1DArray(...)
        elif name == "token_ids_cpu": return Mock2DArray(...)
        elif name == "sampling_metadata": ... 按 logits_map 切采样参数
        return getattr(obj, name)                  # 其他透传
```

- **"1 物理 → W 逻辑"的伪装层**,不改下游代码。采样器照常跑,产出 W 行 logits。

**单元 4 小结**:Scheduler 裁 prefix_len;mega 信号经 `mega_data` → `MEGA_DATA_VAR` 传到 attention;position 重写让 W beam 共享前缀段;每 beam 取末尾 logits → W 行;`InputBatchProxy` 伪装成 W 行。

---

## 单元 5:Cascade Attention

### 5.1 这块解决什么问题

position 重写后,W 条 beam 共享前缀 KV、各自后缀独立。cascade 拆两段:**prefix 段**(W beam query 一起 attention 共享前缀,非因果)+ **suffix 段**(各 beam query attention 各自后缀,因果)+ **merge**(lse 合并)。

### 5.2 整体结构(`beam_attn.py`)

`build`(`:544-660`):从 `MEGA_DATA_VAR` 取 mega_data + req_ids → `_segment_requests`(分 std/mega 组)→ `_build_std_mappings` + `_build_mega_mappings` → `_create_*_tensors` → 组装 `BeamAttentionMetadata`。

`cascade_attention`(`:730-826`)三段:std(因果) + mega-prefix(非因果,paged) + mega-suffix(因果,连续)+ merge。

⚠️ **命名陷阱**:`mega_prefix_indices` 名字带 prefix,但实际是**所有 mega query 的索引**(prefix/suffix 段共用)。

### 5.3 `_segment_requests`(`:289-316`)

按 `req_ids[i]` 查 `mega_data`:`is_mega_decode=True` → mega 组。⚠️ `req_ids` 来自 `InputBatchProxy`——mega 请求 W 行都被映射成同一 session_id,按它查能命中。**proxy 和 segment 在这里接上**。

### 5.4 ⭐ `_build_mega_mappings`:prefix/suffix 布局(`:330-426`)

```python
remaining_prefix = prefix_len - cache_len                  # 前缀本步要新算的部分
shared_mapping = np.arange(out_slot, out_slot + remaining_prefix)   # 前缀共享 slot
for b in range(w):
    b_uncached = (remaining_prefix + steps) if b == 0 else steps    # beam0 可能含未算前缀
    # prefix_groups: 相邻 beam query 在前缀段连续等价 → 合并成大组
    if prefix_groups: prefix_groups[-1][1] += b_uncached
    else: prefix_groups.append([q_curr, b_uncached, cache_len])
    # full_mapping: 每 beam 的 [共享前缀 + 私有后缀] slot 索引
    full_mapping = np.empty(remaining_prefix + steps, dtype=np.int32)
    full_mapping[:remaining_prefix] = shared_mapping                 # 前缀指向共享
    full_mapping[remaining_prefix:] = np.arange(out_slot, out_slot + steps)  # 后缀指向私有
    mega_suffix_slot_mapping_out_list.extend(full_mapping)
    mega_suffix_q_lens.append(b_uncached); mega_suffix_seq_lens_list.append(remaining_prefix + steps)
# 收尾:prefix 组转索引
for g_q_start, g_q_len, g_cache_len in prefix_groups:
    mega_prefix_indices.extend(range(g_q_start, g_q_start + g_q_len))
    mega_prefix_q_lens.append(g_q_len); mega_prefix_seq_lens_list.append(g_cache_len)
    mega_prefix_block_tables.append(bt[:max_prefix_blocks_count])
```

- 产出两套布局:**prefix 段**(合并 query 组 + 共享前缀 block_table + seq_len=cache_len)+ **suffix 段**(各 beam 独立 q_len/seq_len + gather 目标 `slot_mapping_out`)。两段共用同一批 query。

### 5.5 `_create_*_tensors`(`:428-542`)

prefix 段用 **paged block_table + seq_lens**(K 在 paged cache);suffix 段用 **连续 K/V + cu_seqlens_k**(K 被 gather)。`mega_suffix_slot_mapping_out` = `full_mapping` 转张量(gather 目标布局)。

### 5.6 ⭐ `cascade_attention` 三段执行(`:730-826`)

```python
# 段1 std: 普通请求,因果 varlen (paged block_table)
if std_indices is not None:
    flash_attn_varlen_func(std_query, key_cache, value_cache, causal=True, block_table=std_block_table, ...)

# 段2 mega prefix: 非因果 (paged KV)
if mega_prefix_indices is not None:
    mega_query = query[mega_prefix_indices]
    mega_prefix_out, mega_prefix_lse = flash_attn_varlen_func(
        q=mega_query, k=key_cache, v=value_cache, causal=False,           # ← 非因果
        block_table=mega_prefix_block_table, return_softmax_lse=True, ...)
    # gather 后缀 KV 成连续
    contig_k, contig_v = extract_suffix_kv(key_cache, value_cache, slot_mapping,
                                           mega_suffix_slot_mapping_out, mega_suffix_total_tokens)
    # 段3 mega suffix: 因果 (连续 KV)
    if mega_suffix_total_tokens > 0:
        mega_suffix_out, mega_suffix_lse = flash_attn_varlen_func(
            q=mega_query, k=contig_k, v=contig_v, causal=True,            # ← 因果
            block_table=None, cu_seqlens_k=mega_suffix_cu_seqlens_k, return_softmax_lse=True, ...)
        merge_attn_states(mega_out, mega_prefix_out, mega_prefix_lse, mega_suffix_out, mega_suffix_lse)
        output[mega_prefix_indices] = mega_out
    else:
        output[mega_prefix_indices] = mega_prefix_out
```

- prefix 段复用**共享前缀 KV**(paged 高效),非因果;suffix 段处理各 beam 后缀(gather+连续),因果。
- **`merge_attn_states`**(vllm 已有)用 lse 合并,数学上等价于对全序列做一次 attention:
  `out = (out_p·exp(lse_p) + out_s·exp(lse_s)) / (exp(lse_p) + exp(lse_s))`。

### 5.7 triton gather kernel(`:61-146`)

`paged_kv_to_contig_suffix_kernel`:二级间接索引 `out_token → src_idx(SLOT_MAPPING_OUT) → slot_idx(SLOT_MAPPING) → paged KV`,grid `(total_tokens, nheads)`。`extract_suffix_kv` 分配连续 buffer 按 `slot_mapping_out` 布局 gather,产出 `contig_k/v`。

### 5.8 `forward`(`:828-885`)

先 `reshape_and_cache_flash` 写新 KV 进 paged cache(attention 之前,所以段2/3 读 prefix 时本步前缀已写入),再调 `cascade_attention`。

**单元 5 小结**:cascade = std(因果) + mega-prefix(非因果,paged) + mega-suffix(因果,连续) + merge。prefix 用 paged(共享前缀),suffix 用连续(gather),两段 layout 不同是拆分根因。`merge_attn_states` 用 lse 合并等价全序列 attention。

---

## 单元 6:采样后合并 + 结果回传(闭环)

### 6.1 这块解决什么问题

单元 4 把 Mega Request 伪装成 W 行,采样器产出 **W 行 logprobs**。但 engine 要回**一个** RequestOutput(session_id)。本块:(1) worker 把 W 行压回单行(remux);(2) serving 按 stride_k 切回 W beam。

### 6.2 `patched_bookkeeping_sync` 概览(`engine_core_patch.py:715-796`)

```python
if not mega_data or not hasattr(scheduler_output, "_gr_request_row_configs"):
    return _original_bookkeeping_sync(...)             # 无 mega,原样返回(零开销)

orig_input_batch = object.__getattribute__(self.input_batch, "_obj")   # 取回真 batch
try:
    res = _original_bookkeeping_sync(self, scheduler_output, ...)      # 此时 input_batch 还是 proxy → 见 W 行
finally:
    self.input_batch = orig_input_batch                # 恢复(proxy 用完丢弃)
```

- ⭐ **原始 bookkeeping 在 proxy 的 W 行视图下跑**,跑完 finally 恢复——"骗过上游"最后一棒。

### 6.3 ⭐ `_remux_logprobs_tensors`:W 行 → 单行(`:389-482`)

```python
for req_id, _, W_logits in request_row_configs:
    if W_logits == 0:                                  # CASE A: 中间 chunked prefill,占 dummy 行
        new_st_ids.append([[-1]]); new_lps.append(full((1,max_chained_dim), -inf)); ...
        continue
    # CASE B: 活跃采样
    new_st_ids.append(st_ids[gpu_row_cursor:gpu_row_cursor+1, :])      # 取首行 sampled id
    req_lps = orig_lps[gpu_row_cursor:gpu_row_cursor+W_logits, :target_K].reshape(1, W_logits*target_K)  # W行×K → 1行×(W·K)
    if W_logits*target_K < max_chained_dim: req_lps = pad(req_lps, ..., value=-inf)
    new_lps.append(req_lps)
    # lp_ids/ranks 同理(pad value=0)
    gpu_row_cursor += W_logits                         # 推进对齐下一请求
return torch.cat(...), ...                             # 每逻辑请求 1 行
```

- ⭐ **remux 精髓**:把"每请求 W 行 × K 列"重排成"每请求 1 行 × (W·K) 列"(pad 到 max_chained_dim)。`gpu_row_cursor` 按 W_logits 推进对齐每请求的行块。
- CASE A(W_logits==0)填 dummy 行保证每逻辑请求占一行。

### 6.4 回写 sampler_output + 重组返回值(`:744-796`)

```python
actual_K = orig_lps.shape[1]                           # = beam_width
max_chained_dim = max(W * actual_K for _, _, W in request_row_configs)
final_* = _remux_logprobs_tensors(request_row_configs, max_chained_dim, actual_K, ...)
st_ids.resize_(final_st_ids.shape).copy_(final_st_ids)            # ← 原位改写
lp_tensors.logprobs.resize_(...).copy_(...)
...
collapsed_req_ids_copy = [r[0] for r in request_row_configs if r[0] is not None]   # 去重逻辑 req_ids
num_logical_requests = len(collapsed_req_ids_copy)
return (num_nans_in_logits, [[]]*num_logical_requests, [[]]*num_logical_requests,
        prompt_logprobs_dict, collapsed_req_ids_copy, collapsed_req_id_to_index, invalid_req_indices)
```

- remux 后**原位改写 sampler_output** + 重组返回值,对外表现为"每请求 1 行"。W 条 beam 候选 packed 在 session_id 这一行 flat logprobs 里。

### 6.5 回 serving 侧:按 stride_k 切回 W beam(`serving_engine.py:444-514`)

```python
mega_result = output[0]
flat = mega_result.outputs[0].logprobs                 # remux 后一行 packed 的 W·K 候选
total_logprobs = len(raw_token_ids)
stride_k = total_logprobs // len(fork_info)            # 每条 beam 的候选数(主线 = beam_width)
for b_idx in range(len(fork_info)):
    start_offset = b_idx * stride_k; end_offset = start_offset + stride_k
    token_ids_pos, logprobs_pos, ... = _extract_and_dedup_segment(raw_*, start_offset, end_offset, logprobs_num)
    if valid_tokens_sets: logprobs_pos = [lp if tid in valid else -inf ...]   # catalog
    beam_flat_cache.append(...); all_beams_token_id.extend(...); all_beams_logprob.extend(cum + lp ...)
```

- ⭐ **remux 的 reshape 与 serving 的 stride_k 切段是互逆操作**。W 条 beam 候选在 worker packed,在 serving unpack。
- 后续 EOS/top-K/fork_info 与 baseline 共用(`:557-612`)。

### 6.6 收尾(`serving_engine.py:614-666`)

```python
if use_mega_request and prev_beam_internal_ids:
    await _mega_request_cleanup(self.engine_client, prefill_id, prev_beam_internal_ids, rank)   # B=0 清缓存
if sid_end_token_id: [beam.tokens.append(sid_end_token_id) for beam in all_beams]
completed.extend(all_beams)
best_beams = sorted(completed, key=lambda x: x.cum_logprob, reverse=True)[:beam_width]
for beam in best_beams:
    reconstruct_beam_logprobs(beam, initial_logprobs, sid_end_token_id)    # ← 父链回溯收网
    beam.text = tokenizer.decode(tokens)
yield RequestOutput(..., outputs=[CompletionOutput(...) for beam in best_beams], metrics=...)
```

- cleanup 发 B=0;`reconstruct_beam_logprobs`(`logprobs.py:56`)沿父链回溯重建完整 logprobs(单元 1 延迟 logprobs 的收网);yield 最终 RequestOutput → OpenAI 响应。**闭环完成**。

---

## 闭环总结

### 完整数据流

```
HTTP use_beam_search=True
  │  to_beam_search_params(注入 begin/end) → beam_search(入口, 单元1)
  ▼
┌─────────────── 主循环 max_tokens 步 ───────────────┐
│  首步(单元2): ADD_BATCH                              │
│    prepare_request → _add_requests_batch → ADD_BATCH │
│    子进程 _cache_beam_request(种 session 缓存)       │
│    → prefill → baseline 解析 → 展开成 W 条 beam      │
│    → 设 prefill_id=session_id, fork_info(开关翻转)   │
│  后续步(单元3): MEGA                                 │
│    _mega_request_step → register_beam_output(占队列) │
│    → mega_request_step_update(MegaRequestStepUpdate) │
│    子进程 _handle_mega_request_step_update:          │
│      拼 prefill+各beam后缀 → 单条 Request(session_id)│
│      继承 block_hashes(KV 复用) + 打 mega 标记       │
│  调度+Worker(单元4):                                 │
│    scheduler 裁 prefix_len + 透传 mega_data          │
│    _prepare_inputs: 重写 position(W beam 共享前缀)   │
│              + InputBatchProxy(伪装成 W 行)          │
│  Attention(单元5): cascade                          │
│    std + mega-prefix(非因果,paged) + suffix(因果,连续)│
│    + extract_suffix_kv(gather) + merge(lse)          │
│  采样+合并(单元6):                                   │
│    采样器见 W 行 → 产出 W 行 logprobs                │
│    bookkeeping(原,proxy 视图) → _remux(W行→1行)      │
│    回写 sampler_output + 重组返回值                  │
│    → 单个 RequestOutput 回 serving                   │
│  serving 解析: stride_k 切回 W beam → catalog 过滤   │
│    → EOS → np.argpartition top-K → 新 fork_info      │
└───────────────────── 循环 ─────────────────────────┘
  ▼
cleanup(B=0 清缓存) → reconstruct_logprobs(父链回溯)
  → yield RequestOutput → OpenAI 响应
```

### 贯穿全流程的 3 条主线

1. **session_id 单物理请求**:W 条 beam 始终由 `session_id` 一条 Request 承载;parent/child/pruned id 都是逻辑路由。
2. **W 行 ↔ 1 行的可逆变换**:worker 内用 `InputBatchProxy` 展开 W 行(采样),用 `_remux` 压回 1 行(输出);serving 用 `stride_k` 再切回。
3. **KV 复用三重保障**:继承 `block_hashes`(mega 合成)+ scheduler 裁 `prefix_len`(只复用前缀)+ cascade prefix 段(前缀 KV 只算一份)。

---

## 与旧 BEAM_FORK 机制对照

本目录 `online_pipeline.md` 等旧文档基于的 BEAM_FORK/template-clone 机制已被重构为 Mega 机制。主要区别:

| 维度 | 旧(BEAM_FORK / template-clone) | 当前(Mega / MEGA_REQUEST_STEP_UPDATE) |
|---|---|---|
| 解码步请求 | 每条 beam 一个 BEAM_FORK 请求(template-clone 共享 KV) | **单条 Mega Request**(session_id)装下全部 W beam |
| 消息类型 | `BeamForkRequest` | `MegaRequestStepUpdate`(`types.py:21`) |
| EngineCore 枚举 | `BEAM_FORK` | `ADD_BATCH`(b"\x05") + `MEGA_REQUEST_STEP_UPDATE`(b"\x06") |
| 首步 | 普通 ADD | `ADD_BATCH`(force_batch,建 beam_cache) |
| KV 共享 | template-clone(W 份共享 1 份) | 继承 `block_hashes` + cascade prefix 段 |
| attention | 标准 | **cascade attention**(prefix 非因果 + suffix 因果 + lse merge) |
| worker 展开 | — | `InputBatchProxy`(伪装 W 行)+ position 重写 + `_remux`(压回 1 行) |
| logprobs | dict | **FlatLogprobs** + 延迟父链重建 |

> 核心演进:从"多请求共享前缀 KV"(BEAM_FORK)→ "**单请求内部展开 W beam + cascade attention**"(Mega)。后者进一步减少了请求开销和调度复杂度。

---

## 关键文件索引

| 文件 | 作用 |
|---|---|
| `vllm_gr/entrypoints/openai/beam_search_patch.py` | patch 装配(patch_beam_search/patch_sampling/patch_batch_and_fork) |
| `vllm_gr/entrypoints/openai/protocol.py` | to_beam_search_params(注入 begin/end) |
| `vllm_gr/entrypoints/openai/serving_engine.py` | **在线 beam_search 主体** + _add_batch_step + _mega_request_step |
| `vllm_gr/sampling_params.py` | BeamSearchParams 子类(begin/end_token) |
| `vllm_gr/v1/engine/types.py` | MegaRequestStepUpdate 消息结构 |
| `vllm_gr/v1/engine/async_llm.py` | prepare_request/register_beam_output/mega_request_step_update |
| `vllm_gr/v1/engine/core_client.py` | add_requests_async/mega_request_step_update_async |
| `vllm_gr/v1/engine/core_client_patch.py` | apply_batch_fork_patches(挂 AsyncLLM/AsyncMPClient) |
| `vllm_gr/v1/engine/core.py` | _cache_beam_request/_handle_mega_request_step_update/process_input_sockets |
| `vllm_gr/v1/engine/engine_core_patch.py` | scheduler/worker patch + _remux + InputBatchProxy + position 重写 |
| `vllm_gr/v1/attention/backends/beam_attn.py` | **BeamAttention cascade 后端** + triton gather kernel |
| `vllm_gr/logprobs.py` | FlatLogprobs 优化 + reconstruct_beam_logprobs |
| `vllm_gr/patch.py` | `_apply_gr_beam_patches` 总装 |

---

## 深入阅读

本文是**概览级**走读。单元 4(position 重写 + InputBatchProxy)、单元 5(cascade 索引)、单元 6(_remux)这三个最难的机制,在 [`mega_deep_dive.md`](mega_deep_dive.md) 里有**数值与张量级深化**:用一个贯穿例子(W=3 / prefix_len=4 / decode_steps=2)把每一步的张量形状、索引值、reshape/gather/merge 都填实,并附「易踩坑速查」。读本文后想确认底层细节,转去那份。
