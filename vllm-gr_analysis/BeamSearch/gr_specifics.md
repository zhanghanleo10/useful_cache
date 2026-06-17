# GR(生成式推荐)特有机制

**状态**: 基于 `serving_models.py` / `sampling_params.py` / `engine/__init__.py` / `request.py` 核对
**配套**: 在线链路见 `online_pipeline.md`,勘误见 `errata.md`

原文档聚焦 beam search 主线,**完全没写 GR 特有的机制**。本文档补三块:Catalog/Trie(推荐场景候选约束)、begin/end_token(session 标记)、GR 类型扩展链路(gr_features 透传)。

---

## 1. Catalog / Trie — 推荐场景的候选约束

**文件**: `entrypoints/openai/serving_models.py:18-49`

### 本质:前缀树,做"前缀 → 合法下一步"查询

Catalog **不是"合法商品集合",而是用嵌套字典实现的 Trie**(Python 惯用变体,无显式 TrieNode 类)。

```python
class Catalog:
    def __init__(self, tokenizer):
        self.trie: dict[int, Any] = {}    # 嵌套字典 = trie
```

| Trie 概念 | 代码 |
|---|---|
| 一个节点 | 一个 `dict` |
| 节点的出边 | dict 的 **key**(`token_id`) |
| 边指向的子节点 | dict 的 **value**(另一个 dict) |
| 叶子(序列终点) | 空 dict `{}` |

### `load(path)`(`23-40`)— 把一组合法序列建成 trie

```python
def load(self, path):
    catalog_data = json.load(open(path))    # JSON: list of token 序列
    for seq in catalog_data:
        token_ids = self.tokenizer.convert_tokens_to_ids(seq)   # token 字符串 → id
        node = self.trie
        for token_id in token_ids:
            if token_id not in node: node[token_id] = {}        # 没有就建
            node = node[token_id]                                # 有就走(共享前缀)
```

这是**标准教科书 Trie insert**,只是用裸 dict 实现。共享前缀是 Trie 省空间的根本:`[A,B,C]` 和 `[A,B,D]` 共享 A→B。

### `valid(token_ids)`(`42-49`)— 给定前缀,返回合法下一步

```python
def valid(self, token_ids):
    node = self.trie
    for tid in token_ids:            # 沿前缀下钻(标准 trie prefix walk)
        if tid in node: node = node[tid]
        else: return set()           # 前缀本身非法 → 空集
    return set(node.keys())          # 当前节点的子 key = 合法下一步
```

例:cata log 有 `[A,B,C]`、`[A,B,D]`、`[A,X]`,建出的 trie:
```
root
└─ A
   ├─ B
   │  ├─ C   ← 叶子(ABC 结束)
   │  └─ D   ← 叶子(ABD)
   └─ X      ← 叶子(AX)

valid([])     = {A}        # 空前缀:第一步只能 A
valid([A])    = {B, X}     # 已生成 A:可 B 或 X
valid([A,B])  = {C, D}     # 已生成 AB:可 C 或 D
valid([A,B,C])= set()      # 序列完整:无合法下一步
valid([B])    = set()      # B 不能作首步:前缀非法
```

**为什么 `valid` 依赖 `generated_tokens`**:前缀匹配——给定已生成序列,trie 返回合法的下一步。

### 在主循环里怎么用

**① 算合法集**(`serving_engine.py:344-353`)——和 engine step 并行:
```python
def get_valid_tokens_set(beam):
    generated_tokens = beam.tokens[len(prompt_token_ids):]    # 已生成部分(剥掉 prompt)
    return self.models.catalog.valid(generated_tokens)        # 前缀 → 合法下一步
catalog_task = asyncio.create_task(_run_catalog(
    [asyncio.to_thread(get_valid_tokens_set, beam) for beam in all_beams]))
```

**② 过滤候选**(`serving_engine.py:449-454`):
```python
valid_tokens_set = valid_tokens_sets[i]
logprobs_pos = [lp if tid in valid_tokens_set else -float("inf")   # 不合法 → -inf
                for tid, lp in zip(token_ids_pos, logprobs_pos)]
```
不合法候选 logprob 置 `-inf`,Top-K 必然淘汰。**beam search 被硬约束在 catalog 合法路径上**,生成结果一定是 catalog 里某条合法序列的前缀。

### 三个工程要点

1. **硬约束**(掩码式 `-inf`),不是软的。
2. **CPU 查询、GPU 并行**:`valid()` 是纯 Python 字典遍历(CPU 串行),用 `asyncio.to_thread` 丢线程池,和 decode 同时跑。catalog 大时可能成瓶颈,值得 profile。
3. **配置入口**:`--catalog_path` CLI(`arg_utils_gr.py:41`)→ `model_config.catalog_path` → `patch_OpenAIServingModels_init`(`serving_models.py:62-65`)在 `OpenAIServingModels.__init__` 加载,挂到 `self.catalog`。`self.models.catalog is None` 时整段跳过(非推荐场景)。

### 简化版 Trie 的隐含限制

省了 `is_end` 标记,用"空 dict"隐式表示终点。代价:**无法表达"某前缀既是完整序列、又能继续扩展"**。比如 catalog 同时有 `[A,B]`(短)和 `[A,B,C]`(长):插入 `[A,B]` 后 B 节点=`{}`,再插 `[A,B,C]` 会把 B 改成 `{C:{}}`,"AB 是完整序列"信息丢失。对 beam search 用途无害(只问"下一步合法啥",不关心"前缀是否完整")。

---

## 2. begin_token / end_token — Session 标记(SID)

**文件**: `sampling_params.py:14-15`(字段定义)+ `serving_engine.py:204-222, 268-270, 531-533`

```python
class BeamSearchParams(BaseBeamSearchParams, ...):
    begin_token: str | None = None    # session 起始 token
    end_token: str | None = None      # session 结束 token
```

这是 vllm-gr 对原生 `BeamSearchParams` 的**唯一结构扩展**。语义:推荐序列以 `begin_token` 起、`end_token` 止,标记一个 session/交互边界。

在主循环里:
- `sid_begin_token_id`(`:206-213`):转成 id,初始化时追加到 `initial_tokens` + `initial_logprobs`(`:268-270`)
- `sid_end_token_id`(`:214-222`):转成 id,收尾时追加(`:531-533`),`reconstruct_beam_logprobs` 也用它(`:542`)
- `pre_calc`(`:293-301`):max_tokens 要为 begin/end 各预留 1 步

普通文本 beam search 用不到(设 None),这是推荐场景特有。

---

## 3. GR 类型扩展链路 — gr_features 全链路透传

**文件**: `engine/__init__.py` / `request.py` / `core/sched/output.py` / `gr_features_codec.py`

这是 patch.py 里**第一个、最重要**的 patch(`_apply_gr_core_patches`),让"生成式推荐的特征"穿过整个 vLLM 管线。原文档完全没写,但它是理解 `priority`、`gr_features` 等字段来源的钥匙。

### 三个 GR 类型

```python
# engine/__init__.py:28
class GREngineCoreRequest(EngineCoreRequest):
    gr_features: bytes | None = None    # type-erased GR 特征,array_like 序列化末位

# request.py:17
class GRRequest(Request):
    def __init__(self, gr_features=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gr_features = gr_features    # 携带 GR 特征穿过 Scheduler → ModelRunner

    @classmethod
    def from_engine_core_request(cls, request, block_hasher=None):
        return cls(gr_features=getattr(request, "gr_features", None), ...)

# core/sched/output.py
class GRNewRequestData(NewRequestData): ...    # 扩展调度输出数据
```

### 替换机制(`patch.py:_apply_gr_core_patches:99-167`)

```python
replacements = [
    (EngineCoreRequest, GREngineCoreRequest, "EngineCoreRequest"),
    (Request, GRRequest, "Request"),
    (NewRequestData, GRNewRequestData, "NewRequestData"),
]
# Phase 1: 先 patch source module,保证后续 `from X import Y` 拿到 GR 版本
# Phase 2: patch 所有已加载的 consumer module
for module_name, module in list(sys.modules.items()):
    for original, replacement, attr_name in replacements:
        if getattr(module, attr_name, None) is original:
            setattr(module, attr_name, replacement)
```

**双阶段替换**(source-first):先改源模块,再扫已加载模块。保证后加载模块的 `from vllm.v1.request import Request` 拿到 GRRequest。

### 验证(`patch.py:verify_patches:716-751`)

检查关键 consumer(`vllm.v1.engine`、`vllm.v1.engine.core`、`vllm.v1.core.sched.scheduler` 等)是否已替换,未替换则 warning("gr_features may be lost")。

### gr_features 编解码

`gr_features: bytes` 是 type-erased 的 msgpack 编码。每个 adapter family(HSTU 等)自定义 payload 结构,通过 `gr_features_codec.py` 编解码。这样 vllm-gr 核心不依赖具体模型,各 adapter 自己定义特征格式。

### 为什么这条链路重要

- `priority` 字段(`BeamForkRequest`、`Request`):虽然不是 GR 独有,但 GR beam search 重度依赖它做分组(`beam_prefix_groups` 按 priority 聚合同一 beam search 的请求)。
- `gr_features`:推荐场景的用户特征/上下文,需要穿过 EngineCore→Scheduler→ModelRunner 传到模型 forward。
- GR 类型替换是 vllm-gr 的**地基**:beam search 优化是在这个地基上的应用层。原文档只讲应用层(beam search),漏了地基。

---

## 4. 其他 GR 模块(简述)

| 模块 | 位置 | 作用 |
|---|---|---|
| HSTU 模型 | `model_executor/models/` + `adapters/hstu/` | 生成式推荐模型架构,`register_gr_model_architectures` 注册 |
| FuXi 模型 | `config/fuxi_vllm_registration.py` | 另一种 GR 模型配置 |
| adapters 自动发现 | `adapters/registry.py` + `_register_adapter_families` | 扫描 `vllm_gr/adapters/*/`,每个 family 有 Layer1(`__init__.py` register_family)+ Layer2(`adapter.py` register_adapter_class) |
| NPU/昇腾支持 | `patch.py:_apply_yuanrong_npu_patch` + `_apply_kv_merge_patch` | yuanrong KV transfer backend + HSTU 风格 K+V 合并 cache(`kv_cache_per_layer=1`)。配合 `vllm-ascend/` |
| LMCache 集成 | `patch.py:patch_lmcache_adapter` | 可选,环境变量 `VLLM_GR_LMCACHE_PATCH` 或 `LMCacheConnectorV1` 触发 |
| 推荐场景 API | `entrypoints/pooling/recommend/` | 推荐专用 serving / protocol / api_router |

beam search 是通用框架,具体推荐模型(HSTU 等)和硬件后端(NPU)在各自模块里。本文档聚焦 beam search 链路用到的 GR 机制(Catalog/Token/类型扩展),其余模块按需深入。
