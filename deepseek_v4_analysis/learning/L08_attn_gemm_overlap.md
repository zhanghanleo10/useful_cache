# L8 · attn_gemm_parallel_execute(多流 GEMM overlap)

> 单元④注意力。attn 的输入阶段有 **4 个独立 GEMM**,DSV4 用 1 个 default + 3 个 aux CUDA stream 并行算它们。本篇讲这套 overlap 机制。

## 🎯 学习目标

学完能回答:
1. attn 输入阶段有哪 4 个 GEMM?为什么它们能并行?
2. 3 个 aux stream 各跑什么?default stream 跑什么?
3. `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` 是干什么的?为什么小 batch 要关掉多流?
4. `ln_events` 在 overlap 里起什么作用?

## 📖 讲解

### 背景:4 个独立输入 GEMM(`attention.py:366-424`)

attn 的输入准备阶段要算 4 个矩阵乘,它们的**输入都是同一个 `hidden_states`**,输出互不依赖:

| GEMM | 产出 | 产出给谁 | stream |
|------|------|---------|--------|
| `fused_wqa_wkv(hidden)` | `qr_kv` = [qr, kv] | MLA 主体(最重) | **default** |
| `compressor.fused_wkv_wgate(hidden)ᵀ` | `kv_score` | Compressor | aux[0] |
| `indexer.weights_proj(hidden)` | `indexer_weights` | Indexer | aux[1] |
| `indexer.compressor.fused_wkv_wgate(hidden)ᵀ` | `indexer_kv_score` | Indexer 的 Compressor | aux[2] |

> 串行算这 4 个会成为 attn 的瓶颈;它们读同一份 hidden、写不同输出 → **天然可并行**。

### 机制:`execute_in_parallel`(`:414-422`)

```python
qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
    fused_wqa_wkv,                    # default stream
    aux_fns,                          # [compressor_kv_score, indexer_weights_proj, indexer_compressor_kv_score]
    self.ln_events[0],                # fan-out 起始事件
    self.ln_events[1:4],              # 每个 aux 的完成事件
    aux_streams,                      # 3 个 aux CUDA stream
    enable=num_tokens <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
)
```

- default stream 跑最重的 `fused_wqa_wkv`
- aux[0..2] 各跑一个轻 GEMM,和 default **并发执行**
- 末尾 join,所有结果一起返回

### `ln_events` 的同步作用(`:266-270`)

```python
self.ln_events = [torch.cuda.Event() for _ in range(4)]
# [0]: fan-out 起始事件(GEMM 开始)
# [1..3]: 每个 aux GEMM 的完成事件
```

- aux stream 启动前 `wait(ln_events[0])`:等 default 把 hidden 准备好(其实 hidden 已就绪,这里是统一同步点)
- aux GEMM 完成后 `record(ln_events[1..3])`
- default 在 join 时 `wait(ln_events[1..3])`:确保 aux 结果可见

### 小 batch 关闭多流(`:420-421`)

```python
enable=num_tokens <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD
```

> token 数小于阈值时(典型 decode 小 batch),每个 GEMM 都很小,**多流调度的开销(stream 切换、event 同步)可能超过并行收益**。此时 `enable=False`,`execute_in_parallel` 退化为串行,反而更快。大 batch(prefill)才开。

### ROCm 退化(`:375,421` 注释)

ROCm 上 `aux_stream_list is None`(aux stream 未创建),`execute_in_parallel` 串行执行 —— AMD 平台暂不做这个 overlap。

## 🔑 小结

> attn 输入 4 个 GEMM 读同份 hidden、写不同输出 → 天然可并行。DSV4 用 **default(最重 fused_wqa_wkv)+ 3 aux stream(compressor/indexer 的 GEMM)** 并发,`ln_events` 做 fan-out/join 同步;**小 batch 关闭**(省 stream 开销),**ROCm 退化串行**。

## ✅ 自测

<details>
<summary><b>Q1</b>:为什么把最重的 <code>fused_wqa_wkv</code> 放 default stream,而不是也丢给 aux?</summary>

两个原因:① default stream 是主路径,后续 `fused_q_kv_rmsnorm` 和 `attention_impl` 都在 default 上,且要消费 `qr_kv`,放 default 避免跨流同步延迟;② 把最重活留 default,让轻 GEMM 在 aux 上和它并发,这样 default 的执行时间 = 最长路径,aux 只要更短就能藏进去,overlap 收益最大。如果把最重的也丢 aux,default 空等,反而浪费。
</details>

<details>
<summary><b>Q2</b>:这 4 个 GEMM 读同一份 <code>hidden_states</code>,并行时会有写冲突吗?</summary>

不会。它们都只**读** hidden、写各自不同的输出 tensor(qr_kv / kv_score / indexer_weights / indexer_kv_score),没有写后读依赖。读同一份只读数据天然安全,无需同步。同步点(`ln_events`)只用于保证「输出就绪后下游才用」,不是为了读冲突。
</details>

**下一步**:L8 把输入 GEMM 并行算完,产出 `qr/kv` 等。L9 讲这些 latent 怎么做 norm+rope+quant 并写进 KV cache。
