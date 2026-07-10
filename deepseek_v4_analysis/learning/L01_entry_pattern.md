# L1 · vLLM 入口范式

> 单元①骨架。理解模型 forward 的**接口契约**,再看代码会顺畅很多。

## 🎯 学习目标

学完能回答:
1. DSV4 的 `forward` 返回什么?为什么不是 logits?
2. `compute_logits` 在什么时候被调用?
3. Pipeline Parallel 下,`forward` 的输入在不同 stage 有什么不同?

## 📖 讲解

### 顶层类结构(`nvidia/model.py:1251`)

```python
class DeepseekV4ForCausalLM(nn.Module, SupportsPP, DeepseekV4MixtureOfExperts):
    model_cls = DeepseekV4Model

    def __init__(self, *, vllm_config, prefix=""):
        self.model = self.model_cls(...)                    # 主干 transformer
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(...)               # 仅末 PP rank
        else:
            self.lm_head = PPMissingLayer()                  # 中间 stage 不要
        self.logits_processor = LogitsProcessor(...)
```

三个组件:`model`(主干)、`lm_head`(词表投影,仅末 rank)、`logits_processor`。

### forward 极薄(`model.py:1315`)

```python
def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
    return hidden_states      # ← 返回 hidden_states,不是 logits!
```

### logits 分离(`model.py:1308`)

```python
def compute_logits(self, hidden_states):
    logits = self.logits_processor(self.lm_head, hidden_states)
    return logits
```

**为什么分离?** vLLM 的推理分两阶段:
- **forward**:跑 transformer 产 hidden(每个 token 都要跑)
- **采样阶段**:才需要 logits(只有**最后采样的那几个 token** 需要 lm_head + softmax)

多数 token 的 hidden 算完就丢(只留最后需要采样的)。把 lm_head 从 forward 里拆出来,避免给所有 token 都算一遍词表投影(词表很大,很贵)。MTP / 投机解码更是只在验证时算 logits。

### PP 接口(`SupportsPP`)

`forward` 的两个输入参数对应不同 PP stage:

| stage | 有效输入 | 来源 |
|-------|---------|------|
| 首 rank | `input_ids` / `inputs_embeds` | 用户请求 |
| 中间 rank | `intermediate_tensors` | 上一 stage 的输出(`IntermediateTensors`) |
| 末 rank | `intermediate_tensors` | 同上,且本 rank 产出 hidden 给 `compute_logits` |

配套的 `make_empty_intermediate_tensors`(`model.py:1279`)告诉框架 PP 边界上传什么形状的张量(DSV4 是 `(N, hc_mult, hidden)`)。

## 🔑 小结

> DSV4 forward 遵循 vLLM 范式:**返回 hidden_states,logits 由 `compute_logits` 按需算**(省掉非采样 token 的词表投影);`forward` 把 `input_ids` 和 `intermediate_tensors` 都收下以适配 Pipeline Parallel。

## ✅ 自测

<details>
<summary><b>Q1</b>:为什么 <code>lm_head</code> 只在末 PP rank 创建,<code>compute_logits</code> 只在末 rank 有意义?</summary>

lm_head 把 hidden → 词表 logits,只有最终采样需要。中间 PP stage 的输出还要喂给下一 stage 继续算 transformer,不需要 logits。所以在中间 rank 用 `PPMissingLayer()` 占位省显存,`compute_logits` 也只在末 rank 被调用。代码:`model.py:1270-1277`。
</details>

<details>
<summary><b>Q2</b>:如果 forward 直接返回 logits,在「每个序列只采样最后 1 个 token」的 decode 场景会浪费什么?</summary>

浪费一次 `[N, hidden] × [hidden, vocab]` 的词表投影(vocab 通常 10w+),而其中只有最后 1 个 token 的 logits 有用。分离 `compute_logits` 后,只对需要采样的 token 调用,省掉其余 token 的词表投影计算。
</details>

**下一步**:L1 搞清了「forward 返回 hidden」,L2 就钻进 `self.model(...)` 里看 hidden 是怎么算出来的。
