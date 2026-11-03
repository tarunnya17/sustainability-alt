# #45473 [pull_request] Fix EP: RouterParallel shape, tp_plan property, grouped_mm sentinels
https://github.com/huggingface/transformers/pull/45473  ·  opened by AmineDiro on 2026-04-16

## [body] AmineDiro (MEMBER) · opening post · 2026-04-16
While benchmarking Qwen3-30B-A3B SFT training with Expert Parallelism (EP) using TRL, I found three bugs that combine to produce silently wrong results or NaN loss. Every existing test uses `tp_plan="auto"` which bypasses `RouterParallel` and EP sentinel handling entirely if I understand correctly.

These fixes are related to  #45436 (will update with the expert-only EP config once this lands 👍🏼 )

### Bug 1: `RouterParallel` breaks score/index pairing (`tensor_parallel.py`)

The router produces three element-wise paired tensors: `router_logits (seq, num_experts)`, `router_scores (seq, top_k)`, `router_indices (seq, top_k)`. `_prepare_output_fn` is supposed to remap these to local expert space per EP rank.

The old code did `scatter` scores into a full `(seq, num_experts)` matrix then sliced to `(seq, num_local_experts)`, changing the shape to e.g. `(3, 64)` while indices stay `(3, 8)`. All expert forward implementations (`grouped_mm`, `batched_mm`, eager) flatten both with `reshape(-1)` and rely on element-wise pairing. 
This means we got mismatched shapes, the flattened tensors have different lengths, and every routing weight gets paired with the wrong expert.

**Fix**: Replace `scatter+slice` with `masked_fill`. We zero out non-local scores in-place, preserving the `(seq, top_k)` shape. I verified using  EP=1,2,4 logits match non-EP ground truth within bf16 precision.

### Bug 2: Weight loading uses wrong plan (`modeling_utils.py:4281`)

`convert_and_load_state_dict_in_model` builds a regex from `tp_plan.keys()` to decide *which* params to shard, then looks up `model.tp_plan[pattern]` to decide *how*. The call site passed `model._tp_plan` (raw TP plan), but the lookup uses the `tp_plan` property which returns `_ep_plan` when EP is enabled. When using an expert-only EP plan (no attention entries), the regex matches attention params from `_tp_plan` but the lookup in `_ep_plan` raises `KeyError`.

**Fix**: `model._tp_plan` → `model.tp_plan` so both regex and lookup use the same plan.

### Bug 3: `grouped_mm_experts_forward` doesn't handle EP sentinels (`moe.py`)

After Bug 1 fix, `RouterParallel` correctly sets non-local expert indices to a sentinel value (`num_local_experts`). `batched_mm` already handles this (clamp + masked_fill). But `grouped_mm` passes sentinel IDs straight to `torch.histc` which doesn't count them (out of bin range) → `grouped_mm` never writes those output row. So if we have got uninitialized memory NaN propagating through residual connections to every token.

**Fix**: Same pattern as `batched_mm`, clamp sentinel IDs before sort, `masked_fill` after weighted multiply.

### Tests added

- `test_router_parallel_preserves_shape`:  regression test for Bug 1
- `test_router_parallel_score_index_pairing` : verifies element-wise score/index pairing across all EP ranks
- `test_grouped_gemm_shard_tensor_uses_rank_not_device` : verifies shard uses mesh-local rank
- `test_grouped_mm_experts_sentinel_handling`: verifies no NaN from sentinel expert IDs

### End-to-end validation

Tested on 8×H100 with Qwen3-30B-A3B, input `[[1, 2, 3]]`, comparing EP against non-EP ground truth.

### Before submitting
- [x] This is not a pure code-agent PR:  I found these bugs while benchmarking and wrote the fixes.
- [x] Read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request).

### Who can review?

@3outeille @ArthurZucker

## [c3109248402] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:4281 · 2026-04-20
great catch! 🫡 
we did introduce the API then removed it and did not check

## [c3109256292] ArthurZucker (COLLABORATOR) · review_comment on tests/tensor_parallel/test_tensor_parallel.py:1 · 2026-04-20
IDK if we want this or just an explicit test of a model setting ep_plan! Would rather explicit small test but fine otherwise

## [c4138383236] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-04-20
A nice follow up to #43730!
Ty for working on this!

## [c3110494284] AmineDiro (MEMBER) · review_comment on tests/tensor_parallel/test_tensor_parallel.py:1 · reply to [c3109256292] · 2026-04-20
I removed these unit integration,  [6f91136](https://github.com/huggingface/transformers/pull/45473/commits/6f911366c38a1a3ff952d9cb63cdc1388f62e2d1)  should tests EP end to end,  https://github.com/huggingface/transformers/pull/45436 PR for qwen3_moe will also be tested 👍🏼

## [c4139772520] AmineDiro (MEMBER) · review COMMENTED · 2026-04-20 (no text)

## [c3110567932] ArthurZucker (COLLABORATOR) · review_comment on tests/test_tensor_parallel_mixin.py:352 · 2026-04-20
```suggestion
        ep_plan = "auto",
```
might also be "more intuitive"?

## [c3110569546] ArthurZucker (COLLABORATOR) · review_comment on tests/test_tensor_parallel_mixin.py:352 · reply to [c3110567932] · 2026-04-20
(a note but we can fix this if it makes sense for you!) 🤗

## [c3110571296] ArthurZucker (COLLABORATOR) · review_comment on tests/test_tensor_parallel_mixin.py:1 · 2026-04-20
perf!

## [c4139869833] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-04-20 (no text)

## [c3110633981] AmineDiro (MEMBER) · review_comment on tests/test_tensor_parallel_mixin.py:352 · reply to [c3110567932] · 2026-04-20
From what I can understand `tp_plan` can be "auto" OR a dict for config ? There's no user-supplied dict for EP, the plan always comes from the config so it's not consistent with the other params.

I also saw the comment  in `src/transformers/distributed/configuration_utils.py`
```python
@dataclass
class DistributedConfig:
    enable_expert_parallel: bool = False
    # TODO: add tp_plan, pp_plan, device_mesh etc..
``` 

which makes me think thats the forward direction of the api? not sure really.

## [c4139954422] AmineDiro (MEMBER) · review COMMENTED · 2026-04-20 (no text)

## [c4280928449] AmineDiro (MEMBER) · issue_comment · reply to [c4139869833] · 2026-04-20
@ArthurZucker  I see tests failing in CI. 
  - All 5 failures: PreTrainedModel.__init__ → _can_set_experts_implementation. It calls sys.modules[cls.__module__] and then key error  (modeling_utils.py:1971, from #42697).
  - I ran claude through and I think `tests/utils/test_auto_docstring.py` benchmarks import cost and deletes sys.modules[k] for every transformers.models.* without restoring them. The classes stay live (other namespaces hold refs), but sys.modules loses the entry.
  - On main, these two test files land in different CircleCI shards.

I can fix (one-liner, same pattern applies to `_can_set_attn_implementation (1952)`):  47497ce
```python
  class_module = sys.modules.get(cls.__module__)
  if class_module is None or not hasattr(class_module, "__file__"):
      return False
```
I fold the None case into the existing branch that already returns False when `__file__` is missing. Semantically, False means "we can't verify this class supports the new experts API, so assume it doesn't", which is exactly what the fallback does.

!! **tradeoff:** if the MoE model is being instantiated after something has evicted its entry from `sys.modules` (today, only `test_auto_docstring._clear` does this). In that case, the model silently falls back to "eager" experts
  instead of `grouped_mm`. That's a perf regression, never a correctness issue. It only affects MoE models, non-MoE models (Bert, Bart, Mistral, etc., i.e. the 5 tests failing here) don't use experts at all, so "eager" is a no-op for them. If a user explicitly asked for experts_implementation="grouped_mm", they still get a clear ValueError rather than a silent downgrade.

## [c3115609818] ArthurZucker (COLLABORATOR) · review_comment on tests/test_tensor_parallel_mixin.py:352 · reply to [c3110567932] · 2026-04-21
Yep, I was wondering what was more intuitive for you!

## [c4145716997] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-04-21 (no text)

## [c4145727876] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-04-21
sounds good! merging

## [c3116765952] BenjaminBossan (MEMBER) · review_comment on src/transformers/models/gpt_oss/configuration_gpt_oss.py:28 · reply to [c4145727876] · 2026-04-21
Hey @ArthurZucker @qgallouedec This addition leads to PEFT tests crashing. After investigating, it stems from `trl-internal-testing/tiny-GptOssForCausalLM`:

```python
from transformers import AutoModelForCausalLM
model_id = 'trl-internal-testing/tiny-GptOssForCausalLM'
model = AutoModelForCausalLM.from_pretrained(model_id)
```

```
[transformers] GptOssForCausalLM LOAD REPORT from: trl-internal-testing/tiny-GptOssForCausalLM
Key                                               | Status   | Details                                                                                     
--------------------------------------------------+----------+---------------------------------------------------------------------------------------------
model.layers.{0, 1}.mlp.router.weight             | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([128, 8]) vs model:torch.Size([4, 8])        
[… 35 lines of code omitted]

With `openai/gpt-oss-20b`, the same code will pass. So probably it's a misconfiguration of the tiny model, but I wanted to flag this and also wonder if there is a quick fix. Changing all tests to `ignore_mismatched_sizes=True` isn't really an option and changing `"num_local_experts": 4` in the model config is not enough for the error to go away.

## [c4146948630] BenjaminBossan (MEMBER) · review COMMENTED · 2026-04-21 (no text)

## [c3117801792] qgallouedec (MEMBER) · review_comment on src/transformers/models/gpt_oss/configuration_gpt_oss.py:28 · reply to [c3116765952] · 2026-04-21
Thanks, I'll look into it

## [c4148060051] qgallouedec (MEMBER) · review COMMENTED · 2026-04-21 (no text)

## [c3118829483] qgallouedec (MEMBER) · review_comment on src/transformers/models/gpt_oss/configuration_gpt_oss.py:28 · reply to [c3116765952] · 2026-04-21
it makes the CI fail for TRL as well

## [c4149217804] qgallouedec (MEMBER) · review COMMENTED · 2026-04-21 (no text)

## [c3118837384] qgallouedec (MEMBER) · review_comment on src/transformers/models/gpt_oss/configuration_gpt_oss.py:28 · reply to [c3116765952] · 2026-04-21
@AmineDiro do you mind taking a look when possible https://github.com/huggingface/trl/actions/runs/24731559460/job/72347133688#step:9:6880

## [c4149226897] qgallouedec (MEMBER) · review COMMENTED · 2026-04-21 (no text)

## [c3122959882] AmineDiro (MEMBER) · review_comment on src/transformers/models/gpt_oss/configuration_gpt_oss.py:28 · reply to [c3116765952] · 2026-04-22
I'll look into it !

## [c4153590579] AmineDiro (MEMBER) · review COMMENTED · 2026-04-22 (no text)

## [c3124296477] qgallouedec (MEMBER) · review_comment on src/transformers/models/gpt_oss/configuration_gpt_oss.py:28 · reply to [c3116765952] · 2026-04-22
@albertvillanova fixed it in https://github.com/huggingface/trl/pull/5622

## [c4155040222] qgallouedec (MEMBER) · review COMMENTED · 2026-04-22 (no text)

## [c3124362798] BenjaminBossan (MEMBER) · review_comment on src/transformers/models/gpt_oss/configuration_gpt_oss.py:28 · reply to [c3116765952] · 2026-04-22
So IIUC, with that PR, a new model will be pushed to `trl-internal-testing/tiny-GptOssForCausalLM` which should work with latest Transformers? Thanks for taking care.

## [c4155111759] BenjaminBossan (MEMBER) · review COMMENTED · 2026-04-22 (no text)

## [c3124646243] AmineDiro (MEMBER) · review_comment on src/transformers/models/gpt_oss/configuration_gpt_oss.py:28 · reply to [c3116765952] · 2026-04-22
Ok great, I went ahead and removed the attr map but I can close if thats not necessary https://github.com/huggingface/transformers/pulls/AmineDiro

## [c4155426636] AmineDiro (MEMBER) · review COMMENTED · 2026-04-22 (no text)

## [c3131192132] albertvillanova (MEMBER) · review_comment on src/transformers/models/gpt_oss/configuration_gpt_oss.py:28 · reply to [c3116765952] · 2026-04-23
Yes @BenjaminBossan, I have regenerated the tiny GPT-OSS model and pushed it to the Hub: https://huggingface.co/trl-internal-testing/tiny-GptOssForCausalLM/discussions/2/files

Now, the number of experts is correctly set to 4 with `num_local_experts=4`. Before, `num_experts=4` was silently ignored. This ensures the checkpoint is consistent with the intended configuration and works properly with all transformers versions.

---
Additionally, I've created a GitHub team for TRL maintainers, so you can now notify all TRL maintainers at once by tagging `@huggingface/trl`.

## [c4162867937] albertvillanova (MEMBER) · review COMMENTED · 2026-04-23 (no text)

---
outcome: state=closed, merged=True
