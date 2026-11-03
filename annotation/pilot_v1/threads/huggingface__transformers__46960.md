# #46960 [pull_request] Fix silent SDPA math-kernel fallback for GQA when key/value head_dim > 256 or differ
https://github.com/huggingface/transformers/pull/46960  ·  opened by Butterfingrz on 2026-06-29

## [body] Butterfingrz (CONTRIBUTOR) · opening post · 2026-06-29
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=46960)

## What does this PR do?

For attention layers that use GQA with head_dim > 256 (notably Gemma4's head_dim=512 full-attention layers), when no attention_mask is passed (mask=None) during training/prefill, sdpa_attention_forward passes enable_gqa=True to SDPA. However, PyTorch's mem-efficient (EFFICIENT_ATTENTION) and flash SDPA kernels do not support the enable_gqa broadcast at head_dim > 256, so SDPA silently falls back to the MATH kernel, which materializes the S×S score matrix (O(S²) memory) and OOMs at long sequence lengths (e.g. 8192).

Because this fallback is completely silent (no warning), it is very hard to diagnose — it is effectively a "should run on the efficient path, but quietly degrades" correctness/resource bug. This PR makes use_gqa_in_sdpa return False for head_dim > 256, so we take the existing repeat_kv (MHA) path, which the mem-efficient kernel handles fine at head_dim=512, avoiding the silent MATH fallback. (This does not change behavior for head_dim ≤ 256 models; for head_dim > 256 no kernel supports the enable_gqa broadcast anyway, so repeat_kv is the strictly better and correct fallback.)

The same silent fallback also affects MLA-style models where the key and value head_dim differ (e.g. DeepSeek-V3 with qk=192 / v=128), as well as MLA where only `value` head_dim exceeds 256. Under `enable_gqa=True` the Flash kernel is disqualified (it requires q/k/v to share the last dim ≤ 256) and the mem-efficient kernel lacks the GQA broadcast, so only cuDNN can serve them — and they drop to MATH on any build/GPU without the cuDNN SDPA backend. This PR guards all of these: it checks both key and value head_dim, and requires them to be equal.

## Root cause

The root cause is that `use_gqa_in_sdpa` decides whether to use `enable_gqa=True` based only on:

* `torch >= 2.5`
* `attention_mask is None`

It does not take the attention head dimension into account.

```python
def use_gqa_in_sdpa(attention_mask, key):
    ...
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None
```

As a result, `sdpa_attention_forward` skips the existing `repeat_kv` path and passes `enable_gqa=True` whenever `mask is None`.

For layers with `head_dim > 256`, this requests an unsupported SDPA GQA-broadcast path. PyTorch's efficient/flash SDPA kernels reject that path, so SDPA silently falls back to the MATH kernel, which materializes the full `S × S` attention score matrix.

### Trigger chain for Gemma4

```text
Gemma4TextAttention full-attention layer
  ├─ config.global_head_dim = 512
  └─ no sliding_window / local_size
        ↓
[… 19 lines of code omitted]

Sliding-window layers are unaffected because they always materialize a 4D attention mask when `kv_length > sliding_window`. Therefore, they do not enter the `attention_mask is None` path and do not trigger `enable_gqa=True`.

This is why the issue is specifically observed on Gemma4's `head_dim=512` full-attention layers, while sliding layers do not OOM through this path.

## The fix

`use_gqa_in_sdpa` now also takes `value` and only opts into `enable_gqa=True` when the key and value head_dim are equal and ≤ 256:

```diff
-def use_gqa_in_sdpa(attention_mask, key):
+def use_gqa_in_sdpa(attention_mask, key, value):
     ...
     if _is_torch_xpu_available:
         return _is_torch_greater_or_equal_than_2_8
-    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None
+    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None and key.shape[-1] == value.shape[-1] <= 256
```

The call site in `sdpa_attention_forward` passes `value` too. A comment documents the boundary: GQA-in-SDPA is kept only when the Flash kernel can portably serve it (equal head_dim ≤ 256); at head_dim > 256, or when key/value head_dim differ, no kernel except cuDNN supports the `enable_gqa` broadcast, so `repeat_kv` (MHA) is the correct fallback.

The fix lives in the shared dispatch helper rather than a model file because the enable_gqa decision happens entirely there and the model side has no hook to influence it; this is also a general SDPA kernel limitation, not Gemma4-specific. This PR does not touch any modular_*.py/modeling_*.py (the change is confined to the integrations layer).

## Reproduction (Gemma4-shaped tensors, fp32 on CPU / bf16 on CUDA)

```python
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.integrations.sdpa_attention import use_gqa_in_sdpa

[… 32 lines of code omitted]

## Before/after

* head_dim=512 (equal key/value): on `main` the dispatch decision is `True` (still takes the enable_gqa path); with this PR it is `False` (takes repeat_kv). Signature note: on `main` the helper is `use_gqa_in_sdpa(mask, key)`; this PR adds `value` → `use_gqa_in_sdpa(mask, key, value)`.
* MLA differing head_dim (both ≤ 256, e.g. 192/128): on `main` → `True` (only cuDNN can serve enable_gqa, so it silently drops to MATH where cuDNN is unavailable); with this PR → `False` (repeat_kv, served by the mem-efficient kernel). Verified on H20 / torch 2.11: `can_use_flash_attention` and `can_use_efficient_attention` are both False for 192/128.
* CUDA (needs a GPU): under the EFFICIENT_ATTENTION backend, enable_gqa=True at head_dim=512 is rejected by the kernel (the exact point where the buggy path silently falls back to MATH), while the repeat_kv (MHA) path succeeds — concretely showing that EA supports 512 but not enable_gqa.

## Added test

`tests/utils/test_modeling_utils.py::TestAttentionImplementation` adds three tests:

- `test_use_gqa_in_sdpa_skips_large_head_dim` — head_dim > 256 (checking both key and value) disables GQA-in-SDPA; equal head_dim ≤ 256 with no mask stays enabled; an explicit mask never enables it. (CPU)
- `test_use_gqa_in_sdpa_skips_mismatched_head_dim` — differing key/value head_dim ≤ 256 (e.g. 192/128) disables it too. (CPU)
- `test_gqa_in_sdpa_does_not_fall_back_to_math` (`@require_torch_gpu`) — forces the mem-efficient backend (excluding MATH) so a silent fallback becomes a hard error: the raw `enable_gqa=True` call raises, while `sdpa_attention_forward` (repeat_kv path) runs, for both head_dim=512 and 192/128.

The first two fail on main and pass with this fix. (The XPU branch is out of scope for this fix and is skipped.)

## Test commands and results (local, torch 2.12.1+cpu)

```bash
# Dispatch unit tests (CPU)
$ python -m pytest tests/utils/test_modeling_utils.py -k use_gqa_in_sdpa -q
2 passed

[… 21 lines of code omitted]

utils/tests_fetcher.py maps the change to ~477 model test files because it lives in the core shared file integrations/sdpa_attention.py; the runs above are a high-signal subset covering the changed path (head_dim=512), the unchanged path (head_dim ≤ 256 GQA), and the interface. ruff is clean.

## Scope of impact

* Models with equal key/value head_dim ≤ 256: behavior unchanged.
* GQA models with head_dim > 256 (e.g. Gemma4 head_dim=512): with mask=None they now take the repeat_kv (MHA) path, avoiding the silent MATH fallback and O(S²) OOM; numerical results are unchanged.
* MLA-style models with differing key/value head_dim (e.g. DeepSeek-V3 192/128), or with value head_dim > 256: also take repeat_kv instead of relying on cuDNN, which the mem-efficient kernel serves at differing head_dim; numerical results unchanged.
* Only the SDPA path is affected: eager / flash_attention / flex_attention / *_paged and the XPU dispatch are untouched.

## Code Agent Policy

* [✔ ] I confirm that this is not a pure code agent PR.

## Before submitting

* [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
* [✔ ] Did you read the contributor guideline (https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#start-contributing-pull-requests) and the Pull Request checks (https://huggingface.co/docs/transformers/pr_checks)?
* [ ] Was this discussed/approved via a Github issue or the forum? Please add a link to it if that's the case.
* [✔ ] Did you make sure to update the documentation with your changes according to the guidelines?
* [ ✔] Did you write any new necessary tests?

## Review request
 Would appreciate a review from @vasqu @Cyrilvallez @zucchini-nlp Thank you!

## [c3493205545] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/sdpa_attention.py:36 · 2026-06-29
```suggestion
    #   - head_dim <= 256 (otherwise also fallback to math kernel)
```

## [c3493209337] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/sdpa_attention.py:41 · 2026-06-29
Would we also have issues with different head dims between q and k/v?

## [c3493214849] vasqu (COLLABORATOR) · review_comment on tests/utils/test_modeling_utils.py:2927 · 2026-06-29
imo we can just use requires decorators to make sure we are on the correct torch version etc

## [c4593571952] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-29
It reminds me of #45776 😢 agree this one does make sense regardless as base FA does not support this

Just a few smaller comments first

## [c3498722191] Butterfingrz (CONTRIBUTOR) · review_comment on src/transformers/integrations/sdpa_attention.py:36 · reply to [c3493205545] · 2026-06-30
Thanks for the review! I’ve removed the redundant comments

## [c4600320556] Butterfingrz (CONTRIBUTOR) · review COMMENTED · 2026-06-30 (no text)

## [c3498725669] Butterfingrz (CONTRIBUTOR) · review_comment on tests/utils/test_modeling_utils.py:2927 · reply to [c3493214849] · 2026-06-30
Thanks for the guidance — updated accordingly.

## [c4600324853] Butterfingrz (CONTRIBUTOR) · review COMMENTED · 2026-06-30 (no text)

## [c3498927951] Butterfingrz (CONTRIBUTOR) · review_comment on src/transformers/integrations/sdpa_attention.py:41 · reply to [c3493209337] · 2026-06-30
Thanks for pointing this out.

`query.shape[-1] == key.shape[-1]` is required by SDPA, so checking `key` also covers `query`. However, `value.shape[-1]` can differ: MLA models pad `value` to the QK dim only on the flash path, not on SDPA. Therefore, the key-only guard could miss a large `value`.

No shipping model hits this case:

- **Real GQA broadcast (`num_key_value_groups > 1`)**: Gemma3/4 (`256`, global `512`), DeepSeek-V4 (`num_kv_heads=1` → `groups=64`, `head_dim=512`), and Llama/Qwen/Mistral all have `qk_head_dim == v_head_dim`, so `key.shape[-1]` is the true head dim and the `512` layers correctly fall back to `repeat_kv`.

- **Asymmetric `qk != v`**: MLA models such as DeepSeek-V2/V3, MiniCPM3, GLM-MoE-DSA, and GLM4-MoE-Lite all have `num_key_value_heads == num_attention_heads`, so `groups == 1`; `enable_gqa=True` is a no-op broadcast, and `True`/`False` give identical numerics.

So this is hardening against a latent footgun rather than fixing an active bug. I’ve added `value.shape[-1] <= 256` next to the key check.

## [c4600561178] Butterfingrz (CONTRIBUTOR) · review COMMENTED · 2026-06-30 (no text)

## [c4843895824] Butterfingrz (CONTRIBUTOR) · issue_comment · reply to [c4593571952] · 2026-06-30
> It reminds me of #45776 😢 agree this one does make sense regardless as base FA does not support this
> 
> Just a few smaller comments first

@vasqu  Thanks for your review , Gemma4's non-standard attention has been a real headache. I hit this on the AutoModel Gemma4 benchmark: when the attention path passes no mask (`attention_mask is None`), we opt into `enable_gqa=True`, and at `head_dim=512`, SDPA's memory-efficient (EA) kernel silently falls back to the `math` backend because it does not support the GQA broadcast there.

This showed up with a fully packed dataset: every sample was already at `max_length`, so there was no padding mask. That triggered an OOM I would not see in normal training, purely from the `O(S^2)` math fallback.

Hence the minimal change here to keep us off that path.

## [c3499460880] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/sdpa_attention.py:43 · 2026-06-30
`key.shape[-1] != value.shape[-1]` would also cause the math fallback?

## [c4601197455] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-30 (no text)

## [c3503896054] Butterfingrz (CONTRIBUTOR) · review_comment on src/transformers/integrations/sdpa_attention.py:43 · reply to [c3499460880] · 2026-07-01
Sorry for the late reply! I didn’t see a notification for this earlier.

Thanks so much for raising this — I hadn’t noticed this detail before, and it’s a great catch.

* **Scope:** the only transformers-side change is the `use_gqa_in_sdpa` guard, which just decides the K/V shape handed to `torch.nn.functional.scaled_dot_product_attention` (`enable_gqa=True` vs `repeat_kv`); kernel selection and any fallback are PyTorch's own SDPA backend dispatch, untouched by this PR.

* **Problem (differing k/v ****`head_dim`****, e.g. MLA):** Flash requires q/k/v to share the last dim and mem-efficient doesn't support the `enable_gqa` broadcast, so `enable_gqa=True` is served only by cuDNN — and silently drops to the math kernel (`O(S²)`) where cuDNN is unavailable.

* **Guard:** `key.shape[-1] == value.shape[-1] <= 256` enables GQA-in-SDPA only when Flash can portably serve it, otherwise routing to `repeat_kv`, which mem-efficient handles at differing `head_dim` without relying on cuDNN.

* **Verified (H200 / torch 2.11):** for MLA 192/128, `can_use_flash_attention` and `can_use_efficient_attention` are both False.

## [c4606558687] Butterfingrz (CONTRIBUTOR) · review COMMENTED · 2026-07-01 (no text)

## [c3505257251] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/sdpa_attention.py:36 · 2026-07-01
```suggestion
    #   - key head_dim == value head_dim <= 256 (otherwise it will fall back to the math kernel)
```

## [c3505260753] vasqu (COLLABORATOR) · review_comment on tests/utils/test_modeling_utils.py:2920 · 2026-07-01
We will drop torch 2.5 either way, let's not introduce it here

## [c4608188561] vasqu (COLLABORATOR) · review APPROVED · 2026-07-01
Nice tests in general just a few small nits 🤗

## [c3506201611] vasqu (COLLABORATOR) · review_comment on tests/utils/test_modeling_utils.py:2936 · 2026-07-01
Ah i just noticed that it might be better to wrap the sdpa call and force the memory efficient / fa backends. I.e. the context manager from sdpa

This way we know that it worked if we don't error out, wdyt?

## [c4609337485] vasqu (COLLABORATOR) · review COMMENTED · 2026-07-01 (no text)

## [c3506872641] Butterfingrz (CONTRIBUTOR) · review_comment on src/transformers/integrations/sdpa_attention.py:36 · reply to [c3505257251] · 2026-07-01
Thanks, I’ve updated the comment.

## [c4610157291] Butterfingrz (CONTRIBUTOR) · review COMMENTED · 2026-07-01 (no text)

## [c3506893197] Butterfingrz (CONTRIBUTOR) · review_comment on tests/utils/test_modeling_utils.py:2920 · reply to [c3505260753] · 2026-07-01
Thanks, I’ve removed the redundant PyTorch version check.

## [c4610181027] Butterfingrz (CONTRIBUTOR) · review COMMENTED · 2026-07-01 (no text)

## [c3506913821] Butterfingrz (CONTRIBUTOR) · review_comment on tests/utils/test_modeling_utils.py:2936 · reply to [c3506201611] · 2026-07-01
Thanks for the pointer, that's a much cleaner way to test it 🙏 !

I wrapped the call in `sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])` so the math backend is off the table — if it tries to fall back, torch raises `No available kernel` instead of silently degrading. The new `test_gqa_in_sdpa_does_not_fall_back_to_math` checks exactly that: the raw `enable_gqa=True` call errors out (the trap), while going through `sdpa_attention_forward` (the `repeat_kv` path) runs fine. No error = it really stayed off the math kernel, just like you said.

One thing I'd like your take on: the new test is GPU-only (skipped on the CPU CI) and only exercises the `mask=None` / `repeat_kv` path, so the `<= 256` check and the `mask is None` branch are still covered by the boolean test `test_use_gqa_in_sdpa_skips_large_head_dim` on CPU. I'm leaning towards keeping both (unit + integration) so that if someone reverts the `<= 256` guard, the CPU pipeline still goes red. What do you think — keep both, or would you rather just replace the old one with the new test?

## [c4610204908] Butterfingrz (CONTRIBUTOR) · review COMMENTED · 2026-07-01 (no text)

## [c3513113668] vasqu (COLLABORATOR) · review_comment on tests/utils/test_modeling_utils.py:2952 · 2026-07-02
I'd rather we expand on this test and remove the other tests so that we have the different edge cases here if that makes sense

## [c4617547214] vasqu (COLLABORATOR) · review APPROVED · 2026-07-02
One last comment but then we can merge imo

## [c3514085827] Butterfingrz (CONTRIBUTOR) · review_comment on tests/utils/test_modeling_utils.py:2952 · reply to [c3513113668] · 2026-07-02
Makes sense — thanks!

Done. I folded everything into `test_gqa_in_sdpa_does_not_fall_back_to_math` and removed the two standalone tests.

The single test now covers all edge cases in one place:

* `use_gqa_in_sdpa` boolean checks up front:

  * matched `head_dim <= 256` keeps GQA enabled
  * `head_dim > 256` disables it
  * mismatched key/value dimensions, e.g. MLA, disable it
  * an explicit mask disables it
* forced-backend behavior afterward:

  * with the mem-efficient backend forced, `enable_gqa=True` raises, while the `repeat_kv` path through `sdpa_attention_forward` does not

Verified on an H200 with `torch==2.11`: the test passes.

## [c4618780330] Butterfingrz (CONTRIBUTOR) · review COMMENTED · 2026-07-02 (no text)

## [c3514200824] vasqu (COLLABORATOR) · review_comment on tests/utils/test_modeling_utils.py:2944 · 2026-07-02
```
        small, large = torch.empty(1, 2, 4, 256), torch.empty(1, 2, 4, 512)
        self.assertTrue(use_gqa_in_sdpa(None, small, small))
        self.assertFalse(use_gqa_in_sdpa(None, large, large))
        self.assertFalse(use_gqa_in_sdpa(None, large, small))
        self.assertFalse(use_gqa_in_sdpa(None, small, large))
        self.assertFalse(use_gqa_in_sdpa(None, torch.empty(1, 2, 4, 192), torch.empty(1, 2, 4, 128)))  # MLA k/v differ
        self.assertFalse(use_gqa_in_sdpa(torch.empty(1, 1, 4, 4), small, small))  # a mask disables it
```
I meant more like we have a list of cases and the expected backend to be used for each. We can then parametrize the test for all cases
1. They should never raise an error
2. We just create the head dims based on the parametrized values

## [c4618912400] vasqu (COLLABORATOR) · review COMMENTED · 2026-07-02 (no text)

## [c3514533231] Butterfingrz (CONTRIBUTOR) · review_comment on tests/utils/test_modeling_utils.py:2944 · reply to [c3514200824] · 2026-07-02
Sorry about the rough test code style! I’ve cleaned it up now.

## [c4619290807] Butterfingrz (CONTRIBUTOR) · review COMMENTED · 2026-07-02 (no text)

## [c4619324294] vasqu (COLLABORATOR) · review APPROVED · 2026-07-02
Thanks, now it's clean 🫡 thanks for bearing with me

## [c4868471308] Butterfingrz (CONTRIBUTOR) · issue_comment · reply to [c4619324294] · 2026-07-02
@vasqu  Thanks for looking! The `tests_peft_integration [shard 0/1]` failure isn't from this PR — it's an incompatibility between current `main` and `PEFT 0.19.1`, the minimum pinned by `#46442`.

CI installs `peft==0.19.1`, whose `build_peft_weight_mapping` calls:

```python
WeightConverter(
    ...,
    distributed_operation=...,
    quantization_operation=...,
[… 23 lines of code omitted]

## [c4868482070] vasqu (COLLABORATOR) · issue_comment · 2026-07-02
Yep on it, not on you! Will merge as soon as I can

## [c4868486780] vasqu (COLLABORATOR) · issue_comment · 2026-07-02
#47027

## [c4868581271] Butterfingrz (CONTRIBUTOR) · issue_comment · 2026-07-02
Thank you so much!

## [c4874755902] albertvillanova (MEMBER) · issue_comment · reply to [c4868581271] · 2026-07-03
Hi @Butterfingrz and @vasqu,

Just wanted to mention that DeepseekV3Attention derives `num_key_value_groups` from `num_key_value_heads`, which MLA never honors, so any `kv_heads != attn_heads` config silently produces a broken model.

I don't know if you would consider to take this into account: maybe force `groups=1` or validate in `DeepseekV3Config.__post_init__`?

On the other hand, I think this PR is correct and shouldn't be touched.

## [c4875060391] vasqu (COLLABORATOR) · issue_comment · reply to [c4874755902] · 2026-07-03
Hey @albertvillanova, that would affect pretty much any model using MLA correct? `num_key_value_heads` is actually never used it seems but still derived the `num_key_value_groups`.

Imo, we should add something to `validate_architecture` that raises an error directly when the config has mismatching num heads vs num kv heads. Wdyt?

---
outcome: state=closed, merged=True
