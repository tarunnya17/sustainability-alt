# #45742 [pull_request] feat(llama): add has_weight parameter to LlamaRMSNorm for FlashNorm-folded checkpoints (+12.77% e2e on Llama-3.2-1B at bf16/A100)
https://github.com/huggingface/transformers/pull/45742  ·  opened by fm1320 on 2026-05-02

## [body] fm1320 (NONE) · opening post · 2026-05-02
# What does this PR do?

Adds an opt-in `has_weight: bool = True` parameter to `LlamaRMSNorm`, declares a `flashnorm_folded: bool = False` field on `LlamaConfig`, and plumbs the flag through `LlamaDecoderLayer` and `LlamaModel` so that FlashNorm-folded checkpoints (the [`weightless-rmsnorm`](https://huggingface.co/models?other=weightless-rmsnorm) collection on the Hub) automatically dispatch through PyTorch's existing weightless `F.rms_norm` C++ kernel on load. No user-side runtime patching needed.

## Motivation

[FlashNorm](https://arxiv.org/abs/2407.09577) (Graef et al., 2024) is an algebraic identity that folds the per-channel RMSNorm weight into the next linear layer's weight matrix. The transformation is exact and produces a checkpoint with all-ones norm weights. A kernel that supports `weight=None` can skip the per-channel multiply entirely.

PyTorch's `torch.nn.functional.rms_norm(x, shape, weight=None, eps)` C++ kernel already implements this branch (skip multiply when weight is None). This PR wires `LlamaRMSNorm` to dispatch through that kernel when `has_weight=False`.

End-to-end measurement on Colab A100, bf16, 256-token greedy decode, median of 20 trials with 3 warmup, eager attention, on `open-machine/Llama-3.2-1B-FlashNorm`:

| Variant | tok/s |
|---|---|
| Default `LlamaRMSNorm.forward` (manual variance + multiply by all-ones weight) | 45.00 |
| Runtime monkey-patch routing `forward` through `F.rms_norm(weight=None)` | 50.75 |

Delta: **+12.77% end-to-end**. Same checkpoint, same hardware, same decode protocol; only the kernel-dispatch path differs.

This PR ships the same routing as a first-class HF Transformers feature, automatic on FlashNorm-folded checkpoints.

## What this PR changes

| File | Change |
|---|---|
| `src/transformers/models/llama/modeling_llama.py` | Add `has_weight: bool = True` to `LlamaRMSNorm.__init__`. When `False`, register `weight` as a non-persistent buffer of ones (so `.to(device)` works) and route `forward` through `F.rms_norm(weight=None)`. Add `getattr(config, "flashnorm_folded", False)` plumbing to `LlamaDecoderLayer` and `LlamaModel` so the three RMSNorm call sites pass `has_weight=not flashnorm_folded`. |
| `src/transformers/models/llama/configuration_llama.py` | Declare `flashnorm_folded: bool = False` on `LlamaConfig` with a docstring describing the field's semantics. |
| `tests/models/llama/test_modeling_llama.py` | Add `LlamaFlashNormFoldedTest` with three unit tests: bit-equality between `LlamaRMSNorm(has_weight=False)` and `F.rms_norm(weight=None, ...)`; buffer-not-parameter for the weightless case; round-trip from `LlamaConfig(flashnorm_folded=True)` to a `LlamaModel` with all-norms weightless and no `*norm.weight` keys in state_dict. |

## Behavior

- **`has_weight=True` (default)**: unchanged. The existing manual variance + rsqrt + multiply path runs as before. No regression for existing checkpoints.
- **`has_weight=False`**: weight is a buffer (not a Parameter), not in `state_dict`, and the forward routes through PyTorch's weightless `F.rms_norm` C++ kernel.
- **Auto-detection from config**: existing Llama checkpoints that do not set `flashnorm_folded` in their `config.json` are unaffected (default `False`).

## Cross-architecture propagation

Several models in `transformers` reuse `LlamaDecoderLayer` and `LlamaModel` (Mistral, Yi, certain Codestral variants). Those derived classes now also honor `flashnorm_folded` if their config has it set. This is the intended behavior (any RMSNorm-based checkpoint that has been FlashNorm-folded gets the speedup automatically) but flagging it explicitly here for reviewer awareness in case any downstream architecture has special expectations on `LlamaRMSNorm`.

For architectures that do not inherit from these classes (Gemma, Qwen2/3, Phi3, etc.), an analogous one-liner per architecture's RMSNorm constructor would extend the feature. Happy to follow up with separate small PRs once this Llama-side change is reviewed.

## Interaction with `@use_kernel_forward_from_hub("RMSNorm")`

The decorator wraps `LlamaRMSNorm` to allow hub-loaded RMSNorm kernel implementations to replace the forward. Hub kernels are opt-in (typically via env var or model config). When a hub kernel is enabled, that kernel runs *instead* of our branched forward and may not honor `has_weight=False` unless the kernel author specifically supports it.

For correctness of the FlashNorm path: when running a FlashNorm-folded checkpoint, ensure either (a) no hub kernel is enabled for `RMSNorm`, or (b) the configured hub kernel respects `has_weight=False`. Documented in the docstring.

## Reproducibility

Notebook in the FlashNorm paper's companion repo: https://github.com/OpenMachine-ai/transformer-tricks/blob/main/notebooks/flashNorm_hf_a100.ipynb (~3 min on Colab A100; reproduces the +12.77%).

## Tests

Three unit tests added in `tests/models/llama/test_modeling_llama.py`. Run locally:

```bash
pytest tests/models/llama/test_modeling_llama.py::LlamaFlashNormFoldedTest -v
```

## Reference

- FlashNorm paper: https://arxiv.org/abs/2407.09577
- vLLM equivalent loader PR (same `has_weight=False` pattern, model-specific): https://github.com/vllm-project/vllm/pull/41431
- llama.cpp equivalent feature request: https://github.com/ggml-org/llama.cpp/issues/22486

## Checklist

- [x] Read the contributor guidelines.
- [x] Documentation: new parameter and config field have docstrings; cross-architecture propagation and decorator interaction documented in the PR body.
- [x] Tests added (`LlamaFlashNormFoldedTest`, three tests).
- [x] `ruff check` and `ruff format --check` clean on the modified files.
- [x] CLA: will sign on first PR comment from the bot.

## [c4363430959] Qodo-Free-For-OSS (NONE) · issue_comment · 2026-05-02
Hi, With `flashnorm_folded=True`, RMSNorm weights are registered as non-persistent buffers and therefore absent from `state_dict`, so checkpoints containing `*.norm.weight` will be reported as unexpected keys during `from_pretrained` load.

**Severity:** action required | **Category:** reliability

**How to fix:** Ignore norm.weight when folded

**Agent prompt to fix** - you can give this to your LLM of choice:

> ### Issue description
> When `flashnorm_folded=True`, `LlamaRMSNorm` intentionally has no persistent `weight` parameter. If a checkpoint still contains `*.norm.weight` tensors, loading reports them as `unexpected_keys`.
> 
> ### Issue Context
> Transformers loading uses `model.state_dict()` as the expected key set. Non-persistent buffers are excluded, so the checkpoint keys will never match.
> 
> ### Fix Focus Areas
> - Add Llama-specific ignore patterns for these keys when folded (e.g., `_keys_to_ignore_on_load_unexpected` or an override that conditionally filters `loading_info.unexpected_keys` based on `config.flashnorm_folded`).
> - Ensure it covers:
>   - `model.norm.weight`
>   - `layers.*.input_layernorm.weight`
>   - `layers.*.post_attention_layernorm.weight`
> 
> - src/transformers/models/llama/modeling_llama.py[52-108]
> - src/transformers/models/llama/modeling_llama.py[329-417]
> - src/transformers/modeling_utils.py[4719-4752]

We noticed a couple of other issues in this PR as well - happy to share if helpful.

---
*Found by [Qodo](https://github.com/marketplace/qodo-merge-pro-for-open-source). Free code review for open-source maintainers.*

## [c4363879674] fm1320 (NONE) · issue_comment · 2026-05-02
Thanks for the catch. Addressed in 4fdcb3917c by switching the `has_weight=False` weight buffer from `persistent=False` to `persistent=True`.

Reasoning: real-world flashnorm-folded HF checkpoints (the `weightless-rmsnorm` collection on the Hub) still carry the all-ones `*.norm.weight` tensors for HF compatibility, so the cleanest key-for-key match is to keep the all-ones in `state_dict` too. The buffer's all-ones value is loaded into the buffer's all-ones (no-op); save -> reload is symmetric. The buffer stays a buffer (not a Parameter, no gradient, not optimized), so `has_weight=False` semantics are preserved everywhere except `state_dict` membership.

Forward path is unchanged: the runtime still skips the per-channel multiply via `F.rms_norm(weight=None)`. The buffer's value is irrelevant at runtime because the weightless path never reads it. State_dict size grows by `hidden_size * 4` bytes per norm (~264 KB for Llama-3.2-1B, ~0.01% of the checkpoint).

Tests updated (now assert `weight` IS in `state_dict`, IS NOT a `Parameter`, IS in `named_buffers`, IS NOT in `named_parameters`).

## [c4379371615] Rocketknight1 (MEMBER) · issue_comment · 2026-05-05
Flashnorm is a clever idea but it adds a huge amount of code complexity to a central model (which is inherited by many others via the modular system) to support a very small model collection, and the speedups probably get less and less significant at larger model sizes. I'm not sure we want it, sorry!

## [c4379501758] fm1320 (NONE) · issue_comment · reply to [c4379371615] · 2026-05-05
> Flashnorm is a clever idea but it adds a huge amount of code complexity to a central model (which is inherited by many others via the modular system) to support a very small model collection, and the speedups probably get less and less significant at larger model sizes. I'm not sure we want it, sorry!

Hi @Rocketknight1, thank you for the reply and I appreciate the concerns.

1) Would a lighter version with less code work, for example editing LlamaRMSNorm.__init__ to detect at load time whether self.weight is all ones and if so, replace self.forward with the weightless closure ? 
2) If the savings prove to be significant enough with larger model sizes, do you feel like that would be enough to potentially revisit this discussion ?

## [c4380873752] Rocketknight1 (MEMBER) · issue_comment · 2026-05-05
Hey, I discussed this internally! There was general agreement that we don't want to modify the modeling code itself, **but** this might be a candidate for [fusion mapping](https://github.com/huggingface/transformers/blob/main/docs/source/en/fusion_mapping.md). That would let us keep the modeling code intact, but dynamically monkey patch the model a bit to get the faster path when users specify.

If you can get something working as a fusion transform, we might be interested! I closed this a little too hastily, so I'll reopen for now.

---
outcome: state=open, merged=False
