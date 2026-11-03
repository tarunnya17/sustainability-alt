# #47235 [pull_request] Fix double-shift training loss in CohereAsrForConditionalGeneration
https://github.com/huggingface/transformers/pull/47235  ·  opened by sridhar-3009 on 2026-07-10

## [body] sridhar-3009 (NONE) · opening post · 2026-07-10
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47235)

Fixes #46894

## Problem

`CohereAsrForConditionalGeneration.forward()` has a double-shift bug that corrupts training loss:

1. When `labels` are provided, `shift_tokens_right()` is called to produce `decoder_input_ids` (line 483 of `modular_cohere_asr.py`).
2. The model runs with these shifted inputs and produces `logits`.
3. `self.loss_function()` is then called — but this maps to `ForCausalLMLoss`, which **shifts labels again** internally (`labels[..., 1:]`) in `loss/loss_utils.py`.

The net effect is training against `labels[..., 1:]` instead of `labels`, causing silent gradient corruption and divergent training loss.

This is the same bug that was fixed in Moonshine in #46784 (commit d8c235494e). `CohereAsr` inherits from Moonshine and overrides `forward`, but the fix was not carried over.

## Fix

Replace `self.loss_function(logits=logits, labels=labels, ...)` with a direct `CrossEntropyLoss()` call — identical to the Moonshine fix:

```python
loss_fct = CrossEntropyLoss()
loss = loss_fct(logits.reshape(-1, self.config.vocab_size), labels.reshape(-1))
```

Applied to both `modular_cohere_asr.py` (source) and `modeling_cohere_asr.py` (generated).

---
outcome: state=closed, merged=False
