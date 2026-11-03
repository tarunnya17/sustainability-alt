# #44651 [pull_request] Fix Qwen3.5 num_labels not propagated to text_config
https://github.com/huggingface/transformers/pull/44651  ·  opened by gambletan on 2026-03-13

## [body] gambletan (NONE) · opening post · 2026-03-13
## Summary

Fixes #44625

When passing `num_labels` to `AutoConfig.from_pretrained` for Qwen3.5, the value is set on the outer `Qwen3_5Config` but not propagated to `text_config`. This causes `AutoModelForSequenceClassification` to use the default `num_labels=2` from `text_config` instead of the user-specified value.

## Fix

After the parent `__init__` completes (which sets `num_labels` on the outer config and creates `text_config` with defaults), propagate `num_labels` to `text_config` when they differ. This is done in both:
- `modular_qwen3_5.py` (source of truth)
- `configuration_qwen3_5.py` (auto-generated)

## Reproduction

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    "onnx-internal-testing/tiny-random-Qwen3_5ForConditionalGeneration",
    num_labels=1,
)
print(config.num_labels)              # 1
print(config.text_config.num_labels)  # Before fix: 2, After fix: 1
```

🤖 Generated with [Claude Code](https://claude.com/claude-code)

## [c4055115924] Rocketknight1 (MEMBER) · issue_comment · 2026-03-13
cc @zucchini-nlp, bug seems real but not sure if you think this is the right way to do it

## [c4055134139] zucchini-nlp (MEMBER) · issue_comment · 2026-03-13
Not really the way we want. We don't want to fix under a single model class as such. Closing as have the fix locally, opening a PR already

---
outcome: state=closed, merged=False
