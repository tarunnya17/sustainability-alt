# #47597 [pull_request] Fix position misalignment when past_key_values is externally built
https://github.com/huggingface/transformers/pull/47597  ·  opened by Piggy343288 on 2026-07-28

## [body] Piggy343288 (NONE) · opening post · 2026-07-28
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47597)

## Problem

`generate()` accepts a user-built `past_key_values`, but `_prefill()` silently mishandles it when the `attention_mask` length differs from the `input_ids` length — exactly the case when prefill and decoding are done separately:

```python
# Step 1: prefill
out = model(**inputs)           # input_ids=[1, 1800], mask=[1, 1800]

# Step 2: compress / rebuild cache
[… 43 lines of code omitted]

## [c5100984734] Cyrilvallez (MEMBER) · issue_comment · 2026-07-28
Please see the comments/doc on cache continuation:
```
# When restarting from previous cache, the `input_ids` are either the FULL sequence, including previous inputs,
# or only the new tokens but in this case the attention_mask still contains the FULL sequence (because otherwise we may
# lose some early padding tokens information). So slice inputs according to that if needed
# When restarting from `inputs_embeds`, it's always the FULL sequence, and we always need to slice
```

---
outcome: state=closed, merged=False
