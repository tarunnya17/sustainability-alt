# #45215 [pull_request] [Qwen3_5]Remove unnecessary masked_fill_ in torch_chunk_gated_delta_rule attention computation: "attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)"
https://github.com/huggingface/transformers/pull/45215  ·  opened by ENg-122 on 2026-04-03

## [body] ENg-122 (NONE) · opening post · 2026-04-03
# What does this PR do?

Remove unnecessary masked_fill_(mask, 0) call in torch_chunk_gated_delta_rule.

The decay_mask computed earlier already encodes the causal/lower-triangular structure (upper-triangle values are zero), so masking the attention scores again with masked_fill_(mask, 0) is redundant and adds unnecessary overhead.

Before:
attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)

After:
attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i])

- [√] I confirm that this is not a pure code agent PR.

## Before submitting
- [×] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [√] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [×] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [√] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [×] Did you write any new necessary tests?


## Who can review?

@ArthurZucker @Cyrilvallez

## [c4075274126] Rocketknight1 (MEMBER) · review APPROVED · 2026-04-08
This looks right to me! There is a minor danger of NaNs, I guess, but I think it's fine - multiplication of float32 Q/K values in any remotely reasonable range should not yield NaN

---
outcome: state=closed, merged=True
