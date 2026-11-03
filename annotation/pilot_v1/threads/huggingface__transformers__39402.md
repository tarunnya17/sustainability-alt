# #39402 [pull_request] No repeat kv
https://github.com/huggingface/transformers/pull/39402  ·  opened by Rocketknight1 on 2025-07-14

## [body] Rocketknight1 (MEMBER) · opening post · 2025-07-14
Draft PR for now, please don't merge!

The goal of this PR is to experiment with an `einsum` approach for `eager_attention_forward` that avoids a lot of reshapes and transposes that introduce intermediate tensors and unnecessary copies. In notebook benchmarks it seems to be 25-50% faster, but I'm making a proper PR so I can test more thoroughly.

## [c3070780829] vasqu (COLLABORATOR) · issue_comment · 2025-07-14
Fyi, I think #36218 also tried to not materialize the kv repetitions; you could take a look at https://github.com/mseeger/transformers/blob/b49651192f55ce86981c5851aaf9afc209ea3407/src/transformers/models/llama/modeling_llama.py#L138-L192 for reference / comparison w/o using einsum.

## [c3553374136] Rocketknight1 (MEMBER) · issue_comment · 2025-11-19
Admitting defeat on this one - on the latest version of `torch` the compiler can recognize our existing code as SDPA and automatically lowers it to `torch.ops.aten._scaled_dot_product_flash_attention.default`, which makes it much faster + more memory efficient. However, it can't recognize the version with `einsum`, and so we take a big penalty. Although the code in this PR is better in the eager case or when we need to return attentions, the `torch.compile()` case takes precedence!

---
outcome: state=closed, merged=False
