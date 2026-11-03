# #47571 [pull_request] Vectorize NoRepeatNGramLogitsProcessor and remove its host sync
https://github.com/huggingface/transformers/pull/47571  ·  opened by hameedibrh on 2026-07-27

## [body] hameedibrh (CONTRIBUTOR) · opening post · 2026-07-27
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47571)


# What does this PR do?

`NoRepeatNGramLogitsProcessor` rebuilds a dict of **every** ngram in the sequence on **every** decoding step, then throws it away. Only one prefix is ever looked up. It also calls `.tolist()` to build that dict, which is a device to host sync on every step.

This looks up that one prefix directly, with tensor ops only.

### How

Only ngrams whose first `ngram_size - 1` tokens match the current suffix can be completed by the next token. So `unfold` gives every window of `ngram_size` tokens, all windows are compared against that prefix at once, and a matching window bans its own last token.

The ban mask gets one column past the vocab, and this bit is load bearing. `scatter_` is last write wins, so without somewhere to discard them, a non-matching window ending in an already banned token silently unbans it. There's a test for exactly that.

## Device sync and torch.compile

Removing the per step `.tolist()` is the main win. It stalls the pipeline on GPU whether or not anything is compiled.

`torch._dynamo.explain` on the processor also goes from 2 graph breaks / 3 graphs to **0 breaks / 1 graph**. Being precise about what that buys, since it's easy to overstate: `generate` compiles `model_forward` only (`get_compiled_call`), and logits processors run in eager outside that region, so this doesn't by itself change the built in compiled path. It matters if you compile the sampling loop yourself, and it removes a reason the processor couldn't be pulled into a graph later.

`cur_len` grows every step, but automatic dynamic shapes absorb that: 20 successive lengths give 2 unique graphs by default, 1 with `dynamic=True`, not one per length. `cache_implementation="static"` produces output ids identical to eager.

## Speed

CPU, single thread, vocab 128k, `ngram_size=3`, median of 11 rounds, ms per call:

| batch x length | before | after |
|---|---|---|
| 1 x 512 | 0.15 | **0.17** |
| 1 x 4096 | 1.45 | **0.21** |
| 8 x 2048 | 30.90 | **1.42** |
| 32 x 1024 | 60.08 | **6.05** |
| 4 x 16384 | 120.99 | **1.26** |

The old timings are also very noisy, presumably GC on all those short lived dicts. At 1 x 4096 it ranges 1.32 to 12.82 across rounds where the new one stays in 0.21 to 0.36, so tail latency improves more than the medians show.

Below ~640 tokens the old version is very slightly ahead, by hundredths of a millisecond, since there's now a fixed tensor op cost instead of a tiny Python loop. Memory is a transient `batch x windows x (ngram_size - 1)` bool tensor, about 40MB at batch 64 / length 32k / `ngram_size=20`.

## Correctness

Diffed against the old implementation, tensors compared exactly: 3000 randomized cases (batch 1-4, length 1-12, `ngram_size` 1-6) and a 1500 case sweep (vocab 2-64, `ngram_size` 1-10, length 1-257, batch 1/3/8), **0 mismatches**. Tiny vocabs are deliberate, they force the token collisions the spare column exists to handle. End to end `generate()` greedy and 3 beams at `no_repeat_ngram_size` 2/3/5 gives **identical output ids** in all 6 combinations.

That covers `ngram_size=1` and the `cur_len == ngram_size - 1` boundary, where the early return condition shifted from `cur_len + 1 < ngram_size` to `cur_len < ngram_size`. Same result, different route.

## Tests

Two new cases in `tests/generation/test_logits_process.py`, both for invariants nothing covered before. I checked each actually fails when you break what it guards:

| test | fails when |
|---|---|
| `..._banned_token_reappears` | spare mask column removed, bans `[]` instead of `[5]` |
| `..._sequence_shorter_than_ngram` | early return removed, `RuntimeError` from `unfold` |

Full file: 46 passed.

## Also

* `_calc_banned_ngram_tokens` is unused after this so it's removed. Nothing else references it.
* `EncoderNoRepeatNGramLogitsProcessor` is untouched, it builds its ngrams once in `__init__` rather than per step.
* The early return passes `scores` through instead of cloning, matching `WatermarkLogitsProcessor` and others. Nothing aliases badly: `LogitsProcessorList` reassigns, `generate` already passes a copy, and no 2D processor writes to `scores` in place. Happy to clone if you'd rather keep strict parity.

## Before submitting
- [x] Did you read the [contributor guideline](https://huggingface.co/docs/transformers/contributing) and the [Pull Request](https://huggingface.co/docs/transformers/pr_checks) checks?
- [x] Did you write any new necessary [tests](https://huggingface.co/docs/transformers/testing)?

## Acknowledgement
I came up with the logic myself but want to acknowledge that I used AI Agents to optimize it, write test cases and draft this PR.

## Who can review?

@Cyrilvallez @zucchini-nlp @molbap

## [c4808462903] Rocketknight1 (MEMBER) · review APPROVED · 2026-07-29
Sorry for the delay, but this is a nice vectorized optimization and I'm happy to accept it!

---
outcome: state=closed, merged=True
