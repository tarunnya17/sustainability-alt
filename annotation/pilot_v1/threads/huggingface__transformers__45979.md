# #45979 [pull_request] [Generation] Add static ensemble verification for lossy speculative decoding
https://github.com/huggingface/transformers/pull/45979  ·  opened by kasakh on 2026-05-14

## [body] kasakh (CONTRIBUTOR) · opening post · 2026-05-14
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=45979)

## What does this PR do?

Adds support for **static ensemble verification** in speculative decoding (assisted generation), a training-free method that increases draft token acceptance rates by relaxing the verification distribution.

Fixes #45865

### Motivation

Standard speculative decoding rejects many plausible tokens because it strictly enforces that the output distribution matches the target model. Static ensemble verification blends the target and draft distributions during verification:

```
v(x) = w * p_target(x) + (1 - w) * q_draft(x)
```

This provably achieves the Pareto-optimal tradeoff between acceptance rate and distributional bias (Proposition 1 in the paper). The acceptance probability increases from `1 - TV(q,p)` to `1 - w*TV(q,p)`.

### Changes

- Adds `assistant_ensemble_weight` parameter to `GenerationConfig` (float in (0, 1], default `None` = lossless)
- Modifies `_speculative_sampling` to use ensemble acceptance ratio: `v(x)/q(x) = 1 - w + w*(p(x)/q(x))`
- Supports greedy decoding (`do_sample=False`) by comparing against `argmax(v)` instead of `argmax(p)`
- Raises clear error when used with candidate generators that do not return logits (e.g., prompt lookup)
- Adds numerical stability guard for the fallback distribution
- Adds 8 fast synthetic unit tests
- Adds documentation section to `assisted_decoding.md`

### Usage

```python
# Sampling mode
outputs = model.generate(
    **inputs,
    assistant_model=assistant_model,
[… 11 lines of code omitted]

### Design decisions

1. **Fallback distribution unchanged**: Since `[v-q]+ = w*[p-q]+` normalizes to the same distribution as `[p-q]+`, we compute the standard fallback directly for numerical stability.
2. **Acceptance ratio form**: Uses `1 - w + w*(p_i/q_i)` which is algebraically equivalent to `v_i/q_i` but avoids constructing the full ensemble distribution.
3. **Bonus token**: When all candidates are accepted, the bonus token uses `p` (target) rather than `v`, since `q` at position N+1 is unavailable. This is conservative and documented.
4. **Greedy ensemble**: For `do_sample=False`, compares draft tokens against `argmax(v)` — "greedy under the ensemble verifier."

### Tests

8 fast synthetic tests (all pass, ~0.15s total):
- `w=None`/`w=1.0` equivalence (backward compatibility)
- Deterministic acceptance ratio (w=0.7 accepts where w=1.0 rejects)
- Fallback distribution finite and normalized
- Numerical stability with near-zero residual
- Greedy ensemble: `argmax(v)` accepts where `argmax(p)` rejects
- ValueError for missing candidate logits
- GenerationConfig round-trip serialization
- Default value is None

### Results (from the paper)

On CNN/DailyMail with Llama-3.1-8B-Instruct (target) + Llama-3.2-1B-Instruct (draft), temperature=0:
- Standard SD (w=1.0): ~65% acceptance rate
- Static ensemble (w=0.7): ~78% acceptance rate, ROUGE-L within 0.5pt of target

### References

- Paper: [DIVERSED: Relaxed Speculative Decoding via Dynamic Ensemble Verification](https://arxiv.org/abs/2604.07622) (AISTATS 2026)
- Code: https://github.com/comeusr/diversed
- Section 3.1 describes the static ensemble method

cc @Cyrilvallez

## [c3255867148] stevhliu (MEMBER) · review_comment on docs/source/en/assisted_decoding.md:165 · 2026-05-18
would probably be nicer structurally to nest this as `###` under `## Speculative decoding` as this isn't exactly a separate method but a tweak to speculative decoding

## [c3255875437] stevhliu (MEMBER) · review_comment on docs/source/en/assisted_decoding.md:171 · 2026-05-18
this depth is probably not necessary for the docs. users who're really curious can look at the implementation

## [c3255877754] stevhliu (MEMBER) · review_comment on docs/source/en/assisted_decoding.md:169 · 2026-05-18
maybe also show an example of greedy decoding, but lead with the sampling example first. you can use the `<hfoptions>` tags around (see surrounding text for an example) to show both greedy and sampling

## [c3255880712] stevhliu (MEMBER) · review_comment on docs/source/en/assisted_decoding.md:188 · 2026-05-18
better to have this up front in the first paragraph. maybe something like:

```
Standard speculative decoding is *lossless* — it guarantees the output distribution matches the target model exactly. [Static ensemble verification](https://huggingface.co/papers/2604.07622) relaxes this by verifying against a mixture of the target and draft distributions.
```

## [c3255886681] stevhliu (MEMBER) · review_comment on docs/source/en/assisted_decoding.md:173 · 2026-05-18
may be nice to have a recommended value for `assistant_ensemble_weight`

## [c3255892983] stevhliu (MEMBER) · review_comment on src/transformers/generation/configuration_utils.py:355 · 2026-05-18
could simplify the docstring a bit as well

```suggestion
    Enables static ensemble verification in speculative decoding. If set to a value in `(0.0, 1.0)`,
    the verifier accepts tokens against the mixture `w * p_target + (1 - w) * q_draft` instead of
    `p_target`, trading a controlled distributional bias for a higher acceptance rate. `None` or `1.0`
    keeps decoding lossless. Requires the assistant model to return logits, so it is not compatible
    with prompt lookup decoding.
```

## [c4306701160] stevhliu (MEMBER) · review COMMENTED · 2026-05-18
thanks for the docs!

## [c4473624394] zucchini-nlp (MEMBER) · issue_comment · 2026-05-18
Looks like AI code slop

In any case, I am quite reluctant to accept new feats in generation unless it is requested by community, so I will close this issue. If you are a real human, I recommend to host this as remote code via `custom generation strategies` (https://huggingface.co/docs/transformers/en/generation_strategies#custom-generation-methods)

That way we can see how much usage it has in the community, and consider shipping in core library

## [c4473682011] kasakh (CONTRIBUTOR) · issue_comment · reply to [c4473624394] · 2026-05-18
Hi @zucchini-nlp, I am a real human — I am one of the first authors (Siva Rajesh Kasa) of the DIVERSED paper (AISTATS 2026, https://arxiv.org/abs/2604.07622). I used AI tools to assist with drafting, which is permitted per the contributing guide, but I designed the approach and understand every line.

For context on how this PR came about: I opened issue #45865  first, @Rocketknight1 cc'd the generation team (https://github.com/huggingface/transformers/issues/45865#issuecomment-4421470190), and @Cyrilvallez explicitly encouraged me to submit a PR and pointed to the exact function to modify (https://github.com/huggingface/transformers/issues/45865#issuecomment-4436909551). @stevhliu also reviewed the docs here and left constructive inline feedback. I followed the process as intended.

Regarding community demand: speculative decoding is one of the most actively used features in transformers for inference acceleration, and lossy/relaxed verification has been a known gap. The feature request received positive engagement from multiple HF team members before this PR was submitted.

Would you be open to reconsidering, or could @Cyrilvallez weigh in given he approved the feature? I am happy to address any technical concerns or incorporate @stevhliu's docs suggestions.

## [c4473728458] zucchini-nlp (MEMBER) · issue_comment · 2026-05-18
Ah oke, I didn't see the context here, sorry. The PR looked like a AI agent one, so I closed it

I will re-open and lets wait for Cyril. TBH the idea is quite similar to the lenience factor from the very first paper from Leviathan and I do agree it might be of use for some ppl. My concern is only to keep the code clean, unless requested by several users, As you might know, maintaining cost grows with every new feature, and the field of speculative decoding currently has much more happening which we don't yet shipped in transformers. Anyway, yep, let's wait for more input from Cyril :)

## [c4474385754] kasakh (CONTRIBUTOR) · issue_comment · reply to [c4306701160] · 2026-05-18
Updated the PR to address @stevhliu's docs feedback (simplified docstring, nested heading, added recommended value, inlined paper reference) and fixed a bug where the error guard could fire incorrectly when candidate_logits is temporarily unavailable mid-generation. The ValueError now only triggers at config time for incompatible generators (prompt lookup).

Tested end-to-end on 6 model pairs across 4 families (Qwen, SmolLM, OPT, Pythia) — all show expected speedups with w=0.7.

## [c4744558385] kasakh (CONTRIBUTOR) · issue_comment · reply to [c4473728458] · 2026-06-18
Hi @Cyrilvallez — apologies for the bump. Whenever you have the bandwidth, would you mind taking a look at this PR? CI is green, and @stevhliu's docs feedback along with the candidate_logits bug fix from the @zucchini-nlp thread have all been addressed.

Also cc'ing @Rocketknight1 and @ArthurZucker who were originally tagged on the linked issue (#45865), in case any of you can point me to a more appropriate reviewer.

Thanks for all the work you do maintaining this library — happy to make any further changes whenever you get a chance to review.

## [c3490039503] Cyrilvallez (MEMBER) · review_comment on src/transformers/generation/utils.py:987 · 2026-06-29
IIUC, only values < 1 are allowed, and any non-None value (i.e. user-set) should raise here

```suggestion
            if generation_config.assistant_ensemble_weight is not None:
                raise ValueError(
                    "Setting `assistant_ensemble_weight` requires candidate logits from the assistant model. "
                    "It is not supported with prompt lookup decoding."
                )
```

## [c3490044696] Cyrilvallez (MEMBER) · review_comment on src/transformers/generation/configuration_utils.py:461 · 2026-06-29
IIUC, the value can only be < 1. Let's make sure it's the case in `validate`

## [c3490049004] Cyrilvallez (MEMBER) · review_comment on src/transformers/generation/utils.py:3661 · 2026-06-29
See comment above, the < 1 part should already be enforced in the `validate` of the config

```suggestion
                    if assistant_ensemble_weight is not None and candidate_logits is not None:
```

## [c3490051487] Cyrilvallez (MEMBER) · review_comment on src/transformers/generation/utils.py:3886 · 2026-06-29
nit 
```suggestion
    assistant_ensemble_weight: float | None = None,
```

## [c3490054289] Cyrilvallez (MEMBER) · review_comment on src/transformers/generation/utils.py:3907 · 2026-06-29
same as above
```suggestion
    # Compute acceptance ratio. With ensemble weight w < 1, use v(x)/q(x) = 1 - w + w*(p(x)/q(x))
    if assistant_ensemble_weight is not None:
```

## [c3490059304] Cyrilvallez (MEMBER) · review_comment on tests/generation/test_static_ensemble.py:9 · 2026-06-29
No new test file please, we already have generation test files

## [c4589795588] Cyrilvallez (MEMBER) · review COMMENTED · 2026-06-29
Hey, sorry for the delay! Here are a few remarks!

## [c4835803491] kasakh (CONTRIBUTOR) · issue_comment · reply to [c4589795588] · 2026-06-29
Thanks for the review @Cyrilvallez! Addressed everything in 67ec46dfbf:

1. **Strict `0 < w < 1` validation moved into `GenerationConfig.validate()`** — `None` is the only lossless sentinel now; the boundary values `0.0` and `1.0` raise `ValueError` at config time. Updated the docstring to drop the "or `1.0` keeps decoding lossless" wording.
2. **Simplified runtime guards** — `_speculative_sampling` and the greedy verification path now just check `assistant_ensemble_weight is not None` (and `candidate_logits is not None` where relevant) since validate already enforces the range.
3. **Prompt-lookup guard now raises for any non-None weight** with the exact message you suggested.
4. **`float | None = None` annotation** on `_speculative_sampling`.
5. **Tests folded into existing generation test files** — `tests/generation/test_static_ensemble.py` is deleted. Four ensemble-sampling tests joined `UtilsFunctionsTest` in `tests/generation/test_utils.py` next to the existing `test_speculative_sampling*` methods, and two config-level tests (validate-rejects-invalid + default/round-trip) went into `tests/generation/test_configuration_utils.py`.

CI will rerun. Let me know if anything else needs tweaking.

## [c3504711012] Cyrilvallez (MEMBER) · review_comment on src/transformers/generation/utils.py:3929 · 2026-07-01
Let's make sure this is fully isolated to the new path
```suggestion
            if assistant_ensemble_weight is not None and p_prime_sum <= torch.finfo(p_prime.dtype).tiny:
```

## [c3504717356] Cyrilvallez (MEMBER) · review_comment on src/transformers/generation/utils.py:3902 · 2026-07-01
Let's not use "useless" single letter variables if we can avoid them
```suggestion
        probability_ratio = 1.0 - assistant_ensemble_weight + assistant_ensemble_weight * (p_i / q_i)
```

## [c3504721670] Cyrilvallez (MEMBER) · review_comment on src/transformers/generation/utils.py:3661 · 2026-07-01
Same here
```suggestion
                        p_probs = new_logits[:, :candidate_length, :].softmax(dim=-1)
                        q_probs = candidate_logits.softmax(dim=-1)
                        nu_probs = assistant_ensemble_weight * p_probs + (1.0 - assistant_ensemble_weight) * q_probs
                        # For the bonus token position (candidate_length), use target distribution
                        bonus_logits = new_logits[:, candidate_length:, :]
                        selected_tokens = torch.cat([nu_probs.argmax(dim=-1), bonus_logits.argmax(dim=-1)], dim=-1)
```

## [c4607530846] Cyrilvallez (MEMBER) · review COMMENTED · 2026-07-01
Almost ready! Just a few nits

## [c4860568170] kasakh (CONTRIBUTOR) · issue_comment · reply to [c4607530846] · 2026-07-01
Thanks for the fast turnaround on the second pass, @Cyrilvallez! All three suggestions were spot-on — the code reads much more cleanly now.

All 3 suggestions addressed in `6c41d1fcb4`:

1. **Numerical-stability guard scoped to the ensemble path** — `if assistant_ensemble_weight is not None and p_prime_sum <= torch.finfo(p_prime.dtype).tiny:`. Standard (lossless) SD keeps its original fallback behavior; the guard only fires on the new path when `p ≈ q` and the residual is numerically zero.
2. **Dropped `w = assistant_ensemble_weight` in `_speculative_sampling`** — the parameter name is used directly in the ratio expression.
3. **Same in the greedy verification path** — `nu_probs = assistant_ensemble_weight * p_probs + (1.0 - assistant_ensemble_weight) * q_probs`.

CI running now. Thanks again for the review time — more than happy to keep iterating if there's anything else.

## [c4651461022] Cyrilvallez (MEMBER) · review APPROVED · 2026-07-08
ALright, LGTM! Thanks!

---
outcome: state=closed, merged=True
