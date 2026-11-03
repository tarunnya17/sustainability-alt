# #45865 [issue] [Feature Request] Add lossy speculative decoding via static ensemble verification
https://github.com/huggingface/transformers/issues/45865  ·  opened by kasakh on 2026-05-10

## [body] kasakh (CONTRIBUTOR) · opening post · 2026-05-10
### Feature request

**Is your feature request related to a problem? Please describe.**

Standard speculative decoding (assisted generation) in Transformers is *lossless* — it guarantees the output distribution exactly matches the target model. While this is a strong guarantee, it comes at a cost: many plausible draft tokens are rejected because `p(x)/q(x) < 1`, even when those tokens would lead to correct outputs. This limits the practical speedup achievable with speculative decoding.

In our experiments across multiple model pairs (Llama, Qwen, Gemma families), we observe that wall-clock time decreases monotonically as acceptance rate increases. The rigid verification step is the primary bottleneck.

**Describe the solution you would like**

Add an optional `assistant_ensemble_weight` parameter to `GenerationConfig` that enables **static ensemble verification** — a training-free, single-parameter extension that trades a controllable amount of distributional bias for higher acceptance rates.

The verification distribution becomes a weighted mixture:

```
v(x) = w * p_target(x) + (1 - w) * q_draft(x)
```

A draft token is accepted with probability `min(1, v(x) / q(x))`, and on rejection, we resample from the corresponding fallback distribution.

**Proposed API:**

```python
outputs = model.generate(
    **inputs,
    assistant_model=assistant_model,
    assistant_ensemble_weight=0.7,  # float in (0, 1], default=1.0 (lossless)
)
```

**Key properties:**
- `w=1.0` recovers standard lossless speculative decoding (backward compatible)
- `w<1.0` increases acceptance probability from `1 - TV(q,p)` to `1 - w*TV(q,p)` (Lemma 1 in our paper)
- The method is **Pareto-optimal**: it achieves the best possible tradeoff between acceptance rate and distributional bias (Proposition 1)
- No training required, no extra model weights, just one scalar parameter

**Describe alternatives you have considered**

1. Training a draft model to better match the target (e.g., online distillation) — requires expensive training
2. Dynamic ensemble with learned context-dependent weights (our full DIVERSED method) — requires training an ensemble head, too invasive for a first contribution
3. Simply increasing `num_assistant_tokens` — does not address the fundamental acceptance rate limitation

**Empirical results (from our paper):**

On CNN/DailyMail with Llama-3.1-8B-Instruct (target) + Llama-3.2-1B-Instruct (draft), temperature=0:
- Standard SD (w=1.0): ~65% acceptance rate
- Static ensemble (w=0.7): ~78% acceptance rate, with ROUGE-L within 0.5 points of the target model

Similar improvements observed across GSM8K, XSum, WMT, HumanEval, and MBPP benchmarks.

**Implementation scope:**

The change is minimal (~50-100 lines of logic):
1. Add `assistant_ensemble_weight` to `GenerationConfig` in `configuration_utils.py`
2. Modify the verification step in `utils.py` to blend distributions when `w < 1.0`
3. Tests and documentation

**References:**
- Paper: [DIVERSED: Relaxed Speculative Decoding via Dynamic Ensemble Verification](https://arxiv.org/abs/2604.07622) (AISTATS 2026)
- Code: https://github.com/comeusr/diversed
- The static ensemble method is described in Section 3.1 of the paper

I am happy to implement this and submit a PR if there is interest from the maintainers. I am one of the paper authors.

## [c4421470190] Rocketknight1 (MEMBER) · issue_comment · 2026-05-11
cc @cyrilvallez for generation and @remi-or @ArthurZucker @McPatate for CB

## [c4436909551] Cyrilvallez (MEMBER) · issue_comment · reply to [body] · 2026-05-13
Hey @kasakh! It's indeed an interesting approach, to be fair I've wondered for quite some time why the drafting of assistant tokens was not more relaxed!
Feel free to submit a PR for it, it should be relatively straightforward. You can tweak the `_speculative_sampling` function [here](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L3844) to relax the drafting. This should be a good starting point

## [c4466051560] kasakh (CONTRIBUTOR) · issue_comment · reply to [c4436909551] · 2026-05-16
Thanks @Cyrilvallez for the pointer! I have submitted a PR implementing this: #45979

It modifies `_speculative_sampling` as you suggested, adds greedy support via `argmax(v)`, includes 8 fast synthetic tests, and a docs section. CI is green. Happy to iterate on any feedback.

## [c4744650102] kasakh (CONTRIBUTOR) · issue_comment · 2026-06-18
Hi — this was auto-closed by the stale bot, but the work is actively tracked in PR #45979 (CI green, all reviewer feedback addressed). Could the issue be reopened so the context stays linked? Thanks!

---
outcome: state=closed
