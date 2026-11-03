# #47731 [pull_request] [generate] Stop setting the static cache as an attribute to save memory
https://github.com/huggingface/transformers/pull/47731  ·  opened by Cyrilvallez on 2026-08-03

## [body] Cyrilvallez (MEMBER) · opening post · 2026-08-03
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47731)


# What does this PR do?

As per the title. Keeping the cache as an attribute `self._cache` inside `generate` is a very bad idea in general, because it will keep a big memory footprint without the user noticing, which can be very surprising and unwanted if we want to make 1 call to generate with static cache, and then use the model for something else.
It was previously done to avoid recompilation if we were making several calls to `generate` with several lengths, but simply keeping the old `max_cache_len` as attribute and re-using that later is enough to avoid recompilation.

The following snippet illustrates the issue:

```python
import torch
from transformers import AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-1B-instruct"
[… 37 lines of code omitted]

## [c3704854411] vasqu (COLLABORATOR) · review_comment on src/transformers/generation/utils.py:1858 · 2026-08-03
Do we have any test that checks recompilations in regards to that?

## [c3704870420] vasqu (COLLABORATOR) · review_comment on src/transformers/generation/utils.py:1869 · 2026-08-03
It does mean that we recreate a cache in any case, no? Would there be any issue when we use multiple iterations with a chunked prefill? Just a thought, unsure

## [c4845207942] vasqu (COLLABORATOR) · review COMMENTED · 2026-08-03
One question im a bit hesitant about because I dont follow the implications here 100% but other than that it's a nice optim 👀

## [c3708807671] Cyrilvallez (MEMBER) · review_comment on src/transformers/generation/utils.py:1869 · reply to [c3704870420] · 2026-08-04
Not sure I understand what you mean? Basically, now we create a new StaticCache every `generate` call vs keeping one attached to the model. To be honest, I have no idea why it was ever attached to the model... `compile` caches the graphs based on shapes and ops, not objects addresses, so we can just recreate the object

## [c4849808476] Cyrilvallez (MEMBER) · review COMMENTED · 2026-08-04 (no text)

## [c3708815265] Cyrilvallez (MEMBER) · review_comment on src/transformers/generation/utils.py:1858 · reply to [c3704854411] · 2026-08-04
We don't, but I confirmed it locally at least - will try to add one, but can be tricky to catch recompiles automatically

## [c4849816427] Cyrilvallez (MEMBER) · review COMMENTED · 2026-08-04 (no text)

## [c4853238375] vasqu (COLLABORATOR) · review APPROVED · 2026-08-04
Thanks, I meant that in some cases before we did not recreate the cache itself (nothing to do with the attached cache) but tbh it's weird to have the attached version so rather get that out

---
outcome: state=closed, merged=True
