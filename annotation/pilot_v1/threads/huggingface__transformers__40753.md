# #40753 [pull_request] Fix order of mask functions when using `and/or_mask_function`
https://github.com/huggingface/transformers/pull/40753  ·  opened by Cyrilvallez on 2025-09-08

## [body] Cyrilvallez (MEMBER) · opening post · 2025-09-08
# What does this PR do?

If using `and/or_mask_function` arguments in `create_causal_mask`, it is important to apply those masks before any other deviations such as packed sequence, padding etc, as users expect to slightly deviate from the pattern, but not to take such internals into account!

Also take the opportunity to use `getattr` instead of direct object reference for the `is_compileable` flag of the Cache, so that non-standard caches don't need to have the class attribute

Merging as I own the masking code anyway,  but cc @ArthurZucker for viz!
Also cc @vasqu for viz, it's what I said the other day when talking about the Embedding Gemma integration!

---
outcome: state=closed, merged=True
