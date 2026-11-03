# #42329 [pull_request] [docs] Attention backends + continuous batching
https://github.com/huggingface/transformers/pull/42329  ·  opened by stevhliu on 2025-11-21

## [body] stevhliu (MEMBER) · opening post · 2025-11-21
Refactors the optimization docs, starting with the attention backend and continuous batching docs (more updates to come in subsequent PRs!).

- reorg the `toctree` to create a clearer inference optimization section for inference (adds the `AttentionInterface`, continuous batching, and kernels guides)
- rename the "LLMs" section into a more generic "Generate API" section especially since we have VLMs now
- add `ContinuousMixin` and `ContinuousBatchingManager` to the API reference
- polish the Attention Interface doc (table describing the available backends, using kernels, etc.) - maybe you can review this section @Cyrilvallez 🙏
- polish the Continuous batching doc (break up the `ContinuousBatchingManager` example to better show the different options for using it, adds clearer PagedAttention and sliding window attention sections, high-level description of how the mixin, manager, and scheduler work together, move the monitoring section to the `transformers serve` docs) - maybe you can review this section @McPatate 🙏

## [c2556786648] McPatate (MEMBER) · review_comment on docs/source/en/continuous_batching.md:23 · 2025-11-24
```suggestion
## `generate_batch`
```
?

## [c2556803449] McPatate (MEMBER) · review_comment on docs/source/en/continuous_batching.md:149 · 2025-11-24
```suggestion
config.sliding_window = 4096
```

## [c2556806666] McPatate (MEMBER) · review_comment on docs/source/en/continuous_batching.md:168 · 2025-11-24
> prioritizing decoding requests over prefilling requests

depends on the scheduler choice. maybe worth adding a "by default," here

## [c2556809769] McPatate (MEMBER) · review_comment on docs/source/en/continuous_batching.md:65 · 2025-11-24
Are you removing the backticks for consistency across docs?

## [c3501043962] McPatate (MEMBER) · review COMMENTED · 2025-11-24
thank you for reworking this!

## [c2558077681] stevhliu (MEMBER) · review_comment on docs/source/en/continuous_batching.md:65 · reply to [c2556809769] · 2025-11-25
yeah, section titles don't render in code format on the docs site so i figured i might as well remove it

## [c3502682962] stevhliu (MEMBER) · review COMMENTED · 2025-11-25 (no text)

## [c2562271120] McPatate (MEMBER) · review_comment on docs/source/en/continuous_batching.md:65 · reply to [c2556809769] · 2025-11-26
😢

## [c3507716271] McPatate (MEMBER) · review COMMENTED · 2025-11-26 (no text)

## [c2592908092] Cyrilvallez (MEMBER) · review_comment on docs/source/en/attention_interface.md:18 · 2025-12-05
A bit misleading IMO: all attentions are equivalent in what they compute, it's just how the computation is performed that changes!

## [c3545042250] Cyrilvallez (MEMBER) · review APPROVED · 2025-12-05
Just checked the attention function parts as suggested! Looks very nice, just added a single comment! 🤗

---
outcome: state=closed, merged=True
