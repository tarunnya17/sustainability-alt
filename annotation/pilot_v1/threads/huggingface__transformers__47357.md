# #47357 [pull_request] Add FSDP CI and end-to-end FSDP tests + save fsdp
https://github.com/huggingface/transformers/pull/47357  ·  opened by 3outeille on 2026-07-16

## [body] 3outeille (MEMBER) · opening post · 2026-07-16
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47357)


## Summary

Part of FSDP orchestration stack (2/2). Replaces #46990. #47352 (1/2) → this PR (2/2)

- Add FSDP DCP (distributed_checkpoint=True) paths in save_pretrained
- Add FSDP gather into single checkpoint saving
- Add `FSDPTesterMixin` and `tests/test_fsdp_mixin.py` (load/save/generation)
- Add `base_model_fsdp_plan` to cohere2_moe (only model in this stack)
- Add dedicated `fsdp_ci` CircleCI job and pytest markers

## [c3635813239] ArthurZucker (COLLABORATOR) · review_comment on tests/test_fsdp_mixin.py:529 · 2026-07-23
are you patching to make divisable?

## [c3635815310] ArthurZucker (COLLABORATOR) · review_comment on tests/test_fsdp_mixin.py:584 · 2026-07-23
create model on meta is used only once, not sure we need a func

## [c4761099606] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-07-23
wondering if any of the func in the test fsdp mixin are re-usable at all?

## [c4761165958] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-07-23 (no text)

## [c3636552140] 3outeille (MEMBER) · review_comment on tests/test_fsdp_mixin.py:529 · reply to [c3635813239] · 2026-07-23
no I am just patching to make it a smaller model overall

## [c4762085043] 3outeille (MEMBER) · review COMMENTED · 2026-07-23 (no text)

## [c5056096497] 3outeille (MEMBER) · issue_comment · 2026-07-23
run-slow: cohere2_moe

---
outcome: state=closed, merged=True
