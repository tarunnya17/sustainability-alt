# #38704 [pull_request] Fix `mllama`
https://github.com/huggingface/transformers/pull/38704  ·  opened by ydshieh on 2025-06-09

## [body] ydshieh (COLLABORATOR) · opening post · 2025-06-09
# What does this PR do?

`test_11b_model_integration_forward` has dtype issue, it fails, and affect other tests (GPU OOM).

I also update the expected values for A10.

All tests pass now on T4/A10 + torch 2.6/2.7

## [c2921162748] Cyrilvallez (MEMBER) · review APPROVED · 2025-06-12
Thanks!

---
outcome: state=closed, merged=True
