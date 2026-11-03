# #46054 [pull_request] Fix typo in HiggsAudioV2 processor (lenghts -> lengths)
https://github.com/huggingface/transformers/pull/46054  ·  opened by AI-Safeter on 2026-05-19

## [body] AI-Safeter (NONE) · opening post · 2026-05-19
## What does this PR do?

Fixes a typo in the local variable name inside `HiggsAudioV2Processor.__call__` in `src/transformers/models/higgs_audio_v2/processing_higgs_audio_v2.py`: `lenghts` -> `lengths`. The variable is only used locally within the same `if` block (three references), so no public API is affected.

## Before submitting
- [x] Trivial rename of a local variable (typo fix), no behaviour change.

---
outcome: state=closed, merged=False
