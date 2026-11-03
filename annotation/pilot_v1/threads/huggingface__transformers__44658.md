# #44658 [pull_request] Fix: fix Pipeline's save_pretrained method (issue #44655)
https://github.com/huggingface/transformers/pull/44658  ·  opened by Charly21r on 2026-03-13

## [body] Charly21r (CONTRIBUTOR) · opening post · 2026-03-13
# What does this PR do?

Fixes an issue where the `.modelcard` attribute of a pipeline is not initialized. Without this fix, calling `save_pretrained` on a pipeline (e.g., ASR pipeline) raises an `AttributeError` because `.modelcard` does not exist.

This PR ensures that `pipeline.modelcard` is initialized to `None` if it doesn’t already exist, so that `save_pretrained` works safely.

Fixes #44655 

## Example Usage

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")
pipe.save_pretrained("bug_test")  # works with this fix
[… 16 lines of code omitted]

## [c4055358855] Rocketknight1 (MEMBER) · issue_comment · 2026-03-13
Already fixed by #44621!

---
outcome: state=closed, merged=False
