# #39433 [pull_request] Fix placeholders replacement logic in auto_docstring
https://github.com/huggingface/transformers/pull/39433  ·  opened by yonigozlan on 2025-07-15

## [body] yonigozlan (CONTRIBUTOR) · opening post · 2025-07-15
# What does this PR do?

The placeholders logic in auto_docstring seems to be broken currently, so that placeholders like {image_processor_class} are not replaced properly.
This PR simplifies greatly the logic and fixes the issue

## [c3033329680] Cyrilvallez (MEMBER) · review APPROVED · 2025-07-18
Thanks for fixing!

---
outcome: state=closed, merged=True
