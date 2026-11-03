# #39424 [issue] [YosoConfig] Missing `architectures` field
https://github.com/huggingface/transformers/issues/39424  ·  opened by SankethHonavar on 2025-07-15

## [body] SankethHonavar (NONE) · opening post · 2025-07-15
While contributing to the project and running the `add_missing_architectures.py` utility, I found that `YosoConfig` was missing the `architectures` field.

This issue tracks the fix to add the following field to:

📄 `src/transformers/models/yoso/configuration_yoso.py`

```python
architectures = ["YosoForMaskedLM"]

## [c3078271097] SankethHonavar (NONE) · issue_comment · 2025-07-16
Closing this issue, as I now realize the fix I proposed was based on a misunderstanding.
The `add_missing_architectures.py` script and class-level `architectures` field aren't standard or needed in this repo.
Thank you.

---
outcome: state=closed
