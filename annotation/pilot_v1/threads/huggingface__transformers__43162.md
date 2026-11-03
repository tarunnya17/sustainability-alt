# #43162 [pull_request] fix device 0 memory allocate
https://github.com/huggingface/transformers/pull/43162  ·  opened by jiqing-feng on 2026-01-08

## [body] jiqing-feng (CONTRIBUTOR) · opening post · 2026-01-08
Under the current strategy, the memory allocated to device 0 is too small, which may cause the model to skip it entirely. We should ensure device 0 has enough capacity to accommodate at least the largest layer of the model, guaranteeing that a portion of the model is assigned to it.

---
outcome: state=closed, merged=False
