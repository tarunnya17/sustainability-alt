# #44073 [pull_request] [VisualBert] Refactor output tracing using capture_outputs and can_return_tuple decorators
https://github.com/huggingface/transformers/pull/44073  ·  opened by aman-coder03 on 2026-02-17

## [body] aman-coder03 (NONE) · opening post · 2026-02-17
## What does this PR do?

This PR migrates VisualBert to the new standardized output tracing system using the `@capture_outputs` and `@can_return_tuple` decorators.

Specifically, this PR:

- Adds `_can_record_outputs` to `VisualBertPreTrainedModel`
- Applies `@capture_outputs` to `VisualBertModel.forward`
- Updates downstream task-specific forward methods to use `@can_return_tuple`
- Removes manual handling of `output_hidden_states`, `output_attentions`, and `return_dict`
- Relies on the decorator-based approach for output collection

## Why is this needed?

This aligns VisualBert with the new standardized output tracing framework introduced in #43979 and removes legacy manual output tracing logic.

---
outcome: state=open, merged=False
