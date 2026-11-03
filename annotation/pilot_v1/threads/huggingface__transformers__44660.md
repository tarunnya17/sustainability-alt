# #44660 [pull_request] Fix: avoid late CUDA OOM in load_best_model_at_end with PEFT models
https://github.com/huggingface/transformers/pull/44660  ·  opened by DogWala on 2026-03-13

## [body] DogWala (NONE) · opening post · 2026-03-13
# What does this PR do?

Fixes #44637

This PR makes the PEFT `load_best_model_at_end` path in `Trainer` use a CPU-first adapter reload path during best-model loading.

Previously, when training a PEFT model, `Trainer` could reload the best adapter through a path that materialized adapter weights on CUDA during the final best-model load. Under low remaining GPU memory, this could trigger a late OOM even though the training loop had already completed.

To be specific:

The OOM happens because `PeftModel.load_adapter()` does not load weights directly into the existing adapter parameters in place. Instead, it first calls `load_peft_weights()`, which materializes a full temporary adapter `state_dict` on the target device, and only then passes that `state_dict` into `set_peft_model_state_dict()` / `model.load_state_dict(...)` to copy the values into the actual model parameters.

When `torch_device` is not specified, the current PEFT path infers `cuda`, so the checkpoint tensors are first loaded as a separate set of CUDA tensors. Under low remaining GPU memory, this extra device-side materialization can OOM before the weights are fully copied into the model, even though training itself has already finished.

A CPU-first load path is more memory-safe here: load the adapter checkpoint onto CPU first, then copy the weights into the model parameters. That avoids creating a full temporary CUDA `state_dict` at the most memory-constrained point of `load_best_model_at_end`.

## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [x] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [ ] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [x] Did you write any new necessary tests?

## Who can review?

@SunMarc @BenjaminBossan

## [c4055017437] BenjaminBossan (MEMBER) · issue_comment · 2026-03-13
Comment to reviewers: Let's resolve the discussion in #44637 first before proceeding with this PR.

---
outcome: state=open, merged=False
