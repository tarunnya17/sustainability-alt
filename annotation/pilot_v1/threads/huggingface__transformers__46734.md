# #46734 [pull_request] Fix gradient inflation when combining label smoothing with gradient accumulation
https://github.com/huggingface/transformers/pull/46734  ·  opened by Incheonkirin on 2026-06-17

## [body] Incheonkirin (CONTRIBUTOR) · opening post · 2026-06-17
## What this fixes

`label_smoothing_factor > 0` with `gradient_accumulation_steps > 1` inflates gradients by ~`gradient_accumulation_steps` for any model whose forward accepts loss kwargs. Repro and write-up in #46731.

## Root cause

#34198 wired `num_items_in_batch` into `trainer.py` but not into `LabelSmoother` (`trainer_pt_utils.py`). When `model_accepts_loss_kwargs` is True, `training_step` skips the `1/gradient_accumulation_steps` normalization on the assumption the loss already divided by the effective batch — but `LabelSmoother` mean-reduces over the micro-batch instead. The same `num_items_in_batch` is already threaded into the adjacent `compute_loss_func` branch.

## Fix

- `LabelSmoother.__call__` takes `num_items_in_batch` and uses it as the denominator, falling back to the active-token count when it's `None` (so standalone calls are unchanged).
- `compute_loss` passes `num_items_in_batch` into both `label_smoother` calls.

## Test

`test_gradient_accumulation_grad_norm_with_label_smoothing`: the grad-norm ratio between a large-batch baseline and an equivalent accumulation run goes from ~9x at `gradient_accumulation_steps=8` to ~1.0. Existing `TrainerGradientAccumulationTest` / `TrainerLabelSmoothingTest` still pass.

## Backward compatibility

Changes loss/gradient magnitude by ~`gradient_accumulation_steps` for existing label-smoothing + accumulation runs (previously over-scaled). LRs tuned against the old magnitude may need re-checking.

## Scope

Single-device and default DDP token averaging become exact. Legacy DataParallel (`n_gpu > 1`) keeps the same `n_gpu` approximation the `compute_loss_func` path already has, so this doesn't add DP-specific behavior.

Closes #46731

## [c4946363827] Incheonkirin (CONTRIBUTOR) · issue_comment · 2026-07-11
Replacement PR is now #47261 from a proper fork. It contains the rebased change and current validation; no reopen is needed.

---
outcome: state=closed, merged=False
