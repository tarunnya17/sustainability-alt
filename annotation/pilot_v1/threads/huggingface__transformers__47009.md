# #47009 [pull_request] Add linear_attn entries to Qwen3.5 base_model_tp_plan
https://github.com/huggingface/transformers/pull/47009  ·  opened by zaid646 on 2026-07-01

## [body] zaid646 (NONE) · opening post · 2026-07-01
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47009)

Fixes #46846

Qwen3.5 is a hybrid model: ~75% of decoder layers use linear_attention (Gated DeltaNet) with their own projection matrices (`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`). These were missing from `base_model_tp_plan`, causing:
- OOM at TP>1 (weights not sharded)
- RuntimeError in `model.generate()` (Conv1d channel mismatch after `in_proj_qkv`)

**Fix:** Add all `linear_attn.*` projections with `"colwise_gather_output"` pattern, which shards the weight matrix (fixing OOM) and all-gathers activations before the depthwise Conv1d (fixing the shape mismatch).

Updated both `modular_qwen3_5.py` (source) and `configuration_qwen3_5.py` (auto-generated).

## [c4865252269] Rocketknight1 (MEMBER) · issue_comment · 2026-07-02
cc @cyrilvallez for TP!

## [c4912136321] Cyrilvallez (MEMBER) · issue_comment · 2026-07-08
Already taken care of

---
outcome: state=closed, merged=False
