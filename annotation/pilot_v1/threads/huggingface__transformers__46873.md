# #46873 [pull_request] Fix GatedDeltaNet torch fallback prefill vs decode divergence (#46190)
https://github.com/huggingface/transformers/pull/46873  ·  opened by Aravind-11 on 2026-06-24

## [body] Aravind-11 (CONTRIBUTOR) · opening post · 2026-06-24
## Summary

Fixes [#46190](https://github.com/huggingface/transformers/issues/46190): large logit divergence (~0.125 max abs) on Qwen3.5 between full-sequence forward (`use_cache=False`) and prefill+decode with cache, when using the pure-torch GatedDeltaNet path (no `fla` / `causal-conv1d` / `mamba-ssm`).

### Root cause

`torch_chunk_gated_delta_rule` zero-padded incomplete final chunks to `chunk_size=64`. Those pad positions still updated the recurrent state. Because pad counts differ for sequence length `T` vs `T-1`:

| Call | Real tokens | Pad to 64 | State sees |
|------|-------------|-----------|------------|
| Full forward | T | 64-T pads | T + pads |
| Prefill | T-1 | 64-(T-1) pads | T-1 + different pads |

Decode then runs `torch_recurrent_gated_delta_rule` for the last token with a state that does not match the full-forward path → systematic drift that accumulates across hybrid GDA layers.

This matches the reporter's environment (torch fallback only) and the fact that fp32 does not help (not a precision issue).

### Fix

Make the pure-torch `torch_chunk_gated_delta_rule` delegate to `torch_recurrent_gated_delta_rule` so multi-token prefill and single-token decode use the same formulation. Full forward and prefill+decode are then mathematically aligned.

When FLA is installed, `chunk_gated_delta_rule` / `fused_recurrent_gated_delta_rule` still come from the optimized kernels (unchanged).

### Files

- `qwen3_next/modular_qwen3_next.py` + `modeling_qwen3_next.py`
- `qwen3_5/modeling_qwen3_5.py`
- `qwen3_5_moe/modeling_qwen3_5_moe.py`
- `olmo_hybrid/modeling_olmo_hybrid.py`

### Trade-off

Torch-only prefill becomes O(T) sequential instead of chunk-parallel. That path is already the slow fallback; install FLA for production throughput. Correctness over a silent ~0.125 logit error is the priority here.

## Test plan

- [ ] Repro from #46190 on `Qwen/Qwen3.5-0.8B` without FLA:
  ```python
  # full vs prefill+decode max abs logit diff should drop from ~1e-1 to ~1e-4 or better
  ```
- [ ] Same with FLA installed (optional): behavior unchanged from main (kernel path)
- [ ] Sanity: `Qwen3Next` / MoE / Olmo hybrid still import and run a short generate

Fixes #46190

---
outcome: state=closed, merged=False
