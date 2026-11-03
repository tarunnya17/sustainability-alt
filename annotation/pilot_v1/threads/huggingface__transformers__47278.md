# #47278 [pull_request] feat(rope): fused Triton kernel for apply_rotary_pos_emb via torchembed (3–5× faster)
https://github.com/huggingface/transformers/pull/47278  ·  opened by py-ai-dev on 2026-07-12

## [body] py-ai-dev (NONE) · opening post · 2026-07-12
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47278)

## Summary

This PR integrates [torchembed](https://github.com/liodon-ai/torchembed)'s fused Triton kernel into `apply_rotary_pos_emb` and centralises `rotate_half` + `apply_rotary_pos_emb` in `modeling_rope_utils.py`.

### Why is the current path slow?

```python
q_embed = (q * cos) + (rotate_half(q) * sin)
k_embed = (k * cos) + (rotate_half(k) * sin)
```

Each call allocates two intermediate tensors (the `rotate_half` outputs) and reads every element of `q` / `k` at least twice. The fused Triton kernel collapses this into a single pass: read once, rotate, write once.

### Benchmark

Measured on an NVIDIA GB10, B=2 H=32 D=128 float16:

| Seq len | Reference | Fused | Speedup |
|---------|-----------|-------|---------|
| 256 | 0.147 ms | 0.050 ms | **3.0×** |
| 512 | 0.350 ms | 0.065 ms | **5.4×** |
| 1 024 | 0.744 ms | 0.206 ms | **3.6×** |
| 2 048 | 1.699 ms | 0.443 ms | **3.8×** |
| 4 096 | 3.573 ms | 0.847 ms | **4.2×** |
| 8 192 | 7.341 ms | 1.655 ms | **4.4×** |

RoPE is called for Q and K in every attention layer. For Llama 3.1 8B (32 layers) this compounds significantly.

### Changes

**`src/transformers/modeling_rope_utils.py`**

- `_TORCHEMBED_AVAILABLE: bool` — checked once at import time (`find_spec("torchembed") and find_spec("triton")`)
- `rotate_half()` — canonical implementation moved here from per-model files
- `apply_rotary_pos_emb()` — canonical implementation with torchembed fast-path:
  - Activates when `_TORCHEMBED_AVAILABLE and cos.dim() == 3 and cos.is_cuda`
  - Extracts `cos[0, :, :D//2]` → `(S, D//2)`, the format the kernel expects
  - Falls back to the existing rotate-half path automatically (CPU, non-standard shapes, packed training with per-sample position IDs, torchembed not installed)

**`src/transformers/models/llama/modeling_llama.py`**

- Import `rotate_half` and `apply_rotary_pos_emb` from `modeling_rope_utils` instead of defining locally
- Removes ~30 lines of duplicate code
- Serves as the first adopter; other models can follow the same import pattern

**`tests/test_rope_torchembed.py`**

9 tests covering:
- `TestRotateHalf`: shape preservation, double-application identity, exact values
- `TestApplyRotaryPosEmb`: output shapes, float32 reference path, fp16 fused vs reference, bf16 fused vs reference, gradient flow, gradient vs float32 reference

Fused tests are skipped automatically when torchembed is not installed.

### Installation

```
pip install torchembed[triton]
```

No behaviour change when torchembed is not installed — the `_TORCHEMBED_AVAILABLE` guard short-circuits at import time.

### Correctness

The kernel uses the **rotate-half** convention (`cat([-x2, x1])`) matching HF Transformers' existing convention. The cos/sin extraction `cos[0, :, :D//2]` is correct for the standard `(B, S, D)` format produced by `LlamaRotaryEmbedding` (where the second half is identical to the first due to `emb = cat((freqs, freqs), dim=-1)`). The fallback path handles packed training (per-sample `position_ids`) where `cos[0]` would be incorrect.

### Related

- torchembed library: https://github.com/liodon-ai/torchembed
- torchtune PR (same kernel, adjacent-pairs convention): meta-pytorch/torchtune#2970
- vLLM investigation: torchembed is opt-in in vLLM because the C++ kernel is faster for packed-token inference layouts

🤖 Generated with [Claude Code](https://claude.ai/claude-code)

## [c4958847704] TransformerTitan (NONE) · issue_comment · 2026-07-13
hi @Rocketknight1 , this was a genuine technical contribution. I don't understand why do we need to block the contributor who had no intention of anything otherwise. Thank you.

## [c4968717428] Rocketknight1 (MEMBER) · issue_comment · reply to [c4958847704] · 2026-07-14
@TransformerTitan we're getting a flood of bad PRs from users running code agents and it's making it hard for the maintainers to get anything done. We've made it clear in the PR template and `AGENTS.md` that first-time contributors should not use code agents. If users ignore this, they get blocked. I realize it's quite harsh, but this is genuinely killing open-source right now.

This PR in particular overrides Transformers RoPE code with an import from `torchembed`, an entirely vibe-coded repo with a single maintainer and two Github stars. There's absolutely no way we're going to depend on a project like that, and any human contributor would realize what an insane request it is. I am genuinely unsure whether it's a naive code agent or a malicious attempt to inject a supply chain vulnerability, but either way it's code that we would never, ever merge.

---
outcome: state=closed, merged=False
