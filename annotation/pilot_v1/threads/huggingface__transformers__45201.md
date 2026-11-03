# #45201 [issue] [Gemma 4] Support per-layer FlashAttention: FA2 for sliding layers, SDPA for global layers
https://github.com/huggingface/transformers/issues/45201  ·  opened by samuelazran on 2026-04-02

## [body] samuelazran (NONE) · opening post · 2026-04-02
## Problem

Gemma 4 (26B-A4B) has a **hybrid attention architecture** where different layers use different head dimensions:

- **26 out of 30 layers** use **sliding window attention** with `head_dim=256` — fully compatible with FlashAttention 2
- **4 out of 30 layers** use **global attention** with `global_head_dim=512` — exceeds FA2's maximum supported head dimension of 256

Setting `attn_implementation="flash_attention_2"` causes a `RuntimeError` on the 4 global attention layers because FA2 does not support `head_dim > 256`.

### Error

```
RuntimeError: FlashAttention only supports head dimensions up to 256 
(got head_dim=512 in layer 3 [global attention layer])
```

The error occurs at the first global attention layer during the forward pass.

### Config details

From `config.json`:
```json
{
  "head_dim": 256,
  "global_head_dim": 512,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "sliding_window_pattern": 6,
  "num_hidden_layers": 30
}
```

With `sliding_window_pattern=6`, global attention layers are at indices `[3, 9, 15, 21]` (every 6th layer, 0-indexed, offset by 3). The remaining 26 layers use sliding window attention with `head_dim=256`.

## Current workaround

Fall back to `attn_implementation="sdpa"` for **all 30 layers**. This works but sacrifices the FA2 speedup on the 26 sliding layers (~87% of layers) that are fully FA2-compatible.

## Proposed solution: per-layer attention dispatch

Transformers already supports dict-based `attn_implementation` for some models. Gemma 4 should use this to dispatch attention per-layer:

- **Sliding attention layers** (26/30): use `flash_attention_2`
- **Global attention layers** (4/30): use `sdpa` (or `eager`)

This could be implemented in `Gemma4DecoderLayer` or at model init time by reading the layer index and `sliding_window_pattern` from the config:

```python
# Pseudocode for per-layer dispatch in Gemma4Model.__init__
for i, layer in enumerate(self.layers):
    is_global = (i % config.sliding_window_pattern) == (config.sliding_window_pattern // 2)
    if is_global and config._attn_implementation == "flash_attention_2":
        layer._attn_implementation = "sdpa"  # FA2 incompatible (head_dim=512)
    else:
        layer._attn_implementation = "flash_attention_2"  # FA2 compatible (head_dim=256)
```

Alternatively, the model could automatically detect the FA2 head_dim limit and fall back per-layer, similar to how some models handle mixed attention patterns.

## Impact

- **Performance**: Users lose ~87% of potential FA2 speedup by falling back to SDPA on all layers
- **Usability**: Setting `attn_implementation="flash_attention_2"` crashes instead of gracefully degrading
- **Scope**: Affects all Gemma 4 variants with hybrid sliding/global attention (26B-A4B confirmed)

## Environment

- `transformers` >= 4.52.0 (Gemma 4 support)
- `flash-attn` 2.x (head_dim limit = 256)
- PyTorch 2.x, CUDA

## Related

- FlashAttention head_dim limit: [Dao-AILab/flash-attention#801](https://github.com/Dao-AILab/flash-attention/issues/801) (closed — Tri Dao noted head_dim > 256 is nontrivial)
- FA4 (Hopper) head_dim support: [Dao-AILab/flash-attention#2318](https://github.com/Dao-AILab/flash-attention/issues/2318) (head_dim 256 now works on Hopper fwd, but 512 still unsupported)
- Transformers already has dict-based `attn_implementation` support for some model architectures — extending this to Gemma 4 would be the cleanest path

## [c4198737189] zucchini-nlp (MEMBER) · issue_comment · 2026-04-07
cc @Cyrilvallez, @vasqu for FA

## [c4215857112] vasqu (COLLABORATOR) · issue_comment · 2026-04-09
https://github.com/Dao-AILab/flash-attention/issues/2427 suggests that FA3/4 might get proper support soon. The easiest would be to restrict to those versions then. It depends on the timeline, if I have time I can try to take a stab at a layer types solution: Bind the attention implementation per layer types but that is non-trivial imo and might not align as well with our code base, just gotta try it out and see.

But tbh, I would like to see some benchmarks: SDPA is not as slow as people think (even the memory efficient backend) so I doubt that much perf gains.

## [c4215987946] Cyrilvallez (MEMBER) · issue_comment · reply to [c4215857112] · 2026-04-09
Agreed with @vasqu here! I think making this general would too many changes to the codebase for only this model. Moreover, I also don't think performances change that much! And it will necessary end up being supported soon since gemma4 is an important model, kernels will appear!

---
outcome: state=closed
