# #45253 [pull_request] Fix Gemma4 `use_cache=False` producing bad logits
https://github.com/huggingface/transformers/pull/45253  ·  opened by Charly21r on 2026-04-05

## [body] Charly21r (CONTRIBUTOR) · opening post · 2026-04-05
# What does this PR do?

Fixes a bug where `use_cache=False` produces garbage logits in Gemma 4 models due to broken KV sharing between layers.

Fixes #45242 

## Root cause of the issue
Gemma 4 introduces two architectural features not present in Gemma 3:

1. **KV sharing** (`num_kv_shared_layers`): The last N decoder layers ("receiver" layers) don't compute their own keys/values, instead they reuse K/V states from earlier "donor" layers.

2. **K=V attention** (`attention_k_eq_v`): On global attention layers, keys and values share the same projection weights, so `v_proj` is set to `None`.

Both mechanisms were implemented by piggybacking on `past_key_values` (the KV cache object):
- Donor layers store their K/V into `past_key_values.shared_layers[layer_idx]`
- Receiver layers retrieve shared K/V from `past_key_values.shared_layers[donor_layer_idx]`

Both code paths are guarded by `if past_key_values is not None`.

When `use_cache=True`, a `DynamicCache` is created, `past_key_values` is not `None`, and everything works correctly.

When `use_cache=False`, `past_key_values` remains `None`, so:
1. **Donor layers never store** their K/V, thus the `past_key_values is not None` guard fails
2. **Receiver layers can't retrieve** shared K/V, thus the same guard fails, so they fall into the `else` branch and try to compute their own K/V
3. But receiver layers with `attention_k_eq_v=True` have `v_proj = None`, so the fallback `value_states = self.v_proj(hidden_states) if self.v_proj is not None else key_states` **uses keys as values**
4. The attention now computes with corrupted value states, producing garbage output

## The current fix
The fix creates a `DynamicCache` even when `use_cache=False`, but **after** causal mask creation to avoid affecting mask computation (which also depends on `past_key_values`). The cache is then available internally for KV sharing between layers. The return value is set to `None` when `use_cache=False` to preserve the expected API behavior.

```python
# Before (broken): cache only created when use_cache=True
if use_cache and past_key_values is None:
    past_key_values = DynamicCache(config=self.config)
# ... mask creation uses past_key_values ...
[… 47 lines of code omitted]

## [c4190873902] Charly21r (CONTRIBUTOR) · issue_comment · 2026-04-06
CI is currently failing across many `push_to_hub` tests due to 500 errors from the hub-ci endpoint (`/api/repos/create`). This looks like a broader infrastructure issue rather than something introduced in this PR. Happy to re-run once it's resolved.

## [c4212675015] Cyrilvallez (MEMBER) · issue_comment · 2026-04-09
https://github.com/huggingface/transformers/pull/45312

---
outcome: state=closed, merged=False
