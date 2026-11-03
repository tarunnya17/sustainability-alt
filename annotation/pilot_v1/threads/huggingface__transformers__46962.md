# #46962 [issue] [Qwen3] `create_masks_for_generate` return type change after #41251
https://github.com/huggingface/transformers/issues/46962  ·  opened by hoshibara on 2026-06-29

## [body] hoshibara (NONE) · opening post · 2026-06-29
### System Info

`transformers env` throw an error. I catch the system info with pytorch tools.
```
PyTorch version: 2.14.0.dev20260623+xpu
Is debug build: False
CUDA used to build PyTorch: None
ROCM used to build PyTorch: N/A

[… 131 lines of code omitted]
```

### Who can help?

@vasqu @ArthurZucker @CyrilVallez

### Information

- [ ] The official example scripts
- [x] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [x] My own task or dataset (give details below)

### Reproduction

PR #41251 introduced early returns in `create_masks_for_generate` that change the return type from dict to a bare value when `effective_config.layer_types` has a single pattern or contains unknown types. This silently breaks model forward methods that rely on `isinstance(attention_mask, dict)` to detect whether the mask was already prepared by generate().

Before #41251

When effective_config.layer_types was present, create_masks_for_generate always returned a dict mapping layer type names to masks:

```python
# Line ~1530 before PR#41251
if hasattr(effective_config, "layer_types"):
    causal_masks = {}
    for layer_pattern in set(effective_config.layer_types):
        causal_masks[layer_pattern] = LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING[layer_pattern](**mask_kwargs)
    return causal_masks  # Always a dict
```

After #41251

The PR added two early-return paths that skip the dict wrapper:

```python
if any(layer_type not in LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING for layer_type in layer_patterns):
    return attention_mask  # Bare value — could be None, 2D tensor, etc.

if len(layer_patterns) == 1:
    return LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING[next(iter(layer_patterns))](**mask_kwargs)  # Bare value
```

Concrete impact: Qwen3

`Qwen3Model.forward` checks whether the mask was already prepared using isinstance:

```python
# src/transformers/models/qwen3/modeling_qwen3.py line ~404
if not isinstance(causal_mask_mapping := attention_mask, dict):
    # Mask not yet prepared — call create_causal_mask again
    causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
```

Before PR#41251, `create_masks_for_generate` returned a dict like `{"full_attention": None}`. Qwen3Model.forward's isinstance(attention_mask, dict) check passed, and the model used the pre-computed mask (None) directly with is_causal=True.

After PR#41251, with the early return `if len(layer_patterns) == 1: return ...`, the function returns bare None instead of a dict. Qwen3Model.forward enters the non-dict branch, and calls `create_causal_mask(attention_mask=None)` a second time. In this second call, `attention_mask=None produces padding_mask=None`, which causes `_can_skip_causal_mask_xpu` to fall back to a conservative path that cannot determine the mask is purely causal, so it falls through to building a full 4D mask with `is_causal=False`. It cause a significant performance regression.

Script: 
[repro_contract.py](https://github.com/user-attachments/files/29470254/repro_contract.py)

### Expected behavior

Revert early return path or fix Qwen mask script.

## [c4842947575] Rocketknight1 (MEMBER) · issue_comment · 2026-06-30
cc @arthurzucker since it's your PR!

## [c4861266872] sywangyi (CONTRIBUTOR) · issue_comment · 2026-07-02
@kaixuanliu

## [c4861842229] Cyrilvallez (MEMBER) · issue_comment · reply to [body] · 2026-07-02
Hey, thanks for the issue @hoshibara! Indeed, since xpu may return None for the mask during prefill, this is a new issue. However, all other models without `layer_types`, i.e. for which the mask is always a single Tensor or None, have this issue since a very long time... I'm investigating the best fix to detect such cases! 
I don't have xpu device available, but is the difference in speed so big for the prefill?

## [c4862027349] Cyrilvallez (MEMBER) · issue_comment · reply to [c4861266872] · 2026-07-02
I'm very surprised here in general, as skipping the mask for the prefill will necessarily lead to recompilation on the first decode token, which in my experience would lead to worst perfs than having a full mask on the prefill as well.
Also, how is it even possible that the prefill is correct without a mask, given that for StaticCache, the k/v that will be returned are longer than the query tokens, meaning the attention outputs computed will NOT be equivalent (due to the softmax). Do you somehow rescale the output of QK^T to remove the effect of padding? @sywangyi @kaixuanliu

## [c4862547648] hoshibara (NONE) · issue_comment · 2026-07-02
> Hey, thanks for the issue [@hoshibara](https://github.com/hoshibara)! Indeed, since xpu may return None for the mask during prefill, this is a new issue. However, all other models without `layer_types`, i.e. for which the mask is always a single Tensor or None, have this issue since a very long time... I'm investigating the best fix to detect such cases! I don't have xpu device available, but is the difference in speed so big for the prefill?

In our tests, a tensor mask will take 30% performance regression in end-to-end prefill latency during `model.generate`.

## [c4862640376] kaixuanliu (CONTRIBUTOR) · issue_comment · reply to [c4862027349] · 2026-07-02
Hi @Cyrilvallez , I summarize several points:

1. No matter on XPU or on CUDA with StaticCache, prefill and decode are normally compiled as two separate graphs. StaticCache makes the KV cache storage shape stable, but the prefill step still has query length > 1 while decode has query length = 1, and the attention-mask path may also differ. Therefore, if prefill skips the mask and decode uses a 4D mask, this does not stay in a single compiled graph; decode gets its own graph/compilation, so the recompilation is inevitable. 
2. With the default dynamic cache, transformers are expected to pass attention_mask=None in the common no-padding cases and rely on PyTorch SDPA’s is_causal=True. This lets XPU dispatch to the optimized/flash attention path. The update in https://github.com/huggingface/transformers/pull/41251 violated this expectation and lead to XPU dispatch attention to another path , which caused the performance regression
3. For prefill where q_len == kv_len, no matter dynamic cache or static cache, it follows point 2 as well
4. When using StaticCache, decode stage will correctly create a bool mask of shape (batch, 1, 1, max_cache_len), which will guarantee the correctness.
Hope this can answer your doubts. And I will submit a PR to fix the issue.

## [c4863002399] Cyrilvallez (MEMBER) · issue_comment · 2026-07-02
Hey! Yes sorry, we actually never compile the prefill so we don't have the problem of recompilation lol, I forgot for a second. 
However, the change here is only reflected for StaticCache anyway, not DynamicCache, as otherwise the mask is never prepared in advance by `generate`, so it would never have been a dict before and now not anymore. 

For StaticCache, during prefill we never have q_len == kv_len, because the k/v are padded to the max_len. Which is why I asked about the maths, because then the softmax will make the results different. However, with `is_causal=True`, it actually works because the padding is not reflected anymore, so a None mask is indeed correct. It will however work only and only if the Cache is fully empty, and will mathematically silently fail if the prefill is used with a non-empty Cache, as would be the case for example with some prefix caching or something (you call `generate` with an existing non-empty Cache and some new prefill tokens). This is because `is_causal` uses upper-left matrix, and not lower right by default. So this fast path is a bit brittle and should be restrained anyway

## [c4863428287] kaixuanliu (CONTRIBUTOR) · issue_comment · 2026-07-02
Ah yes, sorry my phrasing was not precise enough, what I meant with `q_len == kv_len` here was that the effective valid token range is the prompt length during prefill stage. So the issue here is mainly the contract change for Qwen3's prepared mask path: with StaticCache, `generate()` used to pass a dict like `{"full_attention": None}`. After the early return, it passes bare `None`, so `Qwen3Model.forward` treats the mask as not precomputed and calls `create_causal_mask()` again. That second call may materialize a full 4D mask, which preserves correctness but causes the XPU performance regression.

## [c4890191961] hoshibara (NONE) · issue_comment · 2026-07-06
Fixed.
Thanks for you PR!!

---
outcome: state=closed
