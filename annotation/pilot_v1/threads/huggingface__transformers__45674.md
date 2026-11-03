# #45674 [issue] [BitsAndBytesConfig] Providing llm_int8_skip_modules clears the default lm_head exclusion, causing AssertionError in 4-bit inference
https://github.com/huggingface/transformers/issues/45674  ·  opened by softguy777 on 2026-04-28

## [body] softguy777 (NONE) · opening post · 2026-04-28
## Environment

- `transformers`: 5.5.4
- `bitsandbytes`: 0.49.2
- `torch`: 2.11.0+cu126
- CUDA: 12.6
- OS: Windows 11
- GPU: NVIDIA RTX 3090

## Bug Description

When specifying `llm_int8_skip_modules` in `BitsAndBytesConfig`, the default module exclusion list (which normally protects `lm_head` from being quantized) is silently cleared. This causes a crash during inference:

```
File "bitsandbytes/nn/modules.py", line 415, in fix_4bit_weight_quant_state_from_module
    assert module.weight.shape[1] == 1
AssertionError
```

## Root Cause

In `transformers/quantizers/base.py`, `get_modules_to_not_convert()`:

```python
if skip_modules is None or add_default_skips:
    modules_to_not_convert = get_keys_to_not_convert(model)  # auto-detects lm_head
else:
    modules_to_not_convert = []  # ← cleared when user provides ANY list!

if skip_modules is not None:
    modules_to_not_convert.extend(skip_modules)
```

When `llm_int8_skip_modules=None` (default), `get_keys_to_not_convert()` automatically finds and excludes `lm_head` and other output projection layers.

**The moment the user provides any list** (e.g., to protect a multimodal audio/vision tower from quantization), the auto-exclusion is disabled and `lm_head` gets quantized → `AssertionError` in bitsandbytes.

This is particularly easy to hit with multimodal models like Gemma 4 E2B-IT, where users need to explicitly skip the audio/vision towers to prevent quality degradation, but are not aware that doing so also removes the `lm_head` protection.

## Minimal Reproducible Example

```python
import torch
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

# ❌ Crashes: providing any list clears the lm_head default exclusion
[… 11 lines of code omitted]

**Workaround** (user must manually re-add `lm_head`):
```python
# ✅ Works: lm_head added explicitly
llm_int8_skip_modules=["lm_head", "model.audio_tower"]
```

## Suggested Fix

Change the behavior so that the user-provided list is **additive** rather than **replacing** the auto-detected defaults:

```python
# In get_modules_to_not_convert():
modules_to_not_convert = get_keys_to_not_convert(model)  # always start with defaults

if skip_modules is not None:
    modules_to_not_convert.extend(skip_modules)  # user additions merged, not replacing

modules_to_not_convert = list(set(modules_to_not_convert))
return modules_to_not_convert
```

At minimum, the docstring for `BitsAndBytesConfig.llm_int8_skip_modules` should warn that `lm_head` must be included manually when this parameter is set.

## Related

- Discovered while fixing: https://github.com/huggingface/transformers/issues/45672

## [c4335047867] SunMarc (MEMBER) · issue_comment · 2026-04-28
Well, if you don't pass `llm_int8_skip_modules`, we use good defaults. Otherwise, specifying `llm_int8_skip_modules` should overwrite that. Maybe the issue is that we should also skip the audio tower when quantizing by default. That should be better no ?

## [c4335192383] softguy777 (NONE) · issue_comment · reply to [c4335047867] · 2026-04-28
> Well, if you don't pass `llm_int8_skip_modules`, we use good defaults. Otherwise, specifying `llm_int8_skip_modules` should overwrite that. Maybe the issue is that we should also skip the audio tower when quantizing by default. That should be better no ?

Yes, I completely agree! Adding the audio tower to the default skip list would be the perfect solution.

If it helps, here are a few technical reasons why skipping it by default makes complete sense based on my testing:

The torch.finfo() Crash: When bitsandbytes quantizes the audio_tower to 4-bit, it packs the weights into uint8. This triggers a TypeError: torch.finfo() requires a floating point input type inside Gemma4AudioFeedForward and Gemma4AudioLightConv1d during inference.

Audio Fidelity: As widely known, quantizing audio encoders significantly degrades transcription quality. Since the audio tower is relatively small (~300M params), keeping it in bfloat16 is very cheap (~0.6GB VRAM) while preserving full accuracy.

The Overwrite Trap: Currently, users hitting the finfo() crash are forced to manually pass llm_int8_skip_modules=["model.audio_tower"]. But doing this overwrites your good defaults. Most users will forget to add "lm_head" back into the list, unintentionally quantizing the LM head and ruining text generation.

Adding ["model.audio_tower", "model.embed_audio"] to your default skip list for Gemma 4 Multi-modal models will easily prevent other users from falling into this same trap.

Thanks again for looking into this!

## [c4335240788] SunMarc (MEMBER) · issue_comment · 2026-04-28
Can you open a PR for that ? you will have to update the get_keys_to_not_convert function

## [c4335297190] softguy777 (NONE) · issue_comment · reply to [c4335240788] · 2026-04-28
> Can you open a PR for that ? you will have to update the get_keys_to_not_convert function

Sure, I'd love to! I will fork the repo, update the get_keys_to_not_convert function to include the audio tower for Gemma 4 models, and open a PR shortly. Thanks for pointing me to the right function!

## [c4335487439] softguy777 (NONE) · issue_comment · reply to [c4335297190] · 2026-04-28
> > Can you open a PR for that ? you will have to update the get_keys_to_not_convert function
> 
> Sure, I'd love to! I will fork the repo, update the get_keys_to_not_convert function to include the audio tower for Gemma 4 models, and open a PR shortly. Thanks for pointing me to the right function!

I opened a PR here: https://github.com/huggingface/transformers/pull/45683

---
outcome: state=closed
