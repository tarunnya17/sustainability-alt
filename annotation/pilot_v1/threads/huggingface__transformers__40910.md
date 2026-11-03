# #40910 [issue] Gemma-3: prepare_inputs_for_generation should forward pixel_values based on image token presence, not cache_position==0
https://github.com/huggingface/transformers/issues/40910  ·  opened by Simone999 on 2025-09-16

## [body] Simone999 (NONE) · opening post · 2025-09-16
### System Info

- `transformers` version: 4.56.1
- Platform: Windows-11-10.0.26100-SP0
- Python version: 3.12.11
- Huggingface_hub version: 0.34.4
- Safetensors version: 0.6.2
- Accelerate version: 1.10.1
- Accelerate config:    not found
- DeepSpeed version: not installed
- PyTorch version (accelerator?): 2.8.0+cu129 (CUDA)
- Tensorflow version (GPU?): not installed (NA)
- Flax version (CPU?/GPU?/TPU?): not installed (NA)
- Jax version: not installed
- JaxLib version: not installed
- Using distributed or parallel set-up in script?: no
- Using GPU in script?: yes
- GPU type: NVIDIA GeForce RTX 3060 Laptop GPU

### Who can help?

@zucchini-nlp 

### Information

- [ ] The official example scripts
- [ ] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [ ] My own task or dataset (give details below)

### Reproduction

[Prefix caching](https://huggingface.co/docs/transformers/kv_cache#prefill-a-cache-prefix-caching) with models that use  `Gemma3ForConditionalGeneration` is impossible when the non-cached prompt has an image due to code in `src/transformers/models/gemma3/modeling_gemma3.py:1175`

```python
# If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
# Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
if cache_position[0] == 0:
    model_inputs["pixel_values"] = pixel_values
```

Step to reproduce:
```python
import copy
from typing import Any, Dict, List
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

[… 53 lines of code omitted]
```

After caching a prefix, when we pass a user turn that includes image tokens at non-zero `cache_position`, `prepare_inputs_for_generation` drops `pixel_values` (it only forwards them when `cache_position[0] == 0`).

### Expected behavior

* `prepare_inputs_for_generation` should **forward `pixel_values` whenever the current `input_ids` slice contains the model’s image special token(s)**, regardless of whether `cache_position[0]` is 0 or >0. I.e. `cache_position[0] == 0` is not a good way to understand if we are in the decoding stage.
* This enables a common workflow: cache a **text system prompt** once, then later process a **user turn with an image** without re-encoding the system text, while still exercising the vision path.
* A minimal change would be to replace the strict `cache_position[0] == 0` gate with a check on image token presence
  * If `(input_ids == config.image_token_index).any():` → pass `pixel_values`.
  * Else, omit `pixel_values` during decode.

## [c3298620607] zucchini-nlp (MEMBER) · issue_comment · reply to [body] · 2025-09-16
@Simone999 that makes sense. Initially this condition was added as a check for prefill vs decoding stages because we don't want to tun vision encoder every generation step. I agree that it misses on cached system prompt and checking for image ids is better

One edge case might be if the inputs in decoding stage contain special image tokens. I believe for pre-trained models that will not happen, though it can cause some test cases to fail. I will check this out and see if it doesn't cause mass test failures

Ideally moving vision encoder to input preparation similar to encoder decoder models would be the perfect solution. We've been discussing it internally in the past, though it will require a lot of time to refactor it 🥲

## [c3298637779] Simone999 (NONE) · issue_comment · 2025-09-16
Thanks for the reply. I've already forked the repo and applied the change on it. I would be great to contribute to this fix in such a way

## [c3298668933] zucchini-nlp (MEMBER) · issue_comment · 2025-09-16
Oh cool, in that case feel free to submit a PR and tag me there. Also you'd probably need to fix all VLMs since we use the same pattern everywhere

## [c3414320923] zucchini-nlp (MEMBER) · issue_comment · 2025-10-17
not stale, keeping it open for now

---
outcome: state=closed
