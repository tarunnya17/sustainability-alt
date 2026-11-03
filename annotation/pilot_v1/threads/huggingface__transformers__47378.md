# #47378 [issue] Question about overriding bnb_4bit_compute_dtype for pre-quantized checkpoints
https://github.com/huggingface/transformers/issues/47378  ·  opened by John6666cat on 2026-07-17

## [body] John6666cat (NONE) · opening post · 2026-07-17
### System information

- `transformers`: 5.13.1
- `bitsandbytes`: 0.49.2
- `accelerate`: 1.14.0
- `huggingface_hub`: 1.23.0
- `safetensors`: 0.8.0
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- Python: 3.12.13
- Platform: Google Colab / Linux
- GPU: NVIDIA Tesla T4
- CUDA capability: 7.5
- Native BF16 support: false
- Distributed or parallel setup: no
- Model: `unsloth/gemma-2-9b-it-bnb-4bit`
- Model revision: `27b027bcbb6b1861b02551d5c699d5d07f29610a`

### Description

Hi, I noticed a small behavior when loading a pre-quantized bitsandbytes checkpoint.

This may be expected behavior, but I could not find a supported way to override the saved 4-bit compute dtype at load time, so I thought it might be worth documenting.

The checkpoint stores:

```json
"bnb_4bit_compute_dtype": "bfloat16"
```

I passed an explicit `BitsAndBytesConfig` with:

```python
bnb_4bit_compute_dtype=torch.float16
```

Transformers emitted a warning indicating that the checkpoint's existing `quantization_config` would be used.

After loading:

- `model.config.quantization_config` still contained `bfloat16`;
- all 294 `bitsandbytes.nn.Linear4bit` modules used `torch.bfloat16`;
- the explicitly requested FP16 compute dtype was not applied.

### Minimal reproduction

```python
import warnings

import bitsandbytes as bnb
import torch
[… 45 lines of code omitted]

Observed:

```text
Requested compute dtype:
torch.float16

Linear4bit module count:
294

Loaded Linear4bit compute dtypes:
{torch.bfloat16}
```

The loaded quantization configuration also retained:

```text
bnb_4bit_compute_dtype = bfloat16
```

### Manual workaround

Changing the modules after loading worked:

```python
for module in model.modules():
    if isinstance(module, bnb.nn.Linear4bit):
        module.compute_dtype = torch.float16
        module.compute_type_is_set = True
```

After this change, all 294 modules reported `torch.float16`.

### Small performance observation

This was only a small diagnostic, not a general benchmark.

On the Tesla T4, two repeated short deterministic generations produced identical output with both compute dtypes:

| Compute dtype | Mean generation speed |
|---|---:|
| BF16 | approximately 6.87 tokens/s |
| FP16 | approximately 7.37 tokens/s |

FP16 was approximately 7% faster in this small test.

The T4 does not have native BF16 support in this environment. The default `torch.cuda.is_bf16_supported()` check returned `True`, while the native-only check returned `False`.

### Expected behavior / question

I understand that preserving the saved quantization configuration for an already quantized checkpoint may be intentional.

Is there currently a supported way to override only `bnb_4bit_compute_dtype` while retaining the saved 4-bit weights and the other quantization settings?

If the current behavior is intended, one of the following small improvements might still be useful:

- document that a passed compute dtype does not override the saved value for pre-quantized checkpoints;
- include the conflicting saved and requested compute dtypes in the warning;
- allow `bnb_4bit_compute_dtype` as a safe runtime loading override;
- optionally warn when BF16 compute is selected on a GPU without native BF16 support.

This did not cause incorrect output in my small test, and the manual workaround works, so I consider this a low-priority usability issue rather than a critical bug.

Issue #25901 appears related to loading attributes for an already quantized configuration, but it concerns GPTQ and does not appear to cover this specific bitsandbytes compute-dtype case.

## [c5011886718] Zohaibarif69 (NONE) · issue_comment · 2026-07-18
I looked into this. It appears that `BitsAndBytesConfig` doesn't
define a `get_loading_attributes()` method and isn't included in
`LOADING_ATTRIBUTES_CONFIG_TYPES` in `quantizers/auto.py`. That's the
mechanism `merge_quantization_configs()` uses to decide which fields
are safe to override at load time for a pre-quantized checkpoint —
GPTQConfig, AwqConfig, etc. already support this, bnb never did.

I put together a small fix (adding get_loading_attributes() to
BitsAndBytesConfig, returning bnb_4bit_compute_dtype, and registering
it in LOADING_ATTRIBUTES_CONFIG_TYPES) and verified it resolves the
issue by calling merge_quantization_configs() directly — confirmed the
compute dtype is correctly overridden after the fix.

Happy to open a PR with this if that's useful — let me know.

## [c5127933390] John6666cat (NONE) · issue_comment · reply to [c5011886718] · 2026-07-30
Thank you for looking into this, and sorry I missed your reply for two weeks...😱
Your diagnosis makes sense and matches the behavior I observed.

> Happy to open a PR with this if that's useful 

Yes, please feel free to open a PR if the offer still stands. Treating `bnb_4bit_compute_dtype` as the safe load-time override, while preserving the checkpoint’s other quantization settings, sounds like the right scope to me.

If practical, a regression test covering both the merged configuration and the resulting `Linear4bit.compute_dtype` would be especially helpful, though I will leave the exact test scope to you and the maintainers.

Thanks again for investigating this so quickly.

---
outcome: state=closed
