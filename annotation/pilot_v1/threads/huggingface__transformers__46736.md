# #46736 [issue] LoRA fine-tuning of an FP8 checkpoint is blocked: `get_peft_model` does not clear `validate_quantization_for_training` (QuantizationMethod.FP8)
https://github.com/huggingface/transformers/issues/46736  ·  opened by brunopistone on 2026-06-17

## [body] brunopistone (NONE) · opening post · 2026-06-17
### System Info

- `transformers` 5.9.0
- `peft` 0.19.1
- `accelerate` 1.13.0
- `torch` 2.9.1 (CUDA 12.9)
- Python 3.12
- Platform: SageMaker training job, 2× p5.48xlarge (16× H100 80GB), FSDP `full_shard`, `attn_implementation="eager"`
- Model: `deepseek-ai/DeepSeek-V4-Flash` (284B MoE, published **FP8-only** — `quantization_config.quant_method == "fp8"`, e4m3, 128×128 weight blocks)

### Who can help?

@SunMarc @MekkCyber (quantization), @BenjaminBossan (PEFT)

### Description

Fine-tuning an FP8-quantized checkpoint with LoRA fails at `Trainer.__init__`:

```
ValueError: The model you are trying to fine-tune is quantized with QuantizationMethod.FP8
but that quantization method do not support training. Please open an issue on GitHub ...
```

The error message and the docs both say the supported route for a quantized base model is to **attach trainable LoRA adapters** on top. We do exactly that via `peft.get_peft_model(model, LoraConfig(...))` **before** constructing the `Trainer`, yet the guard still raises.

Root cause (transformers 5.9.0, `trainer_utils.py::validate_quantization_for_training`):

```python
_is_quantized_and_base_model = getattr(model, "is_quantized", False) and not getattr(
    model, "_hf_peft_config_loaded", False
)
...
elif _is_quantized_and_base_model and not _quantization_method_supports_training:
    raise ValueError(... "do not support training" ...)
```

- `get_peft_model` wraps the model in a `PeftModel`, but **does not set `_hf_peft_config_loaded` on the (inner) base model** (confirmed: that attribute is never assigned anywhere in `peft`; it is only set by transformers' native `PeftAdapterMixin.add_adapter`/`load_adapter`, `integrations/peft.py`).
- `PeftModel.__getattr__` proxies `is_quantized` to the FP8 base → `True`.
- So `_is_quantized_and_base_model` stays `True`, and because the FP8 quantizer reports `is_trainable == False`, the **`elif` branch raises** — even though a LoRA adapter is attached.

Note the earlier `if` branch *does* exempt PEFT models (`... and not _is_peft_model(model) ...`), but the `elif` (quant-method-not-trainable) has **no `_is_peft_model` / `_hf_peft_config_loaded` exemption** when the wrapper came from `get_peft_model`. As a result there is **no documented way** to LoRA-fine-tune an FP8 base via the standard `get_peft_model` flow.

### Reproduction

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset
[… 13 lines of code omitted]

### Expected behavior

One of the following:

1. A `get_peft_model`-wrapped model (`_is_peft_model(model) is True`) should **bypass the `elif` "quant-method-not-trainable" branch**, since the frozen FP8 base is never updated — only the bf16 LoRA adapters are. (The first `if` branch already exempts PEFT models; the `elif` arguably should too.)
2. Or: clear, official documentation that FP8 checkpoints cannot be LoRA-fine-tuned at all, and that `model.add_adapter(...)` (which sets `_hf_peft_config_loaded`) is/ isn't a supported substitute — including whether autograd actually backpropagates through FP8 block-quantized linears to the adapters.

### Open questions for maintainers

- Is `model.add_adapter(lora_config)` (native API, sets `_hf_peft_config_loaded=True`) the intended supported path for LoRA-on-FP8, or does it merely bypass the guard while the backward pass through FP8 linears is still unsupported?
- Does this overlap with #46295 (Training Support for FP8) and the closed-without-resolution #41516 / #37927? If FP8 LoRA is intended to work, the `get_peft_model` flow appears to be the blocker.

### Related issues

- #46295 (open) — Training Support for FP8
- #41516 (closed, no resolution) — DeepSeek V3.1 LoRA → same FP8 ValueError
- #37927 (closed) — request FP8 training support
- #39410 (open) — FP8 training for MP/TP

## [c4741332838] matdou (NONE) · issue_comment · 2026-06-18
It seems intentional. `FineGrainedFP8HfQuantizer.is_trainable` returns `False` unconditionally. The comments in `finegrained_fp8.py`'s MoE experts path say it directly:

```py
# FP8 quantized weights are inference-only, so no bwd pre-mask is needed.
# ...
# Inference-only path, so `to_local()` autograd-awareness is moot.
```

`FP8Linear.forward()` follows the same pattern. The real, non-dequantized path calls the Triton and DeepGEMM kernels as plain functions, with no `torch.autograd.Function` or `torch.library` backward registered anywhere.

So even if you bypass the `get_peft_model()` guard with `model.add_adapter()` then the training will start, but the frozen layer drops its gradient contribution to earlier layers without raising an error. The adapter on that layer still trains, the signal reaching upstream layers just gets weaker.

FineGrainedFP8 and DeepGEMM are built for inference speed, not training. That's a different goal from FP8 training methods like Transformer Engine or torchao. A checkpoint published as FP8 only was quantized for serving cost, not for fine-tuning. That distinction is probably part of why this keeps coming up across #41516, #37927, and #46295.

## [c4741445236] BenjaminBossan (MEMBER) · issue_comment · 2026-06-18
I think it should in theory be possible to train LoRA even with float8 base model weights. We improved the support in PEFT recently (https://github.com/huggingface/peft/pull/3055) but I can't say if that's enough for DeepSeek v4 or not (if there is a tiny checkpoint I can use for testing, please let me know). In general, I don't see why training LoRA on top of int8 or int4 base weights should be possible but not on top of fp8.

@SunMarc Is it possible that the check should allow PEFT models to pass through or am I missing something?

## [c4787950527] BenjaminBossan (MEMBER) · issue_comment · 2026-06-24
Ping @SunMarc

## [c4790165983] SunMarc (MEMBER) · issue_comment · 2026-06-24
Even if we allow peft models to pass through, the backward of the FP8Linear will fail as the autograd is not registered. Let me see what i can do to fix this.

## [c4957646009] sanketbadgujar007-byte (NONE) · issue_comment · 2026-07-13
I think you should try Domain Alignment - www.nugen.in

## [c4979817726] michaelmanly (NONE) · issue_comment · reply to [body] · 2026-07-15
@brunopistone 
Hey, saw the FP8 LoRA job is running across 16 H100s on SageMaker.

Is this part of a repeated training pipeline, or mainly a one-off validation run for this checkpoint?

---
outcome: state=closed
