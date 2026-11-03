# #47211 [pull_request] Also size the device_map buffer from the largest leaf module (follow-up to #47203)
https://github.com/huggingface/transformers/pull/47211  ·  opened by dhruv7477 on 2026-07-09

## [body] dhruv7477 (CONTRIBUTOR) · opening post · 2026-07-09
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47211) [[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47211)

## What does this PR do?

Follow-up to #47203, which fixed the `device_map="auto"` under-packing from #46823 for same-class decoder layers of different sizes (MoE) by sizing the `get_balanced_memory` buffer from the largest no-split block.

A large childless leaf can be an even bigger un-splittable atom. `embed_tokens` and `lm_head` are not part of a no-split block, and are not shrunk by 4-bit quantization, so for large-vocab models they dominate. When such a leaf exceeds the balanced per-device caps, `infer_auto_device_map` skips every capped device, the model collapses onto the last device and then spills to cpu, and the 4-bit quantizer rejects the result.

### Reproduction (current `main`, meta device, no GPU needed)

Llama-3-8B loaded in 4-bit:

| module | size |
| --- | --- |
| `embed_tokens` / `lm_head` (fp16, not quantized) | ~2004 MiB each |
| one decoder layer (4-bit) | ~104 MiB |

The buffer is sized from the 104 MiB decoder layer, roughly 19x too small for the 2 GiB embedding, so `device_map="auto"` raises `Some modules are dispatched on the CPU or the disk...` even with 6x the model size in total GPU budget:

```
sum(GPU budget) = 6x model size  ->  ValueError (embedding spilled to cpu)
sum(GPU budget) = 4x model size  ->  ValueError
sum(GPU budget) = 3x model size  ->  ValueError
```
[… 17 lines of code omitted]

## [c4957413646] Rocketknight1 (MEMBER) · issue_comment · 2026-07-13
cc @cyrilvallez I was going to close this as an agent PR, but after re-reading I think it's sort of real, but it only applies on models with massive embedding layers. Basically the error occurs when `embed_size > model_size/num_devices`, placement on every device fails and it all falls through to the last GPU or the CPU.

## [c4957844046] dhruv7477 (CONTRIBUTOR) · issue_comment · 2026-07-13
Glad the re-read landed. Your mechanism is exactly right, so I ran the numbers to put a boundary on it.

One refinement: "massive embedding layers" undersells the blast radius. 4-bit is what drags this into the mainstream. Quantization shrinks the decoder stack roughly 4x and leaves `embed_tokens`/`lm_head` in fp16, so on a perfectly ordinary model the embedding quietly becomes the biggest atom in the room. Llama-3-8B in 4-bit: ~2 GB of embeddings against ~104 MB per decoder layer. Plug that into your inequality and `embed > model/n` flips at n = 4. Verified on current `main` with generous budgets: clean placement on 2 and 3 GPUs, `ValueError` on 4 through 8. Which is precisely the `load_in_4bit` + `device_map="auto"` + `max_memory` combo from #46823.

The fix stays deliberately boring: one line, keeps the #47203 MoE handling intact, and the tests fail on `main` and pass with it.

Also, since agent PRs came up: the diagnosis, code, and every number above are mine and verified end to end on my machine. I do use AI to polish the write-ups because I'd rather they read professionally. If the polish is doing more harm than good here, say the word and you'll get my unpolished typing instead.

## [c5101902002] Cyrilvallez (MEMBER) · issue_comment · 2026-07-28
This embedding will only be bigger than layers for very small models that do not need to be splitted as they are much smaller than any gpu size in general, so it's kind of a no-issue

## [c5107426408] dhruv7477 (CONTRIBUTOR) · issue_comment · 2026-07-28
The imbalance isn't a function of model size it's a function of quantization: 4-bit shrinks every decoder layer ~4× while embed_tokens/lm_head stay fp16, so embeddings dominate for any model at any scale. Llama-3-8B: ~1 GB embedding atoms next to ~104 MB layers.

And the users passing load_in_4bit + device_map="auto" + max_memory are, by definition, the ones without a comfortably large GPU (4×T4 Kaggle, paired 8 GB cards, VRAM reserved for KV cache) #46823 is one hitting it in the wild as a hard ValueError.

Given that, and a one-line fix with tests that fail on main and pass with it: seems worth taking as a bugfix even if the config is judged niche.

## [c5244422187] Cyrilvallez (MEMBER) · review APPROVED · 2026-09-18
Alright let's go

---
outcome: state=closed, merged=True
