# #46239 [pull_request] fix: cache warmup `RuntimeError` on mps
https://github.com/huggingface/transformers/pull/46239  ·  opened by McPatate on 2026-05-27

## [body] McPatate (MEMBER) · opening post · 2026-05-27
Skip warmup on MPS: there is a limit of the maximum size a single buffer can have on MPS, which from testing seems to be about 2/3 of the total device memory (tested on apple silicon). This causes the warmup function to return a `RuntimeError: Invalid buffer size: XX.XX GiB`.

NOTE: not tested on intel macs, but I assume the same issue arises since it's also an `mps` backend.

EDIT: from [this old thread](https://developer.apple.com/forums/thread/61188), it appears the mtlbuffer limit on intel macs is capped to a hardcoded value, so this PR is even more so needed on that platform.

Running:
```py
import torch
from transformers import AutoModelForCausalLM


AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B", dtype=torch.bfloat16, device_map="mps")
[… 60 lines of code omitted]
```

## [c3310820287] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:5146 · 2026-05-27
please provide a repro, it does not fail for me AFAIK! Loading a big mixtral to max capa! 
Also no skip maybe reduce allocate + bench speed loss please

## [c4372394252] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-05-27 (no text)

## [c4557017338] hclsys (NONE) · issue_comment · 2026-05-27
makes sense — warmup's just a perf opt so skipping on mps loses the speedup but won't break correctness, and the buffer-size limit is real.

the cuda branch right above caps `byte_count` instead of bailing — could mps do the same (cap at ~2/3 device mem) and keep partial warmup, rather than `continue`-ing out entirely? or is the per-buffer limit low enough that capping isn't worth it vs just skipping?

## [c3315608792] McPatate (MEMBER) · review_comment on src/transformers/modeling_utils.py:5146 · reply to [c3310820287] · 2026-05-28
We won't need pre-allocation with safetensors on mps once https://github.com/safetensors/safetensors/pull/767 is merged. We allocate the mtlbuffers, fill them with `pread` and then hand them 0-copy to torch with dlpack. So as we don't go through torch's allocation stack, it's going to become unnecessary, at least for mps.

As we discussed by message, you in fact cannot allocate a buffer of size over 58gb out of your 96 available. It'd be interesting to see what `total_byte_count`'s value is when you load your Mixtral model.

## [c4378216850] McPatate (MEMBER) · review COMMENTED · 2026-05-28 (no text)

## [c3315676155] McPatate (MEMBER) · review_comment on src/transformers/modeling_utils.py:5146 · reply to [c3310820287] · 2026-05-28
Added more details in the PR desc

## [c4378294004] McPatate (MEMBER) · review COMMENTED · 2026-05-28 (no text)

## [c3315732480] McPatate (MEMBER) · review_comment on src/transformers/modeling_utils.py:5146 · reply to [c3310820287] · 2026-05-28
The only reason I can see your load not crash is because you don't set `device_map="mps"`, which skips the cache warmup function altogether.

## [c4378361952] McPatate (MEMBER) · review COMMENTED · 2026-05-28 (no text)

## [c4400203732] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-06-01
Ty confirmed on MPS its not slower anyways!

---
outcome: state=closed, merged=True
