# #38662 [issue] Whisper models appear to be broken with Flash Attention 2
https://github.com/huggingface/transformers/issues/38662  ·  opened by Anjum48 on 2025-06-07

## [body] Anjum48 (NONE) · opening post · 2025-06-07
### System Info

- `transformers` version: 4.52.4
- Platform: Linux-6.8.0-60-generic-x86_64-with-glibc2.39
- Python version: 3.12.3
- Huggingface_hub version: 0.32.4
- Safetensors version: 0.4.5
- Accelerate version: 1.7.0
- Accelerate config:    not found
- DeepSpeed version: not installed
- PyTorch version (GPU?): 2.7.1+cu128 (True)
- Tensorflow version (GPU?): not installed (NA)
- Flax version (CPU?/GPU?/TPU?): not installed (NA)
- Jax version: not installed
- JaxLib version: not installed
- Using distributed or parallel set-up in script?: <fill in>
- Using GPU in script?: <fill in>
- GPU type: NVIDIA GeForce RTX 5090

flash-attn: Built from main

### Who can help?

@eustlb 

### Information

- [x] The official example scripts
- [ ] My own modified scripts

### Tasks

- [x] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [ ] My own task or dataset (give details below)

### Reproduction

Code to reproduce:
```python
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v3.5"

[… 23 lines of code omitted]
```

The audio file is long (~2 hours). Note that the model runs fine if flash attention is disabled. Have tested with various Whisper models and get the same error. 

Rolling back to 4.46.1 seems to run ok, but I'm not sure flash attn is actually working. Transcription on a 5090 for some models (e.g. openai/whisper-base) is significantly slower than a 3090, but this might be due to hardware optimisations on the flash attn side...

Error:
```
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[13], line 28
     15 processor = AutoProcessor.from_pretrained(model_id)
     17 pipe = pipeline(
[… 332 lines of code omitted]
```

### Expected behavior

Expect the pipeline to run in the same way as when flash attention is disabled (but faster!)

## [c3014717172] marcndo (CONTRIBUTOR) · issue_comment · 2025-06-28
Hi @eustlb, I’d like to grab this, please.

## [c3018471923] eustlb (COLLABORATOR) · issue_comment · 2025-06-30
cc @vasqu since you've been working on Whisper-related attention issues

## [c3018514499] vasqu (COLLABORATOR) · issue_comment · reply to [body] · 2025-06-30
@Anjum48 Can you try the latest transformers version (`4.53.0`)? We have rewritten the attention module in #38235 to work with all attention types (fa2 was known to not work as expected).

Regarding speed: The transformers implementation with flash attention might be slow(er) on small inputs (<2k) because of how we have to handle attention masks. I would suggest SDPA in most cases. #36585 might be interesting to you (as there are perf comparisons on whisper parts)

## [c3033583224] Anjum48 (NONE) · issue_comment · reply to [c3018514499] · 2025-07-03
Thanks @vasqu! Can confirm that FA2 is now working and on benchmarking, sdpa is slightly faster as discussed above

## [c3039123898] Anjum48 (NONE) · issue_comment · reply to [c3018514499] · 2025-07-05
Hi @vasqu - reopening this as after some benchmarking between 4.46 and 4.53, it looks like the new attention module is slowing things down quite significantly (times are in M.SS):

```
transformers==4.46.1 sdpa

openai/whisper-base: 4.50
distil-whisper/distil-medium.en: 1.20
[… 30 lines of code omitted]

This is benchmarking with 125 mins of audio from [here](https://traffic.libsyn.com/secure/skepticsguide/skepticast2025-05-31.mp3)

Code used for tests:

```python
import torch
from whisper.utils import get_writer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pathlib import Path
[… 79 lines of code omitted]

It looks like in these tests `distil-whisper/distil-large-v3` is about 2x slower :(

## [c3044343871] vasqu (COLLABORATOR) · issue_comment · reply to [c3039123898] · 2025-07-07
@Anjum48 This is indeed not so nice 👀 Since it's a few minor versions, it might not be related to the attention calculation tbh - there have been some changes to the mask creation and the generation code as well, for example. Not sure when I would get to dig deeper into this; if you'd like to, you could also investigate potential bad commits with `git bisect`.

## [c3079983514] Anjum48 (NONE) · issue_comment · reply to [c3044343871] · 2025-07-16
Hi @vasqu, I wasn't sure how to tackle a `git bisect` since I'm not 100% sure which files to watch for changes.

However, I did a manual binary search :) and I located that the performance regression happened between 4.51.3 -> 4.52.1. My tests included 2 repeats and confirmed the 2x slower times I was seeing above.

The [release notes](https://github.com/huggingface/transformers/releases/tag/v4.52.1) show a number of PRs related to attention and a few other things that might be relevant. Probably worth you scanning through those PRs to see if anything jumps out to you as a potential culprit.

This is the only Whisper related PR in that release:
> Fix incorrect attention mask truncate in WhisperFlashAttention2 by @OliBomby in https://github.com/huggingface/transformers/pull/36477

Tagging @eustlb too for visibility.

This is obviously turning into a different issue than the one I originally opened. Happy to rename this issue or open a new one - whichever you all think is best.

## [c3083685630] vasqu (COLLABORATOR) · issue_comment · reply to [c3079983514] · 2025-07-17
Indeed, it would be nice to track this in a separate issue. 

Thanks for already looking into it @Anjum48 :D I'm a bit suprised that it's v4.52 because a lot of changes happened during v4.53 and it makes me believe that it's not attention-related but something else. For `git bisect`, [stackoverflow](https://stackoverflow.com/questions/4713088/how-do-i-use-git-bisect) has a nice answer here - you will need to work on the git repository instead of via pip and you could just test with your performance test each time to determine good vs bad commit.

## [c3090153685] Anjum48 (NONE) · issue_comment · 2025-07-18
Opened a new issue so closing this one

---
outcome: state=closed
