# #39208 [issue] Unknown Model (mobilenetv5_300m_enc) when loading Gemma 3n
https://github.com/huggingface/transformers/issues/39208  ·  opened by HenryNdubuaku on 2025-07-03

## [body] HenryNdubuaku (NONE) · opening post · 2025-07-03
### System Info

```bash
- `transformers` version: 4.53.0
- Platform: Linux-6.1.141+-x86_64-with-glibc2.35
- Python version: 3.10.12
- Huggingface_hub version: 0.33.2
[… 10 lines of code omitted]

Does Gemma 3n require special setups? That is not sustainable. 

### Who can help?

I upgraded to the latest transformers to try Gemma 3n and it would seem there is not implementation of mobilenetv5_300m when i try to run the model as described on the official huggingface page.

```bash
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
[<ipython-input-2-ac178faa1642>](https://localhost:8080/#) in <cell line: 8>()
      6 model_id = "google/gemma-3n-e4b-it"
[… 14 lines of code omitted]

Does Gemma 3n require special setups? That is not sustainable. 

### Information

- [x] The official example scripts
- [ ] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [ ] My own task or dataset (give details below)

### Reproduction

The example code on the model card 
```python
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
import requests
import torch

[… 36 lines of code omitted]
```

### Expected behavior

This is the basic example on the model card.

## [c3040992618] ghost (NONE) · issue_comment · 2025-07-06
https://huggingface.co/google/gemma-3n-E4B-it/discussions/8 this page maight help you

## [c3045362027] qubvel (CONTRIBUTOR) · issue_comment · 2025-07-07
Hey @HMUNACHI, please update `timm` to the latest version, because mobilenet is imported from there

## [c3045369153] qubvel (CONTRIBUTOR) · issue_comment · 2025-07-07
If anyone is down, we should add `timm` version check while instantiating gemma3n with mobilenet v5 vision encoder

## [c3092229607] tanuj-rai (CONTRIBUTOR) · issue_comment · reply to [c3045369153] · 2025-07-19
Hey @qubvel, can I work on this issue?

## [c3096046323] qubvel (CONTRIBUTOR) · issue_comment · reply to [c3092229607] · 2025-07-21
Hey @tanuj-rai! @VIGNESH15103 is already working on it in the following PR:

https://github.com/huggingface/transformers/pull/39264

in case they have no longer bandwidth, feel free to continue, but first let's wait for their feedback

## [c3102171884] tanuj-rai (CONTRIBUTOR) · issue_comment · reply to [c3096046323] · 2025-07-22
> Hey [@tanuj-rai](https://github.com/tanuj-rai)! [@VIGNESH15103](https://github.com/VIGNESH15103) is already working on it in the following PR:
> 
> [#39264](https://github.com/huggingface/transformers/pull/39264)
> 
> in case they have no longer bandwidth, feel free to continue, but first let's wait for their feedback

Thanks @qubvel for letting me know.  No problem, I’ll wait for the feedback.

## [c3201193248] jetnet (NONE) · issue_comment · 2025-08-19
the same error with the latest docker image `v0.10.1`

## [c3201296098] qubvel (CONTRIBUTOR) · issue_comment · reply to [c3201193248] · 2025-08-19
@jetnet make sure to install the latest `timm`

## [c3201346707] jetnet (NONE) · issue_comment · reply to [c3201296098] · 2025-08-19
@qubvel , I got it, but why haven't you installed the required dependencies into the public docker image?

## [c3201358172] qubvel (CONTRIBUTOR) · issue_comment · reply to [c3201346707] · 2025-08-19
@jetnet not sure which docker image you are talking about, can you please point it out?

## [c3201380284] jetnet (NONE) · issue_comment · reply to [c3201358172] · 2025-08-19
@qubvel , I appologize, wrong project here :) The latest vLLM throws the same error. I found the issue via Google and didn't pay attention to the project name..

## [c3209633323] tanuj-rai (CONTRIBUTOR) · issue_comment · reply to [c3102171884] · 2025-08-21
Hey @qubvel! Can I work on it now because it's been a month when 

> Hey [@tanuj-rai](https://github.com/tanuj-rai)! [@VIGNESH15103](https://github.com/VIGNESH15103) is already working on it in the following PR:
> 
> [#39264](https://github.com/huggingface/transformers/pull/39264)
> 
> in case they have no longer bandwidth, feel free to continue, but first let's wait for their feedback

Hi @qubvel! Is this issue still open for work on it?

## [c3255574711] deep1401 (NONE) · issue_comment · 2025-09-04
Hi, 
I am on transformers 4.55.2 and still facing this issue!

---
outcome: state=closed
