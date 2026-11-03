# #40365 [pull_request] add scattermoe kernel for fast MoE training
https://github.com/huggingface/transformers/pull/40365  ·  opened by mayank31398 on 2025-08-22

## [body] mayank31398 (CONTRIBUTOR) · opening post · 2025-08-22
(empty)

## [c3214161941] Rocketknight1 (MEMBER) · issue_comment · 2025-08-22
cc @arthurzucker

## [c3144398000] ArthurZucker (COLLABORATOR) · review COMMENTED · reply to [body] · 2025-08-22
Hello @mayank31398 ! Nice pr, happy to add something like that, do you mind using `kernels` like what we do for GPT_OSS?! 
This way we keep a slow path, compatible with all torch, all hardwares etc and don't have code changes for the core modeling, and just have the kernel on the hub! 

WDYT? 🤗

## [c3214967506] mayank31398 (CONTRIBUTOR) · issue_comment · reply to [c3144398000] · 2025-08-22
@ArthurZucker scattermoe doesnt support bias for now, I will add this soon!
meanwhile supporting every model is hard since some models have expert weights as a moduleList instead of a 3D tensor :/

## [c3216084172] shawntan (CONTRIBUTOR) · issue_comment · reply to [c3214967506] · 2025-08-23
> Hello @mayank31398 ! Nice pr, happy to add something like that, do you mind using `kernels` like what we do for GPT_OSS?! This way we keep a slow path, compatible with all torch, all hardwares etc and don't have code changes for the core modeling, and just have the kernel on the hub!
> 
> WDYT? 🤗

Is there an existing triton kernel you could point to that I could follow?

## [c3219430563] ArthurZucker (COLLABORATOR) · issue_comment · 2025-08-25
This one: https://huggingface.co/kernels-community/megablocks/tree/main/torch-ext/megablocks (might not be triton) and otherwise https://huggingface.co/kernels-community/triton_kernels fully triton!

## [c3369411020] shawntan (CONTRIBUTOR) · issue_comment · 2025-10-05
Sorry it took a while. I've tried to piece together the various guides for community kernels to package `scattermoe`. 

https://huggingface.co/shawntan/scattermoe

This is what I have so far, and the following seems to work:
```python
from kernels import (
    LocalLayerRepository,
    use_kernel_mapping, 
    Mode,
    use_kernel_forward_from_hub,
[… 110 lines of code omitted]
```

## [c3373305975] shawntan (CONTRIBUTOR) · issue_comment · reply to [c3219430563] · 2025-10-06
@ArthurZucker  any thoughts?

## [c3375882045] 3outeille (MEMBER) · issue_comment · 2025-10-07
@MekkCyber

## [c3376043809] MekkCyber (CONTRIBUTOR) · issue_comment · reply to [c3373305975] · 2025-10-07
Hey @shawntan thanks for this contribution, the kernel looks good to me! We can have it in `kernels-community`, then we need some tests, and benchmarks to see if we include it in the kernel mapping in transformers

## [c3377051164] shawntan (CONTRIBUTOR) · issue_comment · 2025-10-07
Alright! What do you need specifically? I can come up with some benchmarks for the Granite models.

## [c3310413145] MekkCyber (CONTRIBUTOR) · review COMMENTED · 2025-10-07
Yes that would be great! we need some latency & memory benchmarks with and without the kernel for different seq len sizes to see if we have some speedups or improved memory consumption

## [c3377561630] shawntan (CONTRIBUTOR) · issue_comment · 2025-10-07
https://arxiv.org/pdf/2403.08245 The report for ScatterMoE has some benchmarks on previous models compared against Megablocks. 

It includes both speedup and memory usage comparisons.

[image]

## [c3380688295] MekkCyber (CONTRIBUTOR) · issue_comment · reply to [c3377561630] · 2025-10-08
Very nice performance @shawntan ! Thanks for sharing ! So the kernel is used for training not inference  ?

## [c3381614996] shawntan (CONTRIBUTOR) · issue_comment · 2025-10-08
It can be used for both. But gains will mainly come for training, or in cases of prefill. 

I've made the draft changes for `transformers`: https://github.com/shawntan/transformers/commit/3297b5e8aef6feaba38d81b1dcc8a89613110184

## [c3382657842] shawntan (CONTRIBUTOR) · issue_comment · 2025-10-08
Started another PR here #41458 because it is significantly different from the current one.

## [c3317950263] MekkCyber (CONTRIBUTOR) · review COMMENTED · 2025-10-09
Let’s close this PR then it will be superseded by https://github.com/huggingface/transformers/pull/41458

## [c3387240358] mayank31398 (CONTRIBUTOR) · issue_comment · 2025-10-09
ciao

---
outcome: state=closed, merged=False
