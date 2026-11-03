# #38702 [issue] Incorrect scaling of Gemma embeddings in float32 regime
https://github.com/huggingface/transformers/issues/38702  ·  opened by norpadon on 2025-06-09

## [body] norpadon (CONTRIBUTOR) · opening post · 2025-06-09
### System Info

Irrelevant

### Who can help?

@ArthurZucker

Google Gemma implementation casts embedding scale to bloat16, which rounds 33.9411 to 34.0

To match this behaviour, [HF implementation](https://github.com/huggingface/transformers/blob/d7b87b415a5dd4a3152051e1a0abd098a02c5bfa/src/transformers/models/gemma3/modeling_gemma3.py#L133) does
```python
super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)
```

This results in incorrect scaling behaviour if the model is loaded in float32 precision.

Relevant PR: https://github.com/huggingface/transformers/pull/29402

### Information

- [x] The official example scripts
- [x] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [ ] My own task or dataset (give details below)

### Reproduction

```python
>>> hf_model = AutoModelForCausalLM.from_pretrained(
    model_repo,
    torch_dtype=torch.float32,
    device_map="cpu",
[… 15 lines of code omitted]

## [c2958770791] Rocketknight1 (MEMBER) · issue_comment · reply to [body] · 2025-06-10
Hi @norpadon, this looks legit! Can you link to the code in the Gemma implementation showing the original behaviour that always casts to `bfloat16`?

## [c2958913631] norpadon (CONTRIBUTOR) · issue_comment · 2025-06-10
The original Gemma implementation in JAX also has the same scaling code:

https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/_modules.py#L111

And I think this is a bug on Google's part, because it causes quick and catastrophic accumulation of errors in activations and a huge divergence between fp32 and bf16 logits. I believe models shouldn't behave like this when upcasted to a higher precision. Model was originally trained with rounded-up constant scale factor, so this scale should not change depending on precision

## [c2959074189] Rocketknight1 (MEMBER) · issue_comment · 2025-06-10
Hmmn, this is tricky. At Hugging Face we place a very strong emphasis on **duplicating the behaviour of the original model**, even when we think it's suboptimal in some ways. 

However, I agree that in this case, it may be unintended - it's increasingly uncommon for people to run models in full `float32`, so that path may just not have been tested. Do you have any evidence that swapping to `float32` significantly degrades generation quality? We could probably justify the change if so.

## [c2959266900] norpadon (CONTRIBUTOR) · issue_comment · 2025-06-10
I have not checked the quality of generations, but this consistently increases the number of times the bf16 argmax token disagrees with the fp32 argmax token, as well as metrics such a max absolute deviation between f32 and bf16 logits (I am seeing increases of the relative difference between logits of 10-20%)

## [c2959326596] Rocketknight1 (MEMBER) · issue_comment · 2025-06-10
Hmm, okay. Given that the model was trained in `bfloat16`, this is almost certainly unintended.

cc @cyrilvallez as core maintainer, WDYT? 

tl;dr `embed_scale` is a non-persistent buffer that's computed on `__init__` as `self.config.hidden_size**0.5`, and then this is cast to `weight.dtype`. This value is `34` for `bfloat16` but `33.9411` in `float32`. Since this value is used for scaling, in `float32` the model diverges rapidly from `bfloat16`. The model was trained in `bfloat16` so it expects `34`.

I think always computing `embed_scale` in `bfloat16` should make `float32` models have correct outputs again, but this is different from Google's codebase. Probably they just never even tested `float32` training though 😅

## [c2963038955] Cyrilvallez (MEMBER) · issue_comment · reply to [c2959326596] · 2025-06-11
Humm, indeed as mentionned by @Rocketknight1  this is a tricky one. Given that the Google team chose to explicitly scale with the dtype instead of using a simple constant (i.e. 34), I am not really in favor of changing it in Transformers without explicit validation from the Gemma team that this is unintended. 
Also, doing it this way allows to use lower precision dtypes such as fp8 which would not work otherwise as the ops would fail

You could maybe open an issue on the Gemma repo to ask if this is intended or not? 🤗

## [c2963062032] norpadon (CONTRIBUTOR) · issue_comment · 2025-06-11
Do people ever quantize the residual stream to fp8? Sounds like a weird thing to do.

I am fine with you leaving it as it is, but it would be really nice to have a big warning about this behaviour somewhere in the code and in the documentation. It could potentially save some people many hours of debugging time

## [c2963318673] Cyrilvallez (MEMBER) · issue_comment · 2025-06-11
Ha yes sorry, it's the embeddings 🙃 was a bit too fast. However, checking more closely it's a known behavior (see the comments [here](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/modeling_gemma3.py#L479-L480)) in all Gemma implementations. In a way, you can think of this as if the scale was merged into the embedding directly and the weights were saved to the hub in fp32, and then cast to bf16 when loading -> you would need to truncate as well, resulting in different weights when loading in fp32/bf16

So this is kind of expected, although the amplitude of the weights of Gemma3 exacerbates the problem

## [c2963795278] norpadon (CONTRIBUTOR) · issue_comment · 2025-06-11
Yes but the explanation in the comments is very misleading because it implies that the code *always* casts scale to bf16, while in reality it only does this in particular circumstances (namely torch_dtype=torch.bfloat16 or torch_dtype="auto")

## [c3054071679] ArthurZucker (COLLABORATOR) · issue_comment · 2025-07-09
Very happy to update the comment! Do you want to open a pr to make it more understandable?

---
outcome: state=closed
