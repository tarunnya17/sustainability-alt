# #44602 [pull_request] Remove `cache_position` in more models (2)
https://github.com/huggingface/transformers/pull/44602  ·  opened by Cyrilvallez on 2026-03-11

## [body] Cyrilvallez (MEMBER) · opening post · 2026-03-11
# What does this PR do?

As per the title. Follow up of https://github.com/huggingface/transformers/pull/44330

Also take the opportunity to simplify t5 and its children, because the way they compute`position_bias` was super convoluted/overcomplicated, and required several additional unnecessary arguments etc...

## [c4041846745] Cyrilvallez (MEMBER) · issue_comment · 2026-03-11
run-slow: bridgetower mt5 qwen2_5_vl xlm_roberta audioflamingo3 data2vec_text gpt2 qwen2_5_omni roberta smolvlm camembert gpt_neo paddleocr_vl pix2struct idefics3 llama4 bark mllama bert_generation qwen2_vl whisper xlm_roberta_xl electra esm clip bert decision_transformer idefics2 pop2piano t5 blt blip_text idefics udop bloom roc_bert roberta_prelayernorm autoformer qwen2_audio ernie switch_transformers longt5 xmod

## [c4044963994] Cyrilvallez (MEMBER) · issue_comment · 2026-03-12
The 2 failed idefics2 tests are false positive: they actually both pass locally, on this PR and on main.
So no new failures at all!

## [c2927173429] vasqu (COLLABORATOR) · review_comment on src/transformers/models/autoformer/modeling_autoformer.py:447 · 2026-03-12
```suggestion
        **kwargs: Unpack[TransformersKwargs],
```

## [c2927177198] vasqu (COLLABORATOR) · review_comment on src/transformers/models/autoformer/modeling_autoformer.py:730 · 2026-03-12
Same here re let's type the kwargs just to be consistent, mentioning it the last time before it gets repetitive

## [c2927179416] vasqu (COLLABORATOR) · review_comment on src/transformers/models/bert/modeling_bert.py:642 · 2026-03-12
❤️

## [c2927198074] vasqu (COLLABORATOR) · review_comment on src/transformers/models/clip/modeling_clip.py:555 · 2026-03-12
Oh yea I remember that one 😬 that was a workaround

## [c2927202206] vasqu (COLLABORATOR) · review_comment on src/transformers/models/ernie/modular_ernie.py:270 · 2026-03-12
Nice

## [c2927208339] vasqu (COLLABORATOR) · review_comment on src/transformers/models/llama4/modeling_llama4.py:383 · 2026-03-12
We dont really have many tests for llama4, just want to raise awareness to double check that one especially

## [c2927209790] vasqu (COLLABORATOR) · review_comment on src/transformers/models/longt5/modeling_longt5.py:396 · 2026-03-12
That's the cleanup I suppose

## [c2927211838] vasqu (COLLABORATOR) · review_comment on src/transformers/models/longt5/modeling_longt5.py:1048 · 2026-03-12
Interesting, was this not needed before?

## [c2927218956] vasqu (COLLABORATOR) · review_comment on src/transformers/models/mllama/modeling_mllama.py:438 · 2026-03-12
This cross attention update looks weird to me; not really related to this PR tbh but wondering where this came from

## [c2927231412] vasqu (COLLABORATOR) · review_comment on src/transformers/models/mllama/modeling_mllama.py:1412 · 2026-03-12
A bit akward tbh but we cannot really avoid it because embeddings only are guaranteed within the language model - on the other hand why not move this to the language model and simplify?

## [c3939519196] vasqu (COLLABORATOR) · review APPROVED · 2026-03-12
Just a few smaller comments, looks overall good, thanks

Probably needs to resolve some merge conflicts because of the capturing PR I merged, sorry 😬

## [c2927509906] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/longt5/modeling_longt5.py:1048 · reply to [c2927211838] · 2026-03-12
Wrong leftover! Good catch!

## [c3939879239] Cyrilvallez (MEMBER) · review COMMENTED · 2026-03-12 (no text)

## [c2927516185] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/longt5/modeling_longt5.py:396 · reply to [c2927209790] · 2026-03-12
Well that and all the weird stuff about `real_seq_length`, passing a `query_length` as argument to the Attention's `forward` which is not the actual query length, etc...

## [c3939886283] Cyrilvallez (MEMBER) · review COMMENTED · 2026-03-12 (no text)

## [c2927532320] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/mllama/modeling_mllama.py:438 · reply to [c2927218956] · 2026-03-12
Yup agreed...

## [c3939902984] Cyrilvallez (MEMBER) · review COMMENTED · 2026-03-12 (no text)

## [c2927555054] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/mllama/modeling_mllama.py:1412 · reply to [c2927231412] · 2026-03-12
The ternaries are not so pretty, but they don't hurt, so I'd rather avoid moving too many things in this PR!

## [c3939928796] Cyrilvallez (MEMBER) · review COMMENTED · 2026-03-12 (no text)

---
outcome: state=closed, merged=True
