# #41103 [pull_request] Simplify and improve model loading logic
https://github.com/huggingface/transformers/pull/41103  ·  opened by Cyrilvallez on 2025-09-23

## [body] Cyrilvallez (MEMBER) · opening post · 2025-09-23
# What does this PR do?

Quite a bit of improvements in this PR to keep making things simpler:

- completely remove `offload_state_dict`: it was added several years ago to avoid loading 2x memory on cpu between the model and the state dict -> this is no longer needed from some time as the model is loaded on meta device, then params are loaded one after the other -> removes quite a bit of convoluted logic
- remove distinction between `model_to_load` and `model` in `_load_pretrained_model` -> this was there originally and is only confusing -> removes quite a bit of logic here as well
- rename `check_quantized_param() -> bool` to `param_needs_quantization() -> bool` in the quantizers -> muuuuch more self-explanatory, the old name was quite hard to understand
- removes `logger.info` messages -> if everything went normal, no need to tell me... Only tell me what went wrong
- rework the `mismatched_shapes` tests -> they were redundant and quite badly written, with a ton of unneeded exceptions and skips
- several other improvements in terms of logic/readability, as well as performances during init

## [c3333304452] Cyrilvallez (MEMBER) · issue_comment · 2025-09-25
run-slow: fbgemm_fp8, finegrained_fp8, higgs, hqq, mxfp4

## [c3333466219] Cyrilvallez (MEMBER) · issue_comment · 2025-09-25
run-slow: fbgemm_fp8, finegrained_fp8, higgs, hqq, mxfp4, bnb

## [c3333535873] Cyrilvallez (MEMBER) · issue_comment · 2025-09-25
run-slow: fbgemm_fp8, higgs, hqq, bnb, fp_quant

## [c3333790822] Cyrilvallez (MEMBER) · issue_comment · 2025-09-25
slow tests on quantizers are similar as on main (though main as quite a lot of failures!)

## [c2379019451] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:5111 · 2025-09-25
single for loop is enough no?

## [c2379023341] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:5788 · 2025-09-25
going through unexpected keys 3 times.

## [c3267559642] ArthurZucker (COLLABORATOR) · review APPROVED · 2025-09-25
core part very happy, leaving @SunMarc to review quantization part

## [c2379056751] Cyrilvallez (MEMBER) · review_comment on src/transformers/modeling_utils.py:5788 · reply to [c2379023341] · 2025-09-25
Indeed, they should be small but let me make an effort on that haha

## [c3267614115] Cyrilvallez (MEMBER) · review COMMENTED · 2025-09-25 (no text)

## [c2379174358] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/base.py:151 · 2025-09-25
why remove those ? compressed-tensors needs these.

## [c3267797347] SunMarc (MEMBER) · review COMMENTED · 2025-09-25
Nice cleanup ! Left a small comment

## [c2379254808] Cyrilvallez (MEMBER) · review_comment on src/transformers/quantizers/base.py:151 · reply to [c2379174358] · 2025-09-25
Are you sure? I don't see it being used anywhere in the library!

## [c3267917196] Cyrilvallez (MEMBER) · review COMMENTED · 2025-09-25 (no text)

## [c2379291673] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/base.py:151 · reply to [c2379174358] · 2025-09-25
mb, it was indeed removed from compressed-tensors in their last commit.

## [c3267969556] SunMarc (MEMBER) · review COMMENTED · 2025-09-25 (no text)

## [c3267970533] SunMarc (MEMBER) · review APPROVED · 2025-09-25
LGTM !

## [c3346525768] lclichen (NONE) · issue_comment · 2025-09-29
I am getting an error when I Load Qwen2.5-VL-72B-Instruct-AWQ:

```sh
Loading checkpoint shards:   0%|                                                                                                         | 0/11 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/deepspeed/workdir/shl/EvalScope/test_0918.py", line 4, in <module>
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
[… 30 lines of code omitted]

## [c3346682227] Cyrilvallez (MEMBER) · issue_comment · 2025-09-29
Indeed, it slipped in this PR! We're working on the quantizers before next release! Will be solved by https://github.com/huggingface/transformers/pull/41138 🤗

---
outcome: state=closed, merged=True
