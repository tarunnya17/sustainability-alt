# #40304 [pull_request] Gpt oss optim
https://github.com/huggingface/transformers/pull/40304  ·  opened by jiqing-feng on 2025-08-20

## [body] jiqing-feng (CONTRIBUTOR) · opening post · 2025-08-20
Enabling fast indexing for CPU. This optimization can bring 3x speed-up for [lmsys/gpt-oss-20b-bf16](https://huggingface.co/lmsys/gpt-oss-20b-bf16) on Intel 6th Gen Xeon.

## [c3204053754] jiqing-feng (CONTRIBUTOR) · issue_comment · 2025-08-20
run-slow: gpt_oss

## [c3204282997] jiqing-feng (CONTRIBUTOR) · issue_comment · 2025-08-20
run-slow: gpt_oss

## [c3204317725] jiqing-feng (CONTRIBUTOR) · issue_comment · 2025-08-20
Hi @yao-matrix , please review this PR. Thanks!

## [c3138876922] yao-matrix (CONTRIBUTOR) · review APPROVED · 2025-08-21 (no text)

## [c3208831826] jiqing-feng (CONTRIBUTOR) · issue_comment · 2025-08-21
Hi @SunMarc . Could you please review this PR? Computing single expert one by one is more friendly to CPU as CPU does not have extra flops to compute all experts for all tokens.

## [c3211325230] SunMarc (MEMBER) · issue_comment · 2025-08-21
cc @ArthurZucker

## [c2293667693] ArthurZucker (COLLABORATOR) · review_comment on tests/models/gpt_oss/test_modeling_gpt_oss.py:192 · 2025-08-22
yep fullgraph is not a must

## [c3144412172] ArthurZucker (COLLABORATOR) · review APPROVED · 2025-08-22
More than happy to add this, do you mind me asking if this is valid in a broad general sense (meaning for consumer's gpus!)

## [c3218571533] jiqing-feng (CONTRIBUTOR) · issue_comment · reply to [c3144412172] · 2025-08-25
> More than happy to add this, do you mind me asking if this is valid in a broad general sense (meaning for consumer's gpus!)

I have no customer's gpu to test it. But A100 shows that computing all experts together is better.

## [c3151041998] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-08-25
I can confirm that on MPS this gives a huge perf boost indeed: 
[image]! 

And ~7x for batched input

---
outcome: state=closed, merged=True
