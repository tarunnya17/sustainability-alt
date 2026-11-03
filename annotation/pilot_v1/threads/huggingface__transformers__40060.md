# #40060 [pull_request] Avoid CUDA stream sync
https://github.com/huggingface/transformers/pull/40060  ·  opened by cyyever on 2025-08-10

## [body] cyyever (CONTRIBUTOR) · opening post · 2025-08-10
# What does this PR do?

Remove more synchronizing CUDA operations reported.

## [c3183522846] Rocketknight1 (MEMBER) · issue_comment · 2025-08-13
Cool idea! How are you searching for CUDA sync points in the code?

## [c3184037017] cyyever (CONTRIBUTOR) · issue_comment · reply to [c3183522846] · 2025-08-13
@Rocketknight1 I use `torch.cuda.set_sync_debug_mode("warn")`.

## [c3184169497] Rocketknight1 (MEMBER) · issue_comment · 2025-08-13
Cool method, in that case! Let me know what else you find with it, and when this PR is ready for review.

## [c3184178274] cyyever (CONTRIBUTOR) · issue_comment · reply to [c3184169497] · 2025-08-13
@Rocketknight1 I have other fixes but it's better to split them into smaller ones. This PR is one of them and is complete, no more to add here. 
PS: the CI failures should be unrelated.

## [c3184199113] Rocketknight1 (MEMBER) · issue_comment · 2025-08-13
Yup - try leaving it for a bit or rebasing/rerunning, ping me once it's green and I'll merge!

## [c3116959454] ArthurZucker (COLLABORATOR) · review APPROVED · 2025-08-13
LGTM, wondering if we should put this in a patch as I think we juste shiped this

## [c3191386841] Rocketknight1 (MEMBER) · issue_comment · 2025-08-15
Merging now the CI is green

## [c2278983827] ydshieh (COLLABORATOR) · review_comment on src/transformers/modeling_flash_attention_utils.py:359 · 2025-08-15
just curious: if `indices_q = torch.arange(...)` create something in python `position_ids.size(0)` then the creation of tensor with this value on cuda device the source of stream sync, while the new change perform all work on cuda itself?

## [c3123846996] ydshieh (COLLABORATOR) · review COMMENTED · 2025-08-15 (no text)

## [c2279022258] cyyever (CONTRIBUTOR) · review_comment on src/transformers/modeling_flash_attention_utils.py:359 · reply to [c2278983827] · 2025-08-15
Yes, the creation of values on cuda device would mean that there is a blocking CPU to GPU memory copy. The new change is also blocking because  `nonzero` has to decide the resulting tensor shape. However, it is slightly faster than before. And the readability has improved.

## [c3123895984] cyyever (CONTRIBUTOR) · review COMMENTED · 2025-08-15 (no text)

## [c2279024009] cyyever (CONTRIBUTOR) · review_comment on src/transformers/modeling_flash_attention_utils.py:359 · reply to [c2278983827] · 2025-08-15
There is possibly more room for optimization in this function. I will keep monitoring it.

## [c3123898314] cyyever (CONTRIBUTOR) · review COMMENTED · 2025-08-15 (no text)

## [c2285240010] ydshieh (COLLABORATOR) · review_comment on src/transformers/modeling_flash_attention_utils.py:359 · reply to [c2278983827] · 2025-08-19
That would be very great 🙏 This cuda sync is definitely not a simple taks 💪

## [c3132296402] ydshieh (COLLABORATOR) · review COMMENTED · 2025-08-19 (no text)

---
outcome: state=closed, merged=True
