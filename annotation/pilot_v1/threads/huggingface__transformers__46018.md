# #46018 [issue] DeepSeek-V4 shared expert not gated.
https://github.com/huggingface/transformers/issues/46018  ·  opened by yuzhongw-nvidia on 2026-05-18

## [body] yuzhongw-nvidia (NONE) · opening post · 2026-05-18
SwiGLU is not clamped for shared expert. Is it a bug or just a divergence between training and inference?

Deepseek inference ref code: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py#L627
transformers implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py#L1076-L1077

## [c4478272759] Rocketknight1 (MEMBER) · issue_comment · 2026-05-18
cc @arthurzucker for the V4 port

## [c4496924640] ArjunSrivastava1 (CONTRIBUTOR) · issue_comment · 2026-05-20
also linking this issue to the pr for easier tracking
#45892

## [c4531446776] ArthurZucker (COLLABORATOR) · issue_comment · 2026-05-25
Ah yeah I am pretty sure this was removed because it does not really make sense when you are not using fp8-4 mixes. 
But we can add it it should stabilize training

## [c4531826750] ArthurZucker (COLLABORATOR) · issue_comment · 2026-05-25
cc @IlyasMoutawwakil to put it in #45634

## [c4532318138] ArjunSrivastava1 (CONTRIBUTOR) · issue_comment · reply to [c4531826750] · 2026-05-25
@ArthurZucker, i got a question and its a bit unrelated to the issue, any plans there to select a different platform other than github for repos and stuff? seeing eus push for their own systems, and well - codeberg surging followed by githubs incidents(80 in the last 90 days or something)... are there any talks bout switching to something else other than github? 

i hear rumours and stuff in many places, so i thought, hey, might as well ask someone who has better chances to know

## [c4533914199] IlyasMoutawwakil (MEMBER) · issue_comment · 2026-05-25
already in #45634 btw

---
outcome: state=closed
