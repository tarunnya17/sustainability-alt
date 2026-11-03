# #46861 [pull_request] Use meshgrid for brevity
https://github.com/huggingface/transformers/pull/46861  ·  opened by zucchini-nlp on 2026-06-24

## [body] zucchini-nlp (MEMBER) · opening post · 2026-06-24
# What does this PR do?

Realized we could let torch handle the `arange(N).repeat(x, y)` even for 3D inputs, accidentally came out in a discussion with Pablo 😄 

IMO this is more readable and less error-prone since we don't have to choose between `repeat_interleave(a * b)/repeat(a * y)/etc`

## [c4787182020] zucchini-nlp (MEMBER) · issue_comment · 2026-06-24
run-slow: cosmos3_omni, ernie4_5_vl_moe, glm46v, glm4v, glm4v_moe, glm_image, glm_ocr, paddleocr_vl, qwen2_5_vl, qwen2_vl, qwen3_5, qwen3_5_moe, qwen3_vl, qwen3_vl_moe

## [c4562100068] molbap (COLLABORATOR) · review APPROVED · 2026-06-24
So not only is it more readable (in my opinion, YMMV) but it looks like it's faster? I ran a quick bisect profile and it seems it's faster especially for long video contexts no? Curious if you have done some measurements on your end (I just ran the standalone part and get ~30% speedup on this op, but haven't checked memory etc)

Ok for me, I think worth an eye from @vasqu as well! Thanks for doing this

## [c4789263331] zucchini-nlp (MEMBER) · issue_comment · 2026-06-24
Not really, I suppose `torch.meshgrid` is doing some kind of optim internally tho I wasn't expecting it. Let me run check it as well in a while

## [c3467701501] vasqu (COLLABORATOR) · review_comment on src/transformers/vision_utils.py:75 · 2026-06-24
any reason we dont do these casts anymore?

## [c3467724679] vasqu (COLLABORATOR) · review_comment on src/transformers/vision_utils.py:81 · 2026-06-24
ah ok so we do the same reshapes here then, quite neat

## [c4562920860] vasqu (COLLABORATOR) · review APPROVED · 2026-06-24
Really like this as well! 

Would be interestiing to see how it performs isolated and with torch compile maybe? But anyways, I like this definitely more than the previous version since the repeats ares super awkward where you have to keep track of their patterns

## [c3472864716] zucchini-nlp (MEMBER) · review_comment on src/transformers/vision_utils.py:75 · reply to [c3467701501] · 2026-06-25
Looks like they dont affect anything, the code runs with single tensors as well and casting `int` might cause unnecessary moving back and forth between devices

## [c4569105242] zucchini-nlp (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c3472866582] zucchini-nlp (MEMBER) · review_comment on src/transformers/vision_utils.py:75 · reply to [c3467701501] · 2026-06-25
unless it has smth to do specifically with export, though the slow CI is fine

## [c4569107183] zucchini-nlp (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c4797209126] zucchini-nlp (MEMBER) · issue_comment · 2026-06-25
Ha, indeed it is up to ~35% faster. I did a small bench with batch-size in (20, 40, 100) and random grids, offsets, merge sizes 😄

## [c3472871232] zucchini-nlp (MEMBER) · review_comment on src/transformers/vision_utils.py:75 · reply to [c3467701501] · 2026-06-25
maybe just to be safe, cc @IlyasMoutawwakil 
Then I will merge

## [c4569112417] zucchini-nlp (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c3473125757] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/vision_utils.py:75 · reply to [c3467701501] · 2026-06-25
they are unnecessary, just explicit, if you don't apply them, torch will do it implicitly for you (like max cu seqlen int on fa)

## [c4569413623] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c4797541628] IlyasMoutawwakil (MEMBER) · issue_comment · 2026-06-25
beautiful!

---
outcome: state=closed, merged=True
