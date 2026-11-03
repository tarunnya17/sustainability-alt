# #42188 [pull_request] [`BLT`] Fix cache usage
https://github.com/huggingface/transformers/pull/42188  ·  opened by vasqu on 2025-11-13

## [body] vasqu (COLLABORATOR) · opening post · 2025-11-13
The issue is/was that the cache wasn't utilized at all while several generation methods rely on it. This resulted in a mismatch between actual input and attention mask
- the mask was correctly expanded one by one
- after the prefill phase, the cache wasn't used resulting in kv always being of size 1

Why did it even work?
- because we silently cut the mask to fit the kv shape, i.e. a silent bug and another good reason why enforcing the correct shape is highly needed

The fix:
- apply self caches for both the encoder and decoder

Future considerations:
- add cross cache for both encoder + decoder --> not used atm (but also wasn't used before as far as I can see)

## [c3528467310] vasqu (COLLABORATOR) · issue_comment · 2025-11-13
run-slow: blt

## [c3528618946] vasqu (COLLABORATOR) · issue_comment · 2025-11-13
run-slow: blt

## [c3532400192] vasqu (COLLABORATOR) · issue_comment · 2025-11-14
run-slow: blt

## [c3532490698] vasqu (COLLABORATOR) · issue_comment · 2025-11-14
run-slow: blt

## [c3532690358] vasqu (COLLABORATOR) · issue_comment · 2025-11-14
Fixes the normal fast tests; weird issue atm on main where it takes significantly more memory / it's not properly cleaned up. Locally the slow tests pass except for those that already didnt

## [c3464969275] zucchini-nlp (MEMBER) · review APPROVED · 2025-11-14
Very interesting model! Approving since the PR fixes CI

I think it's fine if we don't have cache on cross attention module, and instead add a cache for Global Transformers. I skimmer over the models' tech report and seems that global transformer is a much bigger module than local encoder-decoders.

## [c3465226393] ArthurZucker (COLLABORATOR) · review APPROVED · 2025-11-14
thanks

---
outcome: state=closed, merged=True
