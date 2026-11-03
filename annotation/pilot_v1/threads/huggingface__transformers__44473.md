# #44473 [pull_request] fix FSDP loading with meta devices
https://github.com/huggingface/transformers/pull/44473  ·  opened by winglian on 2026-03-05

## [body] winglian (COLLABORATOR) · opening post · 2026-03-05
# What does this PR do?

supersedes #44446

on `main`, when loading to cpu and using meta devices for non-rank0 processes, it now re-initializes weights on those processes as well as uses more CPU memory. In testing with loading llama3-8b.

main; both on CPU, uses 16GB system RAM, slow to load, re-inits weights on rank1
#44446: rank0 on CPU, rank1 on meta, uses 1.5GB system RAM
v4.57.6, both on CPU, uses 1.5GB system RAM
this PR, both on CPU, uses 1.5GB system RAM, same behavior and training loss as main and v4.57.6





Fixes # (issue)


## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [ ] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [ ] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [ ] Did you write any new necessary tests?


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

## [c4006390682] winglian (COLLABORATOR) · issue_comment · 2026-03-05
This PR basically restores the logic from v4.57.6 per https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/modeling_utils.py#L5853-L5877

(this was removed in #41580 https://github.com/huggingface/transformers/pull/41580/changes#diff-6b72b98c4c2dcfc6cc606843917733f5d858374fbc22a735ff483bbc0c1e63eaL5110-L5142)

## [c2895328598] SunMarc (MEMBER) · review_comment on src/transformers/modeling_utils.py:4575 · 2026-03-06
cc @Cyrilvallez 
I'm not sure we want to bring it back for all cases, can you maybe do it only when fsdp is enabled and check if the local rank 0 check influences this or not

## [c3903231065] SunMarc (MEMBER) · review COMMENTED · 2026-03-06 (no text)

## [c2895841611] winglian (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:4575 · reply to [c2895328598] · 2026-03-06
I managed to simplify this and wrapped it with a conditional for fsdp + non-rank0

## [c3903790809] winglian (COLLABORATOR) · review COMMENTED · 2026-03-06 (no text)

## [c2904323766] Cyrilvallez (MEMBER) · review_comment on src/transformers/modeling_utils.py:4563 · 2026-03-09
Should we simply `return` here instead, to completely avoid calling `initialize_weights` later in the function? Would be easier than setting all weights as initialized before calling initialize which will be skipped anyway as params are marked as already initialized

## [c3914167429] Cyrilvallez (MEMBER) · review COMMENTED · 2026-03-09
I believe we should just completely skip the init in this case rather than mark everything as initialized, then try to initialize??

## [c2905675899] winglian (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:4563 · reply to [c2904323766] · 2026-03-09
[image]
the early return leads to unstable training with NaN grad_norm on first step and 0.0 loss on second step.

## [c3915687921] winglian (COLLABORATOR) · review COMMENTED · 2026-03-09 (no text)

## [c3915997888] Cyrilvallez (MEMBER) · review APPROVED · 2026-03-09
LGTM! Thanks a lot for this!
For posterity, the issue is only the non-persistent buffers, as they are NOT gathered from rank0 (only the state_dict), so we need to go through all inits for them (but skipping everything in the state_dict)!

---
outcome: state=closed, merged=True
