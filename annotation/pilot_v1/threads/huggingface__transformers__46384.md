# #46384 [pull_request] DSv4 static cache
https://github.com/huggingface/transformers/pull/46384  ·  opened by IlyasMoutawwakil on 2026-06-03

## [body] IlyasMoutawwakil (MEMBER) · opening post · 2026-06-03
# What does this PR do?

probably still ugly, but works, still needs benching. made with claude's help.


## Code Agent Policy

The Transformers repo is currently being overwhelmed by a large number of PRs and issue comments written by
code agents. We are currently bottlenecked by our ability to review and respond to them. As a result, 
**we ask that new users do not submit pure code agent PRs** at this time. 
You may use code agents in drafting or to help you diagnose issues. We'd also ask autonomous "OpenClaw"-like agents
not to open any PRs or issues for the moment.

PRs that appear to be fully agent-written will probably be closed without review, and we may block users who do this
repeatedly or maliciously. 

This is a rapidly-evolving situation that's causing significant shockwaves in the open-source community. As a result, 
this policy is likely to be updated regularly in the near future. For more information, please read [`CONTRIBUTING.md`](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).

- [x] I confirm that this is not a pure code agent PR.

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

## [c3388874925] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/modeling_utils.py:4715 · 2026-06-10
many patches need to land in torch before this can work natively, in torch 2.13 maybe, but for now torch.compiler.nested_compile_region doesn't seem worth it (the compile speedup is also not that crazy)

## [c4468475991] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-06-10 (no text)

## [c4676077168] Suraj-1207 (NONE) · issue_comment · 2026-06-11
The diff adds two new classes, `DeepseekV4HCAStaticCache` and `DeepseekV4CSAStaticCache`, which are static counterparts to `DeepseekV4HCACache` and `DeepseekV4CSACache` respectively. These classes add compressor-state API on top of pre-allocated buffers to make the whole forward dynamo-traceable. The diff also sets `_can_compile_fullgraph` to `True` in `DeepseekV4PreTrainedModel`. The changes are related to the integration of DeepSeek V4 with the Transformers library.

## [c3395544712] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/deepseek_v4/modeling_deepseek_v4.py:345 · 2026-06-11
let's find better names!

## [c3395546821] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/deepseek_v4/modeling_deepseek_v4.py:348 · 2026-06-11
this is a bitweird / looks wasteful

## [c3395552927] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/deepseek_v4/modeling_deepseek_v4.py:393 · 2026-06-11
let's make this human redable / human made :) :)

## [c3395557096] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/deepseek_v4/modeling_deepseek_v4.py:303 · 2026-06-11
we might want to add to `generation_deepseek_v4.py` its not core modeling!

## [c3395562653] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/deepseek_v4/modeling_deepseek_v4.py:479 · 2026-06-11
same let's put effort in making this human like  !

## [c4476325188] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-06-11
Nice! Can you spend some time on the `doc` so its more human! Otherwise LGTM, we can default the max compress lenght by dividing max_new_tokens by the compression rate, probably already done?

## [c3435639725] ArthurZucker (COLLABORATOR) · review_comment on tests/models/deepseek_v4/_distributed_worker.py:1 · 2026-06-18
to remove no?

## [c4524707512] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-06-18
Nice! can you put the benches here as well! 🤗

## [c3435741037] IlyasMoutawwakil (MEMBER) · review_comment on tests/models/deepseek_v4/_distributed_worker.py:1 · reply to [c3435639725] · 2026-06-18
it's more practical for testing than a long string

## [c4524821818] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-06-18 (no text)

---
outcome: state=open, merged=False
