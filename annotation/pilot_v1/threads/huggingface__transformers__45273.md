# #45273 [pull_request] fix: liger unnecessarily materializes logits in VRAM during eval, causing OOM
https://github.com/huggingface/transformers/pull/45273  ·  opened by excepshenal on 2026-04-06

## [body] excepshenal (CONTRIBUTOR) · opening post · 2026-04-06
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=45273)

# What does this PR do?

Liger kernel unnecessarily materializes logits in VRAM during eval with `prediction_loss_only=True`, causing OOM. We explicitly tell Liger to `skip_logits`.



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

- [ ] I confirm that this is not a pure code agent PR.

## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [ ] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [x] Did you write any new necessary tests?
No, but I used this in my own training setup and observed the fix.


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

## [c4207022385] Rocketknight1 (MEMBER) · issue_comment · 2026-04-08
cc @sunmarc but not sure trainer is the place for this?

## [c4215459708] SunMarc (MEMBER) · issue_comment · 2026-04-09
I think we can add it, there were another PR that also wanted to have this. https://github.com/huggingface/transformers/pull/44981. I'll probably merge the other one as it was there before

## [c4995106906] originalgremlin (NONE) · issue_comment · 2026-07-16
is there any intention to merge this PR?  or [44981](https://github.com/huggingface/transformers/pull/44981)?  i've been maintaining a patched version of transformers using liger on dozens of models and haven't had any problems.  would be better to stop maintaining that patch and use the official version directly, though.

## [c4746347310] SunMarc (MEMBER) · review APPROVED · reply to [c4995106906] · 2026-07-21
Thanks ! I will merge this one then @originalgremlin

---
outcome: state=closed, merged=True
