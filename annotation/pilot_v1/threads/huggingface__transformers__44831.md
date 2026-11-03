# #44831 [pull_request] Fix loading issue in Sam3
https://github.com/huggingface/transformers/pull/44831  ·  opened by zucchini-nlp on 2026-03-18

## [body] zucchini-nlp (MEMBER) · opening post · 2026-03-18
# What does this PR do?

Fix loading in Sam3 which currently doesn't match the state dict keys from checkpoint. Adding a correct base model prefix will add it to all state dict keys, making the ckpt load-able





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

## [c4083157606] zucchini-nlp (MEMBER) · issue_comment · 2026-03-18
run-slow: sam3, sam3_video, sam3_tracker_video

## [c3968798318] ydshieh (COLLABORATOR) · review APPROVED · 2026-03-18
I know you are busy, but a 1 - 3 sentences to explain what was wrong and the fix etc would be nice.

Always have a bit of context - make the history more accessbible.

Thank you for the fix!!

---
outcome: state=closed, merged=True
