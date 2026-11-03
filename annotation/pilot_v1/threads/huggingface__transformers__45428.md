# #45428 [pull_request] [`fix`] PEFT integration fixes preventing save/load & integration
https://github.com/huggingface/transformers/pull/45428  ·  opened by tomaarsen on 2026-04-14

## [body] tomaarsen (MEMBER) · opening post · 2026-04-14
# What does this PR do?



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
- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [ ] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [x] Did you write any new necessary tests?

## Details
I fixed 2 issues in this PR, and had an agent write some matching tests.

### 1. `KeyError` in PEFT loading for non-MoE conversion-mapped architectures

When integrating https://huggingface.co/nomic-ai/nomic-embed-multimodal-3b into Transformers / Sentence Transformers, I ran into the following issue:

```python
import tempfile
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

[… 33 lines of code omitted]

## [c4105048515] BenjaminBossan (MEMBER) · review APPROVED · 2026-04-14
Thanks for fixing this, Tom. Changes LGTM.

Regarding your first issue, we fixed the corresponding code in PEFT in https://github.com/huggingface/peft/pull/3127 but forgot to fix it here (for context, these functions will all be imported from PEFT eventually but we currently duplicate them in transformers for backwards compatibility).

## [c4243048271] tomaarsen (MEMBER) · issue_comment · 2026-04-14
Very glad to see that we found the same solution here. I'm looking forward to 0.19 when we can rely on peft fully for these. I'll wait for the tests to hopefully pass, and then I'll try to ping a maintainer 🤗 

- Tom Aarsen

## [c4243091343] tomaarsen (MEMBER) · issue_comment · 2026-04-14
Test failures are due to 404's on this model repository: https://huggingface.co/AI-Sweden-Models/gpt-sw3-126m
It's not indicative of any issues in transformers, and I think this should be good to go otherwise. cc @zucchini-nlp do you know if I/you can merge?

- Tom Aarsen

## [c4243283159] zucchini-nlp (MEMBER) · issue_comment · 2026-04-14
CI was fixed yesterday, rebasing will help

I don't see anything super core, and since Benjamin agrees imo can be merged. For the second point, it is also related to `base_model_prefix` so tagging @yonigozlan and @vasqu (just fyi and related to internal thread, maybe Yoni's PR can fix it without regex)

---
outcome: state=closed, merged=True
