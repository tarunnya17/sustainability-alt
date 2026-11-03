# #40403 [pull_request] Customizable Logit Warping Strategies for Generation #40010
https://github.com/huggingface/transformers/pull/40403  ·  opened by PamelaBha on 2025-08-23

## [body] PamelaBha (NONE) · opening post · 2025-08-23
# What does this PR do?

Feature request
Improve the generate() API by supporting custom, declarative logit warping strategies. Make it easier for users to plug in standard and custom LogitProcessors via configuration or arguments without needing to subclass or dive into internals.

Motivation
The generation module already supports rich logit manipulation through LogitProcessorList, but:

It is undocumented and hard to use for casual users
Requires advanced subclassing to customize behaviors (e.g., word bans, domain constraints)
Doesn’t support JSON- or dict-style configuration like many other parts of Transformers
Making logit warping more accessible enables:

Prompt engineers and power users to fine-tune generation behavior
Safer generation via blacklists or probability shifting
Dynamic controls like repetition penalties or temperature annealing





Fixes # (issue): 40010


## Before submitting
- [No ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [ Yes] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [ Yes] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case. https://github.com/huggingface/transformers/issues/40010
- [ Yes] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [ Yes] Did you write any new necessary tests?


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

## [c3220466050] Rocketknight1 (MEMBER) · issue_comment · 2025-08-25
cc @gante

## [c2300507445] gante (CONTRIBUTOR) · review_comment on .gitignore:196 · 2025-08-26
```suggestion
```

(let's not change this file :) I believe you can avoid having them in the commit if your conda environment is created outside the `transformers` folder in your machine)

## [c2300518194] gante (CONTRIBUTOR) · review_comment on src/transformers/generation/configuration_utils.py:61 · 2025-08-26
(this diff seems related to a merge conflict, we can probably remove these added lines)

## [c2300521612] gante (CONTRIBUTOR) · review_comment on src/transformers/generation/configuration_utils.py:570 · 2025-08-26
```suggestion
```

## [c2300528892] gante (CONTRIBUTOR) · review_comment on src/transformers/generation/configuration_utils.py:1168 · 2025-08-26
Is this needed? if `self.logit_processors` is a list of dicts, then returning it directly should be fine :)

## [c2300538360] gante (CONTRIBUTOR) · review_comment on src/transformers/generation/configuration_utils.py:1057 · 2025-08-26
Can we move this function to `utils.py`?

Goal: let's keep classes as independent as possible -- the generation config doesn't need to know about the expected format that `logit_processors` takes in `generate`. It's mostly a data storage class.

## [c2300539441] gante (CONTRIBUTOR) · review_comment on examples/pytorch/text-generation/logit_processors_example.py:1 · 2025-08-26
Very cool examples file! 👌

## [c2300548544] gante (CONTRIBUTOR) · review_comment on src/transformers/generation/logits_process.py:27 · 2025-08-26
Not needed? (`LogitsProcessor` and `LogitsProcessorList` are also defined in this file)

## [c2300549872] gante (CONTRIBUTOR) · review_comment on src/transformers/generation/logits_processor_base.py:1 · 2025-08-26
This file should not be needed 🤔 (see import comment in `logits_process.py`)

## [c2300554831] gante (CONTRIBUTOR) · review_comment on src/transformers/generation/logits_process.py:3248 · 2025-08-26
```suggestion
    """
    Registry for LogitProcessor classes to enable configuration-based instantiation with external classes.
    """
```

(adding a mention of external classes, which is quite important 🔥 )

## [c2300560619] gante (CONTRIBUTOR) · review_comment on src/transformers/generation/logits_process.py:3597 · 2025-08-26
given that we dynamically try to register a class if it is defined and not yet registered, can we skip the manual registrations below?

alternatively, can we define a function were we register all logits processors in `transformers`? e.g. `LogitProcessorRegistry.register_transformers_processors()`, which would also be called here.

(no registrations at import-time -> faster imports and fewer import-time bugs)

## [c2300567234] gante (CONTRIBUTOR) · review_comment on src/transformers/generation/logits_process.py:3338 · 2025-08-26
missing: an example in the docstring

## [c2300570036] gante (CONTRIBUTOR) · review_comment on src/transformers/generation/utils.py:1329 · 2025-08-26
merging with the custom `logits_processor` already done in `self._merge_criteria_processor_list` call a few lines above

## [c2300571757] gante (CONTRIBUTOR) · review_comment on src/transformers/generation/utils.py:1325 · 2025-08-26
Perhaps we can reuse the logic in `_merge_criteria_processor_list `?

## [c2300574932] gante (CONTRIBUTOR) · review_comment on tests/generation/test_logits_process.py:1407 · 2025-08-26
Let's have a single class for everything related to the configurable logits processors 🤗

## [c2300583754] gante (CONTRIBUTOR) · review_comment on examples/pytorch/text-generation/logit_processors_example.py:70 · 2025-08-26
all the imports in the example should be:
1 - at the top of the file
2 - `transformers` imports should be top-level imports, i.e. `from transformers import LogitsProcessor, LogitProcessorRegistry`. See the global PR comment I added about this 🤗

## [c3154943250] gante (CONTRIBUTOR) · review COMMENTED · 2025-08-26
Very cool PR! I like where this feature is going 🔥 

I've added a few comments, but I think they are simple to solve 🤗 Let me know if you have questions/counter-proposals regarding my comments.

Other global comments:
- Missing: documentation of `logits_processor` in `GenerationConfig`. I would add a link to the example file inside the docs, your examples are quite clear!
- Suggestion: in the documentation of `logits_processor` in `generate()`, I would add a mention about the new format (and a link to the examples)
- Some classes should be directly imported from the top level, i.e. we should be able to do `from transformers import LogitProcessorRegistry`. This is done by adding the classes to the outermost `__init__.py`. I think the only class we need to add there is `LogitProcessorRegistry` (we can already to `from transformers import LogitsProcessor`)

## [c3224948274] PamelaBha (NONE) · issue_comment · reply to [c3154943250] · 2025-08-26
> Very cool PR! I like where this feature is going 🔥
> 
> I've added a few comments, but I think they are simple to solve 🤗 Let me know if you have questions/counter-proposals regarding my comments.
> 
> Other global comments:
> 
> * Missing: documentation of `logits_processor` in `GenerationConfig`. I would add a link to the example file inside the docs, your examples are quite clear!
> * Suggestion: in the documentation of `logits_processor` in `generate()`, I would add a mention about the new format (and a link to the examples)
> * Some classes should be directly imported from the top level, i.e. we should be able to do `from transformers import LogitProcessorRegistry`. This is done by adding the classes to the outermost `__init__.py`. I think the only class we need to add there is `LogitProcessorRegistry` (we can already to `from transformers import LogitsProcessor`)

thanks a lot for reviewing. I will address these soon and send a new PR

## [c2305741305] PamelaBha (NONE) · review_comment on src/transformers/generation/configuration_utils.py:570 · reply to [c2300521612] · 2025-08-28
Fixed

## [c3162466317] PamelaBha (NONE) · review COMMENTED · 2025-08-28 (no text)

## [c3235323922] PamelaBha (NONE) · issue_comment · reply to [c3154943250] · 2025-08-29
@gante  this test is failing in the latest build but nothing to do with my change I believe: FAILED tests/models/gpt_oss/test_modeling_gpt_oss.py::GptOssModelTest::test_assisted_decoding_matches_greedy_search_1_same - AssertionError: False is not true

any pointers for me?

## [c3246520814] PamelaBha (NONE) · issue_comment · reply to [c3235323922] · 2025-09-02
> @gante this test is failing in the latest build but nothing to do with my change I believe: FAILED tests/models/gpt_oss/test_modeling_gpt_oss.py::GptOssModelTest::test_assisted_decoding_matches_greedy_search_1_same - AssertionError: False is not true
> 
> any pointers for me?

Ping on this @gante

## [c3271239959] gante (CONTRIBUTOR) · issue_comment · reply to [c3246520814] · 2025-09-09
@PamelaBha don't worry about tests that you believe are unrelated to your changes. That may be temporary issues in our codebase, and I can help you with them when we're ready to merge 🤗 

A request: the diff in the PR has many style-related changes, likely the outcome of an automated tool (like the one I paste below). Make sure your environment is up-to-date (`pip install -e .[dev]`), and make sure the diff only contains strictly needed changes. Minimal diff = we can better track why changes were made = easier long-term maintenance :D

[image]

_________________________ 

Ping me when the PR is ready for a re-review

## [c3276794614] PamelaBha (NONE) · issue_comment · reply to [c3271239959] · 2025-09-10
@gante can you help reopen my PR? I was trying to undo all the styling bot changes and somehow accidentally closed the PR. I am hoping you have a reopen permission

## [c3284464087] gante (CONTRIBUTOR) · issue_comment · 2025-09-12
haha done 👍

## [c2349365991] PamelaBha (NONE) · review_comment on examples/pytorch/text-generation/logit_processors_example.py:70 · reply to [c2300583754] · 2025-09-15
Resolved

## [c3225194549] PamelaBha (NONE) · review COMMENTED · 2025-09-15 (no text)

## [c2349378302] PamelaBha (NONE) · review_comment on tests/generation/test_logits_process.py:1407 · reply to [c2300574932] · 2025-09-15
Resolved

## [c3225216280] PamelaBha (NONE) · review COMMENTED · 2025-09-15 (no text)

## [c2349381249] PamelaBha (NONE) · review_comment on src/transformers/generation/logits_process.py:27 · reply to [c2300548544] · 2025-09-15
Resolved

## [c3225222054] PamelaBha (NONE) · review COMMENTED · 2025-09-15 (no text)

## [c2349382342] PamelaBha (NONE) · review_comment on src/transformers/generation/logits_process.py:3597 · reply to [c2300560619] · 2025-09-15
Resolved

## [c3225224146] PamelaBha (NONE) · review COMMENTED · 2025-09-15 (no text)

## [c3293285317] PamelaBha (NONE) · issue_comment · reply to [c3284464087] · 2025-09-15
@gante please take a look at the latest PR

## [c3303929333] PamelaBha (NONE) · issue_comment · reply to [c3284464087] · 2025-09-17
@gante ping on this

## [c3386191714] PamelaBha (NONE) · issue_comment · reply to [c3284464087] · 2025-10-09
Hi @gante - can you please help review and complete the PR?

## [c3491525655] zucchini-nlp (MEMBER) · issue_comment · reply to [c3386191714] · 2025-11-05
@PamelaBha I will take a look tomorrow, sorry for a delay. With the major v5 release, we might have missed this PR. In the meanwhile, review would be easier if you can revert style changes which aren't related. I see the configuration file has no changes expect for style and shorter line length

It'll help me spot the actual files to review

---
outcome: state=open, merged=False
