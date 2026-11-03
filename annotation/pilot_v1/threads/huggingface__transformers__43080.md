# #43080 [pull_request] Add EXAONE-MoE implementations
https://github.com/huggingface/transformers/pull/43080  ·  opened by nuxlear on 2026-01-02

## [body] nuxlear (CONTRIBUTOR) · opening post · 2026-01-02
# What does this PR do?





Add EXAONE-MoE architecture for the [K-EXAONE model](https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B) released by LG AI Research.

This PR adds the modeling code of EXAONE-MoE (K-EXAONE), which is available at the fork of the LG AI Research:
https://github.com/Aim-Highest/transformers
Test code and documentation will be updated.

## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [x] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [x] Did you write any new necessary tests?


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

@ArthurZucker

## [c2683003433] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/exaone_moe.md:1 · 2026-01-12
```suggestion

## [c2683026897] vasqu (COLLABORATOR) · review_comment on src/transformers/models/auto/tokenization_auto.py:134 · 2026-01-12
```suggestion
```
I suspect that you need the tokenizers backend, please see #42894 for more details. Can you double-check? 

As a side note, this does not require any changes on the hub repo (we autodetect this). Only if you notice that you indeed need the gpt2 tokenizer, then we will need to add this to the mapping here

## [c2683034935] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone4/modular_exaone4.py:211 · 2026-01-12
Oh wow, that's a good catch :sweat_smile:

## [c2683053855] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:115 · 2026-01-12
Similar to attention layers (sliding window, full etc), we also introduced it similarly for moe layers, see https://github.com/huggingface/transformers/blob/5f70e191f8710c21e358af9ff6cf225b94fed36f/src/transformers/models/ernie4_5_vl_moe/configuration_ernie4_5_vl_moe.py#L227-L231

Can you change it to that logic?

## [c2683062535] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:128 · 2026-01-12
Unsure if we still need this

## [c2683071679] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:153 · 2026-01-12
Let's inherit from https://github.com/huggingface/transformers/blob/61d7f8a4495822604f14b621beb65f5e41bebee5/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L145-L158

Should be matching perfectly

## [c2683083322] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:161 · 2026-01-12
```suggestion
        self.weight = nn.Parameter(torch.empty((config.num_experts, config.hidden_size)))
```
Simplify a bit more?

## [c2683092439] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:171 · 2026-01-12
We no longer want modulelists for our experts but directly fused parameters. Take a look over here https://github.com/huggingface/transformers/blob/35fe3419c18f40cfd767979dda10b658a7509da1/src/transformers/conversion_mapping.py#L70-L84

The goal is to take these weights and merge them into one singular parameter. It's likely that you just need to add this conversion such as https://github.com/huggingface/transformers/blob/35fe3419c18f40cfd767979dda10b658a7509da1/src/transformers/conversion_mapping.py#L224
(changing the name ofc)

## [c2683098094] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:166 · 2026-01-12
Can we not use the super call (it's mostly the same - if something is added that is not used, you can use `del attr` or if you need to add something on top you simply define them as is `self.attr = ...`)

## [c2683104839] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:199 · 2026-01-12
Let's highlight the diff to deepseek with a comment (maybe there are more?)

## [c2683108171] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:208 · 2026-01-12
Will be moved to a separate module like deepseek here https://github.com/huggingface/transformers/blob/61d7f8a4495822604f14b621beb65f5e41bebee5/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L153-L154 (see my comment about the experts and the conversion)

This makes it compatible with fp8 as well! And we can directly inherit more from deepseek

## [c2683111424] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:235 · 2026-01-12
We can inherit the forward when we convert (looks like this is the only difference)

## [c2683119930] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:246 · 2026-01-12
Is it not exactly the same? Imo we can remove the forward and just let it be inherited

## [c2683122310] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:243 · 2026-01-12
Any reason we need this, should also be inheritable with modular, no?

## [c2683123622] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:244 · 2026-01-12
See my comment about `mlp_layer_types` (in the config)

## [c2683126977] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:288 · 2026-01-12
See https://github.com/huggingface/transformers/blob/61d7f8a4495822604f14b621beb65f5e41bebee5/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py#L546-L548

If we get the conversion working, we can compile fullgraph

## [c2683129187] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:292 · 2026-01-12
Can be added to the pretrained model instead? More of a nit

## [c2683130379] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:309 · 2026-01-12
Nit: Do we really need this? If we can, I'd like to avoid these

## [c3651643091] vasqu (COLLABORATOR) · review COMMENTED · 2026-01-12
Leaving some initial comments
- Missing tests but you already say that they will be added 
- Our Moe implementation has changed for v5 <-- this is the biggest thing to change IMO but it comes with nice benefits (fullgraph compile, boosted moe performance, fp8 support OOB etc)

## [c2688688286] nuxlear (CONTRIBUTOR) · review_comment on src/transformers/models/auto/tokenization_auto.py:134 · reply to [c2683026897] · 2026-01-14
Sure. I will check whether the tokenizer backend works well with EXAONE MoE (and EXAONE 4 as well).

## [c3658567880] nuxlear (CONTRIBUTOR) · review COMMENTED · 2026-01-14 (no text)

## [c2688696492] nuxlear (CONTRIBUTOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:171 · reply to [c2683092439] · 2026-01-14
I understood. I'll try the weight converter for it. 😃

## [c3658577076] nuxlear (CONTRIBUTOR) · review COMMENTED · 2026-01-14 (no text)

## [c2688898030] nuxlear (CONTRIBUTOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:246 · reply to [c2683119930] · 2026-01-14
Oh, the code was different from its superclass, but now it is also inheritable. I'll fix it.

## [c3658795252] nuxlear (CONTRIBUTOR) · review COMMENTED · 2026-01-14 (no text)

## [c3799529620] vasqu (COLLABORATOR) · issue_comment · reply to [c3658795252] · 2026-01-26
@nuxlear just ping me again when it's ready for review

## [c3801204850] nuxlear (CONTRIBUTOR) · issue_comment · reply to [c3799529620] · 2026-01-26
@vasqu I think it's ready for review, but `make fix-repo` does not seem to be consistent. (it adds extra tabs after overwriting the `configuration_exaone_moe.py`, which makes a CI failure)

## [c2728882513] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/exaone_moe.md:1 · reply to [c2683003433] · 2026-01-26
To replace :p

## [c2728885380] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/exaone_moe.md:135 · 2026-01-26
Wrong link?

## [c2728886848] vasqu (COLLABORATOR) · review_comment on src/transformers/models/auto/tokenization_auto.py:134 · reply to [c2683026897] · 2026-01-26
Bumping

## [c2728891074] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:45 · 2026-01-26
I found the issue with `fix-repo` - essentially it loops into first making modular applying it changes (some are wrong), we then fix the docstrings (fixes configuration file but not modular) --> modular and config are out of sync

## [c2728898833] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:280 · 2026-01-26
We need to update the init weights function https://github.com/huggingface/transformers/blob/fa5dfe06c3ceb8379d7d2a3ce108e5f6746704bf/src/transformers/models/deepseek_v3/modular_deepseek_v3.py#L312-L320

## [c2728901606] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:265 · 2026-01-26
```suggestion
```
nit: shouldnt be needed

## [c2728922956] vasqu (COLLABORATOR) · review_comment on tests/models/exaone_moe/test_modeling_exaone_moe.py:55 · 2026-01-26
This will be too big for our CI, can we create a dummy model instead? (up to 24GB Vram as it's an A10 GPU)

## [c2728924419] vasqu (COLLABORATOR) · review_comment on tests/models/exaone_moe/test_modeling_exaone_moe.py:55 · reply to [c2728922956] · 2026-01-26
The tests are nice tho!

## [c2728928912] vasqu (COLLABORATOR) · review_comment on tests/models/exaone_moe/test_modeling_exaone_moe.py:129 · 2026-01-26
Would need to change `get_model` to pass the implementation? It should load with sdpa currently this way - we can also just rename the test

## [c2728930897] vasqu (COLLABORATOR) · review_comment on tests/models/exaone_moe/test_modeling_exaone_moe.py:115 · 2026-01-26
```suggestion
            generated_ids = model.generate(**input_ids, max_new_tokens=20, do_sample=False)
```
nit: just our preferred way to do it

## [c3707564519] vasqu (COLLABORATOR) · review COMMENTED · 2026-01-26
Looks already super clean, just a few small nit + a dummy model for our CI

## [c2728934640] vasqu (COLLABORATOR) · review_comment on tests/models/exaone_moe/test_modeling_exaone_moe.py:115 · reply to [c2728930897] · 2026-01-26
below as well

## [c3707623160] vasqu (COLLABORATOR) · review COMMENTED · 2026-01-26 (no text)

## [c2729101555] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:177 · 2026-01-26
Ah missed this: this should be mlp layer types with a list of the types. (Similar to layer types for attention)

## [c3707821266] vasqu (COLLABORATOR) · review COMMENTED · 2026-01-26 (no text)

## [c2729962111] nuxlear (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/exaone_moe.md:135 · reply to [c2728885380] · 2026-01-27
Oh, I'll fix it :)

## [c3708800173] nuxlear (CONTRIBUTOR) · review COMMENTED · 2026-01-27 (no text)

## [c2730052804] nuxlear (CONTRIBUTOR) · review_comment on tests/models/exaone_moe/test_modeling_exaone_moe.py:55 · reply to [c2728922956] · 2026-01-27
Is it necessary to upload a dummy model to the HF hub?
We don't have a proper model for this, and it feels a bit awkward to upload dummy weights under our official organization. 
Would it be okay if I uploaded it under my personal account instead?

## [c3708892908] nuxlear (CONTRIBUTOR) · review COMMENTED · 2026-01-27 (no text)

## [c2731159410] vasqu (COLLABORATOR) · review_comment on tests/models/exaone_moe/test_modeling_exaone_moe.py:55 · reply to [c2728922956] · 2026-01-27
Yes sure, I can also move it to our internal testing repo afterwards

## [c3710182382] vasqu (COLLABORATOR) · review COMMENTED · 2026-01-27 (no text)

## [c2732930007] nuxlear (CONTRIBUTOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:265 · reply to [c2728901606] · 2026-01-27
When I remove that line, the converter generates `ExaoneMoeNaiveMoe` instead of using `ExaoneMoeExperts`.

## [c3712312109] nuxlear (CONTRIBUTOR) · review COMMENTED · 2026-01-27 (no text)

## [c2733351498] nuxlear (CONTRIBUTOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:177 · reply to [c2729101555] · 2026-01-27
You mean one of 'dense' and 'sparse', right?

## [c3712867410] nuxlear (CONTRIBUTOR) · review COMMENTED · 2026-01-27 (no text)

## [c2733355068] nuxlear (CONTRIBUTOR) · review_comment on tests/models/exaone_moe/test_modeling_exaone_moe.py:55 · reply to [c2728922956] · 2026-01-27
https://huggingface.co/nuxlear/EXAONE-MoE-Dummy-7B-A1B
just uploaded, but I need to do more test with it.

## [c3712871973] nuxlear (CONTRIBUTOR) · review COMMENTED · 2026-01-27 (no text)

## [c2735670189] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:177 · reply to [c2729101555] · 2026-01-28
Yes, exactly

## [c3715545087] vasqu (COLLABORATOR) · review COMMENTED · 2026-01-28 (no text)

## [c3810056814] vasqu (COLLABORATOR) · issue_comment · 2026-01-28
You can ping me when it's ready for review

## [c2735674166] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:265 · reply to [c2728901606] · 2026-01-28
Ah then no worries

## [c3715549501] vasqu (COLLABORATOR) · review COMMENTED · 2026-01-28 (no text)

## [c3812044991] nuxlear (CONTRIBUTOR) · issue_comment · 2026-01-28
Should I update the test code with a dummy model? I think everything else is ready.

## [c3812083626] vasqu (COLLABORATOR) · issue_comment · reply to [c3812044991] · 2026-01-28
> Should I update the test code with a dummy model? I think everything else is ready.

Yes, please :pray: taking a look in a second then

## [c2737437260] vasqu (COLLABORATOR) · review_comment on src/transformers/models/auto/tokenization_auto.py:134 · reply to [c2683026897] · 2026-01-28
Any update here? Can this be removed?

## [c2737438187] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/__init__.py:1 · 2026-01-28
```suggestion
# Copyright 2026 The LG AI Research and The HuggingFace Team. All rights reserved.
```

## [c3812789540] nuxlear (CONTRIBUTOR) · issue_comment · 2026-01-28
It seems the current dummy model needs to be updated, so I’ll notify you when it’s ready.

## [c2742220685] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/configuration_exaone_moe.py:91 · 2026-01-29
I'd like to avoid this if possible, and just use layertypes directly. We also start to do the same for mlp layers (moe) and it gives more flexibility with other attention flavors (e.g. linear attention (gated delta net))

## [c2742223891] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/configuration_exaone_moe.py:98 · 2026-01-29
In the same spirit to my comment before, let's remove this and only use mlp layer types directly

## [c2742231130] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/configuration_exaone_moe.py:24 · 2026-01-29
Probably needs to sync with main, I recently made the Rope mixin explicit for models that use it - can you check

E.g. https://github.com/huggingface/transformers/blob/071e178be163917777dac272e8e26525bc20db08/src/transformers/models/exaone4/configuration_exaone4.py#L25 (modular should do it automatically for you, just need to merge with main and apply modular again)

## [c2742266639] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:228 · 2026-01-29
Sorry I commented directly on the config file but should be done here ofc

## [c3717669535] vasqu (COLLABORATOR) · review APPROVED · 2026-01-29
Leaving some small last comments, imo it looks very much ready! Let's cleanup the config a tad more and wrap up the integration tests then we are good to go

Just ping me again when ready, great work

## [c3818536967] vasqu (COLLABORATOR) · issue_comment · 2026-01-29
Also sorry about the CI, it's still flaky here and there but it should be more stable on main

## [c2743071073] nuxlear (CONTRIBUTOR) · review_comment on src/transformers/models/exaone_moe/configuration_exaone_moe.py:91 · reply to [c2742220685] · 2026-01-29
I understand, and it would be better to remove them.

However, since these configs (including those below) are often used by other libraries such as `llama.cpp`, they should remain in the model’s `config.json`.

If that is acceptable, we have no reason to keep them in the config implementation. 😃

## [c3724472704] nuxlear (CONTRIBUTOR) · review COMMENTED · 2026-01-29 (no text)

## [c2743115048] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/configuration_exaone_moe.py:91 · reply to [c2742220685] · 2026-01-29
Yea, no worries not super important 👍 would be just the ideal case

## [c3724523231] vasqu (COLLABORATOR) · review COMMENTED · 2026-01-29 (no text)

## [c3825034139] nuxlear (CONTRIBUTOR) · issue_comment · reply to [c3724523231] · 2026-01-30
I’ve updated the dummy test model and the docstrings.
Could you please do a final check? @vasqu

## [c3832717288] nuxlear (CONTRIBUTOR) · issue_comment · 2026-02-02
@ArthurZucker @Rocketknight1 could you kindly review this PR?

## [c3834494136] ArthurZucker (COLLABORATOR) · issue_comment · 2026-02-02
Yes!

## [c3836280527] vasqu (COLLABORATOR) · issue_comment · 2026-02-02
run-slow: exaone4, exaone_moe

## [c2755213112] vasqu (COLLABORATOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:93 · 2026-02-02
Took this from https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B/blob/main/generation_config.json

A bit confused since the values were different would be nice if you could confirm these or if it should be the previous values, see 0e1e5bc6ca95da30982e4c9374342767b5f2425e

## [c2755581376] vasqu (COLLABORATOR) · review_comment on tests/models/exaone_moe/test_modeling_exaone_moe.py:86 · 2026-02-02
Logits don't match on our CI, I think it's a GPU diff so let me know if I should update them myself

## [c3740587622] vasqu (COLLABORATOR) · review APPROVED · 2026-02-02
Some last comments from my side, fixed a few smaller issues (checking with run slow again in a second)

## [c3836786515] vasqu (COLLABORATOR) · issue_comment · 2026-02-02
run-slow: exaone4, exaone_moe

## [c2755765330] nuxlear (CONTRIBUTOR) · review_comment on src/transformers/models/exaone_moe/modular_exaone_moe.py:93 · reply to [c2755213112] · 2026-02-02
We use 53 as the `end-of-turn` token, while 2 is used as `EOS`.
Either can be used as the default value, so you can set it to 53.

## [c3741281984] nuxlear (CONTRIBUTOR) · review COMMENTED · 2026-02-02 (no text)

## [c2755771098] nuxlear (CONTRIBUTOR) · review_comment on tests/models/exaone_moe/test_modeling_exaone_moe.py:86 · reply to [c2755581376] · 2026-02-02
I agree with that. It looks like you’ll need to update them in your CI environment.

## [c3741288110] nuxlear (CONTRIBUTOR) · review COMMENTED · 2026-02-02 (no text)

## [c2755786257] vasqu (COLLABORATOR) · review_comment on tests/models/exaone_moe/test_modeling_exaone_moe.py:86 · reply to [c2755581376] · 2026-02-02
Gotcha, let me update them tomorrow then :+1: (and also copy the repo to our internal testing)

## [c3741305769] vasqu (COLLABORATOR) · review COMMENTED · 2026-02-02 (no text)

## [c3743482743] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-02-03
Nice! 🤗

## [c3842306796] vasqu (COLLABORATOR) · issue_comment · 2026-02-03
run-slow: exaone_moe

## [c3746290009] vasqu (COLLABORATOR) · review APPROVED · reply to [c3741288110] · 2026-02-03
I updated the values and made a copy of the repo to our internal testing repos so feel free to remove your private one @nuxlear 

Merging in a second, thanks a lot for iterating :hugs:

---
outcome: state=closed, merged=True
