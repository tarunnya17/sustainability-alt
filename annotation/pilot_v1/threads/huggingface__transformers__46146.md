# #46146 [pull_request] Added cosmos3 model
https://github.com/huggingface/transformers/pull/46146  ·  opened by MaciejBalaNV on 2026-05-21

## [body] MaciejBalaNV (CONTRIBUTOR) · opening post · 2026-05-21
# What does this PR do?





This PR adds a support for Cosmos3 Reasoner model (not released yet). It's a Mixture Of Transformers model, where we have a Generator and a Reasoner tower in a unified checkpoint. The Reasoner tower has Qwen3-VL architecture, so we can directly reuse it. However, we need extra code to handle the checkpoint mapping, since the final checkpoint will be in a unified Reasoner+Generator diffusers format.

Additionally, this PR fixes one issue which currently is present on top of tree - when using latest vllm and latest transformers build from source, even basic `vllm serve Qwen/Qwen3-VL-8B-Instruct` fails during dummy run. This root-cause of the bug is this commit: `ba06e3fbdf355c363ac067ebcda210017e90a852`, reverting it also fixes Qwen-VL.

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
- [x] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [x] Did you write any new necessary tests?


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

@yonigozlan for a vision model review

## [c4522903665] MaciejBalaNV (CONTRIBUTOR) · issue_comment · 2026-05-22
Converting to draft for now until we decide on the final checkpoint format

## [c3295760571] zucchini-nlp (MEMBER) · review_comment on docs/source/en/model_doc/cosmos3.md:30 · 2026-05-25
we don't need to add it in docs

Also, can you share the general pipe? IIUC we are loading a VLM and use it to embed conditioning input for a diffusion model. So is it a single checkpoint atm which holds keys for VLM and for DiT?

Usually we save the transformers in its own folder, and it has its own weights file. If that is the case, we shouldn't need to skip unexpected keys

## [c3295765452] zucchini-nlp (MEMBER) · review_comment on src/transformers/conversion_mapping.py:760 · 2026-05-25
so it is identical to qwen3-vl arch and only the conversion changes. IMO we don't need to add a new model and instead can allow the model pass its own `key_mapping` via config

So we can get `config.key_mapping` if it exists and use it for conversion

## [c3295767446] zucchini-nlp (MEMBER) · review_comment on src/transformers/processing_utils.py:888 · 2026-05-25
i don't see any changes on processor, can you explain why this is needed? 

When there is no replacement needed for a VLM, we can define `self.get_image_replacement` to return a single `image_token`. Thus it will replace an image token with itself (see Mllama for an example)

## [c4353870315] zucchini-nlp (MEMBER) · review COMMENTED · 2026-05-25
I am not sure if we need a whole new model for it since the arch is identical to qwen3-vl from what I see. The only diff seems to be the conversion mapping and `keys_to_ignore_at_load`, for which I added questions below

Since you are still deciding on ckpt, maybe we could release it by re-using a qwen3-vl model. WDYT?

## [c4535132457] MaciejBalaNV (CONTRIBUTOR) · issue_comment · 2026-05-25
The code works now with the planned Cosmos3 checkpoint format, reopening for review.

## [c3298777186] MaciejBalaNV (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/cosmos3.md:30 · reply to [c3295760571] · 2026-05-25
@zucchini-nlp The transformer model is indeed in `transformers/` subdirectory. However, the entire model is a Mixture Of Transformers, like e.g. Bagel. We release only one unified checkpoint with Understanding and Generator towers. In order to get a Qwen3-VL-like model, we need to strip off the Generator tower and some extra auxiliary layers. All of the weights are in the `transformers/` and the weights from different towers can be in the same `.safetensors` file, so we cannot simply skip some files.

## [c4357402274] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-25 (no text)

## [c3298785391] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/processing_utils.py:888 · reply to [c3295767446] · 2026-05-25
I added this as on the ToT on the day of this PR even the basic Qwen3-VL model was failing, without any changes from my side. Let me check if it's still needed, or if it was fixed.

## [c4357414840] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-25 (no text)

## [c3298798820] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/conversion_mapping.py:760 · reply to [c3295765452] · 2026-05-25
Could you please elaborate here a bit? How would we define this `key_mapping` for Qwen3-VL without defining a new `Cosmos3` model?

## [c4357432904] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-25 (no text)

## [c3298953840] MaciejBalaNV (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/cosmos3.md:30 · reply to [c3295760571] · 2026-05-25
Deleted these lines from the doc

## [c4357654662] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-25 (no text)

## [c3298956475] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/processing_utils.py:888 · reply to [c3295767446] · 2026-05-25
Double checked it now - building vllm from main from source and then building transformers from main from source still results in an error when launching "vllm serve Qwen/Qwen3-VL-8B-Instruct". This change fixes that.

## [c4357658881] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-25 (no text)

## [c3309657795] zucchini-nlp (MEMBER) · review_comment on src/transformers/processing_utils.py:888 · reply to [c3295767446] · 2026-05-27
Hmmm, I think I get it now if the error comes from vllm because they encode text/mm data separately sometimes. If the root issue is as I think, then it is not only an image or qwen-vl issue

Could you share an offline inference script that reproduces the issue?

## [c4370886764] zucchini-nlp (MEMBER) · review COMMENTED · 2026-05-27 (no text)

## [c3309667993] zucchini-nlp (MEMBER) · review_comment on src/transformers/conversion_mapping.py:760 · reply to [c3295765452] · 2026-05-27
In `config.json` in the hub add a field `key_mapping: {mapping_dict}`. The in this file, check if config has the field and use config values if present. Otherwise keep values from kwargs

https://github.com/huggingface/transformers/blob/0588858f54c8c79d28497d3ad6eac3417b716c49/src/transformers/modeling_utils.py#L4146

cc @Cyrilvallez @vasqu to get your opinions as well, as this is core change

## [c4370899493] zucchini-nlp (MEMBER) · review COMMENTED · 2026-05-27 (no text)

## [c3310663046] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/processing_utils.py:888 · reply to [c3295767446] · 2026-05-27
Sure, steps to reproduce:

With `docker.io/vllm/vllm-openai:v0.21.0` docker image and `vllm` and `transformers` repos up-to-date, do 

`cd vllm; uv venv --python 3.12 --seed --managed-python; source .venv/bin/activate; VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto; cd ../transformers; uv pip install -e .; vllm serve Qwen/Qwen3-VL-8B-Instruct`

This will throw

`(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159]   File "/root/vllm/vllm/multimodal/processing/processor.py", line 1188, in _apply_hf_processor_text_only
(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159]     prompt_ids, _, _ = self._apply_hf_processor_text_mm(
(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159]                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159]   File "/root/vllm/vllm/multimodal/processing/processor.py", line 1153, in _apply_hf_processor_text_mm
(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159]     processed_data = self._call_hf_processor(
(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159]                      ^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159]   File "/root/vllm/vllm/model_executor/models/qwen3_vl.py", line 1332, in _call_hf_processor
(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159]     processed_outputs = super()._call_hf_processor(
(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159]                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159]   File "/root/vllm/vllm/multimodal/processing/processor.py", line 1110, in _call_hf_processor
(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159]     return self.info.ctx.call_hf_processor(
(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159]   File "/root/vllm/vllm/multimodal/processing/context.py", line 276, in call_hf_processor
(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159]     raise ValueError(msg) from exc
(EngineCore pid=1941) ERROR 05-27 12:05:36 [core.py:1159] ValueError: Failed to apply Qwen3VLProcessor on data={'text': '<|vision_start|><|image_pad|><|vision_end|>'} with kwargs={'truncation': False, 'return_tensors': 'pt'}`

## [c4372186017] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-27 (no text)

## [c3310839232] vasqu (COLLABORATOR) · review_comment on src/transformers/conversion_mapping.py:760 · reply to [c3295765452] · 2026-05-27
Hmm, seeing that there is still a modular file we cannot avoid(?), does it really make sense to overcomplicate it and move it to the hub? Imo, this will make model variations based on this harder

Not familiar with the kwarg but it kind of feels like a good addition to remote code but if we have the native option, I rather prefer that tbh

## [c4372426789] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-27 (no text)

## [c3310897352] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/conversion_mapping.py:760 · reply to [c3295765452] · 2026-05-27
@zucchini-nlp 
I think this approach would require adding new field, similar to `key_mapping`, that would tell the loader which keys to ignore, so that we don't get a warning excess keys. It would also require heavy changes to `config.json`. 

Another drawback is that we already set `architectures=['Cosmos3ForConditionalGeneration']` for Cosmos3 model, since the entire `config.json` references the entire Reasoner+Generator model. It would not be compatible with the approach you're describing, since we'd need to keep it as `Qwen3VLForConditionalGeneration`, which is not really accurate and would cause issues for other integrations of the Generator, e.g. in vllm-omni.

## [c4372509167] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-27 (no text)

## [c3316822205] zucchini-nlp (MEMBER) · review_comment on src/transformers/processing_utils.py:888 · reply to [c3295767446] · 2026-05-28
Yep, reproduced and it comes from how vllm tries to cache mm-data and calls processor on text/images separately. We recently refactored processors to be more strict and error out when the multimodal placeholder tokens don't match inputs

I am talking internally to vllm maintainers to see what we can do, we'll need to fix it anyway so they can bump the next transformers release

## [c4379802474] zucchini-nlp (MEMBER) · review COMMENTED · 2026-05-28 (no text)

## [c3317631854] zucchini-nlp (MEMBER) · review_comment on src/transformers/processing_utils.py:888 · reply to [c3295767446] · 2026-05-28
Will be fixed in https://github.com/huggingface/transformers/pull/46258 :)

Lets revert this change

## [c4380772832] zucchini-nlp (MEMBER) · review COMMENTED · 2026-05-28 (no text)

## [c3317649767] zucchini-nlp (MEMBER) · review_comment on src/transformers/conversion_mapping.py:760 · reply to [c3295765452] · 2026-05-28
I discussed it internally with core maintainers and we decided to keep the current changes you have in the PR. Sadly, we don't have an option to override completely `key_mapping`, so even if we add it in config, that won't be BC

Keeping a new model just for the sake of a few cls instances seems a bit too much to me, so I will keep an eye on it and maybe we will add a better way to override renaming in next releases

## [c4380797375] zucchini-nlp (MEMBER) · review COMMENTED · 2026-05-28 (no text)

## [c3317653584] zucchini-nlp (MEMBER) · review_comment on docs/source/en/model_doc/cosmos3.md:40 · 2026-05-28
ultra nit: lets remove dtype and attn_implementation, these are default values and usually are loaded from model's config

## [c3317661045] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/auto/auto_mappings.py:692 · 2026-05-28
can we update config on the hub to use `cosmos3` as model type as model is not yet released? I'd really prefer to have consistent naming for model types, class names etc

## [c3317676126] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/cosmos3/modular_cosmos3.py:75 · 2026-05-28
nit: we can define `key_to_ignore` once and drop off `config: Cosmos3Config`, as it should be copied by modular

so smth like will be enough:

```python
class Cosmos3PreTrainedModel(Qwen3VLCosmos3PreTrainedModel):
    # no need to add lm head, users expect that lm head will be unexpected when loading a base model
    _keys_to_ignore_on_load_unexpected = _COSMOS3_DROPPED_UNIFIED_CHECKPOINT_KEYS

class Cosmos3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    pass
```

## [c3317686184] zucchini-nlp (MEMBER) · review_comment on src/transformers/conversion_mapping.py:591 · 2026-05-28
we can't support \1 and \2 yet, and instead expect a single match group.  This is better handled with `PrefixChange` imo

https://github.com/huggingface/transformers/blob/bc8f70a9f107f158b1e710a42ce7bbb9ff6e8b53/src/transformers/core_model_loading.py#L642-L650

## [c3317687011] zucchini-nlp (MEMBER) · review_comment on src/transformers/conversion_mapping.py:595 · 2026-05-28
same, can we do `PrefixChange`

## [c3317691827] zucchini-nlp (MEMBER) · review_comment on src/transformers/conversion_mapping.py:1027 · 2026-05-28
iiuc the model is not yet released, so could we make sure that configs carry the correct model type?

## [c3317694314] zucchini-nlp (MEMBER) · review_comment on tests/models/cosmos3/test_modeling_cosmos3.py:1 · 2026-05-28
we dont need any tests since it is a 100% copy from qwen3-vl

## [c3317695443] zucchini-nlp (MEMBER) · review_comment on tests/utils/test_processing_utils.py:1 · 2026-05-28
same here, no need for tests

## [c4380801460] zucchini-nlp (MEMBER) · review COMMENTED · 2026-05-28
Nice, let's merge after addressing a few nitty-picky comments :)

## [c4564123038] vasqu (COLLABORATOR) · issue_comment · 2026-05-28
Btw, we reverted some bigger PR, need to sync with main for the CI to pass again

## [c3317887864] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/models/auto/auto_mappings.py:692 · reply to [c3317661045] · 2026-05-28
The model on the hub is broader - it contains configs and weights for Reasoner+Generator towers in a Mixture of Transformers architecture. In vllm we want to handle Reasoner-only, hence this PR. I think using the same exact `model_type` here as in the hub might be confusing, since it would suggest that we are supporting the full model that's on the hub. Let me actually rename the `transformers` folders to `cosmos3_reasoner` to make this distinction more clear.

## [c4381091301] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-28 (no text)

## [c3318012513] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/auto/auto_mappings.py:692 · reply to [c3317661045] · 2026-05-28
I see what you mean, so there is another bigger config with `model_type="cosmos3"` which has all components. Then we can use a more explicit model class name `Cosmos3ReasonerModel` to be consistent

## [c4381241026] zucchini-nlp (MEMBER) · review COMMENTED · 2026-05-28 (no text)

## [c3318338758] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/processing_utils.py:888 · reply to [c3295767446] · 2026-05-28
Reverted

## [c4381654364] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-28 (no text)

## [c3318341134] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/models/auto/auto_mappings.py:692 · reply to [c3317661045] · 2026-05-28
Renamed to `Cosmos3ReasonerModel` and a directory to `cosmos3_reasoner`

## [c4381657534] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-28 (no text)

## [c3318342355] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/models/cosmos3/modular_cosmos3.py:75 · reply to [c3317676126] · 2026-05-28
Done

## [c4381659221] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-28 (no text)

## [c3318369666] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/conversion_mapping.py:591 · reply to [c3317686184] · 2026-05-28
I think `PrefixChange` cannot work out of the box - it would result in excess `model.language_model` keys for ViT keys. I simplified the regex to use only \1, since \2 wasn't needed anyway.

## [c4381691541] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-28 (no text)

## [c3318371017] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/conversion_mapping.py:595 · reply to [c3317687011] · 2026-05-28
Answered above

## [c4381693199] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-28 (no text)

## [c3318372853] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/conversion_mapping.py:1027 · reply to [c3317691827] · 2026-05-28
It was unnecessary here, removed

## [c4381695280] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-28 (no text)

## [c3318373779] MaciejBalaNV (CONTRIBUTOR) · review_comment on tests/models/cosmos3/test_modeling_cosmos3.py:1 · reply to [c3317694314] · 2026-05-28
Done

## [c4381696230] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-28 (no text)

## [c3318374532] MaciejBalaNV (CONTRIBUTOR) · review_comment on tests/utils/test_processing_utils.py:1 · reply to [c3317695443] · 2026-05-28
Done

## [c4381697140] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-28 (no text)

## [c3318377048] MaciejBalaNV (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/cosmos3.md:40 · reply to [c3317653584] · 2026-05-28
Done

## [c4381700144] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-28 (no text)

## [c3318467866] zucchini-nlp (MEMBER) · review_comment on docs/source/en/model_doc/cosmos3_reasoner.md:39 · 2026-05-28
ig you meant to delete `dtype` (as it's used from config) and keep `device_map="auto"`?

## [c3318470088] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/auto/auto_mappings.py:108 · 2026-05-28
ideally it should be mapped only once, as `("cosmos3_reasoner", "Cosmos3Config"),`, while ` ("cosmos3_omni", "cosmos3_reasoner"),` is deleted. For that we just need to be consistent everywhere in naming, including the underscores etc. Class names are camel cased, and model-type is a lower case version of it usually

## [c3318473810] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/cosmos3_reasoner/modular_cosmos3_reasoner.py:71 · 2026-05-28
not needed

## [c3318477007] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/cosmos3_reasoner/modular_cosmos3_reasoner.py:29 · 2026-05-28
we need to use the same name everywhere in classes, and in model-type

## [c4381814163] zucchini-nlp (MEMBER) · review COMMENTED · 2026-05-28 (no text)

## [c3318785267] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/models/cosmos3_reasoner/modular_cosmos3_reasoner.py:29 · reply to [c3318477007] · 2026-05-28
Done

## [c4382217360] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-28 (no text)

## [c3318786426] MaciejBalaNV (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/cosmos3_reasoner.md:39 · reply to [c3318467866] · 2026-05-28
Done

## [c4382218717] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-28 (no text)

## [c3318843596] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/models/cosmos3_reasoner/modular_cosmos3_reasoner.py:71 · reply to [c3318473810] · 2026-05-28
Without this wouldn't we only have `Qwen3VLModel` class, so e.g. `AutoModel.from_pretrained("nvidia/Cosmos3-Nano")` would return `Qwen3VLModel`? Can we keep this mapping so that it's clear we're using a Cosmos model?

I also renamed it to `Cosmos3ReasonerModel`

## [c4382294777] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-28 (no text)

## [c3318854953] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/models/auto/auto_mappings.py:108 · reply to [c3318470088] · 2026-05-28
I've done some extra name unification to include `Reasoner` everywhere. However, the `model_type` from `config.json` is still going to be `cosmos3_omni`, as it's a unified Generator+Reasoner checkpoint, as explained above. Not sure if we can do `("cosmos3_reasoner", "Cosmos3ReasonerConfig")` mapping in this case? We need to keep `("cosmos3_omni", "Cosmos3ReasonerConfig"),`, right?

## [c4382308753] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-05-28 (no text)

## [c3319092110] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/cosmos3_reasoner/modular_cosmos3_reasoner.py:71 · reply to [c3318473810] · 2026-05-28
hmm interesting, modular should copy all the dependencies and rename. I will check tmrw

## [c4382603230] zucchini-nlp (MEMBER) · review COMMENTED · 2026-05-28 (no text)

## [c3319110916] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/auto/auto_mappings.py:108 · reply to [c3318470088] · 2026-05-28
I want to push for consistent naming a bit more

Not clear to me, why the model type on the hub cannot be changed, or is there another class in remote code that maps from `cosmos3_omni` to custom module?

IIUC model is not released, there is no remote code at hand, and all the loading will happen via transformers. The config maps to Reasoner class and loads qwen3-vl model

How the generator is loaded, since it is also identical to qwen3-vl in impl? Do we load it via transformers or is it not released for usage?

## [c4382628308] zucchini-nlp (MEMBER) · review COMMENTED · 2026-05-28 (no text)

## [c3334392311] MaciejBalaNV (CONTRIBUTOR) · review_comment on src/transformers/models/auto/auto_mappings.py:108 · reply to [c3318470088] · 2026-06-01
@zucchini-nlp 
The model is released now under https://huggingface.co/nvidia/Cosmos3-Nano. Perhaps the mapping will be a bit more clear now. We also integrate this model into multiple different frameworks, like vllm-omni, TRT-LLM, SGLang or diffusers (even merged to main already in case of diffusers). Generator is loaded separately into a newly defined model instance in those framework.

For transformers, we only want to add the Reasoner, since Generator is handled in diffusers. It's identical to Qwen3-VL, hence this PR. Please let me know if you have any more concerns regarding naming when it comes to transformers mapping.

## [c4401473795] MaciejBalaNV (CONTRIBUTOR) · review COMMENTED · 2026-06-01 (no text)

## [c4600603162] MaciejBalaNV (CONTRIBUTOR) · issue_comment · reply to [c4382628308] · 2026-06-02
The CI is fixed now, the test failing is unrelated to this PR. @zucchini-nlp Could you please take a look again?

## [c4622980534] zucchini-nlp (MEMBER) · issue_comment · 2026-06-04
Yep, dw, trying to merge asap today !

Sorry, had to take liberty and rename classes to match the library philosophy. Qwen-VL changes are not much related and long due, but we'll need them for Cosmos3

Tested with integration tests that I added for single and batched inference, loading works and the output is sensible

## [c4428617970] zucchini-nlp (MEMBER) · review APPROVED · 2026-06-04 (no text)

## [c4623263850] zucchini-nlp (MEMBER) · issue_comment · 2026-06-04
run-slow: cosmos3_omni, qwen3_5, qwen3_5_moe, qwen3_omni_moe, qwen3_vl, qwen3_vl_moe

## [c4630163618] zucchini-nlp (MEMBER) · issue_comment · 2026-06-05
run-slow: cosmos3_omni

## [c4631704456] zucchini-nlp (MEMBER) · issue_comment · 2026-06-05
run-slow: cosmos3_omni

## [c4632332638] zucchini-nlp (MEMBER) · issue_comment · 2026-06-05
run-slow: cosmos3_omni

## [c4632477366] zucchini-nlp (MEMBER) · issue_comment · 2026-06-05
Thanks god, slow CI decided to turn green. Merging...

## [c4632857668] ydshieh (COLLABORATOR) · issue_comment · 2026-06-05
> FAILED tests/models/auto/test_modeling_auto.py::AutoModelTest::test_from_pretrained_dynamic_model_distant - AssertionError: <class 'transformers_modules.hf_hyphen_internal_hyphen_testing.test_dynamic_model.2efddce40dddaccd37bae208c3c7ca66dbedf68a.modeling.NewModel'> is not <class 'transformers_modules.hf_hyphen_internal_hyphen_testing.test_dynamic_model.c8ae8a2e5845fc3ad4e5237e4346e5a9ba1e87d2.modeling.NewModel'>

i don't think it's relevant, ping me if you want me to force merge.

## [c4632949856] zucchini-nlp (MEMBER) · issue_comment · 2026-06-05
Yep, the CI is fine, it's just me. The model has sampling on by default and I didn't get at first why the CI keeps complaining after I adjust expectations with runners 😅

---
outcome: state=closed, merged=True
