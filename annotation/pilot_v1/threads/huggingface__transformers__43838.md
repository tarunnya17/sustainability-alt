# #43838 [pull_request] Qwen3 ASR and Forced Aligner
https://github.com/huggingface/transformers/pull/43838  ·  opened by mbtariq82 on 2026-02-08

## [body] mbtariq82 (CONTRIBUTOR) · opening post · 2026-02-08
# What does this PR do?

This PR adds Qwen3-ASR to the Transformers library.

Fixes #43837

## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [x] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case. https://github.com/huggingface/transformers/issues/43837
- [ ] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [x] Did you write any new necessary tests?


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

- audio models: @eustlb @ebezzam @vasqu

## [c3871119517] vasqu (COLLABORATOR) · issue_comment · 2026-02-09
I think this is a good one to add! cc @ebezzam @eustlb

## [c3871299306] ebezzam (CONTRIBUTOR) · issue_comment · reply to [body] · 2026-02-09
@mbtariq82 thanks for opening the PR! we're definitely interested to add the model and were planning to work on it. Could you go ahead with the rest of the model? And I can iterate with you on it.

I see you started with a modular file which is great. Below are some pointers of recent audio LM models that may help you with the other files / get an idea of our conventions:
- https://github.com/huggingface/transformers/pull/42875
- https://github.com/huggingface/transformers/pull/40290

thanks 🤗

## [c3884430267] mbtariq82 (CONTRIBUTOR) · issue_comment · reply to [c3871299306] · 2026-02-11
I'm struggling to get test_apply_chat_template_audio from test_processing_common.py to pass. 

Specifically, the final part of the test where we apply_chat_template with continue_final_message=True fails with ValueError(continue_final_message is set but the final message does not appear in the chat after applying the chat template!...)

I've verified that the chat_template is being correctly loaded from the model checkpoint: Qwen/Qwen3-ASR-0.6B. 

According to ChatGPT, the chat_template provided by Qwen is not correctly rendering the final assistant message so I think the only way to solve this is to override the apply_chat_template method and add some custom logic before calling super().apply_chat_template?

@ebezzam

## [c3885230835] ebezzam (CONTRIBUTOR) · issue_comment · reply to [c3884430267] · 2026-02-11
@mbtariq82 it's only about getting the test to pass but the model is behaving as expected, let's avoid overwriting `apply_chat_template`.

For now you can leave the test failing, and if necessary we can overwrite or even skip the test later on

## [c3885262720] ebezzam (CONTRIBUTOR) · issue_comment · 2026-02-11
when you finish the modeling, and integration tests that produce equivalent outputs to the original, e.g. for audio flamingo:
- tests like this to test single and batch inference: https://github.com/huggingface/transformers/blob/64e41924f45d37593c8297b50578f432b6f893da/tests/models/audioflamingo3/test_modeling_audioflamingo3.py#L237
- and reproducers like this for computing the expected outputs with the original model: https://gist.github.com/ebezzam/c979f0f1a2b9223fa137faf1c02022d4#file-reproducer-py

I can already take a look to give some feedback! and we can take a look at the test after that

## [c3905173948] mbtariq82 (CONTRIBUTOR) · issue_comment · 2026-02-15
So between the current version and v4.57.6, the "default" key was removed from ROPE_INIT_FUNCTIONS. Qwen3-ASR was built using v4.57.6 and the checkpoint uses the "default" key. I've changed the rope_type to "linear" in Qwen3ASRThinkerTextRotaryEmbedding for now but I'm not sure if this is correct.

I also changed the "attentions" PyTorch hooks, it was set to Qwen3ASRThinkerTextAttention which is not used at all in the base class - maybe they plan to use it in the future but I'm not sure. I've changed it to Qwen3ASRTextAttention to get the tests to pass.

I've added the entire model. All the tests are passing.

## [c2826773086] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:39 · 2026-02-19
This seems similar to [Qwen3OmniMoeAudioEncoderConfig](https://github.com/huggingface/transformers/blob/556312cd45a5e619c41b0f8adf680eab0d334324/src/transformers/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py#L29).

With modular you can simply do something like below if they are equivalent:
```python
from ..qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig


class Qwen3AsrAudioEncoderConfig(Qwen3OmniMoeAudioEncoderConfig):
  pass
```

And if you need to change default parameters or add/delete parameters, you can do something like in [this example](https://huggingface.co/docs/transformers/en/modular_transformers#config).

## [c2826781108] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:145 · 2026-02-19
Similar as above, inheriting from: https://github.com/huggingface/transformers/blob/556312cd45a5e619c41b0f8adf680eab0d334324/src/transformers/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py#L171

## [c2826782917] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:301 · 2026-02-19
https://github.com/huggingface/transformers/blob/556312cd45a5e619c41b0f8adf680eab0d334324/src/transformers/models/qwen3_omni_moe/configuration_qwen3_omni_moe.py#L351

## [c2826864671] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:492 · 2026-02-19
Can we inherit from Qwen3Omni's processor: https://github.com/huggingface/transformers/blob/556312cd45a5e619c41b0f8adf680eab0d334324/src/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py#L119

You can delete the non-audio related attributes like so:

```python
from ..qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor


class Qwen3AsrProcessor(Qwen3OmniMoeProcessor):
[… 17 lines of code omitted]

## [c2826867506] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:481 · 2026-02-19
this could also probably be copied over view modular from `src/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py`

## [c2826875001] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:656 · 2026-02-19
For all the modeling components, if you can similar find the corresponding/equivalent ones from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py

## [c3824670050] ebezzam (CONTRIBUTOR) · review COMMENTED · reply to [c3905173948] · 2026-02-19
Hi @mbtariq82 thanks for working on this integration! I'm doing a small review because I noticed you started a modular file, but aren't making full use of its functionality to generate the configuration, processing, and modeling _from existing components in Transformers_. I gave some pointers for the configuration and processing but will let you check out the rest for the modeling components.

I encourage reading this page on using modular to contribute models: https://huggingface.co/docs/transformers/en/modular_transformers

And for practical examples you can see other modular files:
- Qwen3OmniMoe, which has a lot of similarity with the ASR model (namely removing the vision modalities): https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py
- A recent Audio LM addition: https://github.com/huggingface/transformers/blob/main/src/transformers/models/glmasr/modular_glmasr.py

_Note: I think you will have to use "Asr" instead of "ASR" in your model naming because the modular script prefers camelcase_.

## [c3928594602] mbtariq82 (CONTRIBUTOR) · issue_comment · 2026-02-19
I made some big changes in the base model's forward in this commit: 0b3248d55bac707b341210eb93e3c1147b5c78cc. I also removed get_rope_index.

## [c2833826826] mbtariq82 (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:145 · reply to [c2826781108] · 2026-02-20
So to inherit from Qwen3OmniMoeTextConfig I need to delete some class-level metadata. Can you confirm?

## [c3832725713] mbtariq82 (CONTRIBUTOR) · review COMMENTED · 2026-02-20 (no text)

## [c2833874268] mbtariq82 (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:301 · reply to [c2826782917] · 2026-02-20
Also for this, we would need to remove sub_configs["vision_config"].

## [c3832782045] mbtariq82 (CONTRIBUTOR) · review COMMENTED · 2026-02-20 (no text)

## [c2834151081] mbtariq82 (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:492 · reply to [c2826864671] · 2026-02-20
Why do the reassignment?
`del self.image_token = self.tokenizer.image_token`

## [c3833110814] mbtariq82 (CONTRIBUTOR) · review COMMENTED · 2026-02-20 (no text)

## [c2836160971] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:492 · reply to [c2826864671] · 2026-02-21
Sorry that was a typo! you just need to delete (i've updated above)

## [c3835535948] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-02-21 (no text)

## [c2836169095] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:145 · reply to [c2826781108] · 2026-02-21
Yes the MOE parameters for example, and whatever else you see that isn't needed for the ASR model!

## [c3835541215] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-02-21 (no text)

## [c2836169368] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:301 · reply to [c2826782917] · 2026-02-21
yes

## [c3835541414] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-02-21 (no text)

## [c2836174698] ebezzam (CONTRIBUTOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:89 · 2026-02-21
It will actually be an "audio-text-to-text" pipeline. You can check the test of GLM ASR for help on the modeling tests: https://github.com/huggingface/transformers/blob/147b7aa040812b079f467e777a2d2e1284167de0/tests/models/glmasr/test_modeling_glmasr.py#L143

## [c3835545079] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-02-21 (no text)

## [c2836180178] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:623 · 2026-02-21
I think so, or maybe from another audio-text-to-text model.

For example, see the modular of GLM ASR: https://github.com/huggingface/transformers/blob/147b7aa040812b079f467e777a2d2e1284167de0/src/transformers/models/glmasr/modular_glmasr.py#L359

The two keys methods will be:
- `get_audio_features` for preparing the audio embeddings
- `forward` which will call `get_audio_features` and insert/combine with the text embeddings for the LLM

`Qwen3OmniMoePreTrainedModelForConditionalGeneration` has implemented `get_audio_features`: https://github.com/huggingface/transformers/blob/147b7aa040812b079f467e777a2d2e1284167de0/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L1971
which could be why it might be best to inherit from there, and delete unnecessary attributes in the `init` and the corresponding methods for other modalities. Hope that's clear

## [c3835549960] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-02-21 (no text)

## [c2848386625] mbtariq82 (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:492 · reply to [c2826864671] · 2026-02-24
I'm still a bit confused here, do we want to add image_processor and video_processor to the arguments in `Qwen3AsrProcessor.__init__`? 

If we do, `Qwen3AsrProcessor.from_pretrained` no longer works because it tries to set image_processor_type = 'WhisperImageProcessor'

## [c3849272153] mbtariq82 (CONTRIBUTOR) · review COMMENTED · 2026-02-24 (no text)

## [c2848738066] mbtariq82 (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:492 · reply to [c2826864671] · 2026-02-24
Do we definitely not want to override `ProcessorMixin.__init__`?

## [c3849664862] mbtariq82 (CONTRIBUTOR) · review COMMENTED · 2026-02-24 (no text)

## [c2854650497] mbtariq82 (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:94 · reply to [c3835549960] · 2026-02-25
As mentioned in the comments, are we certain we want to have image_processor and video_processor in the arguments? I've commented them out and overriden `ProcessorMixin.__init__ `for now because having them causes `Qwen3ASRProcessor.from_pretrained` to fail - it starts looking for things like` image_processor_type = 'WhisperImageProcessor'`.   @ebezzam

## [c3856042458] mbtariq82 (CONTRIBUTOR) · review COMMENTED · 2026-02-25 (no text)

## [c3961244829] mbtariq82 (CONTRIBUTOR) · issue_comment · 2026-02-25
Just to update everyone, I'm mostly done adding all of the classes - I'm still double checking if there are any better models to inherit from. 

The tests are all passing (ProcessorTesterMixin, ModelTesterMixin, single and batch integration tests) apart from a few (test_init_weights_can_init_buffers and test_apply_chat_template_audio). 

I've added some code (marked with ###) in order to pass some tests. Would appreciate if someone can confirm if we need them or not.

Is there anything else that needs to be done that I'm missing?

## [c2858533365] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:492 · reply to [c2826864671] · 2026-02-26
@mbtariq82 sorry for the confusion, let's do a different approach in the end. After some thought, I don't think `Qwen3OmniMoeProcessor` is a good one to copy through modular as it has `image_processor` and `video_processor` that makes modular clunky, and in the end we have to rewrite most of the methods.

Can you take a look at the AudioFlamingo3 processor: https://github.com/ebezzam/transformers/blob/bd59a876a2af101398033ec4e98f0274212adec8/src/transformers/models/audioflamingo3/processing_audioflamingo3.py#L52

Essentially, we just want to the following methods:
- `__init__`, which will also create attributes for the audio tokens
- `__call__`, in which we will replace/expand the audio tokens within the method _without_ defining a separate function like `replace_multimodal_special_tokens`.
- `apply_transcription_request`, a convenience function for transcription prompts, so that a user doesn't have to create the chat template

GLM ASR used this processor via modular: https://github.com/ebezzam/transformers/blob/bd59a876a2af101398033ec4e98f0274212adec8/src/transformers/models/glmasr/modular_glmasr.py#L52

Sorry again for the confusion and thanks for your patience!

## [c3860376628] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-02-26 (no text)

## [c2858638244] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:94 · reply to [c2854650497] · 2026-02-26
Let's switch to something like AudioFlamingo3: https://github.com/huggingface/transformers/pull/43838#discussion_r2858533365

## [c3860490834] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-02-26 (no text)

## [c2858648269] ebezzam (CONTRIBUTOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:120 · 2026-02-26
Could you uncomment?

## [c2858650804] ebezzam (CONTRIBUTOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:123 · 2026-02-26
don't forget to add 🙂

## [c2858652625] ebezzam (CONTRIBUTOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:125 · 2026-02-26
is this need even with `do_sample=False`?

## [c2858654033] ebezzam (CONTRIBUTOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:184 · 2026-02-26
Same as above

## [c2858811620] ebezzam (CONTRIBUTOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:114 · 2026-02-26
here we should be checking your new checkpoint created with your conversion script

## [c2858842309] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:257 · 2026-02-26
Can you see how I've updated `Qwen3ASRTextConfig` above for modular use? The convention is to call the `super().__init__` and do the necessary changes after that, as shown [here](https://huggingface.co/docs/transformers/en/modular_transformers#config).

## [c2858843351] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:309 · 2026-02-26
Same as [above](https://github.com/huggingface/transformers/pull/43838/changes#r2858842309)

## [c2858848480] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:342 · 2026-02-26
As mentioned [here](https://github.com/huggingface/transformers/pull/43838#discussion_r2858533365), let's switch to AudioFlamingo3

## [c3860501049] ebezzam (CONTRIBUTOR) · review COMMENTED · reply to [c3961244829] · 2026-02-26
@mbtariq82 thanks for updating the modular file! I've pushed a couple changes and made some comments on our conventions when using it.

Some other points:
- could you add a script that converts the [original checkpoint](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) and upload to the HF Hub a Transformers compatible checkpoint? Like [this](https://github.com/ebezzam/transformers/blob/bd59a876a2af101398033ec4e98f0274212adec8/src/transformers/models/audioflamingo3/convert_audioflamingo3_to_hf.py)
- could you run `make style`, this will do some code formatting that is in line with the rest of the repo. And you can also run `make check-repo` (this one takes more time since it runs more checks). Those should help you find other things to address!

Thanks for your contributions!

## [c3971332791] ebezzam (CONTRIBUTOR) · issue_comment · 2026-02-27
FYI I pushed a change for `Qwen3ASRThinkerConfig`. You are right that `Qwen3OmniMoeThinkerConfig` can't be used via modular because of the conditional block, so I copied what was done within `Qwen3OmniMoeThinkerConfig` but removed the vision parts. Although we should still check if the other attributes I tagged as `TODO` are still needed.

## [c3977558103] mbtariq82 (CONTRIBUTOR) · issue_comment · 2026-02-28
[ebezzam](https://github.com/ebezzam), I think I accidentally deleted your commit (I didn't see it and I force-pushed). Really sorry about this, do you still have it locally?

## [c3982878726] dhandhalyabhavik (NONE) · issue_comment · reply to [c3971332791] · 2026-03-02
@ebezzam any Idea when Qwen 3 ASR will be available in transformer? (merge ETA?)

## [c3983690328] mbtariq82 (CONTRIBUTOR) · issue_comment · reply to [c3982878726] · 2026-03-02
@dhandhalyabhavik hopefully by the end of this week

@ebezzam I've added everything from [here ](https://github.com/huggingface/transformers/pull/43838#pullrequestreview-3860501049)
A few things:
- reproducer.py generates slightly different tokens to the HF model (exactly one extra token in the example) 
- when loading the model from the newly generated HF checkpoint we get `thinker.lm_head.weight | MISSING |` but I believe this attribute is specific to forced-aligner which we did not port so it's fine. I can remove lm_head completely from the model if you agree?
- I changed Qwen3ASRConfig and Qwen3ASRThinkerConfig back (removed the modular inheritance), sorry again for deleting your commit but did you change Qwen3ASRThinkerConfig to inherit from PretrainedConfig or Qwen3OmniMoeThinkerConfig?

## [c3986664848] ebezzam (CONTRIBUTOR) · issue_comment · reply to [c3983690328] · 2026-03-02
@mbtariq82 thanks I'll take a look tomorrow!

@dhandhalyabhavik end of this week is probably too tight, as there are still some TODOs and we'll need to get a reviews from others in the Transformers team. hopefully we can get a completed PR by the end of this week though!

## [c4087468250] kadirnar (CONTRIBUTOR) · issue_comment · reply to [c3986664848] · 2026-03-19
@ebezzam When will you merge this pull request?

## [c4091395273] ebezzam (CONTRIBUTOR) · issue_comment · reply to [c4087468250] · 2026-03-19
@kadirnar thanks for checking in, it will still need more changes to refactor according to our audio LM conventions + reviews from others on the team. There are several other models in the pipeline so thanks for your patience 🤗 

@mbtariq82 thanks for starting the PR 🙏  I can take it on from here to standardize things; the main things (conversion and integration tests) are running, and the rest is optimization and cosmetic.

## [c3015765037] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py:125 · 2026-03-31
Accounting for the chunk size ([2*n_window](https://github.com/huggingface/transformers/blob/3a4a66278555bbe7d6a617a16191a51e1a998fe7/src/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py#L903)) rather than hardcoding, as discussed [here](https://github.com/huggingface/transformers/issues/45083#issuecomment-4154929642)

## [c3015767534] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py:233 · 2026-03-31
Fixing defaults according to the [checkpoint config](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct/blob/main/config.json#L246)

## [c3015773434] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py:2556 · 2026-03-31
also account for it in the processor, default to checkpoint config

## [c4037179880] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-03-31
@Rocketknight1 re https://github.com/huggingface/transformers/issues/45083#issuecomment-4154929642, were you thinking of something like this to account for `n_window`?

## [c4282560131] ebezzam (CONTRIBUTOR) · issue_comment · 2026-04-20
run-slow: qwen3_asr

## [c4295753435] ebezzam (CONTRIBUTOR) · issue_comment · 2026-04-22
run-slow: qwen3_asr

## [c4296080117] ebezzam (CONTRIBUTOR) · issue_comment · 2026-04-22
run-slow: qwen3_asr

## [c3124739547] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:103 · 2026-04-22
Atm using `Qwen3OmniMoeAudioEncoder`, but it isn't fullgraph compile compatible due to data-dependent ops (chunking) in its forward operation which should rather be in the feature extractor?  

https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py#L714

Either:
- Use as is
- Refactor original
- Create new like was done for GLM (with respect to Whisper's encoder)

## [c3124749897] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:129 · 2026-04-22
If we re-write the encoder, we can perhaps avoid/remove this (not done in other audio LMs)

## [c3124757639] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:33 · 2026-04-22
TODO: update checkpoints in the end

## [c3124854064] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py:2640 · 2026-04-22
Expose to be able to use AutoModel. But maybe we don't if we rewrite the audio encoder in Qwen3 ASR?

## [c3125210240] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/auto/modeling_auto.py:1847 · 2026-04-22
How about new class of forced alignment models?
- Input: audio and text
- output: timestamps

## [c3125240212] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:171 · 2026-04-22
Created private methods for pre-processing needed by both `apply_transcription_request` (preparing inputs for transcription) and `prepare_forced_aligner_inputs`

## [c3125243476] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:558 · 2026-04-22
Similar to VibeVoice ASR, different formats for decoding the ASR output

## [c3125261383] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:309 · 2026-04-22
From here, I've largely kept many of these methods from the [original codebase](https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/qwen3_forced_aligner.py#L62) to the post-processing leads to equivalent outputs. To iterate on what we keep and how so that it fits Transformers convention

## [c3125264437] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:388 · 2026-04-22
Should we keep such try-imports for Japanese and Korean?

## [c3125274338] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:676 · 2026-04-22
Similar in spirit to `apply_transcription_request`: provide a helper function so the user doesn't need to manually call `apply_chat_template`

## [c3125284221] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:754 · 2026-04-22
Things get a bit unconventional... is it ok to have this separate `decode` just for forced alignment? Or should forced alignment have its own processor, but then does that mean it should be in its own model folder?

## [c4155530011] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-04-22
Self-review on Qwen3 ASR particularities (before potential refactor of Qwen3 audio encoder)

## [c4308072982] ebezzam (CONTRIBUTOR) · issue_comment · 2026-04-23
run-slow: qwen3_asr

## [c4311366923] ebezzam (CONTRIBUTOR) · issue_comment · 2026-04-24
run-slow: qwen3_asr

## [c4312520743] ebezzam (CONTRIBUTOR) · issue_comment · 2026-04-24
run-slow: qwen3_asr

## [c3137299777] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/feature_extraction_qwen3_asr.py:35 · 2026-04-24
Essentially this is the same as Whisper's feature extractor + right-padding data-dependent ops that were done in the audio encoder of Qwen3 Omni MoE

## [c3137580528] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:43 · 2026-04-24
Not using [Qwen3OmniMoeAudioEncoderConfig](https://github.com/huggingface/transformers/blob/a66638d854ae536e0ca31e8bcfa480adfaf58284/src/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py#L128) because it has unused conv_chunksize now that it's moved into the feature extractor

## [c3138116919] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:179 · 2026-04-24
Overwrite forward so it is more compatible with torch compile and move what should be in the feature extractor!

## [c3138124576] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:490 · 2026-04-24
Classifier instead of lm head

## [c4170002353] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-04-24
@eustlb ready for review!

1. I've defined a new audio encoder for Qwen3 ASR, instead of using the one from Qwen3OmniMoe. As we saw together Qwen3OmniMoe's audio encoder had operations which should have been in the feature extractor (and which hurt torch compile speedup). I'm made a new feature extractor object for Qwen3ASR, and if you see the torch compile example in the doc, we now get a speed up of 2.5 🚀 (when using the encoder from Omni it was 1.7).
2. There are two types of model in this PR: **ASR** (audio LM approach) and a **forced aligner** (uses the audio encoder + classification layer to predict word durations). I'm sure we will iterate on the latter as it's a new type of model 😄 The processor methods can definitely be improved, I left them mainly as-is from the original to get your input on what is Transformers compatible.

Note there are some comments from a [previous self-review](https://github.com/huggingface/transformers/pull/43838#pullrequestreview-4155530011), but you should see them when you are the "Files changed" tab!

## [c3166200203] kobenaxie (NONE) · review_comment on src/transformers/models/qwen3_asr/modeling_qwen3_asr.py:138 · 2026-04-30
[Qwen3ASRAudioAttention](https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py#L472) requires `bias=True` for `k_proj`, but set `bias=False` here ?

## [c4203106781] kobenaxie (NONE) · review COMMENTED · 2026-04-30 (no text)

## [c3169204911] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modeling_qwen3_asr.py:138 · reply to [c3166200203] · 2026-04-30
thanks for pointing this out! looking into it, strange that the integration tests (between Transformers and original) still produce equivalent outputs 🤔 

```
# from this branch
CUDA_VISIBLE_DEVICES=0 RUN_SLOW=1 pytest tests/models/qwen3_asr/test_modeling_qwen3_asr.py::Qwen3ASRForConditionalGenerationIntegrationTest
```

EDIT: actually it makes sense that it doesn't affect the output because softmax is invariant to adding the same constant to every logit. Which is what happens when adding the key projection bias. Computationally, it's slightly better to have `bias=False` for less parameters/memory/operations. But don't think it makes a big difference. 

Right now this line is generated via [modular](https://github.com/mbtariq82/transformers/blob/0b932ecb3e09c6efa1f0a96c6621bf77be23a08d/src/transformers/models/qwen3_asr/modular_qwen3_asr.py#L152) by directly inheriting from Whisper. We'll see if during the review process, if we move away from Whisper's definition.

## [c4206594708] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-04-30 (no text)

## [c3169224262] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modeling_qwen3_asr.py:138 · reply to [c3166200203] · 2026-04-30
If the bias is small enough or the general distribution is not influenced, you could have just gotten lucky :D

## [c4206616869] vasqu (COLLABORATOR) · review COMMENTED · 2026-04-30 (no text)

## [c3172115320] eustlb (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modeling_qwen3_asr.py:138 · reply to [c3166200203] · 2026-05-01
Nice analysis @ebezzam! @vasqu eric is right here, bias to the key ends up adding same constant _c_i_ along hidden dim of attention logits row *i* : _c_i = (hidden state row i).[(keys bias vector).T]_

I would still follow source of truth here and maybe add a comment on why we're adding it, but it is def something we should consider in the future

## [c4209966770] eustlb (COLLABORATOR) · review COMMENTED · 2026-05-01 (no text)

## [c3172437902] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modeling_qwen3_asr.py:138 · reply to [c3166200203] · 2026-05-01
I've added a note to the modular

## [c4210387325] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-01 (no text)

## [c3172601771] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modeling_qwen3_asr.py:138 · reply to [c3166200203] · 2026-05-01
Well, it may be fine on inference but it still changes training dynamics. We should always follow what the original authors did as closely as we can to avoid different behavior.

To change the modular is 3 loc:
```
def __init__(self, config, layer_idx)
    super().__init__(config, layer_idx)
    self.k_proj = ...
```
I don't see a reason why not to do it

## [c4210558800] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-01 (no text)

## [c3172976176] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modeling_qwen3_asr.py:138 · reply to [c3166200203] · 2026-05-01
A bit more because Whisper Attention doesn't take a config, but this could do the trick

```
class Qwen3ASRAttention(WhisperAttention):
    def __init__(
        self,
        embed_dim: int,
[… 21 lines of code omitted]

## [c4210938362] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-01 (no text)

## [c3172997505] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modeling_qwen3_asr.py:138 · reply to [c3166200203] · 2026-05-01
As long as we don't change whisper itself (would be breaking), go for it :saluting_face: can also keep it simple and do the way you've shown here

Not sure about this model, but maybe we could inherit from a different attention module? E.g. if it only uses self attn. Just a thought

## [c4210957878] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-01 (no text)

## [c3193926496] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:68 · 2026-05-06
Setting all bias to True like in [original](https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py#L472)

## [c3193956826] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:167 · 2026-05-06
Changed init definition like this, to pass `config`

## [c3193959254] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modeling_qwen3_asr.py:122 · reply to [c4210957878] · 2026-05-06
Although maybe a different attention module can be used as you suggested @vasqu? as you had also put this TODO in Whisper's which gets propagated here (but maybe not relevant as it is only being used for encoder here)

## [c4234317682] ebezzam (CONTRIBUTOR) · review COMMENTED · reply to [c4210957878] · 2026-05-06
@vasqu could I get your thoughts on the attention definition? thanks 🤗

## [c3202681498] vasqu (COLLABORATOR) · review_comment on src/transformers/models/auto/modeling_auto.py:1847 · reply to [c3125210240] · 2026-05-07
Sgmt, let's make the description a bit longer tho, e.g. `forced audio to timestamp alignment`

## [c3202683557] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:16 · 2026-05-07
We do have a release date, no?

## [c3202685249] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:21 · 2026-05-07
```suggestion
```
will no longer be used as we are torch only either way

## [c3202689706] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:30 · 2026-05-07
```suggestion
A forced aligner model is also included. It can be used to timestamp a provided transcript and its audio. It uses the same audio encoder model with a classification head that predicts a word's length. This model can be used with the transcript from any ASR model (see the example below with Parakeet CTC).
```

## [c3202708010] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:33 · reply to [c3124757639] · 2026-05-07
Just fmi, I guess we had to change the configs too much but would it be possible to have a merged variant of the configs on the hub that are both viable for remote + this? This would require us to use the conversion mapping tho (for the weights)

EDIT: nvm the processors would probably kill it either way 😓

## [c3202711513] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:39 · 2026-05-07
why not have the language code here as well?

## [c3202713875] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:52 · 2026-05-07
Let's use auto model pls

## [c3202725636] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:74 · 2026-05-07
Ig the return format is a new thing?

## [c3202729386] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:74 · reply to [c3202725636] · 2026-05-07
Not sure why we dont do everything all the time and the formats are populated via dict keys

## [c3202759414] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:107 · 2026-05-07
Can we also pass the language code e.g. `zh`?

## [c3202769453] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:131 · 2026-05-07
Could I have a list of something like `language=["auto", "Chinese"]`? Maybe I do not know the full list and want it to be auto in certain instances

## [c3202772616] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:144 · 2026-05-07
Maybe we should cross ref above as well

## [c3202779141] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:245 · 2026-05-07
Also auto models

## [c3202792116] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:301 · 2026-05-07
Do we really need extra en vs zh? Imo, not super relevant and makes the docs longer than needed

## [c3202794898] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:301 · 2026-05-07
This is super cool

## [c3202801068] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:420 · 2026-05-07
Dont see why we have the batch section tbh. We could do that above instead maybe (when we first introduce it)

## [c3202815818] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:390 · 2026-05-07
Not necessarily too bad but usually we only compile the forward 🤔

## [c3202822896] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:529 · 2026-05-07
Not sure if we really want the bench scripts within, imo we should keep it simple, i.e. only the compile way, and maybe have a link for the benchmark script along the gpu that was used

## [c3202829672] vasqu (COLLABORATOR) · review_comment on docs/source/en/_toctree.yml:1134 · 2026-05-07
Hmm ok so the forced aligner and asr are within one modeling file I assume. To me it feels like it would make sense to split the implementations, especially since the aligner can be used as standalone for other ASR models

## [c3202844959] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py:97 · 2026-05-07
Imo, we can just use simple conversion mapping with renames no?

## [c3202857607] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/feature_extraction_qwen3_asr.py:35 · reply to [c3137299777] · 2026-05-07
2 possible things to consider here
1. Would it be possible to move it to the models instead to fully inherit with modular
2. Or could we at least partially inherit via modular?

## [c3202867262] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:43 · 2026-05-07
Yea at some point a general audio utils would be nice. Seems like every other audio model has its own calculation here

## [c3202883195] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:145 · 2026-05-07
```suggestion
        if output_labels:
            labels = data["input_ids"].clone()
            # Skip special tokens
            for token_id in [self.audio_token_id, ..., self.audio_eos_token_id]:
                labels[labels == token_id] = -100
            data["labels"] = labels
```

## [c3202886153] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:150 · 2026-05-07
Yea also looks like something that could be a general util

## [c3202893090] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:171 · reply to [c3125240212] · 2026-05-07
Sgtm, but yea also good for a refactor at some point for general utilities?

## [c3202901295] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:210 · 2026-05-07
AH ok so we do expect proper matches and dont really allow what I talked about in the docs with auto detect interleaved with lang hints

## [c3202908268] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:558 · reply to [c3125243476] · 2026-05-07
Ah ok so it was done before, then nvm my previous comment when it's already established

## [c3202921723] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:316 · 2026-05-07
Let's give full name for abbreviation so `... (CJK)`

## [c3202933096] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:388 · reply to [c3125264437] · 2026-05-07
lets use import utils instead of try catch --> simple check and raise error manually then

## [c3202943453] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:674 · 2026-05-07
Honestly, this looks like a simplified tokenizer in a way 👀 but you only focus on valid chars and spaces ig

## [c3202953694] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:412 · 2026-05-07
not sure if we really should have all these methods to be static. They should be private either way and imo readability will be easier with self.

## [c3202963138] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:418 · 2026-05-07
wow, feel like im back at dsa

## [c3202971936] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:418 · reply to [c3202963138] · 2026-05-07
could we
1. have a bit more comments on the steps
2. is there maybe a paper part we can reference?

## [c3202978249] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:539 · 2026-05-07
this looks like something that may better belong into the modeling code?

## [c3202981994] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:754 · reply to [c3125284221] · 2026-05-07
Yea atp, I feel more confident about splitting the models and using modular to have less repetitions on one implementation

## [c3202986612] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:43 · reply to [c3137580528] · 2026-05-07
You can ignore inherited attributes with `attr = AttributeError()`

## [c3203005124] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:115 · 2026-05-07
```suggestion
            self.audio_config = CONFIG_MAPPING["qwen3_asr_audio_encoder"]()
```
shouldnt we already have proper default values?

## [c3203008828] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:119 · 2026-05-07
here it makes sense because you want a small qwen

## [c3203020793] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:137 · 2026-05-07
Do we really? Iirc we use whisper attention, does it use cross attention because then I would avoid setting it true for now 

(encoder-decoder attentions would need different prefixes to be sorted out as the attentions slightly differ)

## [c3203032459] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:146 · 2026-05-07
Imo, all this should be one method within the module instead so we only need to copy at the end, similar to RoPE

## [c3203041124] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:167 · reply to [c3193956826] · 2026-05-07
Sounds good, so this is a cross attention module indeed

## [c3203047049] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:182 · 2026-05-07
```suggestion
        super().__init__(config=config)
        self.self_attn = Qwen3ASRAttention(config=config)
```
should work as well no?

## [c3203057299] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:171 · 2026-05-07
Might be good to make sure to sync with #45396 so that we don't have any issues

## [c3203071860] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:179 · reply to [c3138116919] · 2026-05-07
Hmm, I see. Yea I think that's what the PR I linked also does - I would like to sync if possible

We still need to decide where these computations truly should be moved - like you said feature extractor / processor or collators. Maybe you have an opinion re audio models and whether it makes more sense one way or another

## [c3203078627] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:270 · 2026-05-07
This looks fairly standard, no model we could inherit from?

## [c3203081412] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:358 · 2026-05-07
Same here re inheritance, maybe voxtral

## [c3203088491] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:470 · 2026-05-07
Yea that would be rather with a subconfig and auto then

## [c3203101148] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:508 · 2026-05-07
Ok, no seeing this imo there should not be a new auto model and we should use for token classification classes here. The extra entries can be added to the normal config imo.

## [c3203113753] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:508 · reply to [c3203101148] · 2026-05-07
We also might be able to use `GenericForTokenClassification` instead and have an attribute map to have things work out (there is some PR from Raushan re VLMs, maybe we can use that as reference here as well)

## [c3203119276] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py:1 · 2026-05-07
Fine to me but let's run slow tests for this as well before merging

## [c4245366390] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-07
Ok a bunch of comments. To avoid confusion, please read all comments before anything

For example, I thought the aligner really might be its own model but at the end its just a token classification model. I was sure to split until I saw that.

## [c3203152939] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modeling_qwen3_asr.py:122 · reply to [c3193959254] · 2026-05-07
Yea I dont think it will be easily fixed by using another module, if I understood the model correctly, then we use cross attention which wont work as smoothly re passing precomputed cu seqlens etc. (which is the todo)

But it is also not that major for the moment

## [c4245921483] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-07 (no text)

## [c3209886190] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:131 · reply to [c3202769453] · 2026-05-08
It is possible to set `[None, "Chinese"]`. I can put that as the example!

## [c4253605818] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3210021145] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:301 · reply to [c3202792116] · 2026-05-08
agree to remove

## [c4253758146] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3210032939] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:301 · reply to [c3202794898] · 2026-05-08
yes 🔥

## [c4253771959] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3210040713] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:420 · reply to [c3202801068] · 2026-05-08
agreed, batch example is enough

## [c4253781411] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3210152125] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:529 · reply to [c3202822896] · 2026-05-08
What you suggest is good! to keep it simple here

## [c4253915383] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3210305014] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:16 · reply to [c3202683557] · 2026-05-08
this was automatically created with the styles checks/fixes! I'll set it accordingly when it's merged

## [c4254108944] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3210327486] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/auto/modeling_auto.py:1847 · reply to [c3125210240] · 2026-05-08
do you mean renaming to `MODEL_FOR_FORCED_AUDIO_TO_TIMESTAMP_ALIGNMENT_MAPPING_NAMES`?

## [c4254139342] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3210400679] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/feature_extraction_qwen3_asr.py:35 · reply to [c3137299777] · 2026-05-08
> Would it be possible to move it to the models instead to fully inherit with modular

This was done in a first version but changed because:
1. Padding should be done in the feature extractor to align with audio model conventions (@eustlb can confirm)
2. the audio encoder from Qwen3 Omni MoE was using operations that were not compatible with torch.compile. This meant we had a speed-up of "only" `1.7` compared to `2.5` with the current implementation.

> Or could we at least partially inherit via modular?

This is the current approach no? With
```
class Qwen3ASREncoder(Qwen3OmniMoeAudioEncoder):
```
init is mostly copied over, but forward is overwritten as padding is now shifted to the feature extractor

## [c4254228985] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3211059745] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py:97 · reply to [c3202844959] · 2026-05-08
do you mean not defining `STATE_DICT_MAPPING_ASR` above? With Eustache, we've agreed to keep such dicts like that at the top of the script so it's clear what gets mapped (rather than having to understand from the code logic). 

Let me know if I misunderstood, otherwise I did simplify the conversion method!

## [c4255021727] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3211103121] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:43 · reply to [c3137580528] · 2026-05-08
nice didn't know that one!

## [c4255070891] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3211281553] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:146 · reply to [c3203032459] · 2026-05-08
Ok I think see what you mean. I had to update `SinusoidsPositionEmbedding` back in `src/transformers/models/qwen2_5_omni/modular_qwen2_5_omni.py` where it first gets defined and propagated through modular

## [c4255285568] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3211305097] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:137 · reply to [c3203020793] · 2026-05-08
the audio encoder doesn't use cross attention but the language model (Qwen3) does and sets this attribute to [True](https://github.com/huggingface/transformers/blob/c9de1097eed992fe205f876415e7a7cfaa06be50/src/transformers/models/qwen3/modeling_qwen3.py#L349). How should it be set for the ASR model in this case?

## [c4255315486] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3211369256] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:358 · reply to [c3203081412] · 2026-05-08
Actually Qwen3 ASR differs from the rest of the ASR models (Qwen3Audio, Voxtral, AudioFlamingo3, GLM ASR, VibeVoiceAsr) which didn't create a base model like `Qwen3ASRModel`.

@eustlb is planning to port the existing models to do this, so that they align with the vision language models like [Llava](https://github.com/huggingface/transformers/blob/c9de1097eed992fe205f876415e7a7cfaa06be50/src/transformers/models/llava/modeling_llava.py#L278) which have a base model like `LlavaModel`. 

I believe Eustache traced the history and found that when Qwen2Audio was created (first audio LM), Llava didn't create this separate model. And while new audio LMs were added based on Qwen2Audio, Llava was refactored to have this base model.

(not something you could have known 🙈)

## [c4255402555] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3211382890] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:270 · reply to [c3203078627] · 2026-05-08
Similar to [below](https://github.com/huggingface/transformers/pull/43838/changes#r3211369256), existing audio LM approaches don't have a base model like this

## [c4255420814] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3211410325] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:167 · reply to [c3193956826] · 2026-05-08
cross attention is passed over from Whisper, but we actually don't use it. Do you think we should simplify the forward? (and so pretty much defining a new attention object if we overwrite from Whisper 😛 )

## [c4255455315] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-08 (no text)

## [c3217256560] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:33 · reply to [c3124757639] · 2026-05-11
that would have been great! but yeah there's differences with the processor and the chat template :/

## [c4261753188] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-11 (no text)

## [c3217322010] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:508 · reply to [c3203101148] · 2026-05-11
Switching to token classification!

## [c4261824616] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-11 (no text)

## [c3218407603] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/_toctree.yml:1134 · reply to [c3202829672] · 2026-05-11
keeping together and using `Qwen3ASRForTokenClassification` which inherits from `GenericForTokenClassification` as discussed in [another comment](https://github.com/huggingface/transformers/pull/43838#discussion_r3203101148)

## [c4263107928] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-11 (no text)

## [c3218414873] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:305 · 2026-05-11
Switched to `GenericForTokenClassification`

## [c3218423479] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/configuration_utils.py:266 · 2026-05-11
I added this because otherwise an existing `self.num_labels` (=5000) was getting overwritten by 2 during conversion.

## [c3218433579] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/modeling_layers.py:251 · 2026-05-11
I added this attribute because Qwen's forced aligner doesn't use bias: https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py#L1086

## [c3218442702] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py:288 · 2026-05-11
setting `num_labels` for `GenericForTokenClassification`. But it gets overwritten so I added [this logic](https://github.com/huggingface/transformers/pull/43838/changes#r3218423479). Is there a better way more in line with the rest of the lib?

## [c3218459116] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:78 · 2026-05-11
Added `score_bias` and `timestamp_token_id` which correspond to token classification (and so no more separate config)

## [c4263116800] ebezzam (CONTRIBUTOR) · review COMMENTED · reply to [c4245921483] · 2026-05-11
@vasqu small self-review on using `GenericForTokenClassification`. I had to change core things, let me know if it makes sense or if there's a better way!

## [c3219465434] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:43 · reply to [c3202867262] · 2026-05-11
Agreed, or at least a way such that is doesn't have to be defined twice in modeling and processing (but maybe for another PR to update all models?)

Other such models for reference:
- Audio flamingo: [processing](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/audioflamingo3/processing_audioflamingo3.py#L90) and [modeling](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/audioflamingo3/modeling_audioflamingo3.py#L372)
- Music flamingo: [processing](https://github.com/huggingface/transformers/blob/main/src/transformers/models/musicflamingo/processing_musicflamingo.py#L102-L105) and [used](https://github.com/huggingface/transformers/blob/main/src/transformers/models/musicflamingo/modeling_musicflamingo.py#L260C44-L260C76) from audio flamingo encoder
- GLM ASR: [processing](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glmasr/processing_glmasr.py#L97) and [modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glmasr/modeling_glmasr.py#L418-L422)
- Qwen2Audio: [processing](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_audio/processing_qwen2_audio.py#L106-L107) and [modeling](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L376)

## [c4264284280] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-11 (no text)

## [c3219588913] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:150 · reply to [c3202886153] · 2026-05-11
totally as it's something done in multiple audio processors: [AudioFlamingo3](https://github.com/huggingface/transformers/blob/main/src/transformers/models/audioflamingo3/processing_audioflamingo3.py#L232-L237), [Vibevoice ASR](https://github.com/huggingface/transformers/blob/main/src/transformers/models/vibevoice_asr/processing_vibevoice_asr.py#L199-L204), [GLM ASR](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glmasr/processing_glmasr.py#L242-L247)

Here I made it a separate method because it's needed for `apply_transcription_request` and `prepare_forced_aligner_inputs`

## [c4264429286] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-11 (no text)

## [c3219598549] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:171 · reply to [c3125240212] · 2026-05-11
probably as we are seeing models with same preparation/pre-processing as mentioned above!

## [c4264441599] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-11 (no text)

## [c3219636309] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:210 · reply to [c3202901295] · 2026-05-11
You can set None for auto detection! (see [response](https://github.com/huggingface/transformers/pull/43838#discussion_r3202769453)) The main requirement is that if a language is provided the length of languages list (None allowed) should be equivalent to the lenght of audio.

## [c4264482774] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-11 (no text)

## [c3219806846] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:674 · reply to [c3202943453] · 2026-05-11
Yes my understanding is that they want to split words (or symbols for CJK) since the forced aligner (namely timestamp prediction) is happening at the word level

## [c4264679954] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-11 (no text)

## [c3219974690] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:412 · reply to [c3202953694] · 2026-05-11
or what about moving out all methods that don't need `self`?

## [c4264870416] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220016491] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:418 · reply to [c3202963138] · 2026-05-11
1. I've added more comments on the steps.

2. Their [report](https://arxiv.org/abs/2601.21337) doesn't have any mention of this timestamp "fixing" step. But yeah it's essentially a dynamic programming (or aka dynamic time warping) approach to correcting outlier timestamps

## [c4264917007] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220022257] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:754 · reply to [c3125284221] · 2026-05-11
this is old comment right? since it's just token classification do we keep these two processing paths for ASR and token classiciation?

## [c4264923302] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220113545] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:539 · reply to [c3202978249] · 2026-05-11
don't think it's needed actually so I've removed (my single and batch tests pass without it)

## [c4265028574] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220204958] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:107 · reply to [c3202759414] · 2026-05-11
I added support for this. It requires a mapping as the model expects the full language name as a hint/prompt

## [c4265136024] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220689287] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:167 · reply to [c3193956826] · 2026-05-11
If we don't use cross attention we could follow a more conservative implementation - e.g. llama is always the gold standard but depending on flags etc sometimes other modules are easier.

I would focus on having something working and then we can brainstorm if we could copy from somewhere

## [c4265695650] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220703498] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py:97 · reply to [c3202844959] · 2026-05-11
I meant https://github.com/huggingface/transformers/blob/f15fc1e8f8939cdd1d4e54c5cd409317c5a9e5ce/src/transformers/conversion_mapping.py#L118-L119

but tbh, since we need different configs for all processing related we probably need another repo either way so may as well not be worth it to transition that to the conversion mappings

## [c4265714459] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220708016] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:210 · reply to [c3202901295] · 2026-05-11
Yea I think that's fair enough, neat feature

## [c4265720581] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220713855] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:412 · reply to [c3202953694] · 2026-05-11
sgtm, could also be its own class; no strong preference tbh. Just not a fan of too many static methods

## [c4265728227] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220725639] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:137 · reply to [c3203020793] · 2026-05-11
Audio encoder is not a big deal, we can ignore that. Imo it depends on this module https://github.com/huggingface/transformers/pull/43838#discussion_r3220689287

If we can have something like llama attention then we can keep this imo. The text backbone just needs to be a simple decoder attention without encoder hidden states

## [c4265743452] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220729246] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:146 · reply to [c3203032459] · 2026-05-11
Ah yea likely a bad precedent, the omni models all have some not so nice stuff at times :sweat:

## [c4265747802] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220733206] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:358 · reply to [c3203081412] · 2026-05-11
oh wow, ok TIL

## [c4265753132] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220738853] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:305 · reply to [c3218414873] · 2026-05-11
Can we override the signature? We did something similar in qwen3_5 for text vs text + vision

## [c4265760793] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220747122] vasqu (COLLABORATOR) · review_comment on src/transformers/modeling_layers.py:251 · reply to [c3218433579] · 2026-05-11
We can handle it with using `attribute_map` in the config instead, no?

## [c4265772706] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220757668] vasqu (COLLABORATOR) · review_comment on src/transformers/modeling_layers.py:251 · reply to [c3218433579] · 2026-05-11
Ah sorry nvm, could we simplify that instead and default to True instead?, no need for the if else

## [c4265786140] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-11 (no text)

## [c3220766774] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py:288 · reply to [c3218442702] · 2026-05-11
should be fine but we can use a gettatr instead with the default

## [c4265797807] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-11 (no text)

## [c3225174522] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/convert_qwen3_asr_to_hf.py:97 · reply to [c3202844959] · 2026-05-12
alright but nice to know this feature 👍 

for reference, here are the two repos:
- new checkpoint: https://huggingface.co/bezzam/Qwen3-ASR-1.7B/tree/main
- current checkpoint: https://huggingface.co/Qwen/Qwen3-ASR-1.7B/tree/main

there are processing changes, but also cleanup in the config

## [c4270872974] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-12 (no text)

## [c3225512088] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:171 · reply to [c3203057299] · 2026-05-12
left a [review](https://github.com/huggingface/transformers/pull/45396#pullrequestreview-4270991241)

## [c4271259447] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-12 (no text)

## [c3226375871] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:305 · reply to [c3218414873] · 2026-05-12
overwriting __init__? I've pushed that

## [c4272249507] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-12 (no text)

## [c3228561506] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:159 · 2026-05-12
Differences:
- is_causal = False
- no RoPE

## [c3228564382] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modeling_qwen3_asr.py:74 · 2026-05-12
but such dead code gets generated

## [c4274786848] ebezzam (CONTRIBUTOR) · review COMMENTED · reply to [c4265797807] · 2026-05-12
@vasqu I could get to a more minimal attention definition in the modular file + inherit from LlamaAttention, but it leads to unused code (in the generated modeling) for RoPE.

Do you have other modular tricks, or know of an attention module that would be better to do modular from?

## [c3232219404] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:358 · reply to [c3203081412] · 2026-05-13
PR that will address it for existing models: https://github.com/huggingface/transformers/pull/45534

## [c4279219517] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-13 (no text)

## [c3232506198] ebezzam (CONTRIBUTOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:102 · reply to [c4265797807] · 2026-05-13
@vasqu I've updated to use the new ALM testers, but I get the following error if I don't skip:
```
    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
[… 12 lines of code omitted]
```

Running the integration tests on multiple GPUs works fine
```
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest tests/models/qwen3_asr/test_modeling_qwen3_asr.py::Qwen3ASRForConditionalGenerationIntegrationTest
```

## [c4279582326] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-13 (no text)

## [c3236364003] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:159 · reply to [c3228561506] · 2026-05-13
let's override the forward then so that we don't use anything RoPE. A bit more code but with a comment we can properly make clear what the difference is

## [c4284114982] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-13 (no text)

## [c3236371450] vasqu (COLLABORATOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:102 · reply to [c3232506198] · 2026-05-13
Could be model split percentages not working correctly. This kind of means that some splits are too big to be registered it seems but yea imo not too important, would definitely keep some comment to note that

## [c4284123843] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-13 (no text)

## [c3236412367] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:106 · 2026-05-13
let's add a small side comment that you can also pass the language code here

## [c3236434913] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/feature_extraction_qwen3_asr.py:233 · 2026-05-13
```suggestion
        input_features = padded_inputs["input_features"].transpose(2, 0, 1)
```
imo if we expect an output we can just directly access

## [c3236438898] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/feature_extraction_qwen3_asr.py:236 · 2026-05-13
Tbh, would just make a wrapper function that makes this distinction (instead of here checking for the lib)

## [c3236443044] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:29 · 2026-05-13
Not sure but might be useful as general util in the future just as heads up

## [c3236447459] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:72 · 2026-05-13
imo can be public

## [c3236456443] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:380 · 2026-05-13
noticing that the processor forces torch but we allow otherwise in the feature exctractor - any reason to maybe not also force it there to simplify a bit?

## [c3236468109] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:127 · 2026-05-13
Do we not have a utility for this? Maybe we should add this in this PR, seems something universally useful

## [c3236480279] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:76 · 2026-05-13
hmm we dont inherit but have it in the attribute map, ig we can remove it in the mapping?

## [c3236483465] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:76 · reply to [c3236480279] · 2026-05-13
similar for the other mappings it seems

## [c3236504367] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:159 · reply to [c3228561506] · 2026-05-13
Ok I think it comes from the decorator 😢 I don't mind either
1. Fully rewrite to avoid anything bad to be inherited
2. Use something else that we just use for the init - be careful about the class decorator which forces the imports

## [c3236511839] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:217 · 2026-05-13
```suggestion
```
I think we already defined that in the parent so no need to override imo

## [c3236517002] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:231 · 2026-05-13
lets comment the key diff please

## [c3236527846] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:358 · reply to [c3203081412] · 2026-05-13
Should we wait for that PR? But yea, just needs to properly communicate between each other not to miss this

## [c3236533015] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:423 · 2026-05-13
this looks standard, dont think we need those

## [c3236534575] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:426 · 2026-05-13
can return tuple?

## [c3236539520] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:499 · 2026-05-13
yea same pattern as in VLMs tbh

## [c3236545777] vasqu (COLLABORATOR) · review_comment on src/transformers/modeling_layers.py:248 · 2026-05-13
yea not necessary but also doesnt hurt to be explicit 🤗

## [c3236551247] vasqu (COLLABORATOR) · review_comment on utils/check_repo.py:286 · 2026-05-13
Hmm, yea maybe we need to adapt the tester class for this 🤔 meaning the ALM Tester etc

## [c3236557244] vasqu (COLLABORATOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:80 · 2026-05-13
We could add the for tokens classification here, no? Ig this is not really covered yet if not

## [c3236563970] vasqu (COLLABORATOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:102 · reply to [c3232506198] · 2026-05-13
Ok seeing the full context, it very likely is model split percentages, try something like `model_split_percents = [0.5, 0.7, 0.8]` or more aggressive `model_split_percents = [0.5, 0.8, 0.9]` (or just [0.5, 0.9])

## [c4284167798] vasqu (COLLABORATOR) · review APPROVED · 2026-05-13
Carefully approving already because most is super solid now, there are a few details but nothing major. Super impressive work

## [c3252368309] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:380 · reply to [c3236456443] · 2026-05-16
that's true, let's force in the feature extractor as well. For the feature extractor, I had essentially copied from Whisper and added the necessary padding, but with Eustache we've agreed that supporting numpy isn't really necessary anymore

## [c4303195547] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-16 (no text)

## [c3252390246] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:127 · reply to [c3236468109] · 2026-05-16
There is [something](https://github.com/huggingface/transformers/blob/3ef278124e47832f34406ca3ca85bc50ad8b79bb/src/transformers/audio_utils.py#L252) for handling numpy/torch arrays, but not strings (local path / URLs) which could be passed to chat templates for the audio.

How about adding something like below to `audio_utils.py` for handling chat template inputs? And indeed several other audio LMs could use this utility!

```python
def make_list_of_audio_chat_template(
    audio: list[AudioInput] | AudioInput,
) -> AudioInput:
    """
[… 18 lines of code omitted]

## [c4303225107] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-16 (no text)

## [c3255924069] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:76 · reply to [c3236480279] · 2026-05-18
I think we need to keep for `d_model`?
- `Qwen3ASRAttention` does modular from `LlamaAttention`, which uses `hidden_size`
- `Qwen3ASREncoderLayer` and `Qwen3ASREncoder` does modular from `WhisperEncoderLayer` and `Qwen3OmniMoeAudioEncoder`, which use `d_model`

The mappings `"encoder_ffn_dim": "intermediate_size"` and `"encoder_attention_heads": "num_attention_heads"` could be removed though

## [c4306778268] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-18 (no text)

## [c3256028729] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:159 · reply to [c3228561506] · 2026-05-18
Going with 1 (full rewrite) so that we can altogether remove the attribute_map between `d_model: hidden_size` and just use `d_model` since it is needed for the components that inherit from Whisper

## [c4306898376] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-18 (no text)

## [c3256067699] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:231 · reply to [c3236517002] · 2026-05-18
overwriting `self.layers` in the init is actually because of modular-generated naming.

The real diff is moving chunking in the forward to the feature extractor. I'll add a comment for that!

## [c4306938048] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-18 (no text)

## [c3256076207] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:231 · reply to [c3236517002] · 2026-05-18
Something like

```python
# Unlike `Qwen3OmniMoeAudioEncoder`, padding of chunks is moved to feature extractor
chunked = (
    input_features.view(batch_size, num_mel_bins, num_chunks, chunk_len)
    .permute(0, 2, 1, 3)
    .reshape(batch_size * num_chunks, 1, num_mel_bins, chunk_len)
)
```

## [c4306946083] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-18 (no text)

## [c3256383391] ebezzam (CONTRIBUTOR) · review_comment on utils/check_repo.py:286 · reply to [c3236551247] · 2026-05-18
This worked! https://github.com/huggingface/transformers/pull/43838#discussion_r3236557244

## [c4307280105] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-18 (no text)

## [c3256402805] ebezzam (CONTRIBUTOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:102 · reply to [c3232506198] · 2026-05-18
hmm neither of those options work :/ I'll leave a comment like below?
```python
# Getting: 'Qwen3ASRForConditionalGeneration' object has no attribute 'hf_device_map'
test_model_parallelism = False
```

## [c4307300963] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-18 (no text)

## [c3259185098] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:127 · reply to [c3236468109] · 2026-05-18
Yea sgtm!

## [c4310473538] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-18 (no text)

## [c3259190989] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:76 · reply to [c3236480279] · 2026-05-18
Gotcha, then let's remove as much as we can at least.

## [c4310479765] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-18 (no text)

## [c3259193319] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:231 · reply to [c3236517002] · 2026-05-18
Yep perfect

## [c4310482390] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-18 (no text)

## [c3259208749] vasqu (COLLABORATOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:102 · reply to [c3232506198] · 2026-05-18
Yea not ideal tbh, would be nice if you could debug a bit - might be a bad split percents with bad no split modules. But yea no worries it's not the most important part, would still add a FIXME or similar

## [c4310499508] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-18 (no text)

## [c4478134105] vasqu (COLLABORATOR) · issue_comment · reply to [c4307300963] · 2026-05-18
Before I'm forgetting it, we will probably soon merge #45493 - maybe it could be helpful to refactor slightly around that as well. But depends on the timeline etc @ebezzam

## [c4478997857] ebezzam (CONTRIBUTOR) · issue_comment · 2026-05-18
run-slow: audioflamingo3, auto, glmasr, musicflamingo, qwen2_5_omni, qwen3_asr, qwen3_omni_moe, vibevoice_asr

## [c3260037356] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:155 · reply to [c4478134105] · 2026-05-18
@vasqu I had to do something like `mbart` [here](https://github.com/huggingface/transformers/blob/aeac7140f5fe407ee9f23f30b0ed34752bae19cf/src/transformers/models/mbart/modeling_mbart.py#L517), otherwise this test was failing
```bash
# pytest tests/models/qwen3_asr/test_modeling_qwen3_asr.py::Qwen3ASRForConditionalGenerationModelTest::test_gradient_checkpointing_backward_compatibility
    def test_gradient_checkpointing_backward_compatibility(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
    
        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue
    
            config.gradient_checkpointing = True
            model = model_class(copy.deepcopy(config))
>           self.assertTrue(model.is_gradient_checkpointing)
E           AssertionError: False is not true
```
do you know what could be the cause or if there's a more elegant solution?

## [c4311444857] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-18 (no text)

## [c3260361685] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:155 · reply to [c3260037356] · 2026-05-18
Checked a bit and it fails on `Qwen3ASRForTokenClassification`, I think some hooks got messed up because you init the same model twice - once through super and once by doing it after the super call in init.

Imo you could just do
```py
class Qwen3ASRForTokenClassification(GenericForTokenClassification, Qwen3ASRPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.score = nn.Linear(config.text_config.hidden_size, config.num_labels, bias=False)
```
that should work, afaik you only wanted to change the bias here, no?

## [c4311823600] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-18 (no text)

## [c3260365447] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:155 · reply to [c3260037356] · 2026-05-18
Tbh, it would be better to make this native to the generic layer

## [c4311827546] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-18 (no text)

## [c3263228383] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:155 · reply to [c3260037356] · 2026-05-19
Adding `score_bias` to `GenericForTokenClassification` since modular doesn't unravel `GenericForTokenClassification` like when using components from other models.

## [c4315177674] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-19 (no text)

## [c3263231317] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:358 · reply to [c3203081412] · 2026-05-19
We might as it might get merged soon! @eustlb

## [c4315180526] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-19 (no text)

## [c3263238107] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py:1 · reply to [c3203119276] · 2026-05-19
No new errors introduced: https://github.com/huggingface/transformers/pull/43838#issuecomment-4479794725

## [c4315190518] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-19 (no text)

## [c3263244283] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/modeling_layers.py:248 · reply to [c3236545777] · 2026-05-19
(old comment) added `score_bias` since then to address [this](https://github.com/huggingface/transformers/pull/43838#discussion_r3260037356)

## [c4315197636] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-19 (no text)

## [c3266108658] vasqu (COLLABORATOR) · review_comment on src/transformers/modeling_layers.py:251 · 2026-05-19
Nit: I would be a bit more specific, maybe `classification_bias` or even more `token_classification_bias`? This is a bit generic so I want to make the distinction a bit clearer

Also we can just do `self.score = nn.Linear(config.get_text_config().hidden_size, config.num_labels, bias=getattr(config, "score_bias", False))` no?

## [c4318696947] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-19 (no text)

## [c3272022572] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/modeling_layers.py:251 · reply to [c3266108658] · 2026-05-20
great suggestions! going with `token_classification_bias`

## [c4326151934] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-20 (no text)

## [c3272033465] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/modeling_layers.py:251 · reply to [c3266108658] · 2026-05-20
but we should default bias to True (as was the case before for BC)

## [c4326167329] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-20 (no text)

## [c3272081619] vasqu (COLLABORATOR) · review_comment on src/transformers/modeling_layers.py:251 · reply to [c3266108658] · 2026-05-20
Oops, yes youre right wrong default :D

## [c4326225462] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-20 (no text)

## [c3281299698] eustlb (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:297 · 2026-05-21
The architecture of the encoder is basically Whisper. It stops at the line `self.ln_post`
This proj act proj is just a `multimodal_projector` and should be defined at the `Model` level

## [c3282268827] eustlb (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:250 · 2026-05-21
let's remove this and use auto docstring for `input_features`. If we want to enforce a shape, we enforce it in the code, like [here](https://github.com/huggingface/transformers/blob/10555512868d663ee1ff627e4f5c5c260114235b/src/transformers/models/whisper/modeling_whisper.py#L612-L616)

## [c3282661937] eustlb (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:273 · 2026-05-21
at 12.5 Hz, time dimension here will always be `13` here since we chunk per 1sec (100 mel frames with 10ms hop length), so why self.positional_embeddings is instantiated with max_source_positions=1500? 

This makes even more since with [this comment](https://github.com/huggingface/transformers/pull/43838/changes#r3282268827), we are sure that time is always `13` so we can just simply do conv_out +=  self.positional_embedding.positional_embedding

## [c4337058491] eustlb (COLLABORATOR) · review COMMENTED · 2026-05-21
Stopping my review early because it looks like there is an important issue: the encoder architecture should be using chunked bidirectional attention, while the current implem **does not use it**.

`qwen3 omni`, from which this encoder originates, does apply the chunking (see Qwen3OmniMoeAudioAttention)

Here, the issue originates in the original repo, and it has been reproduced while we should have catched it: the fa2 path does use cu_seqlens to enable chunked attention, but this is not the case for the sdpa path. There is indeed a [helper](https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py#L649-L666) method supposed to build the mask for it, but it's not used.

## [c3290637450] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:297 · reply to [c3281299698] · 2026-05-22
good point! it wasn't done by the original `Qwen3OmniMoeAudioEncoder` but aligns better with other audio LMs

## [c4348172338] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-05-22 (no text)

## [c4521895944] ebezzam (CONTRIBUTOR) · issue_comment · reply to [c4337058491] · 2026-05-22
@eustlb thanks for the comments! I've done a more direct modular from `Qwen3OmniMoe` to do chunked attention and to use `cu_seqlens`

## [c3298301003] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:297 · reply to [c3281299698] · 2026-05-25
It was recently refactored to follow then qwen vl series more closely. We benchmarked here and there and noticed that creating the mask is not only slower in most cases but also more memory hungry (as you materialize a mask). So now those are all aligned, see #45396 for reference

But a good point nonetheless!

## [c4356832101] vasqu (COLLABORATOR) · review COMMENTED · 2026-05-25 (no text)

## [c4718018462] MohamedAliRashad (NONE) · issue_comment · 2026-06-16
updates ? an addition like this would be cool ^^

## [c3430261966] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/feature_extraction_qwen3_asr.py:196 · reply to [c4356832101] · 2026-06-17
Added this min-length padding like the [original](https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/utils.py#L322), since @vasqu's review 

Without this, I was getting a noticeable difference on the AMI split which has many audio below 0.5 seconds. For many of those samples it was returning an empty string `""` (for 1.7B, such empty predictions went down from 359 to 192)

Note that I don't update the padding mask, as it gave worse results when I did: 9.66 vs. 9.53 for 1.7B
(maybe because original doesn't keep track and account for a padding mask?) But since the WER difference is small, I'm fine to account for it in the padding mask if that is better practice?

Also maybe adding a `min_length` option to self.pad below could be of interest?

## [c3430278251] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:193 · 2026-06-17
Also added this post-processing of the model output, notably to fix repetitions and handling the (language, text) output for different cases

Original: https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/utils.py#L403

## [c4518157167] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-06-18
## Open ASR leaderboard comparison

Same batch size for both implementations.

## 1.7B

Metric | bezzam/Qwen3-ASR-1.7B | Qwen/Qwen3-ASR-1.7B | Δ
-- | -- | -- | --
AVG WER | 5.65 | 5.59 | +0.06
 |  |  | 
AMI WER | 9.53 | 9.26 | +0.27
Earnings22 WER | 9.86 | 9.88 | -0.02
GigaSpeech WER | 7.24 | 7.25 | -0.01
LS Clean WER | 1.26 | 1.24 | +0.02
LS Other WER | 2.93 | 2.92 | +0.01
SPGISpeech WER | 2.63 | 2.58 | +0.05
VoxPopuli WER | 6.07 | 5.99 | +0.08

Notes:
- this implementation is faster (RTFx 695.81 vs 394.13), likely due to a newer and more consistent transformers? original checkpoint uses   ["transformers==4.57.6"](https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/pyproject.toml#L23) but also some unique input chunking [here](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py#L366-L383) (audio is chunked into 1200s, but all audio in the Open ASR Leaderboard is chunked way below that so probably doesn't make a diff).
- WER differences are small except on AMI (+0.27pp): [original AMI outputs](https://huggingface.co/buckets/hf-audio/asr_leaderboard_h200/tree/Qwen-Qwen3-ASR-1.7B/MODEL_Qwen-Qwen3-ASR-1.7B_DATASET_hf-audio-open-asr-leaderboard_ami_test.jsonl) vs [this version](https://huggingface.co/buckets/hf-audio/asr_leaderboard_h200/tree/bezzam-Qwen3-ASR-1.7B/MODEL_bezzam-Qwen3-ASR-1.7B_DATASET_hf-audio-open-asr-leaderboard_ami_test.jsonl).

## 0.6B

Metric | bezzam/Qwen3-ASR-0.6B | Qwen/Qwen3-ASR-0.6B | Δ
-- | -- | -- | --
AVG WER | 6.44 | 6.31 | +0.13
 |  |  | 
AMI WER | 11.31 | 10.57 | +0.74
Earnings22 WER | 10.83 | 10.72 | +0.11
GigaSpeech WER | 7.66 | 7.65 | +0.01
LS Clean WER | 1.70 | 1.69 | +0.01
LS Other WER | 4.02 | 3.97 | +0.05
SPGISpeech WER | 2.78 | 2.74 | +0.04
VoxPopuli WER | 6.76 | 6.80 | -0.04

Notes:
- this version is faster (RTFx 634.21 vs 438.74)
- AMI gap is larger here (+0.74pp vs +0.27pp for the 1.7B): [original AMI outputs](https://huggingface.co/buckets/hf-audio/asr_leaderboard_h200/tree/Qwen-Qwen3-ASR-0.6B/MODEL_Qwen-Qwen3-ASR-0.6B_DATASET_hf-audio-open-asr-leaderboard_ami_test.jsonl) vs [this version](https://huggingface.co/buckets/hf-audio/asr_leaderboard_h200/tree/bezzam-Qwen3-ASR-0.6B/MODEL_bezzam-Qwen3-ASR-0.6B_DATASET_hf-audio-open-asr-leaderboard_ami_test.jsonl)

## AMI examples (1.7B)

AMI has the biggest difference, which (from looking in the data) comes up for very short utterances like "Yeah". This implementation prefers staying quiet, while even the original model sometimes gets it more right, it often hallucinates with laughing 😆 

#### this implementation
```
{"audio_filepath": "AMI_EN2002a_H01_FEO070_0147251_0147568", "duration": 3.17, "time": 0.006370495796203613, "text": "Yeah.", "pred_text": ""}
{"audio_filepath": "AMI_EN2002d_H01_FEO072_0043872_0044186", "duration": 3.14, "time": 0.006370495796203613, "text": "Try that a few times.", "pred_text": ""}
{"audio_filepath": "AMI_EN2002a_H03_MEE071_0099650_0099878", "duration": 2.28, "time": 0.008888706207275391, "text": "Yeah.", "pred_text": ""}
{"audio_filepath": "AMI_EN2002b_H03_MEE073_0024214_0024440", "duration": 2.26, "time": 0.008888706207275391, "text": "Just yeah.", "pred_text": ""}
{"audio_filepath": "AMI_EN2002d_H02_MEE071_0023001_0023214", "duration": 2.13, "time": 0.005291684627532959, "text": "Yeah.", "pred_text": ""}
[… 11 lines of code omitted]
```

#### original
```
{"audio_filepath": "AMI_EN2002a_H01_FEO070_0147251_0147568", "duration": 3.17, "time": 0.01175140030682087, "text": "Yeah.", "pred_text": "Ha ha ha ha ha."}
{"audio_filepath": "AMI_EN2002d_H01_FEO072_0043872_0044186", "duration": 3.14, "time": 0.01175140030682087, "text": "Try that a few times.", "pred_text": "Ha ha ha ha ha ha."}
{"audio_filepath": "AMI_EN2002a_H03_MEE071_0099650_0099878", "duration": 2.28, "time": 0.010120024904608727, "text": "Yeah.", "pred_text": "Yeah."}     # <--CORRECT
{"audio_filepath": "AMI_EN2002b_H03_MEE073_0024214_0024440", "duration": 2.26, "time": 0.010120024904608727, "text": "Just yeah.", "pred_text": "Ha ha ha ha."}
{"audio_filepath": "AMI_EN2002d_H02_MEE071_0023001_0023214", "duration": 2.13, "time": 0.010229310020804405, "text": "Yeah.", "pred_text": "Hahaha."}
[… 11 lines of code omitted]
```

## [c3435876704] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/feature_extraction_qwen3_asr.py:196 · reply to [c3430261966] · 2026-06-18
I wouldn't add such options unless it becomes common practice tbh. I think it's fine for now

This is interesting, but I think keeping closer to the original performance is more important in this case then. It may be a bit weird but original implementations have priority :saluting_face:

## [c4524976361] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-18 (no text)

## [c3436808425] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/feature_extraction_qwen3_asr.py:196 · reply to [c3430261966] · 2026-06-18
agree that if we see other models doing it, then maybe we can consider adding the option

## [c4526093110] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-06-18 (no text)

## [c3469357635] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/qwen3_asr.md:421 · 2026-06-24
pretty sure we can also natively compile via generate btw so not necessarily manually like here but just a nit

## [c3469376553] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/feature_extraction_qwen3_asr.py:228 · 2026-06-24
this is weird: do we actually expect negative windows?

## [c3469389786] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:164 · 2026-06-24
```suggestion
    if text is None or not str(text).strip():
        return {"language": None, "transcription": ""}
```
nit

## [c3469393099] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:183 · 2026-06-24
```suggestion
        return {"language": None, "transcription": transcription.strip()}
```

## [c3469397795] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:217 · 2026-06-24
"O(n\u00b2)" what happened here 👀

## [c3469414624] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:754 · reply to [c3125284221] · 2026-06-24
yep old comment

## [c3469417998] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:48 · 2026-06-24
max position embeddings re was it voxtral to keep things aligned

## [c3469421500] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:112 · 2026-06-24
setter maybe as well?

## [c3469428720] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:177 · 2026-06-24
any reason we do tie = False? it is usually for bc

## [c3469435707] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:247 · 2026-06-24
Are we sure we are not inherting unnecessary parts via this, e.g. get place holder mask etc

## [c3469438713] vasqu (COLLABORATOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:282 · 2026-06-24
super kwargs (modular trick just search for **super_kwargs)

## [c3469452240] vasqu (COLLABORATOR) · review_comment on tests/models/qwen3_asr/test_feature_extraction_qwen3_asr.py:29 · 2026-06-24
pretty sure there should be some util to create a floats list / tensor no?

## [c3469456698] vasqu (COLLABORATOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:105 · 2026-06-24
here we go again 😆 dw, not super important

## [c3469459848] vasqu (COLLABORATOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:268 · 2026-06-24
we can have the base path on init/setup (similar above)

## [c3469464357] vasqu (COLLABORATOR) · review_comment on utils/check_config_attributes.py:206 · 2026-06-24
imo, we can add this attribute to the general list to ignore not model specific

## [c4564859167] vasqu (COLLABORATOR) · review APPROVED · 2026-06-24
That processor is a beast - a lot of workarounds but that is also because of the original implementation

Just went for a last round but I'm overall happy with this state. Just smaller comments on my side

## [c3473527295] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/processing_qwen3_asr.py:217 · reply to [c3469397795] · 2026-06-25
O(n^2)

## [c4569897656] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-06-25 (no text)

## [c3473536848] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:48 · reply to [c3469417998] · 2026-06-25
yep `max_position_embeddings` as we discussed [here](https://github.com/huggingface/transformers/pull/46472#issuecomment-4658811805)

## [c4569908705] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-06-25 (no text)

## [c3473610281] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:247 · reply to [c3469435707] · 2026-06-25
get_placeholder_mask is used within the forward to replace audio tokens with the embeddings

## [c4569993447] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-06-25 (no text)

## [c3473652086] ebezzam (CONTRIBUTOR) · review_comment on tests/models/qwen3_asr/test_feature_extraction_qwen3_asr.py:29 · reply to [c3469452240] · 2026-06-25
you're right, there's one [here](https://github.com/huggingface/transformers/blob/33d1bc04300c804222db4d1a1fa12ed48f89d8b6/tests/test_processing_common.py#L83)

I know some other `test_feature_extraction_MODEL.py` files that can benefit from a refactor 😏

## [c4570044053] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-06-25 (no text)

## [c3474354225] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:177 · reply to [c3469428720] · 2026-06-25
So that the attention are outputted?

Otherwise I get an error with these tests:
```
FAILED tests/models/qwen3_asr/test_modeling_qwen3_asr.py::Qwen3ASRForConditionalGenerationModelTest::test_get_audio_features_attentions - AssertionError: unexpectedly None : attentions should not be None
FAILED tests/models/qwen3_asr/test_modeling_qwen3_asr.py::Qwen3ASRForConditionalGenerationModelTest::test_get_audio_features_hidden_states - AssertionError: unexpectedly None : hidden_states should not be None
```

## [c4570880739] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-06-25 (no text)

## [c3474408868] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/qwen3_asr/modular_qwen3_asr.py:112 · reply to [c3469421500] · 2026-06-25
actually this can be removed now! I had needed it for modular usage, but didn't go with it in the end

## [c4570949213] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-06-25 (no text)

## [c3474438902] ebezzam (CONTRIBUTOR) · review_comment on tests/models/qwen3_asr/test_modeling_qwen3_asr.py:105 · reply to [c3469456698] · 2026-06-25
actually they pass locally 😃 so maybe don't need to skip. I'll try with the slow tests

## [c4570987080] ebezzam (CONTRIBUTOR) · review COMMENTED · 2026-06-25 (no text)

## [c4799493661] ebezzam (CONTRIBUTOR) · issue_comment · 2026-06-25
run-slow: audioflamingo3, glmasr, qwen2_5_omni, qwen3_asr, qwen3_omni_moe, vibevoice_asr

## [c4800454571] ebezzam (CONTRIBUTOR) · issue_comment · 2026-06-25
run-slow: audioflamingo3, qwen2_5_omni, qwen3_omni_moe

## [c4809559969] ebezzam (CONTRIBUTOR) · issue_comment · 2026-06-26
run-slow: qwen3_asr

## [c4810554084] ebezzam (CONTRIBUTOR) · issue_comment · 2026-06-26
run-slow: qwen3_asr

---
outcome: state=closed, merged=True
