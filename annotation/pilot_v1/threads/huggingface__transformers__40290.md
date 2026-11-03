# #40290 [pull_request] [models] Add AudioFlamingo3 integration
https://github.com/huggingface/transformers/pull/40290  ·  opened by lashahub on 2025-08-19

## [body] lashahub (CONTRIBUTOR) · opening post · 2025-08-19
This PR adds support for **AudioFlamingo3 (AF3)** — NVIDIA’s open large audio language model capable of reasoning over speech, sounds, and music.

- **Paper**: [Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models](https://arxiv.org/abs/2507.08128)
- **Model Weights**: [nvidia/audio-flamingo-3 on Hugging Face](https://huggingface.co/nvidia/audio-flamingo-3)
- **Original Code**: [NVIDIA Audio Flamingo GitHub (branch: audio_flamingo_3)](https://github.com/NVIDIA/audio-flamingo/tree/audio_flamingo_3)

It introduces the following components:

- `AudioFlamingo3` model class
- `AudioFlamingo3Processor` for preprocessing text + audio
- Configuration, modeling, and processing utilities
- Example usage

With this integration, AF3 can be loaded directly from the Hugging Face Hub:

```python
from transformers import AudioFlamingo3Processor, AudioFlamingo3

processor = AudioFlamingo3Processor.from_pretrained("nvidia/audio-flamingo-3")
model = AudioFlamingo3.from_pretrained("nvidia/audio-flamingo-3")
[… 13 lines of code omitted]

## [c2291435090] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:63 · 2025-08-21
is [this](https://huggingface.co/nvidia/audio-flamingo-3) the intended HF link or a new one will be made?

## [c2291443548] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:157 · 2025-08-21
Update with correct/public link

## [c2291583690] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/__init__.py:17 · 2025-08-21
Can you add copyright and license (e.g. with `Copyright 2025 Nvidia and The HuggingFace Team...`)?

And you can make sure it is correct on the other relevant files as well. Thanks!

## [c2293977647] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:16 · 2025-08-22
Can such parameters be put in the model config?

## [c2293999916] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:714 · 2025-08-22
Let's keep the configs inside `configuration_audioflamingo3.py`

## [c2294065146] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:211 · 2025-08-22
Let's avoid bare `except` so we can identify the source of potential errors, and are remove as much as possible from inside the `try` statement

## [c2294076189] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:221 · 2025-08-22
We typically don't pass audio paths, but rather audio array themselves. Could you change to that?

For example, with a dummy dataset we often use for testing (see here).
```python
from datasets import Audio, load_dataset

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
#  resample
ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
# get audio array
audio_array = librispeech_dummy[0]["audio"]["array"]

```

## [c2294076925] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:127 · 2025-08-22
You can remove this method as we will only pass audio arrays to the processor

## [c2294079797] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:10 · 2025-08-22
we can remove this dependency as audio will be loaded outside the processor

## [c2294127051] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:29 · 2025-08-22
no need to overwrite `from_pretrained`. If the configuration of the processor is correctly set, it will pull the necessary models.

For example with Dia:
- conversion script which defines configuration of processor and creates model files ([example](https://github.com/huggingface/transformers/blob/29ddcacea3ad9d3cdf6c5d8e51d1d39cbc5e7dfa/src/transformers/models/dia/convert_dia_to_hf.py#L191))
- resulting [model files](https://huggingface.co/nari-labs/Dia-1.6B-0626/tree/main) which have the specified config (there is a `audio_tokenizer_config.json` and `preprocessor_config.json` with corresponding config)

## [c2294127777] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:25 · 2025-08-22
remove `None` as these are required parameters

## [c2294130885] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:20 · 2025-08-22
Rename to `AudioFlamingo3Config`?

## [c2294131315] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:53 · 2025-08-22
Comments like these can be removed

## [c2294139306] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:1 · 2025-08-22
I see that several components are copied from Whisper. Eventually we may want to create a modular script to generate `modeling_audioflamingo3.py` from all components that already exist within Transformers.

Tutorial: https://huggingface.co/docs/transformers/en/modular_transformers

But for now we can focus on reorganizing components between the Model and the Processor

## [c2294144530] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:731 · 2025-08-22
Let's rather store the config of each component inside `AudioFlamingo3Config` so that we can use each of those configs with `AutoModel.from_config`. Like in [Llava](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py#L139-L142).

## [c2294148195] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:732 · 2025-08-22
Let's move this inside the processor. Is this an audio tokenizer? or for text?

## [c2294175144] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:754 · 2025-08-22
I guess this is a sort of audio tokenizer? Let's also put this inside the processor. You can see [here](https://github.com/huggingface/transformers/blob/29ddcacea3ad9d3cdf6c5d8e51d1d39cbc5e7dfa/src/transformers/models/dia/processing_dia.py#L74) how Dia also uses an audio tokenizer

## [c2294200934] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:31 · 2025-08-22
this seems to be an audio tokenizer? If so, let's remove and this will be handled within the processor

## [c2294202511] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:50 · 2025-08-22
Can we move this configuration into the one above?

## [c2294209922] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:1 · 2025-08-22
We should only need a single config `AudioFlamingo3Config` like for [Llava](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/configuration_llava.py), _maybe_ a second for `AudioFlamingo3EncoderConfig` but I think that even can be encapsulated inside `AudioFlamingo3Config`.

## [c2294220950] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:429 · 2025-08-22
We should eventually remove this when the configurations are correctly set

## [c2294221309] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:469 · 2025-08-22
this should also be eventually removed

## [c2294222401] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:519 · 2025-08-22
this can also be removed eventually

## [c2294226219] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:128 · 2025-08-22
combine with above into single `AudioFlamingo3Config`

## [c2294227089] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:717 · 2025-08-22
like [Llava](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py), we will have separate models for `AudioFlamingo3Model` and `AudioFlamingo3ForConditionalGeneration`

## [c3141242705] ebezzam (CONTRIBUTOR) · review CHANGES_REQUESTED · reply to [body] · 2025-08-25
Thanks @lashahub for your PR! This is very a exciting model to add to the Transformers library!

I see that you've taken inspiration from [Llava](https://github.com/huggingface/transformers/tree/main/src/transformers/models/llava), which makes sense as you combine modules of different modalities.

Most of my comments are about rearranging your modules so that they fit the Transformers convention, such that it will be more convenient for others to use and to test your new model. To that end, [Llava](https://github.com/huggingface/transformers/pull/27662/files)'s and [this PR](https://github.com/huggingface/transformers/pull/38405/files) (of another audio model) might serve as a useful example to see what files will be added/modified.

Below are my suggested steps.

## 1. Refactoring / reorganizing current files according to Transformers convention

You can take inspiration from the above models for refactoring your `configuration`, `modeling` and `processing` files, specifically:
- Consolidating your configurations. From what I see you may need only one config like in [Llava](https://github.com/huggingface/transformers/blob/f0e87b436d8bd4dd4225da235ac8f988d4bbe845/src/transformers/models/llava/configuration_llava.py#L24) or two like in [Dia](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dia/configuration_dia.py) for `AudioFlamingo3Config` and `AudioFlamingo3EncoderConfig`.
- Processor: you can take inspiration from [Dia](https://github.com/huggingface/transformers/blob/f0e87b436d8bd4dd4225da235ac8f988d4bbe845/src/transformers/models/dia/processing_dia.py#L62) to group your feature extractor, text tokenizer, and audio tokenizer into a single component. Loading pre-trained feature extractors and LLMs can be directly handled by the processor **without** you (or the user) having to manually load the corresponding weights (like you've done below).
```python
# Load components
fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
tok = AutoTokenizer.from_pretrained(
    llm_dir,
    padding_side="right",
[… 17 lines of code omitted]
```

## [c3241425200] lashahub (CONTRIBUTOR) · issue_comment · 2025-09-01
**Update (ready for another pass) - [Qwen2Audio](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_audio) style**

* **HF-style split**: `AudioFlamingo3ForConditionalGeneration` (LLM + projector) and a lean `AudioFlamingo3Encoder` (audio backbone). No Llava wrappers; no tokenizer in modeling.
* **Processor = arrays in**: accepts NumPy audio arrays, does 30s windowing, builds Whisper-compatible masks, and inserts `<sound>` tokens. No file I/O deps.
* **Configs**: `AudioFlamingo3Config` (+ small `AudioFlamingo3EncoderConfig`). Media tokens, padding/eos/bos IDs, `model_max_length`, etc., read from the config and used in modeling.
* **Long audio**: attention masks + windowing restore support beyond 30s (Whisper stride logic kept), deterministic gen by default (`do_sample=False`).
* **Cleaner surfaces**: removed bare `except`, no custom `from_pretrained` in processor, minimal projector inlined.
* **Weights (temporary)**: `lashahub/audio-flamingo-3` (will switch to NVIDIA on approval).

**Example (arrays, no paths)**

```python
from transformers import AudioFlamingo3Processor, AudioFlamingo3ForConditionalGeneration
import librosa, torch, numpy as np

def load_audio(path, sr=16000):
[… 25 lines of code omitted]

## [c2315195971] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/__init__.py:1 · 2025-09-02
Can you update the year here and in other files?

## [c2315223024] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:154 · 2025-09-02
Let's rename (to fit with Transformers conventions)
- `llm_cfg` to `text_config` (e.g. [Llava](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/configuration_llava.py#L81))
- `sound_tower_cfg` to `encoder_config` (e.g. [Dia](https://github.com/huggingface/transformers/blob/8a091cc07c9808196e3154ab8bd6252d919f929b/src/transformers/models/dia/configuration_dia.py#L332))

The reasoning being: `text_config`, `vision_config`, `audio_config` are meant for pre-trained models used as is, but you've actually trained a specific audio encoder for your task.

So your dict would be:
```python
sub_configs = {"text_config": AutoConfig, "encoder_config": AudioFlamingo3EncoderConfig}
```

## [c2315240875] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:160 · 2025-09-02
`sound_mm_projector_cfg` config feels a bit "opaque". Would it be possible to "break" up its parameters like for [Llava](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/configuration_llava.py#L87C9-L92)? (I'm supposing it has a similar role as the projector in Llava)

_Or you may not have any parameters to define_, if you use suggestion I gave below in your modeling code for `AudioFlamingo3MultiModalProjector`.

## [c2315270994] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:180 · 2025-09-02
Since you "own" the `encoder_config` (being defined above), you can do something shorter like [this](https://github.com/huggingface/transformers/blob/8a091cc07c9808196e3154ab8bd6252d919f929b/src/transformers/models/dia/configuration_dia.py#L348-L352) and default to necessary parameters in the config (which I guess you've already done?)

## [c2315277068] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:2 · 2025-09-02
year

## [c2315277544] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:2 · 2025-09-02
year

## [c2315293667] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:137 · 2025-09-02
perhaps rename as you're rather reshaping?

## [c2315325490] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:73 · 2025-09-02
Since a few things are copied from Whisper, we will eventually want to move to using a modular script to generate `modeling_audioflamingo3.py`.

From what I see the following can be copied over with modular ([this example](https://huggingface.co/docs/transformers/en/modular_transformers#attention) may prove useful):
- `eager_attention_forward` imported from Whisper
- `AudioFlamingo3Attention` for which I see you changed `forward` and added `_shape` function. Above example shows overwriting `forward`.
- `AudioFlamingo3EncoderLayer`, this one is satisfying 😄 
```python
from ..whisper.modeling_whisper import WhisperEncoderLayer

class AudioFlamingo3EncoderLayer(WhisperEncoderLayer):
    pass
```

Components which aren't copied from elsewhere, you can define as you've done here. And then run the following command to generate `modeling_audioflamingo3.py`:
```
python utils/modular_model_converter.py \
--files-to-parse src/transformers/models/audioflamingo3/modular_audioflamingo3.py
```

Can you try that out?

## [c2315458479] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:294 · 2025-09-02
Modular can be used to copy over same parts from [WhisperEncoder](https://github.com/huggingface/transformers/blob/8a091cc07c9808196e3154ab8bd6252d919f929b/src/transformers/models/whisper/modeling_whisper.py#L589)
- for `__init__`, from quick glance seems like you can call super init and define your new `self.avg_pooler = nn.AvgPool1d(2, stride=2)`. And could those hardcoded parameters be added to `AudioFlamingo3EncoderConfig`?

## [c2315471244] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:337 · 2025-09-02
You only seem to use this function once in `_build_square_attn_mask` and only use the first output. Do you need to return the second?

## [c2315480111] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:337 · reply to [c2315471244] · 2025-09-02
That way it would be similar to `WhisperPreTrainedModel` which you can also use modular to overwrite appropriate parts:
- `_init_weights`
- `_no_split_modules`

## [c2315491263] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:388 · 2025-09-02
won't have to define these anymore as they are in `WhisperEncoder` and will be copied over by modular!

## [c2315495779] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:485 · 2025-09-02
I can see that they are convenient, but to keep with Transformers convention, let's remove these. (And hidden size can be obtained from the config)

## [c2315517600] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:472 · 2025-09-02
Can we use `BaseModelOutput` here like in [WhisperEncoder](https://github.com/huggingface/transformers/blob/8a091cc07c9808196e3154ab8bd6252d919f929b/src/transformers/models/whisper/modeling_whisper.py#L729)?

`*LMOutput*` is typically something reserved for outputs of ConditionalGeneration models (e.g. [Whisper](https://github.com/huggingface/transformers/blob/8a091cc07c9808196e3154ab8bd6252d919f929b/src/transformers/models/whisper/modeling_whisper.py#L1330) as well)

## [c2315522206] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:359 · 2025-09-02
Let's remove this method as we prefer to keep a single `forward` method.

## [c2315527589] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:558 · 2025-09-02
instead of calling `forward_tower`, let's call the `forward` method itself and extract `last_hidden_state`.

You can also add a check that `mask is not None` as you do inside `forward_tower` (and then move to appropriate device as you do above)

## [c2315541022] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:2 · 2025-09-02
year

## [c2315569688] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:755 · 2025-09-02
Can you switch to ModuleList? Transformers convention is to not use Sequential.

(Your conversion script will have to account this when mapping original weights to Transformers definition)

## [c2315588726] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:758 · 2025-09-02
`config.sound_hidden_size` and `config.hidden_size` don't seem to be defined within `AudioFlamingo3Config`?

Since they depend on the configs of the LLM and the Audio Encoder, you can add the following properties to `AudioFlamingo3Config` (double-check where hidden size is stored in each config):
```python
@property
def text_hidden_size(self) -> int:
    return self.text_config.hidden_size

@property
def audio_hidden_size(self) -> int:
    return self.encoder_config.hidden_size
```
and possibly add other parameters which you think might be good to expose to the configuration class (e.g. [activation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/configuration_llava.py#L88)).

Or you can directly access like in [Llava](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py#L92-L111).

## [c2315631377] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:525 · 2025-09-02
this seems to be missing from the config, and we'll want to define them there (like in [Dia](https://github.com/huggingface/transformers/blob/8a091cc07c9808196e3154ab8bd6252d919f929b/src/transformers/models/dia/configuration_dia.py#L340-L342) and [Llava](https://github.com/huggingface/transformers/blob/91be12bdc6d312a2d7a44174971f025c4023a91f/src/transformers/models/llava/configuration_llava.py#L87))

## [c2315641038] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:747 · 2025-09-02
Maybe you can add that it corresponds to "Audio Adaptor" in your paper? (I found that name more intuitive but yeah to be consistent with other parts of Transformers "MultiModalProjector" might be better)

## [c2315690982] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:552 · 2025-09-02
In Llava context, they call something like [this](https://github.com/huggingface/transformers/blob/91be12bdc6d312a2d7a44174971f025c4023a91f/src/transformers/models/llava/modeling_llava.py#L157) (tower followed by projector) as `get_image_features`, how about `get_audio_features` for here?

## [c2315725928] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:550 · 2025-09-02
`_process_features` seems to be only called here, so why not simplify this line to:
```python
[torch.cat([f, self.end_emb_token], dim=0) for f in feats]
```
and:
- remove `_process_features`
- pre-compute `self.end_emb_token` in `__init__`

## [c2315910474] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:561 · 2025-09-02
I suppose this function is inspired by [this one](https://github.com/huggingface/transformers/blob/91be12bdc6d312a2d7a44174971f025c4023a91f/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L517C9-L517C45) from Qwen2Audio?

_Note_: This function from Qwen2Audio may not be the best of examples, see that this is [considered legacy](https://github.com/huggingface/transformers/blob/91be12bdc6d312a2d7a44174971f025c4023a91f/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L814) with a preference for preparing the embedding in the processor and performing something like [this](https://github.com/huggingface/transformers/blob/91be12bdc6d312a2d7a44174971f025c4023a91f/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L821). Similarly, the approach done in [Llava](https://github.com/huggingface/transformers/blob/91be12bdc6d312a2d7a44174971f025c4023a91f/src/transformers/models/llava/modeling_llava.py#L296C43-L296C57), with using `masked_scatter`, is the best way to do!

In any case, I'll give some first comments to improve the readability of this function, but we may need to reconsider how we do the expansion/merging.

Can you rename `_embed` to `_merge_input_ids_with_audio_features`? And similarly, it would be helpful to provide a block comment on the expected inputs and the custom processing that is needed for your model 🙏

## [c2315913863] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:564 · 2025-09-02
rename to `audio_features`? like in [Qwen2Audio](https://github.com/huggingface/transformers/blob/91be12bdc6d312a2d7a44174971f025c4023a91f/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L524C13-L524C27)
and is it a list because it's a batch? can you ensure it is tensor instead?

## [c2315932770] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:567 · 2025-09-02
rather than passing dict, can we be more explicit about the input arguments? namely passing `audio_feature_mask` and `audio_embed_mask` instead (also renaming "sound" > "audio")

## [c2315974443] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:575 · 2025-09-02
Let's compute the embeddings before this function, like in [Qwen2](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L806-L808), so purpose of this function is just merging!

## [c2315997983] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:550 · reply to [c2315725928] · 2025-09-02
(This comment will make more sense after seeing `_embed` below)

Since `self.end_emb_token` is needed as part of the merging of text and audio embeddings (and not necessarily a sound feature), how about moving this concatenation into your function for merging for the embeddings?

Namely deleting this method and inside `_emb` (or `_merge_input_ids_with_audio_features` after renaming), performing the above concatenation to the inputted audio features?

## [c2316009165] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:589 · 2025-09-02
as mentioned above, let's make sure inputted `audio_features` is a tensor so no need to do this!

## [c2316077118] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:165 · 2025-09-02
Can you rename to `audio` and accept list of audios as well? I.e. for batch (see [Qwen2](https://github.com/huggingface/transformers/blob/37103d6f22cd25c9eba7297ae79ed8c47a852e54/src/transformers/models/qwen2_audio/processing_qwen2_audio.py#L84))

## [c2316084097] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:166 · 2025-09-02
No need to pass `device` if output of `__call__` is `BatchFeature`. Since we can call `to(device)` on the output. 

For example, desired usage:
```python
inputs = processor(text=prompts, audios=[audio1, audio2], return_tensors='pt', padding=True).to("cuda")
```

## [c2316137946] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:218 · 2025-09-02
Let's output `BatchFeature` so no need to do device moving here (see [Qwen](https://github.com/huggingface/transformers/blob/37103d6f22cd25c9eba7297ae79ed8c47a852e54/src/transformers/models/qwen2_audio/processing_qwen2_audio.py#L175C16-L175C28))

## [c2316160846] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:222 · 2025-09-02
have you defined this method to handle batch decoding? Typically, we have separate methods for `decode` and `batch_decode`. You can see Dia for an example (where [`decode`](https://github.com/huggingface/transformers/blob/8a091cc07c9808196e3154ab8bd6252d919f929b/src/transformers/models/dia/processing_dia.py#L329) would call [`batch_decode`](https://github.com/huggingface/transformers/blob/8a091cc07c9808196e3154ab8bd6252d919f929b/src/transformers/models/dia/processing_dia.py#L258)).

## [c2316195024] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:194 · 2025-09-02
Ideally, this would just be a call to the feature extraction (like in [Qwen](https://github.com/huggingface/transformers/blob/37103d6f22cd25c9eba7297ae79ed8c47a852e54/src/transformers/models/qwen2_audio/processing_qwen2_audio.py#L127C28-L127C50)).

Are you not able to use the mask from `WhisperFeatureExtractor`? (like in [Qwen](https://github.com/huggingface/transformers/blob/37103d6f22cd25c9eba7297ae79ed8c47a852e54/src/transformers/models/qwen2_audio/processing_qwen2_audio.py#L130))

## [c2316205539] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:140 · 2025-09-02
Can you give a reason/comment for why 750? Does it depend on any of the config parameters?

## [c2316206221] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:148 · 2025-09-02
Can you give a reason/comment for why 3000? Does it depend on any of the config parameters?

## [c2316221913] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:118 · 2025-09-02
what you're doing here (preparing audio and attention masks) is meant for feature extractor. So if `WhisperFeatureExtractor` doesn't provide what you need (more or less out of the box), we should define a new feature extractor for AF3.

Modular doesn't work for generating `feature_extraction_audioflamingo3.py` so you would have to do manually, but you can use the same `Copied from...` comments for parts that coped from [`WhisperFeatureExtractor`](https://github.com/huggingface/transformers/blob/8a091cc07c9808196e3154ab8bd6252d919f929b/src/transformers/models/whisper/feature_extraction_whisper.py#L36).

## [c2316257328] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:584 · 2025-09-02
We'll rather want `audio_features_mask` prepared _in the processor_ with the correct shape, so this function can just focus on merging.

And then (if I understand correctly), `media_meta["sound_embed_masks"]` wouldn't be needed by this method?

## [c2316286513] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:671 · 2025-09-02
Let's avoid these calls to separate functions, for improved readability to know what is returned (see [Qwen](https://github.com/huggingface/transformers/blob/37103d6f22cd25c9eba7297ae79ed8c47a852e54/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L712))

## [c2316290472] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modeling_audioflamingo3.py:721 · 2025-09-02
Similar as above renaming `media` to `audio_features` and avoiding a `dict` to be more explicit about what `media_meta` contains.

## [c2316307576] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/auto/configuration_auto.py:935 · 2025-09-02
We can remove the projector.

## [c2316312799] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/auto/modeling_auto.py:1 · 2025-09-02
We will also want to add something similar to:
```python
("qwen2_audio", "Qwen2AudioForConditionalGeneration"),
```
for `AudioFlamingo3ForConditionalGeneration`

## [c2316314516] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/auto/configuration_auto.py:1 · 2025-09-02
Could you add it so it respects alphabetic order?

## [c2316315306] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/auto/modeling_auto.py:1 · reply to [c2316312799] · 2025-09-02
And also adding both so that it is in alphabetic order!

## [c3175256079] ebezzam (CONTRIBUTOR) · review COMMENTED · reply to [c3241425200] · 2025-09-02
@lashahub Thanks for the changes! It's already looking much cleaner 👏 

I've given more comments to improve compatibility, structure, and style according to Transformers.

Regarding your open question, keeping a separate `AudioFlamingo3EncoderConfig` as you've done looks great 👍 

As for next steps, what you've mentioned (conversion script, tests, and docs) is perfect! I would also add (and addressing below first):
- Writing a [modular script](https://huggingface.co/docs/transformers/en/modular_transformers) to make best use of other components in Transformers. I've tried to give pointers on how to best to do that.
- I think a `feature_extraction_audioflamingo3.py` will be necessary to define `AudioFlamingo3FeatureExtractor`, as it seems that `WhisperFeatureExtractor` doesn't prepare audio inputs as is needed for your model?
- Cleaning up your `_embed` function to be more consistent with [Qwen2](https://github.com/huggingface/transformers/blob/37103d6f22cd25c9eba7297ae79ed8c47a852e54/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L517) and ensuring that attention masks are properly prepared in the processor.

Great job so far, and let me know if you have any questions!

## [c3245732286] ebezzam (CONTRIBUTOR) · issue_comment · reply to [c3241425200] · 2025-09-02
@lashahub another tip with regards to code style/formatting. You can run `make fixup` for running repo consistency checks, and `make style` for formatting.

If it passes locally, it should also pass the  [setup_and_quality](https://github.com/huggingface/transformers/pull/40290/checks?check_run_id=49308922594) CI tests.

## [c3269211209] lashahub (CONTRIBUTOR) · issue_comment · 2025-09-09
Current usage:
```python
from transformers import AudioFlamingo3Processor, AudioFlamingo3ForConditionalGeneration
import librosa
import torch


[… 24 lines of code omitted]
```

## [c2332445853] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:133 · 2025-09-09
`AudioFlamingo3EncoderConfig` for `"encoder_config"`?

## [c2332496306] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:96 · 2025-09-09
can you replace assert statements with e.g. raise ValueError(...)? The Transformers convention is to not use asserts.

## [c2332501566] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:86 · 2025-09-09
Can you add the parameters (kernel size and stride) for `AvgPool1d` to `AudioFlamingo3EncoderConfig`?

## [c2332505034] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:88 · 2025-09-09
@eustlb @vasqu is there a similar utility for this in transformers in `src/transformers/masking_utils.py`?

## [c2332535524] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:149 · 2025-09-09
Similar as above, replace assert with `raise ValueError(...)`

(even though I notice original Whisper had it 🙈)

## [c2332543805] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:131 · 2025-09-09
we can remove this, we typically assume inputs are on correct device/etc when passed

## [c2332628859] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:301 · 2025-09-09
Nit for consistency: 
"audio_embeds" and "audio_embeds_mask" 
OR 
"audio_features" and "audio_features_mask"

## [c2332637415] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:308 · 2025-09-09
can you also compute the text embeddings before and pass to the function? (like in [Qwen](https://github.com/huggingface/transformers/blob/a871f6f58d49f3a05ae9dae519caa8aa9d919a07/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L528))

## [c2332645680] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:316 · 2025-09-09
Why not simply making them required inputs instead of defaulting to None?

## [c2332685965] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:270 · 2025-09-09
Can `_compute_audio_embeds` be renamed to `get_audio_features`? And you can remove the above methods `_sound_features` and (old) `get_audio_features` (which seem to do the same thing but `_sound_features` is not called)

## [c2332701755] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:458 · 2025-09-09
Change after renaming as above

## [c2332706659] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:460 · 2025-09-09
as mentioned above computing text embedding before passing to `_merge_input_ids_with_audio_features`. Like in [Qwen](https://github.com/huggingface/transformers/blob/37103d6f22cd25c9eba7297ae79ed8c47a852e54/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L777-L779).

## [c2332865876] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:166 · 2025-09-09
can you put these args inside `audio_kwargs` like in [Dia](https://github.com/huggingface/transformers/blob/37103d6f22cd25c9eba7297ae79ed8c47a852e54/src/transformers/models/dia/processing_dia.py#L50C10-L50C22)

## [c2332868639] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:167 · 2025-09-09
this can be in `common_kwargs` like in [Dia](https://github.com/huggingface/transformers/blob/37103d6f22cd25c9eba7297ae79ed8c47a852e54/src/transformers/models/dia/processing_dia.py#L58)

## [c2332878291] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:224 · 2025-09-09
as for Feature extraction, `return_tensors` can be obtained from `common_kwargs`

## [c2332885760] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:91 · 2025-09-09
if not used anymore, we can delete. Or did you want to use within `__call__`?

## [c2332888091] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:180 · 2025-09-09
is there a reason it can't go more than 20? 

if this is the limit, how about a warning in case someone wants to work with audio longer than this?

## [c2332916704] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:143 · 2025-09-09
Can these be added to `audio_kwargs`? Namely 16000, 160, and 30.

It will help to see the parameters clearly presented above.

See [Dia](https://github.com/huggingface/transformers/blob/37103d6f22cd25c9eba7297ae79ed8c47a852e54/src/transformers/models/dia/processing_dia.py#L56) which also adds the sampling rate

## [c2332966131] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:446 · 2025-09-09
Instead of overwriting the `generate` method, we typically define the [`forward`](https://github.com/huggingface/transformers/blob/37103d6f22cd25c9eba7297ae79ed8c47a852e54/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L715) method, and overwrite [`prepare_inputs_for_generation`](https://github.com/huggingface/transformers/blob/37103d6f22cd25c9eba7297ae79ed8c47a852e54/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L882). I put example links to Qwen.

## [c2332994332] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:471 · 2025-09-09
This is not something we normally have.

I saw that in your most [recent example](https://github.com/huggingface/transformers/pull/40290#issuecomment-3269211209), you're using this to pass the token IDs.

This is something that can be handled if you overwrite  `prepare_inputs_for_generation` as mentioned above.

Moreover, like [Dia](https://github.com/huggingface/transformers/blob/8a091cc07c9808196e3154ab8bd6252d919f929b/src/transformers/models/dia/configuration_dia.py#L340-L342) and [Llava](https://github.com/huggingface/transformers/blob/91be12bdc6d312a2d7a44174971f025c4023a91f/src/transformers/models/llava/configuration_llava.py#L87), let's have the token IDs in the configuration.

## [c2333027895] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:153 · 2025-09-09
Can we just use `n_samples`? and use a more descriptive name than `wo`?

## [c3199999962] ebezzam (CONTRIBUTOR) · review COMMENTED · reply to [c3269211209] · 2025-09-09
@lashahub this is coming together very nicely! Thanks for switching to modular and updating the processor. I've left some comments to better conform to Transformers-convention.

We may still come back to `_merge_input_ids_with_audio_features` if a core maintainer thinks we should move some of the processing to `AudioFlamingo3Processor`, but we can leave it like this for now.

For next steps, how about:
- Adding the conversion script which pushes a model to the hub.
- Starting the unit tests, first with integration tests to make sure that the Transformers-compatible model is consistent with the original model. For example, [here](https://gist.github.com/ebezzam/6382bdabfc64bb2541ca9f77deb7678d) you can see scripts I created to test another Nvidia model to write the expected outputs to a JSON file. And [here](https://github.com/nithinraok/transformers/blob/d04760a7f8a54f60820ec577c0eb229413e8b027/tests/models/parakeet/test_modeling_parakeet.py#L310) are the tests themselves.
- Start the markdown doc file

Let me know if you have any questions or need any pointers!

## [c2333445679] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:88 · reply to [c2332505034] · 2025-09-09
I have an alternative. @lashahub could you change to the following? (It is working when I try your example)

```python
from ...masking_utils import eager_mask, padding_mask_function


def _build_square_attn_mask(self, mask_1d: torch.Tensor, max_mel_seq_len: int) -> torch.Tensor:
[… 28 lines of code omitted]

## [c3201382349] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-09-09 (no text)

## [c3291124179] lashahub (CONTRIBUTOR) · issue_comment · 2025-09-15
- Dropped the legacy `_merge_input_ids_with_audio_features` method
- Perform sound token expansion in the processor
- Switched from `generate()` to `forward()`
- Added the model conversion script
- Added the markdown doc
- _Writing tests in progress_

Updated usage:
```python
from transformers import AudioFlamingo3Processor, AudioFlamingo3ForConditionalGeneration
import librosa, torch


def load_audio(p, sr=16000):
[… 19 lines of code omitted]
```

## [c2362167162] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:42 · 2025-09-19
Nit: we can remove such comments for sections

## [c2362186770] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:164 · 2025-09-19
Can we remove this line? If inputs are properly mapped like in your example (and as we expect in usage):
```python
inputs = processor(texts, audios, padding_side="left", tensor_type="pt").to(device)
```
we shouldn't need this.

## [c2362220133] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:343 · 2025-09-19
Same here as above, can these device mappings be removed?

## [c2362224242] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:364 · 2025-09-19
Can you use the mask utilities from the Transformers library? As mentioned [here](https://github.com/huggingface/transformers/pull/40290#discussion_r2333445679)

And for the device/type mapping, instead of using `dev`, using `input_features`

## [c2362260343] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:349 · 2025-09-19
Since `_get_feat_extract_output_lengths` is only used once, let's remove that method and simply compute here:
```python
pre_lengths = ((Lmel - 1) // 2 + 1).to(dtype=torch.long)
post_lengths = ((Lmel - 2) // 2 + 1).to(dtype=torch.long)
```

## [c2363107024] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/convert_audioflamingo3_to_hf.py:251 · 2025-09-19
Could you add example usage like [this](https://github.com/huggingface/transformers/blob/main/src/transformers/models/xcodec/convert_xcodec_weights_to_hf.py#L264) for reproducibility? (e.g. command to download original model and then to run this script for conversion + upload)

## [c2368507503] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/convert_audioflamingo3_to_hf.py:106 · 2025-09-22
you can do something like [Dia](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dia/convert_dia_to_hf.py#L191-L196) to create the processor + below to pass a Jinja template:
```python
chat_template = """{% set audio_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}{% else %}{% set audio_tokens = namespace(value='') %}{% set text_tokens = namespace(value='') %}{% for content in message['content'] %}{% if content['type'] == 'audio' or 'audio' in content %}{% set audio_count.value = audio_count.value + 1 %}{% if 'path' in content %}{% set audio_tokens.value = audio_tokens.value + '<sound>' %}{% elif 'url' in content %}{% set audio_tokens.value = audio_tokens.value + '<sound>' %}{% elif 'audio' in content %}{% set audio_tokens.value = audio_tokens.value + '<sound>' %}{% elif 'array' in content %}{% set audio_tokens.value = audio_tokens.value + '<sound>' %}{% endif %}{% elif 'text' in content %}{% set text_tokens.value = text_tokens.value + content['text'] %}{% endif %}{% endfor %}{{ audio_tokens.value }}{{ text_tokens.value }}{% endif %}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

processor = AudioFlamingo3Processor(
    feature_extractor=WhisperFeatureExtractor(feature_size=128, return_attention_mask=True),
    tokenizer=AutoTokenizer.from_pretrained(str(llm_dir), chat_template=chat_template)
)
# save and upload processor
```

## [c2368535547] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/convert_audioflamingo3_to_hf.py:167 · 2025-09-22
Could you add a command line argument (`push_to_hub`) + commands like below to push the converted model and processor to the hub? (E.g. where `push_to_hub=lashahub/audio-flamingo-3`)

```python
if args.push_to_hub:
  processor.push_to_hub(args.push_to_hub)
  model.push_to_hub(args.push_to_hub)
```

So your above methods would have to return `model` and `processor`, or do pushing to hub within the methods.

## [c2368602456] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:198 · 2025-09-22
(Note about one of my changes) I added this fallback as you have in your docs:

`"You can omit `<sound>` entirely and let the processor prepend the expanded tokens:" `

## [c2368606295] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:41 · 2025-09-22
(Note about one of my changes) In all your examples, you're setting `padding_side=left`. How about defaulting to this?

## [c3243767006] ebezzam (CONTRIBUTOR) · review COMMENTED · reply to [c3291124179] · 2025-09-22
@lashahub Thanks for your changes, I made a few tweaks to the processor. 

Before I saw you were using the chat template within the processor and then removed it. We indeed don't want to use chat template within the processor, but _we do_ want to be able to support such usage! Below is an example of how we could do that (in a batch):
```python
# Create two separate conversations instead of one conversation with two user messages
conversations = [
    [  # First conversation
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
[… 48 lines of code omitted]
```

## [c2368670681] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/convert_audioflamingo3_to_hf.py:150 · 2025-09-22
I'm getting this warning:
```
The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
```
So maybe they don't need to be set or it's not correctly configured?

## [c3253124433] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-09-22 (no text)

## [c2368686999] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:16 · 2025-09-22
Can you change the link? https://huggingface.co/papers/2507.08128

## [c3253144257] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-09-22 (no text)

## [c2368693544] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:53 · 2025-09-22
Remove? With left padding as default + chat_template removed from within the processor

## [c3253152633] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-09-22 (no text)

## [c2368699278] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:60 · 2025-09-22
"Whisper-style conv front-end → Transformer encoder" --> "Whisper-style feature extractor + encoder"

## [c3253160407] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-09-22 (no text)

## [c2368730759] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:1 · 2025-09-22
Examples can remove `padding_side="left"` if we keep the default in the processor.

`tensor_type` can also be removed (as I've also set to default to `pt`), or renaming to `return_tensors` which is the name typically used in Transformers (see [Qwen example](https://huggingface.co/docs/transformers/v4.56.2/en/model_doc/qwen2_audio#inference)).

## [c3253198751] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-09-22 (no text)

## [c2368753697] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:95 · 2025-09-22
(Note about one of my changes) your paper mentioned possibility of follow-up/clarification prompts where there is no audio input -> so I made audio optional

## [c3253226622] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-09-22 (no text)

## [c2415406703] lashahub (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/convert_audioflamingo3_to_hf.py:150 · reply to [c2368670681] · 2025-10-09
It happens only when `do_sample` is set to `False`.

## [c3317016063] lashahub (CONTRIBUTOR) · review COMMENTED · 2025-10-09 (no text)

## [c2440153882] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:227 · 2025-10-17
Can you update the example to real audio paths? Such that the example is runnable. You may even be able to pass URLs like [here](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507#audio-instruct-1) or [here](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct#voice-chat-inference)

## [c2440167244] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:195 · 2025-10-17
How about just a short text above explaining the example snippet? (Batch processing that does both transcription and music understanding)

## [c2440170480] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:241 · 2025-10-17
could you provide a multi-turn example?

## [c2440178037] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:195 · reply to [c2440167244] · 2025-10-17
and another short example just above that shows just transcription with a single example. In general people like seeing multiple examples for different tasks so they can conveniently copy-paste

## [c2440182314] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:109 · 2025-10-17
You can remove type hints as already indicated in the docstrings.

## [c2440186898] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:101 · 2025-10-17
let's rename to `hidden_size` like other/recent Transformers models. You can see a recent model addition ([Voxtral](https://github.com/huggingface/transformers/blob/main/src/transformers/models/voxtral/configuration_voxtral.py#L35)) for typical naming for a similar model and how it uses [`attribute_map`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/voxtral/configuration_voxtral.py#L72-L78) to make it work with modular

## [c2440196171] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:182 · 2025-10-17
as above, can remove type hints

## [c2440210551] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/convert_audioflamingo3_to_hf.py:216 · 2025-10-17
awesome!

## [c2440213749] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/convert_audioflamingo3_to_hf.py:223 · 2025-10-17
will there be a backup elsewhere? in case you decide to replace those checkpoints with the HF-compatible ones

## [c2440250361] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:493 · 2025-10-17
I think you can avoid returning `attention_mask` since it is the same as the one that is inputed?

That way you you also don't need to define a new `AudioFlamingo3CausalLMOutputWithPast` but can directly use `CausalLMOutputWithPast` like in [Voxtral](https://github.com/huggingface/transformers/blob/39b6d3bf7e30b92b2de50a31ee991557d41ab568/src/transformers/models/voxtral/modular_voxtral.py#L210) and update `loss` output with the value you compute in lines 469-481

## [c2440252162] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:47 · 2025-10-17
You can avoid defining this by directly using [`CausalLMOutputWithPast`](https://github.com/huggingface/transformers/blob/39b6d3bf7e30b92b2de50a31ee991557d41ab568/src/transformers/modeling_outputs.py#L660). See comment below.

## [c2440258977] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:76 · 2025-10-17
Can remove such comments so they don't appear in the modeling file.

## [c2440290982] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:245 · 2025-10-17
Also update here with actual paths/urls

## [c2440329752] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:139 · 2025-10-17
Just noticed that `Qwen2AudioEncoder`  is like yours, albeit with hardcoded values to `AvgPool1d`:

https://github.com/huggingface/transformers/blob/39b6d3bf7e30b92b2de50a31ee991557d41ab568/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L322

Anyway let's use that with modular:
```python
class AudioFlamingo3Encoder(Qwen2AudioEncoder)
...
```
So that you don't need to overwrite `__init__`.

You can then remove the following from your config:
```python
self.avg_pool_kernel_size = avg_pool_kernel_size
self.avg_pool_stride = avg_pool_stride
```
(Or add them to that of [Qwen2AudioEncoderConfig](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_audio/configuration_qwen2_audio.py#L24C7-L24C30) if you think it's important to expose them.)

## [c2440331364] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:141 · 2025-10-17
_Note to self: I may want to come back to this if there's a way to simplify further_

## [c2440398361] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:59 · 2025-10-17
similar to `__init__` can `forward` of `Qwen2AudioEncoder` be used as is, as they use [average pooler before layer norm](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L416) in similar fashion?

## [c2440439191] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:277 · 2025-10-17
This is very similar to `VoxtralMultiModalProjector`:

https://github.com/huggingface/transformers/blob/39b6d3bf7e30b92b2de50a31ee991557d41ab568/src/transformers/models/voxtral/modular_voxtral.py#L115

But I don't think modular can be used since Voxtral has `bias=False` and they didn't expose that parameters in their config.

But could you imitate their code style? (As it's more intuitive)

And add/expose the following parameters in your configs:
```
config.audio_config.intermediate_size
config.projector_hidden_act
config.projector_bias    # NEW one not available with Voxtral
```
Other models maye then use modular with your `AudioFlamingo3MultiModalProjector` as it would be more general!

## [c2440443674] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:340 · 2025-10-17
You can add the decorator `@can_return_tuple` to remove the `return_dict` and corresponding logic.

See [Voxtral](https://github.com/huggingface/transformers/blob/39b6d3bf7e30b92b2de50a31ee991557d41ab568/src/transformers/models/voxtral/modular_voxtral.py#L195)

## [c2440445995] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:330 · 2025-10-17
_note to self: change to `audio_spectrogram` for more intuitive input as per standardization for audio models?_

## [c2440458482] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:433 · 2025-10-17
`S_out_max = audio_feats.shape[1]` is enough?
 
Convention on our side to avoid single letter variable and unused ones

## [c2440466217] ebezzam (CONTRIBUTOR) · review_comment on tests/models/audioflamingo3/test_modeling_audioflamingo3.py:232 · 2025-10-17
Let's put such decorators on methods rather than classes

## [c2440471445] ebezzam (CONTRIBUTOR) · review_comment on tests/models/audioflamingo3/test_modeling_audioflamingo3.py:298 · 2025-10-17
You can use these URLs in the examples I mentioned above

## [c2440478437] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:277 · reply to [c2440439191] · 2025-10-17
Since you change the state dict, you will also have to update the checkpoint conversion script

## [c2440484363] ebezzam (CONTRIBUTOR) · review_comment on tests/models/audioflamingo3/test_modeling_audioflamingo3.py:233 · 2025-10-17
For the integration tests, what we will want to test are rather outputs of single file and batch inference against the expected outputs of the original checkpoint. So in this case, testing against the checkpoint that is currently stored at [`nvidia/audio-flamingo-3`](https://huggingface.co/nvidia/audio-flamingo-3). 

Here are examples for NVIDIA's Parakeet model ([single](https://github.com/huggingface/transformers/blob/7e204ad121e729cac59ea213356c68527cac0a7e/tests/models/parakeet/test_modeling_parakeet.py#L329) and [batched](https://github.com/huggingface/transformers/blob/7e204ad121e729cac59ea213356c68527cac0a7e/tests/models/parakeet/test_modeling_parakeet.py#L354)).

It will also be useful to write/upload script like [this](https://github.com/huggingface/transformers/blob/7e204ad121e729cac59ea213356c68527cac0a7e/tests/models/parakeet/test_modeling_parakeet.py#L331C60-L331C150) to be able to reproduce the expected outputs (in case of regression) and that can write the expected outputs in a JSON file that can be stored in the [`fixtures`](https://github.com/huggingface/transformers/tree/main/tests/fixtures) folder. 

_Similar to the conversion script, we'll probably want to make a backup of the original checkpoint elsewhere in case you decide to update your HF page with this new Transformers-compatible checkpoint._

## [c2440507583] ebezzam (CONTRIBUTOR) · review_comment on tests/models/audioflamingo3/test_modeling_audioflamingo3.py:345 · 2025-10-17
see comment above, namely computing expected outputs against original checkpoint

## [c3350535943] ebezzam (CONTRIBUTOR) · review COMMENTED · reply to [c3317016063] · 2025-10-17
@lashahub thanks for the changes! I've left some comments on:
1. making even better use of modular for the Encoder model
2. standardize with other (recent) Transformer models (namely Voxtral which is similar in architecture)
3. integration tests that compare against expected outputs of the checkpoint current on [your HF page](https://huggingface.co/nvidia/audio-flamingo-3) (and backing it up elsewhere in case you decide to overwrite the HF page with the Transformers-compatible checkpoint)

## [c2450908112] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:211 · 2025-10-22
Does `positions` need to be dynamically created? Or can they be obtained from `self.embed_positions`?

https://github.com/huggingface/transformers/blob/a8cece13e29057fc9712543402d73e78796b306d/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L381

## [c3364298266] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-10-22 (no text)

## [c2450909858] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:59 · reply to [c2440398361] · 2025-10-22
See [this comment](https://github.com/huggingface/transformers/pull/40290/files#r2450908112) for a difference that would need to be addressed

## [c3364300421] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-10-22 (no text)

## [c2451238987] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/convert_audioflamingo3_to_hf.py:190 · 2025-10-22
I didn't get to run the conversion script myself, but if you double-check that this work, that would be great!

## [c2451244645] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:135 · 2025-10-22
Does `positions` need to be dynamically created? Or can they be obtained from `self.embed_positions`? Like below

https://github.com/huggingface/transformers/blob/a8cece13e29057fc9712543402d73e78796b306d/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L381

## [c2451252206] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:259 · 2025-10-22
Is there a reason to not pass to self.language model? Like in Qwen below

https://github.com/huggingface/transformers/blob/a8cece13e29057fc9712543402d73e78796b306d/src/transformers/models/qwen2_audio/modeling_qwen2_audio.py#L821

## [c2451299707] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:335 · 2025-10-22
We still need to iterate on this part. 

The goal would be to simplify this condition for computing audio features to something like Voxtral's

https://github.com/huggingface/transformers/blob/7cd1d2b66ca118ea032d517ddb3ec31543354177/src/transformers/models/voxtral/modular_voxtral.py#L247-L254

## [c2451321700] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:135 · reply to [c2451244645] · 2025-10-22
If not, we could probably also use modular to copy of the `forward` method from `Qwen2AudioEncoder`!

## [c2451337600] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:1 · 2025-10-22
_Note to self to come back to this file after changes (e.g. variable naming, comparing to Qwen/Voxtral)._

## [c2451354301] ebezzam (CONTRIBUTOR) · review_comment on tests/models/audioflamingo3/test_modeling_audioflamingo3.py:1 · 2025-10-22
_Note to self to compare with Voxtral's tests: https://github.com/huggingface/transformers/blob/main/tests/models/voxtral/test_modeling_voxtral.py_

## [c2451369891] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:195 · reply to [c2440167244] · 2025-10-22
You can take example from Voxtral to see the type of snippets that people find useful: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/voxtral.md

## [c2451435845] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:57 · 2025-10-22
For transcription/ASR, if the prompt is always the same (`"Transcribe the input speech."` from what I see?), we can do something like [Voxtral](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/voxtral.md#transcription-mode) and define a convenience method such that the user doesn't have to write `"Transcribe the input speech."` each time. 

(The method won't be like their `apply_transcription_request` as they use a custom method from `MistralCommonTokenizer`)

## [c3364718823] ebezzam (CONTRIBUTOR) · review COMMENTED · reply to [c3317016063] · 2025-10-22
@lashahub I pushed some standardization changes, and left some comments on things to still address, as well as those from the previous [review](https://github.com/huggingface/transformers/pull/40290#pullrequestreview-3350535943) on:
- updating integration tests
- updating example usage
- backing up original weights

hope that helps and thanks for your work!

## [c3452900460] lashahub (CONTRIBUTOR) · issue_comment · reply to [c3364718823] · 2025-10-27
@ebezzam Thanks for the review! I’ve updated the code based on your comments:

- The converted HF model is available [here](https://huggingface.co/nvidia/audio-flamingo-3-hf). The [previous repo](https://huggingface.co/nvidia/audio-flamingo-3) keeps the original weights.
- Added usage examples in the [docs](https://github.com/jsalt-2025/transformers/blob/audioflamingo3/docs/source/en/model_doc/audioflamingo3.md) for loading directly from URLs (audio+text, multi-turn, text-only, audio-only, batched).
- Small patch to align the converted MM projector weight keys.
- Cleaned up `generation_config.json`: correct `bos_token_id`, `eos_token_id`, and `pad_token_id`; default to greedy decoding.
- Use `self.embed_positions.weight` directly in `AudioFlamingo3Encoder.forward()`. We probably can’t reuse `Qwen2AudioEncoder.forward()` as-is, it enforces a fixed mel length, doesn’t slice positions, handles `LayerDrop` differently, and doesn’t expose the pre-pool hidden state we need.
- Added tests comparing generated responses and IDs to the [original model outputs](https://github.com/jsalt-2025/transformers/tree/audioflamingo3/tests/fixtures/audioflamingo3), produced by this [script](https://gist.github.com/lashahub/185b080ca87fe82fd397bc621e74efcb). Some values may differ, as the original model sometimes uses `.half()`.

## [c2469937863] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:38 · 2025-10-28
You can add model contributors here like [this](https://github.com/huggingface/transformers/blob/2f9e3ae7f532f12ff16648fd956687354847e151/docs/source/en/model_doc/voxtral.md?plain=1#L349)

## [c2469949027] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:38 · 2025-10-28
Checkpoint links should also be added, like [this](https://github.com/huggingface/transformers/blob/2f9e3ae7f532f12ff16648fd956687354847e151/docs/source/en/model_doc/voxtral.md?plain=1#L24)

## [c2469974868] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:18 · 2025-10-28
It would be good to make this more prominent, namely that audio is capped at 10 min.

So people don't get unexpected behavior when passing audio larger than this

## [c2470368272] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:245 · reply to [c2440290982] · 2025-10-28
Can you add paths to actual audio files and update the HF compatible model ID? Like you did in the model doc

## [c2470446965] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:128 · 2025-10-28
If this is always False (or never used) can we remove? Here and in other modules

## [c2473767582] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:164 · 2025-10-29
@eustlb what are your thoughts about the mask preparation here?

A main difference with Whisper Encoder (and derivatives in Qwen2Audio and Voxtral) is that the encoder layers of this model take into account an attention mask. 

Some pointers that may help:
- Whisper not using attention mask: https://github.com/huggingface/transformers/blob/4d0b6758b90aa54e4077171e6d42c55e0c01c622/src/transformers/models/whisper/modeling_whisper.py#L681
- Voxtral [not using it](https://github.com/huggingface/transformers/blob/4d0b6758b90aa54e4077171e6d42c55e0c01c622/src/transformers/models/voxtral/modeling_voxtral.py#L332) as it isn't passed to its `self.audio_tower`: https://github.com/huggingface/transformers/blob/4d0b6758b90aa54e4077171e6d42c55e0c01c622/src/transformers/models/voxtral/modeling_voxtral.py#L442

## [c3462221523] ebezzam (CONTRIBUTOR) · issue_comment · 2025-10-29
run-slow: audioflamingo3

## [c3389280974] ebezzam (CONTRIBUTOR) · review COMMENTED · reply to [c3452900460] · 2025-10-29
@lashahub thanks for the changes! 

I pushed some changes to prune code and standardize with more recent Transformers conventions (i.e. like Voxtral):
- using modular as much as possible to copy from previous models
- added a `get_audio_features` method to `AudioFlamingo3ForConditionalGeneration` and shifted some padding mask preparation to the processor
- removed output attentions logic (as it always seems to be False?)
- more descriptive variable names, e.g. `post_pool_len` and `conv_output_len` rather than `K` and `L1`
- fixed some of the tests

Please take a look and let me know if there's any breaking changes for your modeling logic. 

Could you also:
1. Address the open comments (here and in the previous review).
2. Convert and push checkpoints to HF again, to rename `init_std`: https://huggingface.co/nvidia/audio-flamingo-3-hf/blob/main/config.json#L11
3. Double check the integration tests. They are failing on our CI machine (you can see [here](https://github.com/huggingface/transformers/actions/runs/18913136148)) due to `CUDA out of memory`. Maybe you can try with smaller/different inputs? (e.g. the inputs from the [Voxtral tests](https://github.com/huggingface/transformers/blob/a43b36cf802f00616800e0bd4d748679236123ee/tests/models/voxtral/test_modeling_voxtral.py#L230))

On my machine I get the following errors:
```
# RUN_SLOW=1 pytest tests/models/audioflamingo3/test_modeling_audioflamingo3.py::AudioFlamingo3ForConditionalGenerationIntegrationTest

============================================================================================== short test summary info ===============================================================================================
FAILED tests/models/audioflamingo3/test_modeling_audioflamingo3.py::AudioFlamingo3ForConditionalGenerationIntegrationTest::test_fixture_batched_matches - RuntimeError: The frame has 0 channels, expected 1. If you are hitting this, it may be because you are using a buggy FFmpeg version. FFmpeg4 is known to fail here in some valid scenarios. Try to upgrade FFmpeg?
FAILED tests/models/audioflamingo3/test_modeling_audioflamingo3.py::AudioFlamingo3ForConditionalGenerationIntegrationTest::test_fixture_single_matches - AssertionError: The values for attribute 'shape' do not match: torch.Size([1, 21]) != torch.Size([1, 22]).
====== 2 failed, 2 warnings in 27.28s =====
```
_(although FFmpeg is probably something buggy from my environment setup)_

## [c2484937309] eustlb (COLLABORATOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:164 · reply to [c2473767582] · 2025-11-02
Indeed the no-attention mask with attending to padding zeros on the 30sec fixed input window is the big limitation of Whisper encoder, and the one Moonshine adressed an improved on (along with RoPE pos encodings).

In this case, I would do something similar to what is done in [Parakeet encoder](https://github.com/huggingface/transformers/blob/37a629628329217a8f0913fd8b0119086d67058c/src/transformers/models/parakeet/modular_parakeet.py#L408-L472), meaning we handle the mask resizing directly in the audio_tower (so the audio encoder)'s forward. I do agree thought that the `attention_mask` naming becomes misleading, but I'd rather really have the mask passed to the encoder the same length as input_features passed to the encoder than doing an external resizing. This will also ease using the encoder in another model with modular if required. I would agree if possible in renaming what is here named as attention_mask in the encoder's forward to input_features_mask directly.

## [c3408900920] eustlb (COLLABORATOR) · review COMMENTED · 2025-11-02 (no text)

## [c3478999316] lashahub (CONTRIBUTOR) · issue_comment · reply to [c3389280974] · 2025-11-03
Hi @ebezzam thank you for your help!

Changes:
- Added a convenience method `apply_transcription_request` to `AudioFlamingo3Processor`.
- Updated fixtures to include new examples for both single and batch inference.

All tests are passing locally except for:
- `test_eager_matches_fa2_generate`
- `test_flash_attn_2_can_dispatch_composite_models`

But those tests are also failing for `Qwen2Audio` and `Voxtral`. Everything else appears to be working as expected on my end.

## [c2485922301] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:266 · reply to [c3478999316] · 2025-11-03
@lashahub at the moment, the "transcription" would look something like:
```
The transcription of the audio is 'summer follows spring the days grow longer and the nights are warm'.
```
is it possible for a decoding option to return just the transcription, namely just the contents within `'...' `?

## [c2485923946] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:297 · reply to [c3408900920] · 2025-11-03
great, thanks for adding this example! 

@eustlb just want to double-check that this API looks good? (related to offline discussion we had with Lysandre) 

We went for something similar to Voxtral: https://github.com/huggingface/transformers/blob/b9f90dc388fd415a2ba2a6a31a372f451d4a4eed/src/transformers/models/voxtral/processing_voxtral.py#L278

## [c3479783906] ebezzam (CONTRIBUTOR) · issue_comment · 2025-11-03
run-slow: audioflamingo3

## [c3479892292] ebezzam (CONTRIBUTOR) · issue_comment · 2025-11-03
run-slow: audioflamingo3

## [c2486053872] ebezzam (CONTRIBUTOR) · review_comment on tests/models/audioflamingo3/test_modeling_audioflamingo3.py:267 · 2025-11-03
Can you put this file (and others) within an [HF dataset](https://huggingface.co/new-dataset)? 

For example:
- Qwen2Audio ([test](https://github.com/huggingface/transformers/blob/main/tests/models/qwen2_audio/test_modeling_qwen2_audio.py#L344), [dataset](https://huggingface.co/datasets/raushan-testing-hf/audio-test))
- Voxtral ([test](https://github.com/huggingface/transformers/blob/b9f90dc388fd415a2ba2a6a31a372f451d4a4eed/tests/models/voxtral/test_modeling_voxtral.py#L252), [dataset](https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples))

## [c2486054674] ebezzam (CONTRIBUTOR) · review_comment on tests/models/audioflamingo3/test_modeling_audioflamingo3.py:308 · 2025-11-03
HF dataset link

## [c2486055000] ebezzam (CONTRIBUTOR) · review_comment on tests/models/audioflamingo3/test_modeling_audioflamingo3.py:323 · 2025-11-03
HF dataset link

## [c2486055568] ebezzam (CONTRIBUTOR) · review_comment on tests/models/audioflamingo3/test_processing_audioflamingo3.py:147 · 2025-11-03
HF dataset link

## [c2486055895] ebezzam (CONTRIBUTOR) · review_comment on tests/models/audioflamingo3/test_processing_audioflamingo3.py:161 · 2025-11-03
HF dataset link

## [c2486061140] ebezzam (CONTRIBUTOR) · review_comment on tests/models/audioflamingo3/test_modeling_audioflamingo3.py:267 · reply to [c2486053872] · 2025-11-03
If you can also use HF dataset links in the examples, that would also be great.

But at least within the tests.

## [c3410126986] ebezzam (CONTRIBUTOR) · review COMMENTED · reply to [c3478999316] · 2025-11-03
@lashahub thanks for the changes! I also pushed some small changes, and left some comments for some minor things to address:
- Transcription output
- Creating a mini HF dataset for the testing samples (which can also be used to store samples for Music Flamingo)

Regarding the failing tests:
- `test_eager_matches_fa2_generate` and `test_flash_attn_2_can_dispatch_composite_models` are fine on my machine and also on the [GitHub CI](https://github.com/huggingface/transformers/actions/runs/19031008789/job/54345151607) so should be fine.
- On the GitHub CI, there are some issues with loading the model for the integration tests. I'm checking internally about that. Otherwise the tests passed on my machine 👍

## [c2486156402] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:164 · reply to [c2473767582] · 2025-11-03
thanks for the suggestion, I've done just that:
- shift mask preparation to the encoder's `forward` method
- rename attention mask input of the `forward` method to `input_features_mask`

## [c3410574458] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-11-03 (no text)

## [c2486258509] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:159 · 2025-11-03
Ok to spit out this warning to users? Essentially that audio larger than 10 min will be truncated.

## [c2486281860] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:217 · 2025-11-03
Do we want to support this type of logic? Which allows users to provide inputs with/without sound tokens in the prompt

Or do we remove `n_placeholders == 0` case? (which I'm not aware of in other models)

## [c3480261320] ydshieh (COLLABORATOR) · issue_comment · 2025-11-03
run-slow: audioflamingo3

## [c2486494583] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:200 · 2025-11-03
thoughts on this convenient method for transcription/ASR? so that users don't have to manually create the chat template (related to offline discussion we had with Lysandre) 

We went for a method name like Voxtral's: https://github.com/huggingface/transformers/blob/b9f90dc388fd415a2ba2a6a31a372f451d4a4eed/src/transformers/models/voxtral/processing_voxtral.py#L278

Example usage (in the model doc):
```python
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

MODEL_ID = "nvidia/audio-flamingo-3-hf"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AudioFlamingo3ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto")

inputs = processor.apply_transcription_request(audio="https://audioflamingo3.github.io/static/long_speech/t_837b89f2-26aa-4ee2-bdf6-f73f0dd59b26.wav").to(model.device)

generated = model.generate(**inputs)
transcription = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

print(transcription[0])
```

## [c3410718565] ebezzam (CONTRIBUTOR) · review COMMENTED · reply to [c3408900920] · 2025-11-03
@eustlb A self-review for decisions made in the processor, if we can get your feedback. Thanks!

## [c2486536689] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:217 · reply to [c2486281860] · 2025-11-03
For reference:
- Qwen2Audio: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_audio/processing_qwen2_audio.py#L137-L164
- CSM: https://github.com/huggingface/transformers/blob/main/src/transformers/models/csm/processing_csm.py#L280-L293

## [c3411099069] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-11-03 (no text)

## [c3481017536] ebezzam (CONTRIBUTOR) · issue_comment · 2025-11-03
run-slow: audioflamingo3

## [c3481539099] ebezzam (CONTRIBUTOR) · issue_comment · 2025-11-03
run-slow: audioflamingo3

## [c2487584105] lashahub (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:266 · reply to [c2485922301] · 2025-11-03
The model is trained to output transcriptions in that style. We can simply post-process the decoded string with a regex to strip common prefixes like `The transcription of the audio is` or `The spoken content of the audio is` and return only the text inside the quotes.

## [c3412532187] lashahub (CONTRIBUTOR) · review COMMENTED · 2025-11-03 (no text)

## [c3483912451] lashahub (CONTRIBUTOR) · issue_comment · reply to [c3481539099] · 2025-11-04
Hi @ebezzam, I’ve made minor updates to the converter and modified `AudioFlamingo3MultiModalProjector.__init__()` so that the weights remain in BF16.

## [c3484017989] ydshieh (COLLABORATOR) · issue_comment · 2025-11-04
run-slow: audioflamingo3

## [c3485294271] ebezzam (CONTRIBUTOR) · issue_comment · reply to [c3483912451] · 2025-11-04
@lashahub thanks! The model now loads on the CIs without memory issues. 

However, could you try recomputing the expected outputs in BF16? (maybe setting dtype in your reproducer [here](https://gist.github.com/lashahub/185b080ca87fe82fd397bc621e74efcb#file-reproducer-py-L77)) 

I'm wondering if that's what's causing the output differences (see [here](https://github.com/huggingface/transformers/actions/runs/19059173040/job/54435669823#step:14:682))

## [c3487196729] lashahub (CONTRIBUTOR) · issue_comment · reply to [c3485294271] · 2025-11-04
@ebezzam I wasn’t able to reproduce the test failures locally. I reinstalled the latest requirements using `uv pip install -e ".[dev-torch]"`, and both tests are passing on my end. Could you please try running them locally on your side to confirm?

## [c3487551312] ebezzam (CONTRIBUTOR) · issue_comment · 2025-11-04
run-slow: audioflamingo3

## [c3488303156] lashahub (CONTRIBUTOR) · issue_comment · reply to [c3484017989] · 2025-11-04
Hi @ydshieh could you run the slow tests? Thanks

## [c3489910279] ebezzam (CONTRIBUTOR) · issue_comment · 2025-11-05
run-slow: audioflamingo3

## [c3489958417] ebezzam (CONTRIBUTOR) · issue_comment · reply to [c3488303156] · 2025-11-05
@lashahub all passes! let's go with these audio

## [c2493537997] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:31 · 2025-11-05
Could you also add example usage at the model page? This helps increase usage, see [Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) and [Voxtral](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)

## [c2493596925] ebezzam (CONTRIBUTOR) · review_comment on tests/models/audioflamingo3/test_modeling_audioflamingo3.py:250 · 2025-11-05
thanks for updating! and helped me a lot for my debugging 👍

## [c2493618272] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:266 · reply to [c2485922301] · 2025-11-05
I think that could be convenient for users and ASR benchmarks. What do you think @eustlb?

## [c3420609320] ebezzam (CONTRIBUTOR) · review COMMENTED · reply to [c3488303156] · 2025-11-05
thanks a bunch @lashahub for your work! As mentioned below, could you also add example usage to: https://huggingface.co/nvidia/audio-flamingo-3-hf

## [c2493704174] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:266 · reply to [c2485922301] · 2025-11-05
@lashahub confirmed with @eustlb.

Could you add a flag to `processor.batch_decode`? Something like `strip_prefix` to return only the text inside the quotes + a docstring explanation as above on how the model was trained to outputs transcriptions in that style, etc

## [c3420868489] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-11-05 (no text)

## [c2493716184] ebezzam (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/audioflamingo3.md:329 · reply to [c3488303156] · 2025-11-05
@lashahub related to [this discussion](https://github.com/huggingface/transformers/pull/40290/files#r2486281860) which I continued offline with @eustlb , I'm going to remove support for such usage, as we prefer usage through the standardized chat template.

## [c3420888763] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-11-05 (no text)

## [c3490345590] ebezzam (CONTRIBUTOR) · issue_comment · 2025-11-05
run-slow: audioflamingo3

## [c3494017812] lashahub (CONTRIBUTOR) · issue_comment · reply to [c3490345590] · 2025-11-05
Hi @ebezzam, I’ve updated the [model page](https://huggingface.co/nvidia/audio-flamingo-3-hf) with usage examples and added a `strip_prefix` parameter to `AudioFlamingo3Processor.batch_decode` (set to `False` by default) for ASR.

## [c2497734425] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:200 · reply to [c2486494583] · 2025-11-06
Discussion continued here: https://github.com/huggingface/transformers/pull/40290/files#r2485922301

## [c3426552439] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-11-06 (no text)

## [c3495753607] ebezzam (CONTRIBUTOR) · issue_comment · reply to [c3408900920] · 2025-11-06
@ArthurZucker could we get your review on this audio-text-to-text model? (Whisper encoder to Qwen2 LLM)
Model page: https://huggingface.co/nvidia/audio-flamingo-3-hf

Similar models which are used in modular:
- Voxtral ([model](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), [code](https://github.com/huggingface/transformers/tree/main/src/transformers/models/voxtral))
- Qwen2Audio ([model](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct), [code](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_audio))

A difference you'll see in modular (which is a nice aspect here), this model accounts for the attention mask in the encoder. See discussion [here](https://github.com/huggingface/transformers/pull/40290#discussion_r2473767582) with @eustlb

## [c3500537500] Dong-mel (NONE) · issue_comment · reply to [c3494017812] · 2025-11-07
Hi @lashahub , thanks to the open-source model and code support!

However, after installing transformers using pip
```
pip install --upgrade git+https://github.com/huggingface/transformers
```
I found that the current version does not support AudioFlamingo3ForConditionalGeneration.
```
>>> from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ImportError: cannot import name 'AudioFlamingo3ForConditionalGeneration' from 'transformers' (/opt/conda/lib/python3.11/site-packages/transformers/__init__.py)
```
Does it require a specific branch?

## [c3500581791] lashahub (CONTRIBUTOR) · issue_comment · reply to [c3500537500] · 2025-11-07
Hi @Dong-mel , it hasn’t been merged into the main branch yet. You can currently use the [audioflamingo3 branch](https://github.com/jsalt-2025/transformers/tree/audioflamingo3) from the JSALT repo instead.
Hopefully, it’ll be merged within the next few days.

## [c3500939119] Dong-mel (NONE) · issue_comment · 2025-11-07
> Hi @Dong-mel , it hasn’t been merged into the main branch yet. You can currently use the [audioflamingo3 branch](https://github.com/jsalt-2025/transformers/tree/audioflamingo3) from the JSALT repo instead. Hopefully, it’ll be merged within the next few days.

Got it, thank you!

## [c2513892533] eustlb (COLLABORATOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:315 · 2025-11-11
here the is no reasons not to use language model loss directly, which is equivalent. Not that for padding tokens that are masked here using the attention mask, they should simply be prepared before with -100 in the processor.

```suggestion
    outputs: CausalLMOutputWithPast = self.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        labels=labels,
        use_cache=use_cache,
        cache_position=cache_position,
        logits_to_keep=logits_to_keep,
        **kwargs,
    )
    return outputs
```

## [c2513910545] eustlb (COLLABORATOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:41 · 2025-11-11
This can be safely removed (it will be initialized directly thanks to the encoder layer)
```suggestion
```

## [c2513921995] eustlb (COLLABORATOR) · review_comment on src/transformers/models/audioflamingo3/configuration_audioflamingo3.py:91 · reply to [c3495753607] · 2025-11-11
nice @ebezzam! indeed those introduced in Qwen2Audio are not the correct ones. Way better like this where the encoder have standard config key names

## [c2513926170] eustlb (COLLABORATOR) · review_comment on src/transformers/models/voxtral/modular_voxtral.py:89 · 2025-11-11
nice catch

## [c2514493029] eustlb (COLLABORATOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:131 · 2025-11-11
+ no the following renaming of `texts` in `text`
```suggestion
        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")
```

## [c2514507047] eustlb (COLLABORATOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:47 · 2025-11-11
standard is audio_token throughtout the lib. Moreover, they should not be in the kwargs but has processor attributes directly (see[ llava example](https://github.com/huggingface/transformers/blob/2072f3059eea2d5c5061516b8fc089853e50a654/src/transformers/models/llava/processing_llava.py#L63-L86))

## [c2514513101] eustlb (COLLABORATOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:57 · 2025-11-11
overall here for coding style: let's not bloat with typing where it is not adding understanding value

## [c2514714138] eustlb (COLLABORATOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:177 · 2025-11-11
Strongly against this! this is really misleading for the user: the `input_features_mask` **is not** an `input_features_mask`, but something resize to the ouput shape of the encoder convs. Such a reshaping should be done in the model itself, there is no reasons to do it here

## [c3447620131] eustlb (COLLABORATOR) · review COMMENTED · 2025-11-11
Great work guys, thanks a lot for the contribution! 👏
LGTM but still few things, the only truly blocking one is the `input_feature_mask` that is not one! I am going to push directly few changes in order to shortcut the process and I'll approve right after.

## [c2514853269] ebezzam (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/processing_audioflamingo3.py:177 · reply to [c2514714138] · 2025-11-11
That was me! Related to [this discussion](https://github.com/huggingface/transformers/pull/40290#discussion_r2486156402) (partly offline), I should have also moved this mask preparation to the encoder

## [c3448872659] ebezzam (CONTRIBUTOR) · review COMMENTED · 2025-11-11 (no text)

## [c2514884523] eustlb (COLLABORATOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:140 · 2025-11-11
this should definitely not be necessary

## [c3448910409] eustlb (COLLABORATOR) · review COMMENTED · 2025-11-11
forgot this in the former review

## [c2515033719] eustlb (COLLABORATOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:140 · reply to [c2514884523] · 2025-11-11
@lashahub I'd like your input on this, why was this necessary?

## [c3449102141] eustlb (COLLABORATOR) · review COMMENTED · 2025-11-11 (no text)

## [c2515095235] lashahub (CONTRIBUTOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:140 · reply to [c2514884523] · 2025-11-11
@eustlb I noticed that when using the [converter](https://github.com/jsalt-2025/transformers/blob/e63e295c47cc4dde98778e0b9435a3d3730320b8/src/transformers/models/audioflamingo3/convert_audioflamingo3_to_hf.py#L179) the projector weights were [coming out](https://huggingface.co/nvidia/audio-flamingo-3-hf) in `fp32` even though the [originals](https://huggingface.co/nvidia/audio-flamingo-3) were in `bf16`, so this was needed to make sure the dtype stayed consistent during load.

## [c3449183971] lashahub (CONTRIBUTOR) · review COMMENTED · 2025-11-11 (no text)

## [c3449204293] eustlb (COLLABORATOR) · review APPROVED · 2025-11-11
LGTM! 🎉

## [c2515112398] eustlb (COLLABORATOR) · review_comment on src/transformers/models/audioflamingo3/modular_audioflamingo3.py:140 · reply to [c2514884523] · 2025-11-11
strange, not happening on my end when using `from_pretrained(dtype=something)`. Such thing should therefore be handled at conversion, leaving you add it but not critical

## [c3449209916] eustlb (COLLABORATOR) · review COMMENTED · 2025-11-11 (no text)

## [c3518130785] ebezzam (CONTRIBUTOR) · issue_comment · 2025-11-11
run-slow: audioflamingo3

## [c3518256180] ebezzam (CONTRIBUTOR) · issue_comment · reply to [c3449209916] · 2025-11-11
@eustlb thanks for the review and standardization commits!

@lashahub can you look into the changes [here](https://github.com/huggingface/transformers/pull/40290/commits/e63e295c47cc4dde98778e0b9435a3d3730320b8)? I think something might have introduced an error in the integration tests (see [here](https://github.com/huggingface/transformers/actions/runs/19274337788/job/55111004193#step:14:558)) (or similar issue as before and can try different test examples)

Otherwise I can take a look tomorrow

## [c3520262586] lashahub (CONTRIBUTOR) · issue_comment · reply to [c3518256180] · 2025-11-12
@ebezzam The tests were converting the batch to `bf16` before generation, so I removed all those conversions, it’s working fine on my end now. I also replaced one of the audio files, even though the previous one would’ve worked fine.

## [c3520618237] ebezzam (CONTRIBUTOR) · issue_comment · 2025-11-12
run-slow: audioflamingo3

## [c3520972414] ebezzam (CONTRIBUTOR) · issue_comment · reply to [c3520262586] · 2025-11-12
@lashahub we still need to keep the models in bf16 for the tests, otherwise they won't load properly (see [here](https://github.com/huggingface/transformers/actions/runs/19290969625/job/55161919802#step:14:767))

## [c3520974007] ebezzam (CONTRIBUTOR) · issue_comment · 2025-11-12
run-slow: audioflamingo3

## [c3521002267] ebezzam (CONTRIBUTOR) · issue_comment · reply to [c3520262586] · 2025-11-12
@lashahub could you also add this training snippet: https://github.com/huggingface/transformers/pull/40290/commits/8bf40fa607c6b3e920ed6563893e723b80f18ea4

to the model page: https://huggingface.co/nvidia/audio-flamingo-3-hf

## [c3453331074] ArthurZucker (COLLABORATOR) · review APPROVED · reply to [c3521002267] · 2025-11-12
Kudos everyone and thanks @eustlb and @ebezzam very clean 😉

## [c3521850613] ebezzam (CONTRIBUTOR) · issue_comment · 2025-11-12
run-slow: audioflamingo3

## [c3453628420] ebezzam (CONTRIBUTOR) · review APPROVED · reply to [c3453331074] · 2025-11-12
Thanks @lashahub and @Sreyan88 for the great work, and @eustlb and @ArthurZucker for the feedback 🤗 

Merging!

---
outcome: state=closed, merged=True
