# #40962 [pull_request] perceptron: Isaac-0.1 implementation
https://github.com/huggingface/transformers/pull/40962  ·  opened by AkshatSh on 2025-09-18

## [body] AkshatSh (NONE) · opening post · 2025-09-18
# Perceptron Isaac Implementation

Perceptron released open weight models [Isaac-0.1](https://huggingface.co/PerceptronAI/Isaac-0.1) and [Isaac-0.1-Base](https://huggingface.co/PerceptronAI/Isaac-0.1-Base) a 2B dense model  for perception.

## [c3308332674] zucchini-nlp (MEMBER) · issue_comment · 2025-09-18
Nice, lmk if you need any help or a quick review 🤗

## [c3313695570] AkshatSh (NONE) · issue_comment · reply to [c3308332674] · 2025-09-19
@zucchini-nlp  - we are just closing this out feel free to give us a quick review while we just finish up testing!

## [c3314978426] zucchini-nlp (MEMBER) · issue_comment · 2025-09-20
amazing, reviewing on Monday :)

## [c2372259614] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:48 · 2025-09-23
hmm is this for streaming video data? I am not sure we can add it as is, we need to either safe import it or find a way to integrate it in `self.video_processor`

## [c2372261663] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:56 · 2025-09-23
we also need to add the two new args in the docstring

## [c2372298280] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:53 · 2025-09-23
we need to call it `PerceptronVisionConfig` with the model's name

## [c2372300366] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:107 · 2025-09-23
we can prepare it once when model is init, no?

## [c2372304100] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:113 · 2025-09-23
used only once, we don't need to save in var and can move directly to interpolate call

## [c2372307280] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:128 · 2025-09-23
if it should never happen, maybe we need to raise an error here?

## [c2372313050] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:109 · 2025-09-23
the first two var are unused, we can either skip them `*_` or obtain `packed_seq_patches[-1]`

## [c2372315353] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:139 · 2025-09-23
same here, of not used let's just get `packed_seq_patches[0]` and use it below

## [c2372319495] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:83 · 2025-09-23
The name for all modules need to follow the format of "model_name+layer_name": `IsaacXXX` or ``IsaacVisionXXX``

## [c2372329209] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:211 · 2025-09-23
ideally we need to allow all types of attentions with `ALL_ATTENTION_FUNCTIONS`. I believe the tiny diffs in computation will not alter the final preds much, can you update it and check?

## [c2372336122] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:228 · 2025-09-23
`init` looks same as Siglip2EncoderLayer, we can use modular inheritance

## [c2372339157] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:237 · 2025-09-23
should not be re-using existing layers from other models. If we let the modular copy `init`, it will add an `IsaacMLP` automatically by copying from Siglip2 :)

## [c2372341440] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:272 · 2025-09-23
same here, can be copied with modular

## [c2372349006] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:332 · 2025-09-23
`scale_factor = int(scale_factor)` for better naming

## [c2372351213] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:336 · 2025-09-23
`from ...utils import is_torchdynamo_compiling` will do the check :)

## [c2372391017] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:81 · 2025-09-23
used only once, we don't need a utility and can compute cu-seqlens in the forward pass

## [c2372414223] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:424 · 2025-09-23
why we can't apply pixel shuffle directly to the input images, since they are in `(bs, seqlen, dim)` format already? This looks over-engineered, let's do similar to https://github.com/huggingface/transformers/blob/946e5f95ea0894e5c88f7915547ed3ffcafa56cd/src/transformers/models/internvl/modeling_internvl.py#L689-L723

## [c2372430318] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:441 · 2025-09-23
embeddings does not use `seq_sizes`, no need to pass it. Also, I'd prefer to forward seq_patches and  token_grids as separate input args

## [c2372433805] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:481 · 2025-09-23
These have to be serialized in processing config and set as defaults in processor's `init`

## [c2372440136] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:509 · 2025-09-23
we need to add `IsaacImageProcessorFast` which handles loading and preparaing images. All image transform ops should go in there, see Siglip2 as an example

## [c2372569608] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:549 · 2025-09-23
let's move it outside of the fn

## [c2372571244] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:598 · 2025-09-23
same as above, defaults need to be defined in the config and processor's `init`

## [c2372572935] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:662 · 2025-09-23
has to be in `IsaacImageProcessorFast._preprocess`

## [c2372576616] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:730 · 2025-09-23
for simple RoPE, we can use the `ROPE_INIT_FUNCTIONS` similar to llama and other LLMs. It supports most common rope types

## [c2372580853] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:789 · 2025-09-23
let's have rope params in `config.text_config.rope_scaling`. We are planning a huge rope refactor, and until then it will be easier to follow the existing format

## [c2372584636] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:912 · 2025-09-23
not needed to override

## [c2372588193] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:885 · 2025-09-23
the processor class does not expect config and needs to be initialized with `tokenizer ,image-processor, kwargs`

## [c2372591293] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:818 · 2025-09-23
we need a text config as well

## [c2372598488] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1129 · 2025-09-23
if the text model is identical to qwen3, we need to init with `AutoModel.from_config(text_config)`. Otherwise we need to add `IsaacTextModel` by copying with modular

## [c3257910862] zucchini-nlp (MEMBER) · review COMMENTED · 2025-09-23
Nice job. I did not review everything and went over to check if the PR follows transformers API. The most important comments are:

1. We need to add image processor class and wrap image transforms inside it
2. Is the "tensorstream" required to get the model running? AFAIU we can discard it which is prob what we'll need to do. Transformers does not support streaming of multimodals yet
3. The model structure has to follow standards of other VLMs with correct class names

LMK if you have any questions 🤗

## [c3390581396] philippguevorguian (NONE) · issue_comment · reply to [c3257910862] · 2025-10-10
@zucchini-nlp just pushed a round of updates addressing your feedback; give it another look when you get a moment

## [c2420654176] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:424 · reply to [c2372414223] · 2025-10-10
bs is 1 here for varlen attention, so we can’t apply pixel shuffle directly - this design is actually necessary to make it work cleanly

## [c3324072897] philippguevorguian (NONE) · review COMMENTED · 2025-10-10 (no text)

## [c2421631121] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:15 · 2025-10-10
let's add license in the header

## [c2421637242] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:19 · 2025-10-10
this is usually imported with `is_torchvision_available()` from `from ...utils import is_torchvision_available. If it is used only in the fast image processing file, then we don't even need to safe import  :)

## [c2421646363] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:77 · 2025-10-10
the mean and std are available as IMAGENET_STANDARD_MEAN and STD in `image_utils`. The other onecan just be assigned a var when needed. Let's not have global var

## [c2421651564] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:149 · 2025-10-10
these are already defined in base class, no need to duplicate

## [c2421655093] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:160 · 2025-10-10
the class attr here should be all the image args with their default values, i.e. `do_resize, size, image_mean` etc. The attr `slow_image_processor_class` is not needed

## [c2421658631] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:187 · 2025-10-10
it is handled by `self.normalize`, no need

## [c2421660802] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:199 · 2025-10-10
`super().init()` will assign all args as long as we have the defaults set as cls attr

## [c2421662501] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:174 · 2025-10-10
let's have only kwargs with unpacking, these are already defined in the typed dict

## [c2421666113] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:225 · 2025-10-10
torchvision is guaranteed to be installed if fast processor is used, no need to check

## [c2421667589] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:240 · 2025-10-10
same, the super class handles arg preparation in fast processors, not needed

## [c2421670712] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:246 · 2025-10-10
lets follow the format as in most fast processors with grouping

https://github.com/huggingface/transformers/blob/7164924a7e83f223a2bf2e104bef98eabe545091/src/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py#L178-L197

## [c2421672739] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:254 · 2025-10-10
RGB conversion is called before processing in super class

## [c2421675266] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:270 · 2025-10-10
this line is not needed, handled in super

## [c2421677105] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:287 · 2025-10-10
if `self.do_rescale`?

## [c2421681587] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:331 · 2025-10-10
can we delete the slow processor, since we will be using only the fast version by default starting v5. So it is not needed anymore to have both

## [c2421683918] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:128 · reply to [c2372307280] · 2025-10-10
not addressed

## [c2421688703] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:741 · 2025-10-10
Lets try to use the standard attention API with `ALL_ATTENTION_FUNCTIONS` as in llama. It will allow us to support all attention functions inclusing custom ones from users

## [c2421691116] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:778 · 2025-10-10
typo? we can return it as is without tuple

## [c2421703710] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:424 · reply to [c2372414223] · 2025-10-10
if bs is `1` then we can unsqueeze the batch dim, apply pixel shuffle in standard way and then squeeze it back for attention layer? I didn't really read through the whole fn, so lmk if I am missing smth here

We need to try to make it standard way if possible, so it is easier to maintain and debug

## [c2421706194] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:998 · 2025-10-10
same, global vars are better assigned where it is used. In this case prob it's a cls attr of a fast image processor

## [c2421715164] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:789 · reply to [c2372580853] · 2025-10-10
comment still relevant. We will be merging the refactor next week hopefully, and we will add a typing hint with typed dict there. For now let's follow the standard way and type hint as simple `dict`

## [c2421721478] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1233 · 2025-10-10
these are params from processing. Do we really need them for model code? I did not see where they are used and usually modeling does not need to know how processing happened

## [c2421723114] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1234 · 2025-10-10
not needed, the vision config has its own attn implementation attribute

## [c2421725038] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1241 · 2025-10-10
i think the super `get_text_config` works well, or is it raising errors?

## [c3325333673] zucchini-nlp (MEMBER) · review COMMENTED · 2025-10-10
Thanks for iterating! I gave a quick look at the first ~1500 lines. 

We are currently insisting on following certain standards before merging models, as it causes many issues otherwise. Overall the code looks good but I feel like the models still needs more standardization. So I left some comments and I will review the rest on Monday.Not sure if we can keep the `tensorstream`, let me see in detail how it is used

## [c2426587209] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:424 · reply to [c2372414223] · 2025-10-13
Currently the samples can be of unequal length, so additional metadata is used to correctly identify samples boundaries without padding.

## [c3331958105] philippguevorguian (NONE) · review COMMENTED · 2025-10-13 (no text)

## [c2426590688] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:128 · reply to [c2372307280] · 2025-10-13
We currently raise an error on the line below (129)

## [c3331963028] philippguevorguian (NONE) · review COMMENTED · 2025-10-13 (no text)

## [c3406518686] philippguevorguian (NONE) · issue_comment · reply to [c3325333673] · 2025-10-15
@zucchini-nlp just pushed another round of refinements based on the latest feedback. Would love for you to give it another pass when you get a chance

## [c3422759418] philippguevorguian (NONE) · issue_comment · reply to [c3325333673] · 2025-10-20
Hi @zucchini-nlp, gentle ping on this PR when you get a chance. Thanks for your time!

## [c3456460091] philippguevorguian (NONE) · issue_comment · reply to [c3325333673] · 2025-10-28
Hey @zucchini-nlp - just wanted to follow up again on this PR. It’s been a little while since the last round, and your review would help us move things forward. Appreciate it!

## [c2472550020] molbap (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1 · 2025-10-29
Missing date and usual format header

## [c2472576782] molbap (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:111 · 2025-10-29
This is the biggest hurdle for now: if I understand correctly this is an if/else path for TensorStream. We don't want to add a new dependency to `transformers` (and here the import will simply fail).

## [c2472577782] molbap (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:186 · 2025-10-29
should not be needed

## [c2474160887] molbap (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:2255 · 2025-10-29
This is not needed: mappings in `processing_auto` are enough

## [c3392734131] molbap (COLLABORATOR) · review COMMENTED · 2025-10-29 (no text)

## [c3473054167] philippguevorguian (NONE) · issue_comment · 2025-10-31
> Leaving a comment and question from a quick review: Is it absolutely necessary for the core functionality of the model to be preserved to have dependencies in tensorstream?

TensorStream is a core primitive for us; it underpins how we handle multimodal computation, and future open-source releases will include incremental improvements to its ergonomics and performance aligned with model updates. We see it as the right abstraction boundary for multimodal modelling.

That said, we’ll be guarding the imports here to avoid issues on that front. As a heads up, we'll also following up with changes here that address feedback around pattern-reuse standards in transformers, feedback there is much appreciated.

## [c3473374838] molbap (COLLABORATOR) · issue_comment · reply to [c3473054167] · 2025-10-31
Thanks @philippguevorguian , then I will see if we can integrate it. I can't say for now if we'll be able to integrate it as our principles for modeling files is to not abstract too much and keep the models "hackable", meaning a user should be able to intervene at any given point of a model's forward, adding a module, modifying the processing, etc. But if the rest is more `transformers`-aligned, and since the model is relevant in general, it'll be easier.

In particular attention classes that currently are relying on for instance `integrations/flash_attention.py` should be used for FA. To explain: the policy is to have one "naive" path, `eager_attention_forward`, that is a `Callable` explicitly defined in the modeling code and serves as a baseline for attention computation. The optimized paths (sdpa, fa, flex, etc) and associated masks are handled through config keys that swap this `Callable` into another one that wraps an efficient fa kernel. 

For cross-document masking, combinations of existing utils should be sufficient, like `masking_utils.py` that defines `and` and `or` operators, rather than specific utils for this model. If these two "reuse" points are addressed it'll be excellent

## [c3542291529] philippguevorguian (NONE) · issue_comment · reply to [c3473374838] · 2025-11-17
@molbap would appreciate another round of reviews here

## [c3546355176] molbap (COLLABORATOR) · issue_comment · reply to [c3542291529] · 2025-11-18
Hi @philippguevorguian, the `main` branch is very unstable right now as the next major version (v5) is around the corner, but will review soon!

## [c3618044490] molbap (COLLABORATOR) · issue_comment · 2025-12-05
I am currently reviewing the totality but wanted to come back on `tensorstream`. One thing that'll help the CI already will be to protect `tensorstream` imports that are not present in our test images: the majority of failing tests are just because the imports aren't guarded. The best would be to have an equivalent codepath in `PyTorch` for the utilities it covers, akin to how for instance we can run `mamba` models without their dedicated tensor contraction kernels.

## [c3631187431] philippguevorguian (NONE) · issue_comment · reply to [c3618044490] · 2025-12-09
> protect `tensorstream` imports that are not present in our test images: the majority of failing tests are just because the imports aren't guarded.

Added the proper guards around the problematic imports, which clears out most of the CI failures

## [c3632366670] ArthurZucker (COLLABORATOR) · issue_comment · 2025-12-09
Ty will review today !

## [c2603181648] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/model_doc/isaac.md:44 · 2025-12-09
very nice!

## [c2603205542] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/configuration_isaac.py:87 · 2025-12-09
Are you sure this is meant to be in the code? We can only have MIT / Apache2 in the code the rest would change `transformers` whole scope of licence. Put this on the hub not on the code please

## [c2603211796] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:231 · 2025-12-09
should be removed

## [c2603213188] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:231 · reply to [c2603211796] · 2025-12-09
there is no reason for this model to for this vs the usual path

## [c2603218788] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:510 · 2025-12-09
dtype does not look right here?

## [c2603223757] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:489 · 2025-12-09
not sure this fits in the paradigm we have of passing callables! it should be a function like this one: https://github.com/huggingface/transformers/blob/0ec5befc8d394d01a79319ae5e68a7e7bcb9f6a9/src/transformers/masking_utils.py#L143-L151

## [c2603240069] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:814 · 2025-12-09
this is really wrong. The api is designed to support all of this OOB, you are really not leveraging the correct way 😰 

The attention interface you are using is plain / simple the same as the other models we have, you can just copy past what SiglipAttention has with eager forward and etc. 

Supporting `cu_seq_lens` and etc should be completely outside. There are good standards already this should be much much simpler

## [c2603243866] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:987 · 2025-12-09
not the way we usually handle this!

## [c2603249586] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1184 · 2025-12-09
are there any model that don't use this?

## [c2603253907] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1519 · 2025-12-09
its not expected that you have so many things to change here

## [c2603269706] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1878 · 2025-12-09
this should not be done, or we have to fix something on `trnasfomers` side

## [c2603274773] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1899 · 2025-12-09
let's remove sequential please and create a proper class. Renaming can be done easily with the new:
```python
    _checkpoint_conversion_mapping = {
        r"^language_model.model": "model.language_model",
        r"^vision_tower": "model.vision_tower",
        r"^multi_modal_projector": "model.multi_modal_projector",
        r"^language_model.lm_head": "lm_head",
    }
```

or the dynamic weight loader

## [c2603277534] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1944 · 2025-12-09
not usually something we expose

## [c2603280243] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1985 · 2025-12-09
missing the classic decorator `auto_docstring` and `check_model_inputs`

## [c2603280934] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:2010 · 2025-12-09
```suggestion
```

## [c2603285923] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:2173 · 2025-12-09
this is a very old API let's update (see llama)

## [c2603290473] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:2290 · 2025-12-09
this is done in the model as well no?

## [c2603293343] ArthurZucker (COLLABORATOR) · review_comment on tests/fixtures/isaac/isaac_generation_golden.json:1 · 2025-12-09
should be in fixtures if kept

## [c2603295324] ArthurZucker (COLLABORATOR) · review_comment on tests/models/isaac/test_modeling_isaac.py:17 · 2025-12-09
missing the licence we usually have

## [c2603301673] ArthurZucker (COLLABORATOR) · review_comment on tests/models/isaac/test_modeling_isaac.py:1 · 2025-12-09
most important is:
1. follow llava-like model tests
2. have proper integration tests that show how we would use the model 
3. add perceptron to the list of optional deps to instal on test in our workflow, I can help with that!

## [c3558111293] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-12-09
Good BUT let's update the attention interface please and mask to our most recent / recommended way!

## [c2614952910] philippguevorguian (NONE) · review_comment on tests/fixtures/isaac/isaac_generation_golden.json:1 · reply to [c2603293343] · 2025-12-12
Thanks, moved to fixtures

## [c3572788280] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2614954890] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:510 · reply to [c2603218788] · 2025-12-12
We now explicitly cast to dtype min, should be a no-op change

## [c3572790984] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2614956594] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:231 · reply to [c2603211796] · 2025-12-12
Thanks, dropped it

## [c3572793073] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2614958041] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/configuration_isaac.py:87 · reply to [c2603205542] · 2025-12-12
Updated the license !

## [c3572794888] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2614960379] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:489 · reply to [c2603223757] · 2025-12-12
Please take a look at the implementation in use now, should be aligned with paradigm but would appreciate confirmation here

## [c3572798101] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2614961631] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1184 · reply to [c2603249586] · 2025-12-12
Good point, dropped any conditional checks of pixel_shuffle_scale_factor

## [c3572799760] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2614963364] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1519 · reply to [c2603253907] · 2025-12-12
Greatly simplified the config in most recent changes

## [c3572802227] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2614970344] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1878 · reply to [c2603269706] · 2025-12-12
Removed the explicit attn_implementation handling. Without the deepcopy the config gets mutated by AutoModel init; would appreciate any guidance on the idiomatic approach for handling this or if an isolated repro would help internal efforts on resolving this

## [c3572812311] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2614970622] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1899 · reply to [c2603274773] · 2025-12-12
Done !

## [c3572812617] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2614972354] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1944 · reply to [c2603277534] · 2025-12-12
Dropped gradient_checkpointing; embed_text_tokens is a utility we use for readability so would prefer to leave it in but can inline it if this is a high priority change

## [c3572815244] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2614975279] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1985 · reply to [c2603280243] · 2025-12-12
Ah, good catch. Added

## [c3572819750] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2614976372] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:2173 · reply to [c2603285923] · 2025-12-12
Upgraded to new implementation !

## [c3572821318] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2614977687] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:2290 · reply to [c2603290473] · 2025-12-12
Just to clarify, would you prefer an abstraction here to unify the logic?

## [c3572823306] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2614978546] philippguevorguian (NONE) · review_comment on tests/models/isaac/test_modeling_isaac.py:17 · reply to [c2603295324] · 2025-12-12
Updated this to be in line with other test files

## [c3572824582] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2614997682] philippguevorguian (NONE) · review_comment on tests/models/isaac/test_modeling_isaac.py:1 · reply to [c2603301673] · 2025-12-12
On 3, would definitely take you up on that. What's the proper way to go about doing this? 

Also please refer to some of the expanded testing w.r.t. (2), will be adding more but want to align on format

## [c3572849190] philippguevorguian (NONE) · review COMMENTED · 2025-12-12 (no text)

## [c2624177413] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:157 · 2025-12-16
identical args with siglip will be copied by modular, e.g. hidden_size. We can delete

## [c2624180384] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:180 · 2025-12-16
nit: attributes set to `None` and same as in base class can be deleted

## [c2624182506] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:214 · 2025-12-16
any reason for this? Let's add a comment, we'll forget why it's skipped after a few months

## [c2624194282] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:245 · 2025-12-16
Ig this is needed to check what is the resampling below and adjust `align_corners`? We can be sure that the interpolation is already a torchvision `InterpolationMode` and check directly with `mode_key in [InterpolationMode.BILINEAR, etc,etc]`

## [c2624196533] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:274 · 2025-12-16
the private method for preprocessing usually gets keyword args, why do we need this?

## [c2624204801] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:295 · 2025-12-16
in which cases it can be non 4D images? Unless we overwrite image preparation, these will be channel first images in all cases

## [c2624208599] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:300 · 2025-12-16
same, it will be an image with 3 channels already if `do_convert_rgb`, it's handled by base class

## [c2627126422] philippguevorguian (NONE) · review_comment on tests/models/isaac/test_modeling_isaac.py:1 · reply to [c2603301673] · 2025-12-17
@ArthurZucker should be good here w.r.t. tests, could you provide guidance on adding the dependency

## [c3587824815] philippguevorguian (NONE) · review COMMENTED · 2025-12-17 (no text)

## [c2627128615] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:814 · reply to [c2603240069] · 2025-12-17
Updated to attention_interface !

## [c3587827336] philippguevorguian (NONE) · review COMMENTED · 2025-12-17 (no text)

## [c2627129357] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:987 · reply to [c2603243866] · 2025-12-17
we now pass these values down without consuming a stash

## [c3587828271] philippguevorguian (NONE) · review COMMENTED · 2025-12-17 (no text)

## [c2627383814] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/image_processing_isaac.py:31 · 2025-12-17
haha that's not expected but also something that `modular` does because of the naming!

## [c2627414613] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:165 · 2025-12-17
```suggestion
class IsaacImageProcessorFastKwargs(ImagesKwargs, total=False):
```

## [c2627415595] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:204 · 2025-12-17
please remove the ones that are default

## [c2627417362] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:224 · 2025-12-17
cc @yonigozlan I find really weird that people have to do that no?

## [c2627425689] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:458 · 2025-12-17
block mask for `llama4` works the same! I think we don't need this no? again we want the api to be quite standard, and again, let's use the attention API that the we have which allows checking for q / k idx and etc no?

## [c2627427559] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:491 · 2025-12-17
why not use a nn.parameter?

## [c2627430320] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:530 · 2025-12-17
why are the weights not saved in a proper format directly>

## [c2627435540] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:606 · 2025-12-17
sequence packing in general should not be handled here, but in a custom generation loop unless this is very specific to this model, and in that case we need good documentation as to why / when we do this

## [c2627436846] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:627 · 2025-12-17
that is not something we do in transformers 😉

## [c2627438572] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:645 · 2025-12-17
in general should not be done here either, attention interface does that itself

## [c2627440036] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:651 · 2025-12-17
unused by sdpa, eager so should just not appear here

## [c2627441277] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:670 · 2025-12-17
why not leave this in the kwargs then? I don't understand

## [c2627442705] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:690 · 2025-12-17
same here this should not happen / no be here

## [c2627443852] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:698 · 2025-12-17
this should be remove + introduces a sync

## [c2627445560] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:748 · 2025-12-17
1. its always a tuple

## [c2627447891] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:757 · 2025-12-17
2. we no longer capture attention states this way! 
Checkout llama, we use `OutputRecorder` which allow to just not return anything here :)

## [c2627451421] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:785 · 2025-12-17
`return_mask_function=False` is always `False`

## [c2627453866] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:798 · 2025-12-17
why do you need to recreate them? just leave them as kwarg. For document masking read from kwargs

## [c2627456054] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:907 · 2025-12-17
please let's improve the naming

## [c2627458114] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:243 · 2025-12-17
you should inherit from `PreTrainedModel` otherwise this is doing nothing 😉

## [c2627459607] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:959 · 2025-12-17
let's remove this please 😉

## [c2627461054] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:966 · 2025-12-17
we have a decorator for that `@check_model_input` that sets them correctly!

## [c2627463605] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:988 · 2025-12-17
what is this?

## [c2627466119] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1000 · 2025-12-17
we don't use sequential, you can create a mulitmodal projector class! See Llava!

## [c2627468332] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1097 · 2025-12-17
pretty sure lots of other models use this no?

## [c2627470262] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1162 · 2025-12-17
there should be 0 BC comments, or code because the mode is new. Please remove this

## [c2627472373] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1227 · 2025-12-17
no other models in the repo need this. why would this one need that? 😅

## [c2627475249] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1290 · 2025-12-17
```suggestion
    tokenizer_class = ("Qwen2Tokenizer",)
```

## [c2627479536] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1287 · 2025-12-17
if this uses stream etc, we should document. Document the motivations, why this processor is specific / particular

## [c2627484172] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1484 · 2025-12-17
please please let's simplify. We never import from other classes in `transformers` (only for inhertiance)

## [c2627488095] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1572 · 2025-12-17
it is very weird that you have to do this, let's try to remove

## [c2627489783] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1647 · 2025-12-17
I don't think we need this no?

## [c2627492221] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1696 · 2025-12-17
these are in the transformers kwargs, handleed by the check model input decorator!

## [c2627495226] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1763 · 2025-12-17
this does not make sense to me. either its none, you properly create it with good shape with a func, or its passed -> user is responsible for it

## [c2627498154] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1791 · 2025-12-17
anything related to cos and sin should be done before hand in self.rotarty_emb no?

## [c2627501953] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1851 · 2025-12-17
```suggestion
```

## [c2627503851] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1922 · 2025-12-17
the model prepares them as well. Why prepare them twice?

## [c2627516781] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/model_doc/isaac.md:43 · 2025-12-17
quick q: what's the motivation behind this? It adds a lot of code overhead, we still have somewhat non GIL free usage (I don't see async)?

My confusion will be shared by the community: let's explain why it makes sense. How does it revie ROPE coordinates.

## [c2627520792] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/model_doc/isaac.md:89 · 2025-12-17
this supports `.to()`, next two lines are not needed

## [c3588130219] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-12-17
Hey Sorry if I missed some stuff, my confusion will be the same for anyone a bit new to this code and thus clarification is needed: 
- why are we introducing a new paradigm with tensor stream? 
- what are the benefits ? benches ? etc
The attention you are using is more than standard. Its the same as any other vision model except you have the mask. 

Packing sequences should be the responsability of the user in general, if not we can do it but in a custom generate because its something that does not scale that well (with TP for example). 

Let's make sure we isolate what's new with this model compared to other models. Say rope3d: how is it different from qwenvl models -> document the relevant class. 

This is really needed to get a model to be translatable by the rest of the community. Make sure you just look at modeling_llama.py and take "inspiration" from it

## [c2627543495] ArthurZucker (COLLABORATOR) · review_comment on tests/models/isaac/test_modeling_isaac.py:1 · reply to [c2603301673] · 2025-12-17
will have a look later on!

## [c3588325196] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-12-17 (no text)

## [c2627549476] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1944 · reply to [c2603277534] · 2025-12-17
its more like its not part of the transformers API so not sure its really useful

## [c3588332585] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-12-17 (no text)

## [c2627550999] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:2290 · reply to [c2603290473] · 2025-12-17
no juste that its done twice. If model does it why does parent class do it as well?

## [c3588334390] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-12-17 (no text)

## [c2631425510] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/image_processing_isaac.py:31 · reply to [c2627383814] · 2025-12-18
This should be resolved now with the rename, thanks!

## [c3593434737] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631426448] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:204 · reply to [c2627415595] · 2025-12-18
Done

## [c3593436141] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631430183] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:224 · reply to [c2627417362] · 2025-12-18
Also open to other patterns here if this is not idiomatic or unexpected

## [c3593441106] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631435984] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:458 · reply to [c2627425689] · 2025-12-18
Updated to the llama 4 approach of using mask attention functions mapping

## [c3593449573] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631437944] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:627 · reply to [c2627436846] · 2025-12-18
Got it, removed these

## [c3593452102] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631439009] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:645 · reply to [c2627438572] · 2025-12-18
Dropped

## [c3593453382] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631441190] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:651 · reply to [c2627440036] · 2025-12-18
Made this derivation conditional on being an attention implementation that uses it!

## [c3593456251] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631444889] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:670 · reply to [c2627441277] · 2025-12-18
refactored this section to update kwargs conditionally on attention implementation, let me know if this matches what you had in mind

## [c3593461594] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631446338] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:698 · reply to [c2627443852] · 2025-12-18
dropped this + replaced with sync free code

## [c3593463590] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631447736] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:748 · reply to [c2627445560] · 2025-12-18
resolved with new api

## [c3593465480] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631449114] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:757 · reply to [c2627447891] · 2025-12-18
This is clean; migrated to this

## [c3593467054] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631450500] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:785 · reply to [c2627451421] · 2025-12-18
Moving to llama4 style mask has addressed this

## [c3593469016] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631451776] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:798 · reply to [c2627453866] · 2025-12-18
updated this

## [c3593470807] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631454929] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:907 · reply to [c2627456054] · 2025-12-18
done !

## [c3593474725] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631466428] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:243 · reply to [c2627458114] · 2025-12-18
So this attribute is actually accessed in test_sdpa_can_dispatch_composite_models, regardless of PreTrainedModel presence. Is this expected?

## [c3593490219] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631468542] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:966 · reply to [c2627461054] · 2025-12-18
added!

## [c3593493208] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631471919] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1097 · reply to [c2627468332] · 2025-12-18
Identified a model that has something quite similar and adapted to use that

## [c3593497411] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631478203] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1227 · reply to [c2627472373] · 2025-12-18
Was able to do away with this by setting things up in init

## [c3593504854] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631480193] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1484 · reply to [c2627484172] · 2025-12-18
Changed class to inherit from qwen2_5 vl and call forward() instead

## [c3593507537] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631488389] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1647 · reply to [c2627489783] · 2025-12-18
removed a couple of the aliases here, some are accessed via tests as presumed entry points for vision components

## [c3593518735] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631493728] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1696 · reply to [c2627492221] · 2025-12-18
added !

## [c3593524598] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631532129] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1791 · reply to [c2627498154] · 2025-12-18
Took the liberty to just drop these

## [c3593570366] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2631533472] philippguevorguian (NONE) · review_comment on docs/source/en/model_doc/isaac.md:89 · reply to [c2627520792] · 2025-12-18
Ah got it, removed

## [c3593572210] philippguevorguian (NONE) · review COMMENTED · 2025-12-18 (no text)

## [c2634475566] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1922 · reply to [c2627503851] · 2025-12-19
This was a bit more of a refactor but have deduplicated this behavior such that IsaacModel is responsible only for rope_deltas

## [c3597878088] philippguevorguian (NONE) · review COMMENTED · 2025-12-19 (no text)

## [c2634480048] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:959 · reply to [c2627459607] · 2025-12-19
Done !

## [c3597882676] philippguevorguian (NONE) · review COMMENTED · 2025-12-19 (no text)

## [c2634483639] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1162 · reply to [c2627470262] · 2025-12-19
Removed the BC note to reflect that we are looking for compatibility with Qwen3 expectations at init

## [c3597886252] philippguevorguian (NONE) · review COMMENTED · 2025-12-19 (no text)

## [c2634487174] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1763 · reply to [c2627495226] · 2025-12-19
We now raise an error if the modality tensor has improper shape

## [c3597889713] philippguevorguian (NONE) · review COMMENTED · 2025-12-19 (no text)

## [c2635388189] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:491 · reply to [c2627427559] · 2025-12-19
Besides consistency with the prior checkpoint nn.Embedding also handles initialization internally, which removes need for custom initialization logic that gets flagged by `test_can_init_all_missing_weights`

## [c3598951265] philippguevorguian (NONE) · review COMMENTED · 2025-12-19 (no text)

## [c2635390448] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:530 · reply to [c2627430320] · 2025-12-19
This is only for consistency with pre-existing checkpoints, if this is a major sticking point it can be modified

## [c3598954026] philippguevorguian (NONE) · review COMMENTED · 2025-12-19 (no text)

## [c2635396513] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:606 · reply to [c2627435540] · 2025-12-19
Added documentation explaining the role of the sequence packing in this model; let me know if there are further concerns or questions here

## [c3598960747] philippguevorguian (NONE) · review COMMENTED · 2025-12-19 (no text)

## [c2635413949] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1000 · reply to [c2627466119] · 2025-12-19
Done !

## [c3598980196] philippguevorguian (NONE) · review COMMENTED · 2025-12-19 (no text)

## [c2635439219] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1572 · reply to [c2627488095] · 2025-12-19
initializing with `self.text_model = AutoModel.from_config(config.text_config)` causes significant output inconsistencies in the integration tests suggesting that the AutoModel init mutates the config. Any insight here around potential paths to resolution would be appreciated

## [c3599008632] philippguevorguian (NONE) · review COMMENTED · 2025-12-19 (no text)

## [c2646288748] philippguevorguian (NONE) · review_comment on docs/source/en/model_doc/isaac.md:43 · reply to [c2627516781] · 2025-12-24
@ArthurZucker We view tensorstream as a clean way to unify the interface for multimodal model inputs. To keep in line with the philosophy of maintaining a reference implementation here, this dependency has been removed from the modular code. Would appreciate another look

## [c3611609366] philippguevorguian (NONE) · review COMMENTED · 2025-12-24 (no text)

## [c2646289789] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1287 · reply to [c2627479536] · 2025-12-24
TensorStream and related abstractions have since been removed

## [c3611610272] philippguevorguian (NONE) · review COMMENTED · 2025-12-24 (no text)

## [c2675763334] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:303 · 2026-01-09
I really don't think we need a nested function to just call another function with another lamba function. put this out explicitly in a separete func or update reorder image

## [c2675768485] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:315 · 2026-01-09
thanks for updating, tho as I mentioned last time, I don't think this is something the model should handle. `FA2` interface already handles this, and already creates the block diagonal mask when you set the attnetion implementation. If anything we can update the interface / mask to take the cu_seqlens, but it just is not something specifc to this model and should be put out

## [c2675772462] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:491 · reply to [c2627427559] · 2026-01-09
this is not a correct justification! 
Custom init scheme should be handled by `_init_weights` its nt a big issue just not intuitive

## [c2675774704] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:451 · 2026-01-09
you should never have to do this..

## [c2675781538] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:494 · 2026-01-09
🤣 I don't understand why it has to be so complicated. 
If the `flash_attention_2` interface does not compute the max q or k properly, you can create a custom flash warpper for this model, but it has nothing to do with this model per say. 

What I mean here is that the attention used here is absolutely standard W.R.T other attention so lets make sure we use modular / follow convention and isolate `engenieering tricks` from core model conceptts

## [c2675784700] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:534 · 2026-01-09
this is not relevant. 

Also as I said cuseqlens and max_seqlens are only used for FA2 right? so should not appear explicitly ! as above, its hidden under the flash abstraction that we have

## [c2675803766] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:243 · reply to [c2627458114] · 2026-01-09
probably not, but if you have a self.config = config -> you probably want to have a preTrained model tthere no?

## [c2675805449] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:988 · reply to [c2627463605] · 2026-01-09
I re-iterate, this does not make sense for a nn.Module ! should be a PreTrainedModel

## [c2675823205] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1222 · 2026-01-09
I understand where this comes from, and it is actually quite important ! 
But the way its done is not really where I wish it was! 

I don't think we should do all the packing at all in this PR. It will be much simpler with the attention mask interface we already have and will give us time to work on something a bit cleaner / more aligned with `transformers`

Its less a comment for you, more a comment for @Cyrilvallez @vasqu @eustlb @yonigozlan for audio / vision models with interleaved text / image and lots of images / for videos etc we today don't have a clean way to do it. We should! Its something we want to hide as its not specific to the model, but that model here right now probably cannot run without this.

## [c2675824794] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1258 · 2026-01-09
```suggestion
class IsaacRotaryEmbedding(Qwen2_5_VLRotaryEmbedding):
```
this needs to be properly imported

## [c2675826593] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1311 · 2026-01-09
let s try to avoid super here and also modality Type if there are only 2 supporte by this model it does not really make sense!

## [c2675828088] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1347 · 2026-01-09
this is still not related to this model, did you find soemthing we should fix glob aly?

## [c2675831223] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1415 · 2026-01-09
so, there already was processor work to prepare them (in a for loop) then we re-do a for loop, we do device synch for offsets and lenghts when they are on device, we still prepare the inputs, etc.
This does not really make sense to me

## [c2675832420] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1415 · reply to [c2675831223] · 2026-01-09
I mean that in the sense that not packing is probably gonna help for now to simplify!

## [c2675836177] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1455 · 2026-01-09
like we have for some qwen, it would be nice to share what is the intention of this function / explain what it achieves / why it needs the attention mask when usually rope is agnostic to that, why it needs cache positions etc

## [c2675837949] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1595 · 2026-01-09
collection of hidden attention should be hidden with output recorder

## [c2675840255] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/isaac/modular_isaac.py:1596 · 2026-01-09
this layer should be moved outside the text model no? or just tie the weights? it s a bit weird to iterate over the text model's layers and not just pass the inputs to the model

## [c3643449718] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-01-09
thanks for updating!

## [c2691351885] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:494 · reply to [c2675781538] · 2026-01-14
Main thing is we need to generate the cu_seqlens from our seq_sizes tracking token spans for segments of vision tokens. The approach here mirrors the implementation of qwen2_vl, which likewise derives cu_seqlens from processor results and threads it through higher level model components. Is there a reference pointer for using the wrapper approach here?

## [c3661857945] philippguevorguian (NONE) · review COMMENTED · 2026-01-14 (no text)

## [c2691352835] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:451 · reply to [c2675774704] · 2026-01-14
removed

## [c3661859444] philippguevorguian (NONE) · review COMMENTED · 2026-01-14 (no text)

## [c2691353723] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:534 · reply to [c2675784700] · 2026-01-14
Localized this to the attention module

## [c3661861054] philippguevorguian (NONE) · review COMMENTED · 2026-01-14 (no text)

## [c2691943272] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1595 · reply to [c2675837949] · 2026-01-14
Added!

## [c3662643819] philippguevorguian (NONE) · review COMMENTED · 2026-01-14 (no text)

## [c2691944523] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1455 · reply to [c2675836177] · 2026-01-14
Added an explanation here

## [c3662645680] philippguevorguian (NONE) · review COMMENTED · 2026-01-14 (no text)

## [c2691946002] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:303 · reply to [c2675763334] · 2026-01-14
Pulled this out

## [c3662647671] philippguevorguian (NONE) · review COMMENTED · 2026-01-14 (no text)

## [c2691981620] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1222 · reply to [c2675823205] · 2026-01-14
That makes sense. To align with best practices and land cleanly, what’s the currently recommended approach for preparing inputs for batch inference in this format?

## [c3662690849] philippguevorguian (NONE) · review COMMENTED · 2026-01-14 (no text)

## [c2691984867] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:315 · reply to [c2675768485] · 2026-01-14
Went ahead and started the refactor by localizing to vision attention, to proceed here there is the same sticking point which is that it appears the packed sequence masking is necessary, but happy to switch to a higher level abstraction or update the  mask / interface if you have any pointers

## [c3662694422] philippguevorguian (NONE) · review COMMENTED · 2026-01-14 (no text)

## [c2691988524] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1347 · reply to [c2675828088] · 2026-01-14
Dropped it and we're good

## [c3662699337] philippguevorguian (NONE) · review COMMENTED · 2026-01-14 (no text)

## [c2691991078] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1311 · reply to [c2675826593] · 2026-01-14
Removed use of modality type here but can make the change more broad if needed, although I find it useful for readability to have the cases identified

## [c3662702569] philippguevorguian (NONE) · review COMMENTED · 2026-01-14 (no text)

## [c2692144055] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1596 · reply to [c2675840255] · 2026-01-14
Put an alias on the norm here so that this is more clear. Iterating over layers is done to implement multimodal integration logic, rope implementation and masking logic which differs from Qwen3

## [c3662899481] philippguevorguian (NONE) · review COMMENTED · 2026-01-14 (no text)

## [c2755629819] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1222 · reply to [c2675823205] · 2026-02-02
Following up here; @ArthurZucker would really value some guidance on getting batched inference landed. Happy to adapt the approach to push it forward once we’re aligned.

## [c3741116437] philippguevorguian (NONE) · review COMMENTED · 2026-02-02 (no text)

## [c2885513812] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1 · 2026-03-04
the pr has been open for a while 😄  2026 in all file pls

## [c2885520105] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:133 · 2026-03-04
we don't need the args which are identical to Siglip. If the default is identical, we can remove alltogether. Otehrwise, just delete from `__init__`, because calling `super(**kwargs)` will set all attr

## [c2885521674] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:144 · 2026-03-04
the default is already SDPA (i.e. `None`), should not be needed

## [c2885527276] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:162 · 2026-03-04
I believe they can't take a value of `None`, in that case let's keep only `int` type hints. We check against these typings at run-time

## [c2885556147] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:213 · 2026-03-04
what is the reason for this?

## [c2885559522] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:228 · 2026-03-04
This is usually handled by base class before reaching `_preprocess`, shouldn't be needed

## [c2885573596] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:190 · 2026-03-04
only this line is needed in `__init__`, the rest will be copied by modular

## [c2885583482] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:337 · 2026-03-04
not needed here, these decorators are used in a `PreTrainedModel` class

## [c2885619786] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:459 · 2026-03-04
can you move "args/returns" doc under `forward` method?

## [c2885628540] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:334 · 2026-03-04
would be great to decorate with `capture_outputs` and `merge_defaults`. Then at the end return a `BaseModelOutpuWithPooling` dict

Also would need a cls attr `_can_record_outputs` similar to Siglip2VisionModel

## [c2885638389] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:519 · 2026-03-04
nit: unused in forward so we can directly obtain and use in L522, no need to save as self attribute

## [c2885661018] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:538 · 2026-03-04
imo we don't need a class for it. We can init a vision tower and an mm-proj directly in `IsaacModel`

## [c2885700283] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:607 · 2026-03-04
the whole idea here looks quite similar to existing VLMs, e.g. Coheer2Vision/Mllama/InternVL. Prob it can even be copied entirely, could you check 👀 

https://github.com/huggingface/transformers/blob/421c7f6248e28d24d84ee000252a1e71fbc24917/src/transformers/models/cohere2_vision/image_processing_cohere2_vision_fast.py#L83

## [c2885704818] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:648 · 2026-03-04
"text_config: AutoConfig" so we don't import from Qwen in config file

## [c2885707093] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:649 · 2026-03-04
not needed here, usually it's saved in the hub already in `processor_config.json`

## [c2885714325] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:654 · 2026-03-04
Qwen3Config -> PreTrainedConfig pls, again to avoid importing from another model file and to follow "one model - one file" philosophy

## [c2885719128] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:686 · 2026-03-04
all these params are already in text config, we should not be needing them in base config. It causes confusion when the same key is duplicated several times

## [c2885720204] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:699 · 2026-03-04
same here for attn implementation

## [c2885726161] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:718 · 2026-03-04
`@auto_docstring` on the class should cover most args

## [c2885727804] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:726 · 2026-03-04
not needed, we refactored loading logic and in v5 the processor classes are auto-inferred

## [c2885731868] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:726 · 2026-03-04
pad token shuold be in init, as follows `self.pad_token_id = tokenizer.pad_token_id`. Cls attributes can cause issues downstream when they are modified

## [c2885733229] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:732 · 2026-03-04
same q, any reason for this?

## [c2885753568] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:745 · 2026-03-04
we can't use `config` class here. All args should be in signature and saved correctly, so it can be loaded back. I see that visino-token, etc are already in signature, so deleting the config will be enough

## [c2885758571] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:757 · 2026-03-04
casting `int` is not safe in case the token id isn't found and is `None`. Let's just make sure the token actually exist in vocab and simplify as 

```
self.text_pad_token_id = tokenizer.pad_token_id
self.image_pad_token_id = tokenizer.image_pad_token_id
```

## [c2885763145] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:758 · 2026-03-04
nit: duplicate from above

## [c2885765219] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:762 · 2026-03-04
chat template needs to be part of processor, and thus in `__init__` signature

## [c2885766201] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:761 · 2026-03-04
to delete, unused

## [c2885827397] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:770 · 2026-03-04
the whole fn is quite complicated and could be simplified, I will be commenting below on each line

## [c2885834492] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:793 · 2026-03-04
iiuc we expect nested images `(list[list[PIL.Image]])`. We have several models accepting such a format and also checking that number of images matches number of texts, and we can adapt from there
For ex: Gemma3 or Mllama, it already uses a bunch of utillities which simplifies the code a lot

## [c2885841360] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:839 · 2026-03-04
we shouldn't be manipulating more on top of `image_inputs`. Anything that is returned from the image processor should be directly passed to `return` or popped along the way if not used in model

Let's move this part to image processor's `_preprocess` method

## [c2885847748] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:850 · 2026-03-04
if we want to make sure that the token don't go beyond `max_length`, let's just use `tokenizer(truncation=True, max_length=X)`

Later we also can check that it didn't truncate image placeholder tokens and raise an error (via `self._check_mm_tokens` utility)

## [c2885851831] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:871 · 2026-03-04
this looks very much like Qwen-VL 3D rope without a temporal axis. Let's move it to model code and copy from qwen when possible

## [c2885856835] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:876 · 2026-03-04
this is same as `mm_token_type_ids` that we support in all VLMs, let's use the correct notation and move to the very end. Like this for ex:

https://github.com/huggingface/transformers/blob/421c7f6248e28d24d84ee000252a1e71fbc24917/src/transformers/models/llava/processing_llava.py#L130

## [c2885864112] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:895 · 2026-03-04
this can happen in the same way as in Qwen-VL or other model  where the expansion happens on input text, not ids. Can be adapted easily since Isaac also return similar `grid` to compute number of placeholder tokens 

https://github.com/huggingface/transformers/blob/421c7f6248e28d24d84ee000252a1e71fbc24917/src/transformers/models/qwen2_vl/processing_qwen2_vl.py#L151

## [c2885868734] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:910 · 2026-03-04
we don't yet support devices in processors and everything is returned on CPU. We're planning to support `device` in future releases, until then let's delete and ask users to move to GPU after calling the processor

## [c2885872722] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:949 · 2026-03-04
padding also should happen when calling the tokenizer `tokenizer(text_with_expanded_image_tokens, padding=True)`

## [c2885880574] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:982 · 2026-03-04
if we are reshaping and padding image inputs, that should be in `image processor._preprocess` as well. Processor can only gather different modalities and apply modifications that require "text+image" (e.g. adding placeholder tokens requires knowledge about number of image patches)

## [c2885901933] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1075 · 2026-03-04
thus should be prepared correctly before RoPE in `prepare_position_ids` and in model's forward

## [c2885920349] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1092 · 2026-03-04
other than this line, the RoPE layer is identical to Qwen2-VL. Maybe we can uss Qwen2VLTextModel or Qwen3VLTextModel as the lm backbone

I think Isaac doesn't have sliding window, in that case `Qwen3VLTextModel` might be a better fit

## [c2885929920] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1105 · 2026-03-04
Not needed, because they are init as standalone backbones. Instead each backbone should have its own `_can_record_outputs`. Qwen already has, so `IsaacVisionModel` needs one as I commented above

## [c2885931835] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1107 · 2026-03-04
should not be overwritten, we only need to define `_tied_weights_keys` dict

## [c2885937954] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1113 · 2026-03-04
RoPE is part of the lm backbone. We can either:

1. Init a ready lm backbone if there exists any, maybe Qwen3VLTextModel
2. Make a new IsaacTextModel by copying from the closes model and overriding blocks where needed

Option 1 is more preferable

## [c2885942365] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1133 · 2026-03-04
not needed. When setting it in `text_model`, the config is modified by super class

## [c2885948605] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1145 · 2026-03-04
i'd prefer without properties if possible, they can cause issues. Especially the embed token is functionally a duplicate of `set/get input embeddings`

## [c2885952408] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1218 · 2026-03-04
we recently refactored the whole 3D rope thing in all models. Can you check qwen-vl or ernie4.5-VL and update Isaac accordingly, by re-using building blocks

## [c2885973204] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1267 · 2026-03-04
the base VLM model needs only `can_return_tuple` instead of these two decorators. Same applied for the `ConditionalGeneration` model

## [c2885979528] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1336 · 2026-03-04
we need to follow the standards from other VLMs, e.g. call `get_image_features` and `get_placeholder_mask` after which scatter the embeddings

## [c2885984615] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1322 · 2026-03-04
I dont' think we can call vision backbone when any of these are present but no `vision_patches`. The most important thing is `vision_patches`, so we can just rely on `if vision_patches is not None: selfget_image_features`

## [c2885990149] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1344 · 2026-03-04
nit: I'd prefer the standard naming `mm_token_type_ids`. And also, no need for `class ModalityType`, we have assumption across codebase that "0->text, 1->iimage, 2->video, 3->audio" :)

## [c2885994111] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1358 · 2026-03-04
these blocks are better delegated to LM backbone, a bit repetiive

## [c2885999312] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1379 · 2026-03-04
same here, pass it to LM and let it create the 4D mask

## [c2886007008] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1400 · 2026-03-04
same as calling `self.language_model(**inputs)`. But we need to make sure the norm is part of lm in the state dict, which can be done easily by adding the Isaac in `src/transformers/conversion_mapping.py`

## [c2886009790] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1404 · 2026-03-04
we'll need to manually return here `output.hidden_states` etc after lm backbone if called

## [c2886014126] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1412 · 2026-03-04
same here, should not be hardcoded. It is an internal instance attribute, not cls attribute

## [c2886016556] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1439 · 2026-03-04
btw, `cache_position` was removed just recently. Can you remove it everywhere?

## [c2886020820] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1532 · 2026-03-04
we need to check `if not is_first_iteration and use_cache` to cover all edge cases

https://github.com/huggingface/transformers/blob/421c7f6248e28d24d84ee000252a1e71fbc24917/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L1572C9-L1572C49

## [c2886025617] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1525 · 2026-03-04
this part should not be needed and instead we expect users to be consistent with what they pass. Just checking if we're in prefill/decoding phase will do the work, commented below 👇🏻

## [c2886028342] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1537 · 2026-03-04
not needed, if model inherits from GenerationMixin it will have `can_generate = True` in any case

## [c2886029252] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1547 · 2026-03-04
same, can be entirely deleted

## [c2886031202] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/__init__.py:13 · 2026-03-04
nit: we don't need a license header here

## [c2886037204] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_processing_isaac.py:1 · 2026-03-04
also an image processor test in a separate file pls because the model implements a new image processor class

## [c2886047106] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_processing_isaac.py:48 · 2026-03-04
we can use the common testing suite from `ProcessorTestsCommon` and load a pretrained checkpoint. For testing we can also change image patch sizes to output a tiny processed image

I think the rest of the tests will be cleaned up after using the mixin

## [c2886059061] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:82 · 2026-03-04
hmm interesting. Maybe we should support basic post-processing in IsaacProcessor so users can call it and get clean `bboxes` easily

Prob similar to what we have in SAM:

https://github.com/huggingface/transformers/blob/421c7f6248e28d24d84ee000252a1e71fbc24917/src/transformers/models/sam/processing_sam.py#L215

## [c2886061151] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:249 · 2026-03-04
needs vision inputs as well for the test to be complete

## [c2888902790] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1165 · 2026-03-05
this part also not needed, we only have to check once in `forward` that ` vision_patches is not None and vision_token_grids is not None` and then call `get_image_features`

## [c2888938179] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1168 · 2026-03-05
we shouldn't be moving devices and expect users to put inputs in correct device/dtype before calling the model

## [c2888943952] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:541 · 2026-03-05
instead of a tuple, they have to be passed as separate args, here and in the VisionTransformer

## [c2888952290] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1203 · 2026-03-05
Imo the unpadding can be simplified a lot since we only need to remove the padding part by checking each inputs actual `image_size`

So instead of offset, length and a mask, we can keep only one arg which will tell us the length of each image

## [c3584247584] zucchini-nlp (MEMBER) · review COMMENTED · 2026-03-05
Hey 👋🏻 

I think the PR needs a bit more standardization, before we can merge it. I see that the general idea for Isaac is quite similar to existing VLMs and we can try to re-use existing code in transformers. Specifically the processor might be simplified a looot and we can try to copy a few utilities in image processing

Also left a few comments on the recent changes in v5, we have deprecate several things and the model needs to be updated accordingly

## [c2955604244] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1203 · reply to [c2888952290] · 2026-03-18
Tried to move things around here but because we allow truncation within an image span, the length information alone is not sufficient to enable the unpadding

## [c3970174585] philippguevorguian (NONE) · review COMMENTED · 2026-03-18 (no text)

## [c2955694661] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:607 · reply to [c2885700283] · 2026-03-18
I checked those and a couple other vlms; seems to me they're not a proper fit. E.g. mllama/coere2/internvl use tiled, multi-crop or image-placeholder logic. Isaac use a single resize path that preserves exact pre-shuffle `min_num_patches`/`max_num_patches` while enforcing the `patch_size * pixel_shuffle_scale` constraint for pixel shuffle. ernie4.5 is closer, but the budgeting is by pixel area rather than patch-count, so I don’t think we can use these helpers here.

## [c3970265508] philippguevorguian (NONE) · review COMMENTED · 2026-03-18 (no text)

## [c2955761169] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:850 · reply to [c2885847748] · 2026-03-18
This would be a behavior change. Isaac currently supports suffix truncation through image spans and tracks the kept tail with `vision_token_offsets`+ `vision_token_lengths`. Switching to tokenizer truncation + `_check_special_mm_tokens` only would error out on any truncation through image placeholders. I'll enforce that expectation prior to the custom truncation and apply the tokenizer padding afterwards as well to simplify as much as possible while preserving the current behavior

## [c3970330377] philippguevorguian (NONE) · review COMMENTED · 2026-03-18 (no text)

## [c2959951322] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:541 · reply to [c2888943952] · 2026-03-19
Done, now pass these around as separate arguments

## [c3975012068] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2959952504] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1168 · reply to [c2888938179] · 2026-03-19
Removed this

## [c3975013672] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2959959634] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:274 · reply to [c2624196533] · 2026-03-19
dropped it

## [c3975021555] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2959961095] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1 · reply to [c2885513812] · 2026-03-19
made the update

## [c3975023065] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2959962825] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:133 · reply to [c2885520105] · 2026-03-19
made the change, a lot cleaner now

## [c3975024943] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2959963741] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:144 · reply to [c2885521674] · 2026-03-19
dropped this

## [c3975025910] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2959964738] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:162 · reply to [c2885527276] · 2026-03-19
strictly typing them as int now

## [c3975027074] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2959965201] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:213 · reply to [c2885556147] · 2026-03-19
removed

## [c3975027563] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2959966341] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:228 · reply to [c2885559522] · 2026-03-19
Removed this

## [c3975028814] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2959973261] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:190 · reply to [c2885573596] · 2026-03-19
Ah this is great, simplified to just having this

## [c3975036265] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2959993790] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:337 · reply to [c2885583482] · 2026-03-19
Removed, only using on children of PreTrainedModel now

## [c3975059421] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960005534] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:459 · reply to [c2885619786] · 2026-03-19
Updated docstring and moved

## [c3975072661] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960008064] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:334 · reply to [c2885628540] · 2026-03-19
Have a look, but this should be correct now. Looked at reference implementations and matched

## [c3975075255] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960009998] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:519 · reply to [c2885638389] · 2026-03-19
stopped storing these as attributes

## [c3975077273] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960121621] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:538 · reply to [c2885661018] · 2026-03-19
Flattened this so that vision tower and projector are directly under the model

## [c3975196950] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960128473] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:648 · reply to [c2885704818] · 2026-03-19
Made a few changes here including a bespoke IsaacTextConfig that obviates the need for Qwen imports in general

## [c3975204446] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960129557] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:649 · reply to [c2885707093] · 2026-03-19
Got it, removed

## [c3975205528] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960132384] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:654 · reply to [c2885714325] · 2026-03-19
As stated above, the config changes make this unnecessary. No qwen imports

## [c3975208353] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960132965] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:686 · reply to [c2885719128] · 2026-03-19
Dropped!

## [c3975208927] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960133626] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:699 · reply to [c2885720204] · 2026-03-19
Removed !

## [c3975209662] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960134579] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:718 · reply to [c2885726161] · 2026-03-19
Done!

## [c3975210638] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960135502] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:726 · reply to [c2885727804] · 2026-03-19
Awesome, removed this

## [c3975211583] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960136967] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:726 · reply to [c2885731868] · 2026-03-19
dropped !

## [c3975213278] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960137853] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:732 · reply to [c2885733229] · 2026-03-19
Not needed, so removed

## [c3975214350] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960139259] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:745 · reply to [c2885753568] · 2026-03-19
Thanks for catching this, removed

## [c3975215724] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960140774] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:757 · reply to [c2885758571] · 2026-03-19
Modified the config to store the token info, now read directly from tokenizer.

## [c3975217337] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960141449] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:758 · reply to [c2885763145] · 2026-03-19
resolved

## [c3975218172] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960142291] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:762 · reply to [c2885765219] · 2026-03-19
moved chat template to init

## [c3975219040] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960142742] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:761 · reply to [c2885766201] · 2026-03-19
removed

## [c3975219503] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960143632] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:770 · reply to [c2885827397] · 2026-03-19
ack, see below

## [c3975220528] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960146063] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:793 · reply to [c2885834492] · 2026-03-19
Enforced the invariant and made use of nested image handling utility

## [c3975223043] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960148511] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:839 · reply to [c2885841360] · 2026-03-19
moved all image-specific handling that doesn't require external metadata to image processing _preprocess

## [c3975225559] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960152858] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:871 · reply to [c2885851831] · 2026-03-19
Relatively big refactor to get this in with support for beam gen and cache but now works with other standard utils

## [c3975229952] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960154073] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:876 · reply to [c2885856835] · 2026-03-19
Changed name and moved to end of function

## [c3975231180] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960155937] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:895 · reply to [c2885864112] · 2026-03-19
We now expand the text here directly instead of operating on the ids

## [c3975233072] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960156796] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:910 · reply to [c2885868734] · 2026-03-19
Removed all device logic

## [c3975234103] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960160474] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:949 · reply to [c2885872722] · 2026-03-19
Made a suite of changes to allow this logic, we now pad with the tokenizer directly after the custom truncation

## [c3975237941] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960161301] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:982 · reply to [c2885880574] · 2026-03-19
Moved to image processor

## [c3975238870] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960162509] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1075 · reply to [c2885901933] · 2026-03-19
Done!

## [c3975240034] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960164514] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1092 · reply to [c2885920349] · 2026-03-19
Inherit from Qwen3Vl to rely on the logic there, only overriding this logic

## [c3975242212] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960166985] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1105 · reply to [c2885929920] · 2026-03-19
Moved _can_record_outputs to text and vision models and removed manual OutputRecorder spec

## [c3975244939] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960169188] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1107 · reply to [c2885931835] · 2026-03-19
moved this definition to _tied_weights_dict

## [c3975247086] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960170414] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1113 · reply to [c2885937954] · 2026-03-19
Moved to IsaacTextModel

## [c3975248300] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960171319] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1133 · reply to [c2885942365] · 2026-03-19
Removed!

## [c3975249190] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960173348] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1145 · reply to [c2885948605] · 2026-03-19
Removed all properties; norm logic is delegated to text backbone

## [c3975251347] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960176694] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1218 · reply to [c2885952408] · 2026-03-19
Take a look here to make sure; but we now adhere to the compute_3d_position_ids and vision_position_ids standard so should be aligned

## [c3975254533] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960177887] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1267 · reply to [c2885973204] · 2026-03-19
Made this chagne

## [c3975255761] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960179579] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1336 · reply to [c2885979528] · 2026-03-19
we now follow get_image_features -> placeholder_mask -> scatter flow

## [c3975257417] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960181175] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1322 · reply to [c2885984615] · 2026-03-19
simplified all such calls to rely on invariant based on processor output

## [c3975258994] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960181799] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1344 · reply to [c2885990149] · 2026-03-19
done !

## [c3975259659] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960185314] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1358 · reply to [c2885994111] · 2026-03-19
attention_mask is now regenerated where needed for pos ids, same with cache logic. Attention mask handling here is now in the text backbone as stated

## [c3975263245] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960185768] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1379 · reply to [c2885999312] · 2026-03-19
done !

## [c3975263684] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960186407] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1400 · reply to [c2886007008] · 2026-03-19
Done !

## [c3975264197] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960187886] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1404 · reply to [c2886009790] · 2026-03-19
Now feed lm backbone outputs here

## [c3975265774] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960188597] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1412 · reply to [c2886014126] · 2026-03-19
Done !

## [c3975266460] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960190741] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1439 · reply to [c2886016556] · 2026-03-19
All cache_position logic removed following standards from other models

## [c3975268736] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960193573] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1532 · reply to [c2886020820] · 2026-03-19
Changed to rely on is_first_iteration and use_cache for the branching logic here

## [c3975271451] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960197745] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1537 · reply to [c2886028342] · 2026-03-19
Removed this

## [c3975275570] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960199320] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1547 · reply to [c2886029252] · 2026-03-19
dropped all of this

## [c3975277297] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960200228] philippguevorguian (NONE) · review_comment on tests/models/isaac/__init__.py:13 · reply to [c2886031202] · 2026-03-19
removed it !

## [c3975278138] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960202013] philippguevorguian (NONE) · review_comment on tests/models/isaac/test_processing_isaac.py:1 · reply to [c2886037204] · 2026-03-19
added common image processor test suite

## [c3975279966] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960203504] philippguevorguian (NONE) · review_comment on tests/models/isaac/test_processing_isaac.py:48 · reply to [c2886047106] · 2026-03-19
Added the common suite!

## [c3975281427] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960207176] philippguevorguian (NONE) · review_comment on tests/models/isaac/test_modeling_isaac.py:82 · reply to [c2886059061] · 2026-03-19
Implemented this as processor functionality

## [c3975285338] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960208588] philippguevorguian (NONE) · review_comment on tests/models/isaac/test_modeling_isaac.py:249 · reply to [c2886061151] · 2026-03-19
Added vision inputs here

## [c3975286668] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2960211423] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1165 · reply to [c2888902790] · 2026-03-19
Made the changes to standard flow through get_image_features

## [c3975289697] philippguevorguian (NONE) · review COMMENTED · 2026-03-19 (no text)

## [c2965973650] zucchini-nlp (MEMBER) · review_comment on docs/source/en/model_doc/isaac.md:1 · 2026-03-20
nit: 2026 in all files

## [c2965977863] zucchini-nlp (MEMBER) · review_comment on docs/source/en/model_doc/isaac.md:42 · 2026-03-20
maybe the doc is a bit out-dated with all the changes. Let's update it before merging

## [c2965983040] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/auto/image_processing_auto.py:130 · 2026-03-20
oops, this needs a rebase on main, we merged a huge refactor yesterday. The mapping is now a `dict[dict[str, str]]]

## [c2965985842] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/auto/processing_auto.py:50 · 2026-03-20
accidentally removed `TYPE_CHECKING`?

## [c2965992120] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/image_processing_isaac_fast.py:1 · 2026-03-20
this file contents need to be moved to `image_processing_isaac.py` to align with the refactor. I think rebase and `modular_convert` will do eveything for you

## [c2975162962] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:76 · 2026-03-23
we just merged a PR refactoring configs, so we need a rebase and `auto_docstring` and `strict` decorators here. And same for other configs

## [c2975168374] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:224 · 2026-03-23
same here, after a recent refactor the parent class is different. I think it was `TochBackendImageProcessor`

## [c2975177040] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:234 · 2026-03-23
attr deleted in the refactor

## [c2975188995] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:296 · 2026-03-23
do we support it, or should we just raise an error? I can't think if use-case for an empty list

## [c2975194791] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:358 · 2026-03-23
amazing work here, love to see the processor cleaned-up!

## [c2975199785] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:363 · 2026-03-23
nit: `nested_outputs["vision_patches"][0][0]` is easier than iterating no?

## [c2975206484] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:379 · 2026-03-23
nit: could we move out this block to `self.pad` and call padding with `if self.do_pad`: self.pad(patches, grids, mask)?

## [c2975213303] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:249 · 2026-03-23
nit: lets' add a cls attr `config: IsaacVisionConfig` for completeness

## [c2975229330] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:763 · 2026-03-23
nope, they need to be accessed via text config if needed. Otherwise it will be serialized under `text_config` and here, which might because confusion

## [c2975243051] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:913 · 2026-03-23
why do we pad inputs separately from `tokenizer.__call__`?

## [c2975247053] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:975 · 2026-03-23
not: let's just keep the contents of `_build_batch` here, imo there is no need to split up because it's not much code

## [c2975313608] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:985 · 2026-03-23
why is this all, with copy and text_config? It already gets a text config from what I see below

## [c2975318744] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:991 · 2026-03-23
it is `rope_parameters` in recent versions, so you either need a rebase or smth went wrong in config's init

## [c2975342036] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1048 · 2026-03-23
nit: if we keep the name `self.language_model`, we can drop a few overriden methods. If you are worrying about pre-saved weight, they can be easily renamed on the fly if we add them in `conversion_mapping.py`

## [c2975349043] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1080 · 2026-03-23
iniputs have to be already on device, and we don't move across devices ourselves usually

## [c2975354714] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1098 · 2026-03-23
why is the filter? Users shouldn't pass unused kwargs

## [c2975360937] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1135 · 2026-03-23
is this possible, that mask is all zeros? Doesn't look like we should support it, but we can raise an error if that happens for some reasons

## [c2975362869] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:533 · 2026-03-23
😍

## [c2975382111] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1285 · 2026-03-23
I think the functionality is same as in qwen-vl series and only input args are different. Can we move around stuff so it can be copied at these parts?

For ex broadcast the positions in model's forward etc

## [c2975383637] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1318 · 2026-03-23
`merge_with_config_defaults` is not needed here

## [c2975386755] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1369 · 2026-03-23
same comment about moving to devices

## [c2975393111] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1391 · 2026-03-23
do we want to get non-full attention? That would prob not work well to build position ids

## [c2975399558] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1535 · 2026-03-23
there are two args that look suspiciously similar, are they the same thing with different names?

## [c2975404156] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1549 · 2026-03-23
this already happens when we call `generate()`, so we shouldn't need to call manually each decode step

## [c2975407932] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1628 · 2026-03-23
not needed, base class can work it out when it is inside `model`

## [c3981762840] zucchini-nlp (MEMBER) · review COMMENTED · 2026-03-23
Huuge work cleaning it up. It already looks much much better 🤩 

We need to rebase on `main` since there were a few big refactors merged recently. I commented below what needs to be changed for them. And the main question I have is about padding, since we need to try and delegate pad/truncate to tokenizer's `__call__`  method. The rest is mostly nits about styling

## [c4111106963] zucchini-nlp (MEMBER) · issue_comment · 2026-03-23
Also would be great to fix the CI before we request core maintainer's review (the last step before merging)

## [c2983108799] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1628 · reply to [c2975407932] · 2026-03-24
Removed!

## [c4000904217] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983109979] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1549 · reply to [c2975404156] · 2026-03-24
thanks, dropped this

## [c4000905601] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983112006] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1391 · reply to [c2975393111] · 2026-03-24
Made this more opinionated to be all full attention

## [c4000907999] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983113966] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/image_processing_isaac_fast.py:1 · reply to [c2965992120] · 2026-03-24
Yep, after updating the processing related code this was changed

## [c4000910459] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983115701] philippguevorguian (NONE) · review_comment on src/transformers/models/auto/image_processing_auto.py:130 · reply to [c2965983040] · 2026-03-24
Made the update with the TorchVisionBackend

## [c4000912501] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983117323] philippguevorguian (NONE) · review_comment on docs/source/en/model_doc/isaac.md:42 · reply to [c2965977863] · 2026-03-24
Updated the doc here with all of the changes

## [c4000914447] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983118606] philippguevorguian (NONE) · review_comment on docs/source/en/model_doc/isaac.md:1 · reply to [c2965973650] · 2026-03-24
Oops, missed this on the prior pass; resolved

## [c4000916173] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983120790] philippguevorguian (NONE) · review_comment on src/transformers/models/auto/processing_auto.py:50 · reply to [c2965985842] · 2026-03-24
Yes, merge mistake. Reverted

## [c4000918661] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983139504] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:224 · reply to [c2975168374] · 2026-03-24
Changed backend here to TorchVisionBackendProcessor

## [c4000940210] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983149312] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1318 · reply to [c2975383637] · 2026-03-24
Removed

## [c4000951613] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983152587] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1285 · reply to [c2975382111] · 2026-03-24
Moved the broadcast out; added handling for some special cases like continuing from past key values as well

## [c4000955374] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983154179] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1098 · reply to [c2975354714] · 2026-03-24
Right; removed this

## [c4000957376] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983155248] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1080 · reply to [c2975349043] · 2026-03-24
Dropped this device handling code

## [c4000958732] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983156196] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1135 · reply to [c2975360937] · 2026-03-24
cleaned this up

## [c4000959972] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983162805] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1048 · reply to [c2975342036] · 2026-03-24
Doesn't work with self.language_model AFAIK and didn't want to change `base_model_prefix` here. Made an update to the embedding mixin to support nested paths as a general solution for such scenarios, take a look.

## [c4000967519] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983163971] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:991 · reply to [c2975318744] · 2026-03-24
Updated!

## [c4000969088] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983165681] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:985 · reply to [c2975313608] · 2026-03-24
Moved these all such that they are owned by the config passed in without copy

## [c4000971163] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983166893] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:975 · reply to [c2975247053] · 2026-03-24
inlined the build_batch logic

## [c4000972633] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983172458] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:763 · reply to [c2975229330] · 2026-03-24
These attrs now all under text_config

## [c4000979967] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983172997] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:249 · reply to [c2975213303] · 2026-03-24
done !

## [c4000980561] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983174551] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:379 · reply to [c2975206484] · 2026-03-24
refactored the padding logic to pad() method

## [c4000982695] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983176093] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:234 · reply to [c2975177040] · 2026-03-24
dropped!

## [c4000984697] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983178783] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1369 · reply to [c2975386755] · 2026-03-24
Removed device handling logic here

## [c4000988281] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983302670] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:296 · reply to [c2975188995] · 2026-03-24
Added a comment explaining this, essentially we support this to maintain the output schema. Alternative wouldn't simplify much as all branches would still need to be supported

## [c4001160473] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983335308] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:913 · reply to [c2975243051] · 2026-03-24
Added a comment and changed var names to clarify. Essentially, We pad separately here as the the model needs the unpadded sequence prior to this for multimodal metadata derivation. After tokenizing, we compute the image-token spans, apply custom truncation, etc.. Only after that we pad the batch. Using tokenizer's call for padding/truncation directly would make that alignment more complex and could bring image features out of sync from the placeholder tokens in the sequence

## [c4001201834] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2983345689] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:363 · reply to [c2975199785] · 2026-03-24
nested_outputs["vision_patches"] can contain empty sublists for samples with no images, so [0][0] is not guaranteed. The `next` finds the  first non-empty patch across the batch

## [c4001214615] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2984235952] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:76 · reply to [c2975162962] · 2026-03-24
Added both decorators!

## [c4002257163] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2984630970] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1535 · reply to [c2975399558] · 2026-03-24
Standardized this on image_patch_attention_mask

## [c4002695008] philippguevorguian (NONE) · review COMMENTED · 2026-03-24 (no text)

## [c2986854676] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:896 · 2026-03-25
So, tokenizer needs left-side truncation which is the only diff. We support left-side truncation in v5 when using fast tokenizers, so we don't need to pad/truncate manually. I added here a small snippet showing what we need for Isaac

```python
# Set truncation side
AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B', truncation_side="left")

# then we can collect necessary output for vision-overflowing and drop off truncated parts
[… 15 lines of code omitted]

## [c2986871268] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:903 · 2026-03-25
IIUC, image-related offsets are left truncated, and used in model forward to truncate obtained image encoding. We can't encoder a half-image, but still can crop off unwanted "tokens"

Could you add a comment like this to explain, it will make it easier to refactor in the future

## [c2986909284] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:903 · 2026-03-25
this loop can be merged with prev one and re-ordered, so we don't loop twice 

```python
for image_idx, (segment_length, image_span) in enumerate(
                zip(expected_image_lengths, image_spans, strict=True)
            ):
                kept_start = max(start, int(image_span[0].item()))
                kept_end = min(total, int(image_span[-1].item()) + 1)
                if kept_end > kept_start:
                    vision_token_offsets[batch_idx, image_idx] = kept_start - image_start
                    vision_token_lengths[batch_idx, image_idx] = kept_end - kept_start
```

## [c2986921947] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:190 · 2026-03-25
hehe, sorry, I merged yesterday a tiny PR removing `accept_kwargs=True`. CI will complain after rebase anyway, so just so you're not surprised 😅

## [c2986928539] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:296 · reply to [c2975188995] · 2026-03-25
weird, inside processor we shouldn't be routing empty lists. I'll go check processing code

## [c2986932768] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:502 · 2026-03-25
nit: can we rename `x`-es to `hidden_states/image_states/image_embeddings`

## [c2986938507] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:790 · 2026-03-25
ig this comes from saved checkpoints which we don't want to re-save?

## [c2986945734] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:986 · 2026-03-25
it cannot be `None` if `return_attention_mask=True`, so maybe just force it to be `True`

## [c2986961492] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:841 · 2026-03-25
`image_pad_token` and `image_token` is same thing no?

## [c2986965076] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:843 · 2026-03-25
and this is same as `image_token` ig. We usually try to keep a single token as placeholder and it allows better integration with 3rd party libraries

## [c2987011063] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1053 · 2026-03-25
let's instead copy from qwen-VL so identical part can be copied. As mentioned, I am seeing a few similar part sch as `get_placeholder_mask` and `compute3d_position_ids`. Other part like `forward` need new args so can be copied and adapted

IMO things could be simplified

## [c4005164707] zucchini-nlp (MEMBER) · review COMMENTED · 2026-03-25 (no text)

## [c3010463495] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:841 · reply to [c2986961492] · 2026-03-30
Went ahead and unified these

## [c4031334795] philippguevorguian (NONE) · review COMMENTED · 2026-03-30 (no text)

## [c3010464853] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:843 · reply to [c2986965076] · 2026-03-30
Yep, unified this with the image_token attribute as well

## [c4031336252] philippguevorguian (NONE) · review COMMENTED · 2026-03-30 (no text)

## [c3010482589] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:896 · reply to [c2986854676] · 2026-03-30
Went ahead and unified tokenizer calls so truncation and padding occur at the same call site.

## [c4031355439] philippguevorguian (NONE) · review COMMENTED · 2026-03-30 (no text)

## [c3010520899] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:190 · reply to [c2986921947] · 2026-03-30
Done!

## [c4031398427] philippguevorguian (NONE) · review COMMENTED · 2026-03-30 (no text)

## [c3010523194] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:502 · reply to [c2986932768] · 2026-03-30
Done

## [c4031400843] philippguevorguian (NONE) · review COMMENTED · 2026-03-30 (no text)

## [c3010532861] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:1053 · reply to [c2987011063] · 2026-03-30
Inherit from Qwen3VL now after big refactor. After going after a few approaches here, delegating logic of model helpers to the qwen3vl stack generally leads to increased complexity, as the Isaac image truncation logic differs from qwen's policy

## [c4031411629] philippguevorguian (NONE) · review COMMENTED · 2026-03-30 (no text)

## [c3010547131] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:903 · reply to [c2986909284] · 2026-03-30
Good point, went ahead and combined these loops

## [c4031428305] philippguevorguian (NONE) · review COMMENTED · 2026-03-30 (no text)

## [c3010550087] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:903 · reply to [c2986871268] · 2026-03-30
Added a comment explaining the left truncation metadata tracking motivation

## [c4031432062] philippguevorguian (NONE) · review COMMENTED · 2026-03-30 (no text)

## [c3010562752] philippguevorguian (NONE) · review_comment on src/transformers/models/isaac/modular_isaac.py:986 · reply to [c2986945734] · 2026-03-30
Removed this conditional check

## [c4031448382] philippguevorguian (NONE) · review COMMENTED · 2026-03-30 (no text)

## [c3022555420] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:198 · 2026-04-01
a small explanation comment would be nice on `uint8`

## [c3079775327] zucchini-nlp (MEMBER) · review_comment on docs/source/en/model_doc/isaac.md:85 · 2026-04-14
the output from `apply_chat_template` is already a dict of `inputs`

## [c3079818762] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:170 · 2026-04-14
nit: it will be `float` and otherwise error out, as per type annotation. So either we need to annotate (int | float) or we can just delete this line?

## [c3079827200] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:371 · 2026-04-14
these two will be auto-filled by decorators, can delete now

## [c3079850942] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:486 · 2026-04-14
maybe shorter?

```
if image_metadata is not None:
    offsets = [...]
    projected_features = tuple([... for offset in offsets])

return pooler_output=projected_features
```

## [c3079854305] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:489 · 2026-04-14
we don't change the last hidden state, so we can override only pooler output by the last features after projecting

## [c3079868834] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:497 · 2026-04-14
`mm_token_type_ids` here forces us to support and update them during generation, i.e. add new token type for each new token. That was the reason I deleted and copied `get_placeholder_mask` from other models, can we do so?

Then we can also delete a bunch of code later, I will comment

## [c3079879187] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:513 · 2026-04-14
this is same as `super()` except for token offset. It should be fine to copy `get_vision_position_ids` as is and move slicing by offset into `get_rope_index`

## [c3079895864] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:556 · 2026-04-14
`input_ids` cannot be None, no? It is a required input

If yes, both conds can be a bit simplified

## [c3079905423] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:561 · 2026-04-14
initial dtype/device placement happens in processing, and afaik the mask/type ids are long tensors by default. Lets delete if not needed, we try to not process further inputs in `forward`

## [c3079916029] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:581 · 2026-04-14
why not group-by, imo iterating over each token id might be an overkill for very long input sequences

## [c3079927859] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:623 · 2026-04-14
TBH i am quite sure `get_rope_deltas` is similar to super class, but needs extra inputs and offset slicing for vision pos. Can you verify if the draft I shared earlier works or doesn't?

Also, could you add a small comment at the beginning why it is not copied, so the next reviewer doesn't ask the same q again 😅

## [c3079938182] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:625 · 2026-04-14
same here,  quite convinced that copying as I suggested before would work but I might be missing some use-cases

## [c3079942457] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:733 · 2026-04-14
nope, cannot be `0`. I will check processing, but we need to get rid of empty images definitely

## [c3079954356] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:764 · 2026-04-14
Force setting to `None` looks weird, why is it so? IMO we still need correct positions for new tokens. Same q for moving to device, shouldn't be needed because we move it on-the-fly in `accelerate`

## [c3079964960] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:787 · 2026-04-14
can do `class IsaacCausalLMOutputWithPast(Qwen2VLCausalLMOutputWithPast): pass` which is same

## [c3079970344] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:914 · 2026-04-14
this whole block can be deleted if we apply the suggestion on `get_placeholder_mask`

## [c3079982524] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1139 · 2026-04-14
the q remains, sorry if I missed the reply prev. Images are not yet rescaled usually by this time so guaranteed to be `uint`

## [c3079988404] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1150 · 2026-04-14
yeah, I see that we still have `None` images. Can you apply the processing chnages I suggested so the processor always gets a valid non-empty image list?

## [c3079998303] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1354 · 2026-04-14
not needed, templates are better saved as `chat_template.jinja` files in v5 which makes them available to both: tokenizer and processor

## [c3080003836] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1382 · 2026-04-14
we have the power to sync template and processor, no? I mean you have ownership over models or is there a BC issue

The current line can cause us a huge lot of issues when integrating into 3rd party libraries

## [c3080012141] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1385 · 2026-04-14
this part, can you check my suggestion on PR and delete empty images here. It will allow us delte it further in image processing and modeling

In general processors expect a non-empty list

## [c3080075059] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/isaac/modular_isaac.py:1437 · 2026-04-14
can be `None` only when a slow tokenizer is loaded iiuc, and we are slowly dropping off slow tokenizers. Especially in new models

Does Isaac load a slow tokenizer? Maybe we can just assume that offsets are alwasy there, and raise a tiny message if not (e.g. "Make sure you have `tokenizers` installed to load a fast tokenizer. It is important for correct functioning of the model!")

## [c3080084458] zucchini-nlp (MEMBER) · review_comment on src/transformers/modeling_utils.py:1006 · 2026-04-14
kinda unrelated, but I see what you mean. Let's not add it here, model code is already huge. We can keep old-format `get_inputs_embeddings` and you can submit a new PR to improve it, very much welcome :)

## [c3080092898] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:69 · 2026-04-14
let's clean-up global vars before asking final core review. I suppose we will either merge the `pr/5` or have a proper repo for hf-style models?

## [c3080103796] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:96 · 2026-04-14
lets move inside a `modelTester` and delete dead path. Testers are usually simple and deterministic so we can ssume a bunch of stuff

## [c3080106872] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:72 · 2026-04-14
also move inside the specific unitest, used only once

## [c3080111409] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:126 · 2026-04-14
can be in IntegrationTest's `setupClass` and

## [c3080313985] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:213 · 2026-04-14
it is already set to eager if we need attentions no? Which tests were complaining? I can check since we shouldn't force the impl

## [c3080326144] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:345 · 2026-04-14
wait, these tests are skipped for non-video model by default? 

In modular we'll need `def get_video_features(): raise AttributeError()` which deletes the method completely. If not there yet

## [c3080330775] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:372 · 2026-04-14
let's use `testing_utils.torch_device`

## [c3080336764] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:381 · 2026-04-14
why not pass directly as?
`IsaacForConditionalGeneration.from_pretrained(attn_implementation = "flash_attention_2")`

## [c3080347964] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:389 · 2026-04-14
let's not skip if for some reason image isnt loaded. We don't check test outputs in CI and assume green is green, so we can accidentally break Isaac and never notice it

Better to allow it error out if smth goes wrong

## [c3080354936] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:420 · 2026-04-14
`self.assertListEqual` and `self.assertEqual` are much better for us when checking CI outputs, can you fix everywhere

## [c3080361429] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:533 · 2026-04-14
`torch.allclose` would be better, we always get numerical precision diffs due to hardware or pckg versions

## [c3080374015] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:600 · 2026-04-14
testing processor outputs which imo belongs in `test_processor_isaac`

Let's move if we need it, not sure if it is already being tested there. If duplicate or just already tested in similar unitest, then we can simply delete 😅

## [c3080384613] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_modeling_isaac.py:775 · 2026-04-14
same comments on this `IntegrationTest`

## [c3080387649] zucchini-nlp (MEMBER) · review_comment on tests/models/isaac/test_processing_isaac.py:49 · 2026-04-14
same here

## [c4044706834] zucchini-nlp (MEMBER) · review COMMENTED · 2026-04-14
Oke, I have a few questions on processing and 2D RoPE based on prev suggestions. I left questions below, would be great to incorporate the suggestions from before if possible or add a small comment in code about the `main diff is [....]`

Also looked at test file, and seeing a pr ckpt used for integration tests. Do you know what will be the plan after merging the model? (do we merge hub-changes or do we create a new hub repo, so we can clean-up the testing ckpt before a final core review)

## [c4708396819] vasqu (COLLABORATOR) · issue_comment · reply to [c3313695570] · 2026-06-15
What's the current state? @AkshatSh :hugs:

---
outcome: state=open, merged=False
