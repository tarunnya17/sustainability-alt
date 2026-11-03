# #47874 [pull_request] Add WeatherNext 2 weather forecasting model
https://github.com/huggingface/transformers/pull/47874  ·  opened by kashif on 2026-08-10

## [body] kashif (CONTRIBUTOR) · opening post · 2026-08-10
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47874) [[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47874)

Adds WeatherNext 2 (FGN), Google DeepMind's global medium-range weather model. This is the first weather model in the library, so it also adds a `weather-forecasting` task and `AutoModelForWeatherForecasting`.

The model is an encode-process-decode graph network: the lat/lon grid is encoded, projected onto an icosahedral mesh, processed by a transformer with attention restricted to a k-hop neighbourhood on that mesh, and projected back. It's made probabilistic in an unusual way — one 32-dim noise vector per ensemble member modulates the scale and offset of every norm layer, so an ensemble is just several noise draws (the batch dimension here).

Two things worth a look during review:

- The mesh, both grid↔mesh graphs and the attention mask are deterministic functions of the config, so they're built at init into non-persistent buffers and cached on disk rather than shipped in the checkpoint. Mesh, RCM permutation, k-hop mask and ball-query edges are bit-identical to the original.
- After the RCM ordering the attention mask is banded, so attention runs over three block-diagonals instead of the full node set. That keeps a mask that would be 1.7 GB dense at 0.25° down to 1.2 GB and avoids materializing the scores. `eager`, `sdpa` and `flex_attention` agree to 3e-8; Flash Attention can't express this mask.

Checked against the original JAX/Haiku implementation with fixed noise on the published sample data — normalized inputs match to 1e-6 and predictions correlate at 0.999+, though there's still a ~0.04% output difference I haven't tracked down yet.

Converted checkpoints run end-to-end and beat 6h persistence comfortably (0.25°, verified against the HRES analysis):

| field | model | persistence |
|---|---|---|
| 2m_temperature | 0.796 K | 2.622 K |
| mean_sea_level_pressure | 47.3 Pa | 260 Pa |
| geopotential @500hPa | 25.4 m²/s² | 224.5 m²/s² |

Still to do: push the converted weights to the Hub, and add slow integration tests pinned to them.

[collapsed block '🤖 mlinter review state', 7 lines omitted]

## [c3767801150] Rocketknight1 (MEMBER) · review_comment on docs/source/en/model_doc/weathernext2.md:106 · 2026-08-12
Is incrementing valid_time before advance_state correct here? Or does this put the time and state out of sync?

## [c3767812612] Rocketknight1 (MEMBER) · review_comment on docs/source/en/model_doc/weathernext2.md:139 · 2026-08-12
This generator will always be on CPU, no?

## [c3767829741] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/auto/modeling_auto.py:1987 · 2026-08-12
I was mostly kidding about adding a `weather_forecasting` task! We probably shouldn't do it for just one model

## [c3767856960] Rocketknight1 (MEMBER) · review_comment on docs/source/en/model_doc/weathernext2.md:45 · 2026-08-12
We'll definitely have to coordinate and move these to GDM or the right owner before launch

## [c3767871045] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/weathernext2/configuration_weathernext2.py:232 · 2026-08-12
We generally want to avoid these 'Claude headers', same thing for big block comments (Claude loves writing comments to justify its own PR decisions)

## [c3767899648] Rocketknight1 (MEMBER) · review_comment on docs/source/en/model_doc/weathernext2.md:62 · 2026-08-12
Make sure to use `device_map="auto"` or something in doc examples!

## [c4918216171] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-12
There are some issues here that are the same ones I encountered with ESMFold2! I used that PR as a reference since they're both science models with unusual modalities. I commented in a few locations, but other issues are:

1) Rebuilding masks every layer (can we precompute once and just reuse it?)
2) No tests for the feature extractor
3) Do we document anywhere where users can actually get weather data for this? Since it's an unusual modality users might have no idea how to actually use it.
4) Are we careful to either actually support a batch dim, or raise an error if it's passed? What we want to avoid is looking like we half-support it in places, but crashing because it's untested/unsupported in reality. Raising an error if there's a batch dim (or a batch dim > 1) might be good

## [c3784044259] Rocketknight1 (MEMBER) · review_comment on docs/source/en/model_doc/weathernext2.md:106 · reply to [c3767801150] · 2026-08-14
Hey, I see the added comment here, but I'm still not sure about this! I sent a code agent to check the original code and it looks like this deviates from what they do - can we confirm that this is really what we want? This kind of thing will create subtly wrong output

## [c4937602829] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-14 (no text)

## [c3784066065] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/weathernext2/geometry_weathernext2.py:291 · 2026-08-14
This docstring scares me a bit! If we were training a model from scratch then this would be fine, because both tie-breaks are equally valid. But if we train the model with one and then switch to the other this is a distribution shift that could cause divergences. It'd be nice if we could reproduce their original tie-break, or demonstrate that our version isn't causing any output divergence

## [c4937628000] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-14 (no text)

## [c3784082941] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/weathernext2/geometry_weathernext2.py:533 · 2026-08-14
This kind of thing (caching with `np.savez`) is very unusual for Transformers. I think it's fine for science models covering new modalities to do some things differently, but just flagging it as something core maintainers might point out later!

## [c4937643821] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-14 (no text)

## [c3784094304] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/weathernext2/feature_extraction_weathernext2.py:62 · 2026-08-14
Are we registering this with an autoclass anywhere?

## [c4937652514] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-14 (no text)

## [c4937663148] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-14
It's looking better! I added a few more comments. Also, do we have any integration testing that checks actual values, and is there any support for bf16? If so, do we need to make sure some models are added to the `keep_in_fp32` attributes?

## [c5359609250] Rocketknight1 (MEMBER) · issue_comment · 2026-08-20
Looks better now! The two things I'd flag now are 1: Some kind of integration test with real weights if we can, preferably with a smaller model, and 2: Any details about running in bf16 versus fp32. If it's a recent model it was almost certainly trained with bf16, in which case we probably need `keep_in_fp32_strict` entries for things like norms, and it might take some work there to match the reference. We definitely don't want to force users on HF to run in fp32 since it doubles memory usage and it's like 10X slower!

## [c3952207122] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/weathernext2.md:33 · 2026-09-07
Just curious: Do you think this becomes more common, meaning the class; maybe we can have an auto for that in the future 👀

## [c3952208783] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/weathernext2.md:45 · reply to [c3767856960] · 2026-09-07
Whats the state here btw?

## [c3952210841] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/weathernext2.md:51 · 2026-09-07
Maybe we should just have separate repos and one ensemble variation. not a fan of subfolders

## [c3952212606] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/weathernext2.md:56 · 2026-09-07
any expansion of the abbreviation?

## [c3952214875] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/weathernext2.md:80 · 2026-09-07
Could a data collator make more sense instead?

## [c3952221485] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/weathernext2.md:173 · 2026-09-07
would rather like to have a custom generation mixin to handle things like this

## [c3952224731] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/weathernext2.md:277 · 2026-09-07
I mean we have control over the kernels so we can just raise errors based on the type of the mask no?

## [c3952229040] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/hub_kernels.py:303 · 2026-09-07
agentic comments lets shorten. I think only the tested devices are kind of relevant, i.e. xpu is untested but theoretically should work

## [c3952231789] vasqu (COLLABORATOR) · review_comment on src/transformers/models/auto/feature_extraction_auto.py:39 · 2026-09-07
```suggestion
```

## [c3952245711] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/feature_extraction_weathernext2.py:1 · 2026-09-07
Im a bit concerned because iirc feat extraction is usually an audio processor so im not sure. In the end this is kind of needed tho but given we want to rename to audio processor im hesitant here

It feels like something fairly custom in a way and new related to existing feature extractors?

## [c3952249510] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/feature_extraction_weathernext2.py:134 · 2026-09-07
any reason we need all the casts and just force stricter typing?

## [c3952258509] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/geometry_weathernext2.py:381 · 2026-09-07
Any way we could express it with our existing mask api using and/or layers on top of e.g. bidirectional mask creation

## [c3952262156] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/geometry_weathernext2.py:14 · 2026-09-07
Only for conversion or does it have any other usecase? Because maybe we should just make the conversion file bigger but structured

## [c3952267050] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:77 · 2026-09-07
Are there really multiple dimensions?

## [c3952269712] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:59 · 2026-09-07
Why not move the film in this module?

## [c3952272330] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:105 · 2026-09-07
this is clip mlp so we could refactor a bit imo

## [c3952274601] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:124 · 2026-09-07
why not fuse the bias into the linear directly?

## [c3952277932] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:151 · 2026-09-07
is it needed to if here? i.e. do we need that option

## [c3952280870] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:161 · 2026-09-07
hardcoded values, would some intermediate size attrs help or similar?

## [c3952283824] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:215 · 2026-09-07
Makes no sense to me, eager attn expects eager mask no?

## [c3952286524] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:302 · 2026-09-07
Hmm honestly not sure whether this is really the right way and whether it makes more sense as something similar as in minimax 3 and MSA

this is not to say to change it just thinking / wanna discuss

## [c3952288447] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:315 · 2026-09-07
Why not call it before?

## [c3952289634] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:370 · 2026-09-07
llama layer?

## [c3952292871] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:410 · 2026-09-07
can also do hidden states no?

## [c3952299390] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:557 · 2026-09-07
why not one encoder module potentially?

## [c3952302870] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:536 · 2026-09-07
not to familiar but buffering the mask seems a bit weird to me

## [c3952305803] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:698 · 2026-09-07
one module?

## [c5134820926] vasqu (COLLABORATOR) · review COMMENTED · 2026-09-07
Some initial comments before I forget about the PR (I'm sorry 😢)

It's pretty clean but since it's something completely new there is a lot of design questions around it

## [c3961108856] kashif (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/weathernext2.md:277 · reply to [c3952224731] · 2026-09-08
The kernel now rejects BlockMask with a ValueError that points users to SDPA. I also corrected the docs in bf911fd; SDPA batch-one is the fused fast path, while eager and larger batches use the PyTorch fallback.

## [c5145653438] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-08 (no text)

## [c3961109077] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/weathernext2/geometry_weathernext2.py:381 · reply to [c3952258509] · 2026-09-08
Done. The model now builds this through create_bidirectional_mask with the geometry supplied as and_mask_function.

## [c5145653884] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-08 (no text)

## [c3961109392] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:59 · reply to [c3952269712] · 2026-09-08
Done. The FiLM projection now lives directly in WeatherNext2ConditionedNorm.

## [c5145654356] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-08 (no text)

## [c3961109641] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:105 · reply to [c3952272330] · 2026-09-08
Done. WeatherNext2MLP now reuses CLIPMLP, and the conditioned MLP subclasses it.

## [c5145654820] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-08 (no text)

## [c3961109863] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:124 · reply to [c3952274601] · 2026-09-08
Done. The shared bias is now the bias of edge_proj, the projection that is always present.

## [c5145655203] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-08 (no text)

## [c3961110085] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:315 · reply to [c3952288447] · 2026-09-08
Done. The attention interface is resolved before the flatten helper.

## [c5145655589] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-08 (no text)

## [c3961110344] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:370 · reply to [c3952289634] · 2026-09-08
Renamed to WeatherNext2Layer.

## [c5145656028] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-08 (no text)

## [c3961110601] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:698 · reply to [c3952305803] · 2026-09-08
Done. Decoder projection, output projection, and output gating now live in WeatherNext2ForecastHead.

## [c5145656379] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-08 (no text)

## [c3965925585] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:215 · reply to [c3952283824] · 2026-09-09
Yes, agreed. `create_bidirectional_mask` already prepares the additive mask for eager attention. I removed the boolean compatibility branch and added a test asserting that eager receives a floating-point mask.

## [c5151304738] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-09 (no text)

## [c3976848644] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:302 · reply to [c3952286524] · 2026-09-10
I compared the Minimax3/MSA structures. WeatherNext2 has a fixed mesh topology: each query block attends its own and two neighboring blocks, represented as one Q block against three concatenated KV blocks plus the geometry mask. The shared attention interface lets eager, SDPA, and the Hub kernel use the same mask. Minimax3/MSA do not encode this fixed tri-block graph, so I kept the current structure; happy to adjust if there is a closer primitive.

## [c3976848646] kashif (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/weathernext2.md:80 · reply to [c3952214875] · 2026-09-10
I checked the existing collator and processor patterns. A data collator only batches prepared examples, while this object also stores normalization statistics and performs inference-time normalization, calendar forcing construction, residual decoding, NaN restoration, and autoregressive state advancement. A training collator could wrap it later, but cannot replace the serializable transform used here.

## [c3976848650] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/weathernext2/feature_extraction_weathernext2.py:1 · reply to [c3952245711] · 2026-09-10
I checked the existing abstractions. ProcessorMixin is compositional and expects subordinate tokenizer/image/audio components; this is a standalone serializable numerical transform. FeatureExtractionMixin supplies generic save/load and BatchFeature behavior and has non-audio precedent, so it is the closest current base. I am leaving this open because the planned audio rename may warrant a new generic weather abstraction or model-specific processor naming.

## [c3976848655] kashif (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/weathernext2.md:51 · reply to [c3952210841] · 2026-09-10
Agreed. Separate Hub repos plus one ensemble variant is cleaner. This is tied to the pending GDM ownership transfer; I will update the docs to the final repo IDs when those repos exist. Leaving this open until then.

## [c3976848662] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:151 · reply to [c3952277932] · 2026-09-10
Yes. This is one implementation of the two typed graph passes from the original: grid-to-mesh projects and updates mesh receivers and normalizes by mesh receive counts; mesh-to-grid projects and updates grid receivers and uses grid counts. Removing the direction option would duplicate the same message-passing code or change one pass.

## [c3976848663] kashif (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/weathernext2.md:173 · reply to [c3952221485] · 2026-09-10
Done in 09a04d8. WeatherNext2GenerationMixin.generate now owns preprocessing, model calls, postprocessing, clock advancement, and state updates. It accepts fixed per-step noise for reproducible ensembles; tests compare it against the manual rollout.

## [c3976848667] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:536 · reply to [c3952302870] · 2026-09-10
This is fixed topology, not a padding mask. Rebuilding it with the original trimesh algorithm on a different numeric stack selected adjacent faces for 203 of 65,160 edge-tied grid points. Using the checkpoint topology with the same inputs and noise brought JAX/PyTorch normalized mean absolute error to 1.07e-5 and RMSE to 2.68e-5. Buffering preserves the exact conversion-time graph, device moves, and state-dict behavior without runtime scipy/trimesh/rtree dependencies. I added this rationale to the code in 09a04d8.

## [c3976848674] kashif (CONTRIBUTOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:557 · reply to [c3952299390] · 2026-09-10
They are separate learned encoders in the original JAX model and accept different channel layouts and weights for grid nodes versus mesh nodes and edges. A wrapper containing both would only rename state-dict keys and require checkpoint reconversion without sharing computation. I kept them separate, but can group them if module organization itself is preferred.

## [c3976848678] kashif (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/weathernext2.md:33 · reply to [c3952207122] · 2026-09-10
Yes, I expect this to recur. I kept the concrete class for the first weather family; once a second family lands, AutoModelForWeatherForecasting can be added with a real cross-model abstraction rather than locking it in from one implementation.

## [c3976848681] kashif (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/weathernext2.md:45 · reply to [c3767856960] · 2026-09-10
Code and validation are ready. The Hub transfer is still outstanding and needs coordination with GDM/the final owner before launch, so I am leaving this thread open.

## [c5164353721] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-10 (no text)

## [c5164353724] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-10 (no text)

## [c5164353727] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-10 (no text)

## [c5164353730] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-10 (no text)

## [c5164353748] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-10 (no text)

## [c5164353751] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-10 (no text)

## [c5164353756] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-10 (no text)

## [c5164353758] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-10 (no text)

## [c5164353772] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-10 (no text)

## [c5164353774] kashif (CONTRIBUTOR) · review COMMENTED · 2026-09-10 (no text)

## [c5698234321] vasqu (COLLABORATOR) · issue_comment · 2026-09-16
I will probably make a review either today or tomorrow, sorry for the delays!

## [c4064869473] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/weathernext2.md:51 · reply to [c3952210841] · 2026-09-21
Sounds good

## [c4064907622] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/weathernext2.md:97 · 2026-09-21
could we have something explicit here?

## [c4065123916] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/weathernext2.md:118 · 2026-09-21
from pretrained already casts to eval

## [c4065149562] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/weathernext2.md:140 · 2026-09-21
this is a bit weird, I'd expect the processor to (optionally) handle this

## [c4065155480] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/weathernext2.md:157 · 2026-09-21
same here ish, wouldnt it make much more sense to live within the model?

## [c4065169941] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/weathernext2.md:242 · 2026-09-21
small pip install note for the packages that are not default in transformes?

## [c4065235990] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/configuration_weathernext2.py:101 · 2026-09-21
there are a lot of different parts within the models would subconfigs make sense here?

## [c4065247524] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/feature_extraction_weathernext2.py:122 · 2026-09-21
we have a decorator that does this which is more elegant imo

## [c4065269521] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/feature_extraction_weathernext2.py:349 · 2026-09-21
imo we are already torch dependent so dont see why we dilly dally around numpy so much

## [c4065283505] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/feature_extraction_weathernext2.py:385 · 2026-09-21
also a bit weird that we often manually broadcast, why are these not directly broadcasted e.g. in _statistic

## [c4065290509] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/feature_extraction_weathernext2.py:388 · 2026-09-21
ig this is also done in the og? A bit weird but just wanted to know that it's not some workaround from our side w/o explanation

## [c4065312896] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:88 · 2026-09-21
inheritance within the same file is a no no (it acts as normal inheritance), let's rather inherit from clip again here

## [c4065319152] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:110 · 2026-09-21
lets not super call but make it explicit

## [c4065332537] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:153 · 2026-09-21
close to WeatherNext2ConditionedMlp 😬 sad but it is what it is

## [c4065336630] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:167 · 2026-09-21
would rather use this as config attributes than constants

## [c4065349974] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:212 · 2026-09-21
pretty sure we can import from e.g. bert or so

## [c4065360171] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:302 · reply to [c3952286524] · 2026-09-21
No it's fine we should just be careful that the triton kernel rejects an incompatible mask

I expect this to work well if the mask is eager/sdpa but flex's block mask?

## [c4065365259] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:329 · 2026-09-21
not sure but lets check whether we really pass kwargs as downstream 👀

## [c4065386149] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:314 · 2026-09-21
I guess we cannot flatten earlier as the get neighbor states function needs proper blocks?

I'd just like to avoid another function within this function and make it right before getting the interface

## [c4065387319] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:366 · 2026-09-21
dtype cast necessary?

## [c4065392771] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:394 · 2026-09-21
let's make the forward close to llama please, imo we can inherit from llama or similar potentially?

## [c4065400886] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:377 · 2026-09-21
we can shorten this comment massively imo, it's just metadata to accomodate the final folded states which go by bsz x block size instead of the 5d split of bsz block block

## [c4065405094] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:406 · 2026-09-21
```suggestion
```

## [c4065423453] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:591 · 2026-09-21
could potentially be a config entry to simplify the work here?

## [c4065428698] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:543 · 2026-09-21
let's use proper init somehow here tho

## [c4065441747] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:568 · 2026-09-21
could we rather dtype cast where its needed than preemptively?

## [c4065444858] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:576 · 2026-09-21
let's just type it out where we need it, fn within fns is not very transformers esque

## [c4065453230] vasqu (COLLABORATOR) · review_comment on src/transformers/models/weathernext2/modular_weathernext2.py:647 · 2026-09-21
we should already set them at init and only allocate the value with init copy

## [c5269988839] vasqu (COLLABORATOR) · review COMMENTED · 2026-09-21
Ok one last round of desing related questions and comments. I think afterwards it should be one last detail round!

## [c5894606561] Raymondlol (NONE) · issue_comment · reply to [c5164353774] · 2026-09-29
Hi @kashif, great to see WeatherNext 2 coming to Transformers. I've been working on the JAX side ([faster-weathernext](https://github.com/Raymondlol/Faster-WeatherNext): the official code, patched to run at 0.25° on consumer GPUs). Two things that might help here:

**The ~0.04% output difference.** It's worth checking which precision the JAX reference ran at. On Ampere and newer GPUs, JAX's default matmul precision uses TF32; on TPU it uses bf16 passes. With the official WeatherNext 2 modules on the 0.25° Cyclones checkpoint, two implementations that agree to 2e-5 relative RMS at `jax.default_matmul_precision("highest")` differed by up to 4.7e-3 at the default precision ([numbers](https://github.com/Raymondlol/Faster-WeatherNext/blob/main/docs/validation.md)). So a 4e-4 gap could come from the reference side. If the reference ran on CPU, this doesn't apply. Either way, I can dump strict-fp32 JAX activations for the sample input and a fixed noise draw to help bisect: the encoder output, each of the 24 transformer layers, and the decoder output.

**Memory (~50 GB per member in the model card).** In the JAX code, the peak came from per-edge tensors in the grid↔mesh GNNs: the mesh→grid graph has 3.1 M edges, and each per-edge tensor is about 8.9 GiB. Processing edges and receivers in blocks, plus running the 24 layers as a loop, brought one step down to about 6.4 GiB, with results unchanged beyond rounding. The decoder blocks cleanly because each grid point has exactly 3 incoming edges, from its enclosing mesh triangle. The same blocking should carry over to the PyTorch port; there's a write-up in [how-it-works.md](https://github.com/Raymondlol/Faster-WeatherNext/blob/main/docs/how-it-works.md), and I'm happy to share details.

## [c5895125657] kashif (CONTRIBUTOR) · issue_comment · reply to [c5894606561] · 2026-09-29
thanks @Raymondlol checking!

## [c5956935680] kashif (CONTRIBUTOR) · issue_comment · reply to [c5894606561] · 2026-10-02
@Raymondlol i have added some of your suggestions here now if you want to check

---
outcome: state=open, merged=False
