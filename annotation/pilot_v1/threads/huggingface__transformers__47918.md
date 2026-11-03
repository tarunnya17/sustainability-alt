# #47918 [pull_request] Moving mlinter to 0.1.4
https://github.com/huggingface/transformers/pull/47918  ·  opened by tarekziade on 2026-08-12

## [body] tarekziade (COLLABORATOR) · opening post · 2026-08-12
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47918)

# What does this PR do?

Bumps `transformers-mlinter` to [`0.1.4`](https://github.com/huggingface/transformers-mlinter) to pick up the new rules (TRF020 to TRF057), and resolves everything those rules flag.

The base pinned `transformers-mlinter==0.1.2`, whose `utils/rules.toml` carried only TRF001–TRF019. This bump adds **37 new rules in the TRF020–TRF057 range**, and every finding they reported across the repo is resolved: `make check-code-quality` passes with 0 modeling structure violations.

No pre-existing rule changes behavior: the TRF001–TRF019 rule definitions and allowlists are exactly the base's, with justification comments added on top. In particular **TRF018 is untouched** — it is an old rule and the base already allowlists the 16 models that would otherwise trip it.

## Cherry-picked from other PRs

To avoid duplicating work already done elsewhere. Both are cherry-picked with `-x`, preserving original authorship:

- **#47866** (@zucchini-nlp), `5b09535` — renames 9 test files from `test_processor_*.py` to `test_processing_*.py` (`auto`, `glm46v`, `glm4v`, `glm_image`, `kosmos2_5`, `lighton_ocr`, `ovis2`, `sam2`, `sam2_video`). Those tests were never missing, just named where `TRF038` does not look for them. Clears 5 of the 7 `TRF038` findings.
- **#47768** (@guarin), `1ed5317` — adds the missing `@auto_docstring` decorators that **TRF057 is the rule for** (authored alongside it in mlinter PR #21). Without it TRF057 is red across many models; with it, only 4 findings remain, all in models added after that PR was written, fixed below.
- **#47536** — nothing to take. That PR introduces the `TRF020` rule definition and bumps the pin; it contains no model-side fix, and it carries the same `axk2` finding this PR started with. The rule itself already arrives here via mlinter main, so the only open question was the `axk2` finding, handled under *Allow lists* below.

### Conflict resolution worth reviewing

#47768 conflicted in 8 places on the four `Inkling` config classes. The two sides were complementary rather than competing:

- #47768 (Aug 4) enabled `@auto_docstring` and documented ~150 lines of config fields
- `main` (Aug 11, #47902) had `@auto_docstring(checkpoint="thinkingmachines/Inkling")` **commented out** behind a TODO: *"uncomment once all fields below are documented"*

Since #47768 documents exactly those fields, the TODO is discharged. Resolved by enabling the decorator **and** keeping `main`'s newer `checkpoint=` argument (plus `custom_intro` on `InklingConfig`). `utils/check_docstrings.py` reports no Inkling errors, so this is verified rather than assumed.

## Fixes

| rule | model | change |
| --- | --- | --- |
| `TRF029` | axk2 | `AXK2GatedRMSNorm.__init__` no longer takes `eps` alongside `config`; it reads `config.rms_norm_eps` internally. Both call sites were already passing exactly that value. |
| `TRF043` | cohere_compass | Dropped the unused `position_ids` parameter from `CohereCompassAttention.forward`; it flows through `**kwargs` so padding-free flash-attention can consume it. No caller was passing it. |
| `TRF055` | qwen3_omni_moe | `Qwen3OmniMoeThinkerTextPreTrainedModel.config` is now an annotation rather than an assignment, which was shadowing `config_class` resolution. This was the only assignment form left in the repo. |
| `TRF038` | cohere_compass | Added `test_image_processing_cohere_compass.py` and `test_video_processing_cohere_compass.py`, mirroring the qwen2_vl / qwen3_vl tests that these `pass`-subclass processors inherit from. |
| `TRF057` | cohere_compass, muse_glimmer, muse_glimmer_assistant | Added the 4 missing `@auto_docstring` decorators (see below). |

### TRF057 — missing `@auto_docstring`

These 4 are the ones #47768 could not have covered, since all three models were added after it was written. `@auto_docstring` goes **innermost** on methods, as the rule requires:

| target | file |
| --- | --- |
| `CohereCompassVisionModel` | `modular_cohere_compass.py` |
| `MuseGlimmerVisionModel` | `modular_muse_glimmer.py` |
| `MuseGlimmerVisionModel.forward` | `modular_muse_glimmer.py` |
| `MuseGlimmerAssistantModel.forward` | `modular_muse_glimmer_assistant.py` |

`TRF056` (no `.item()` / `.tolist()` in `forward`) reported nothing.

### Docstring fallout from the cherry-pick

#47768 also touches `utils/auto_docstring.py`, and together with the new decorators that surfaced three `check_docstrings` failures in models newer than it. Fixed at the **modular** source, since fixing the generated files would be reverted on the next conversion:

- Redundant `video_grid_thw` / `image_grid_thw` arg docs, now supplied centrally — removed from `modular_kimi_k25.py` and `modular_hunyuan_vl.py`. `muse_glimmer` inherits `get_video_features` from `Kimi_K25Model`, so the kimi_k25 fix covers it too.
- `MuseGlimmerVisionModel.forward` needed `grid_thw` documented once decorated.

## Allow lists — justified in place

Per @ArthurZucker's review: every non-empty allowlist now carries a comment in `utils/rules.toml` saying what is on it and why, so the exemption is readable next to the rule instead of living in a PR thread. Each entry was re-checked against a run with all allowlists emptied (228 findings), so the counts and the model lists in those comments are measured, not estimated. Two entries no longer fire at all and were dropped: `sam2` (TRF038) and `mimo_v2_flash` (TRF041).

Short version of what the comments say:

| rule | why the list exists |
| --- | --- |
| `TRF020`, `TRF034` | Mostly false positives: the rule resolves a class's base chain only inside the linted file, so `modular_*.py` inheritance from another model reads as a missing base while the generated `modeling_*.py` is compliant — checked class by class (13 models for TRF034, `axk2` for TRF020). Upstream: mlinter#27. TRF034 also has 5 real ones (`dinov3_convnext`, `openai`, `tipsv2_dpt`, `x_clip`'s `PromptGeneratorLayer`, `xcodec2`'s `Xcodec2ResNetBlock`), left for a follow-up. |
| `TRF035` | 4 `# noqa`s that ruff needs: one deliberate re-export (`hunyuan_vl`) and three modular-converter artifacts (`step3p7` F401 import that picks Qwen2-VL's 2-axis vision RoPE, `nemotron3_5_asr` quoted forward reference, `zaya` `del` of an inherited name). Deleting the suppression just moves the failure to ruff, so the fix belongs upstream. |
| `TRF038` | Verified by @molbap: only `nougat` and `glmga` genuinely have no test. The rest are tested models whose processor / feature extractor has no dedicated test file, and the rule pairs each source file with its own test. Per @zucchini-nlp, enforce for new models and backfill the old families later. |
| `TRF041` | 35 branches over 13 models (openai alone has 11). Each `# CODEPATH:` note has to name the checkpoints that take each path — only the model owner knows that, and guessing would be worse than the missing note. Follow-up per model. |
| `TRF016`, `TRF019` | Inherited unchanged from the base pin; the live entries need a decision from the flag / checkpoint owner (`do_convert_rgb` that no `_preprocess` reads; `_defaults` that has to move to `processor_config.json` on the Hub first). |
| `TRF018` | 15 of the 16 spell the base call `PreTrainedModel._init_weights(module)` instead of `super()._init_weights(module)` — the form a modular file needs, since `super()` there resolves to the inherited model. 459 files use the spelling the rule accepts, these 15 the other. `radio` is the one real case. |
| `TRF043`, `TRF046`, `TRF051`, `TRF053` | These four have no `cutoff_date`, so they fire across the pre-existing corpus (21 / 26 / 10 / 37 findings). Each fix is a per-model breaking change or behavior change — a public `forward` signature, a stateful module, an attention-dispatch branch, a loss path — so they go out one model per PR. For `TRF043` in particular, a `cutoff_date` upstream would express the legacy half better than a 21-model list. |

Fixing the models on these lists is explicitly **not** in this PR; it is the follow-up series agreed above.

## Review follow-ups on `step3p7`

- The three `# CODEPATH:` notes were cut to two lines each (@ArthurZucker: *"comment can be a lot shorter"*).
- `Step3p7DecoderLayer` no longer builds the MLP through a conditional expression: now that `Step3p7SparseMoeBlock` and `Step3p7MLP` share the `(config, layer_idx)` signature, it picks the class and calls it once. `tests/models/step3p7` passes (189 tests).

## Suppressed in place

**`TRF026`** on `MuseGlimmerTextNormedEmbedding` (`# trf-ignore: TRF026`) — also a false positive. The class is an `nn.Embedding` subclass and its forward does the embedding lookup through `super().forward()` before applying the norm, so it is not a pass-through wrapper and there is no inner module to hoist out.

## Linter blind spots found

8 findings were false positives, and 7 of those share one root cause: the rules resolve a class's base chain only within the file being linted, so normal `modular_*.py` inheritance from another model reads as a missing base. Filed as **huggingface/transformers-mlinter#27** with the evidence for `TRF034`, `TRF020` and `TRF026`, plus a note that `TRF018` accepts only one of the two spellings of the modular `_init_weights` sentinel in use here (459 files vs 15).

The 6 allowlist entries added for `TRF034` and `TRF020` should be removable once that issue is fixed — they record a deviation that does not actually exist.

## Testing

- `make fix-repo` clean, `make check-code-quality` passes (0 modeling structure violations, `ty` clean)
- `cohere_compass`, `muse_glimmer`, `axk2` and `qwen3_omni_moe` suites pass locally, including the two new cohere_compass processor test files
- The #47768 cherry-pick changes docstrings and `utils/auto_docstring.py`, so it leans on CI for full model coverage

## [c5272190184] tarekziade (COLLABORATOR) · issue_comment · 2026-08-12
run-slow: axk2, cohere_compass, gemma4, mbart, muse_glimmer, qwen3_omni_moe

## [c3773776750] zucchini-nlp (MEMBER) · review_comment on docs/source/en/modeling_rules.md:304 · 2026-08-13
wow, did we forget to add it in docs or is this release that big?

## [c3773798059] zucchini-nlp (MEMBER) · review_comment on docs/source/en/modeling_rules.md:350 · 2026-08-13
oh wooow, very nice rule. I see it checks for equivalent names based on hardcoded list, prob could be extended later

## [c3773865934] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/muse_glimmer/processing_muse_glimmer.py:62 · 2026-08-13
i am really not sure about this rule, and I think Pablo is back from short vacation today. Can we wait for Pablo's input on it, seems to be a typo to me as we can get a really long list of special tokens
E.g. on tokenizer-side I remember moving from explicit `property` to a more dynamic way to unclutter it in the past

## [c4925080916] zucchini-nlp (MEMBER) · review APPROVED · 2026-08-13
LGTM, but lets wait for Pablo on TRF054 before merging, want to make sure it is intended

## [c3773880557] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/muse_glimmer/processing_muse_glimmer.py:62 · reply to [c3773865934] · 2026-08-13
cc @molbap , forgot to tag you

## [c4925213858] zucchini-nlp (MEMBER) · review COMMENTED · 2026-08-13 (no text)

## [c5278382590] tarekziade (COLLABORATOR) · issue_comment · 2026-08-13
run-slow: axk2, cohere_compass, cosmos3_edge, dinov3_convnext, dinov3_vit, gemma4, hunyuan_vl, inkling, kimi_k25, mbart, muse_glimmer, nemotron3_5_asr, qwen3_omni_moe, radio, sapiens2, tipsv2_dpt

## [c3774039466] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/hunyuan_vl/modular_hunyuan_vl.py:1339 · 2026-08-13
these should be remove-able now i think, i recently merged PR into main for grid-thw

## [c4925406649] zucchini-nlp (MEMBER) · review COMMENTED · 2026-08-13 (no text)

## [c3793317639] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/modeling_rules.md:304 · reply to [c3773776750] · 2026-08-17
Wow I did not know that ! Its a good decision tho!

## [c3793321729] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/modeling_rules.md:403 · 2026-08-17
hell yeah

## [c3793322791] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/modeling_rules.md:444 · 2026-08-17
super important!

## [c3793323965] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/modeling_rules.md:526 · 2026-08-17
yesss

## [c3793324883] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/modeling_rules.md:581 · 2026-08-17
HELLL YEAHHHH

## [c3793330116] ArthurZucker (COLLABORATOR) · review_comment on docs/source/en/modeling_rules.md:764 · 2026-08-17
super nice

## [c3793333935] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/step3p7/modular_step3p7.py:795 · 2026-08-17
comment can be a lot shorter for this IMO

## [c3793334074] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/step3p7/modular_step3p7.py:827 · 2026-08-17
same here

## [c3793335329] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/step3p7/modular_step3p7.py:908 · 2026-08-17
no need for a if else since we changed the signature

## [c3793338197] ArthurZucker (COLLABORATOR) · review_comment on utils/rules.toml:650 · 2026-08-17
mmm we should either fix or explain each allowed model here

## [c3793338703] ArthurZucker (COLLABORATOR) · review_comment on utils/rules.toml:674 · 2026-08-17
same we can fix imo

## [c3793339566] ArthurZucker (COLLABORATOR) · review_comment on utils/rules.toml:684 · 2026-08-17
Fine AFAIK but needs explanation

## [c3793340058] ArthurZucker (COLLABORATOR) · review_comment on utils/rules.toml:727 · 2026-08-17
nope this is one of the most important one!

## [c3793340459] ArthurZucker (COLLABORATOR) · review_comment on utils/rules.toml:816 · 2026-08-17
Same

## [c3793340782] ArthurZucker (COLLABORATOR) · review_comment on utils/rules.toml:884 · 2026-08-17
same

## [c3793341375] ArthurZucker (COLLABORATOR) · review_comment on utils/rules.toml:975 · 2026-08-17
same

## [c3793341597] ArthurZucker (COLLABORATOR) · review_comment on utils/rules.toml:1015 · 2026-08-17
same

## [c4947838945] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-08-17
Super nice BUT let's split into 1 PR per new rule? And justify better each rules allow list please

## [c5313966674] tarekziade (COLLABORATOR) · issue_comment · reply to [c4947838945] · 2026-08-17
> Super nice BUT let's split into 1 PR per new rule? And justify better each rules allow list please

switching to the new mlinter requires us to update the toml file with all new rules, so we have two options to split one PR per rule:

- have this PR land all rules but disabled, and then activate each rule in a separate PR
- land the existing patch and do follow ups

the benefit of the later is that we cover now new models, like step3p7

## [c5314023957] ArthurZucker (COLLABORATOR) · issue_comment · reply to [c5313966674] · 2026-08-17
Okay, 
> land the existing patch and do follow ups

SG

## [c3796860143] molbap (COLLABORATOR) · review_comment on utils/rules.toml:684 · reply to [c3793339566] · 2026-08-17
I verified, only `nougat` and `glmga` do not apply. The others have no processing test, so the linter flags them because it's directory-based. All the others are OK

## [c4952214576] molbap (COLLABORATOR) · review COMMENTED · 2026-08-17 (no text)

## [c3796872970] zucchini-nlp (MEMBER) · review_comment on utils/rules.toml:684 · reply to [c3793339566] · 2026-08-17
i was just lazy to add missing tests for all old models 
better to enforce for new models, and add old tests if someone feels bored

## [c4952230475] zucchini-nlp (MEMBER) · review COMMENTED · 2026-08-17 (no text)

## [c5317052015] molbap (COLLABORATOR) · issue_comment · 2026-08-17
Agreed as it stands TRF 054 does hold up at all :eyes: but 67 processors do assign token ids, none serialize

## [c5317191076] zucchini-nlp (MEMBER) · issue_comment · 2026-08-17
Niice, thanks for checking it, didn't really know they aren't serialized atm. Should we deactivate the TRF054 then?

## [c5317201735] tarekziade (COLLABORATOR) · issue_comment · 2026-08-17
Yeah as agreed with Pablo in a discussion I will remove it. Thanks both for looping back !

---
outcome: state=closed, merged=True
