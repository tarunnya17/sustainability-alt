# #41116 [pull_request] Add MiniCPM3
https://github.com/huggingface/transformers/pull/41116  ·  opened by bzantium on 2025-09-24

## [body] bzantium (CONTRIBUTOR) · opening post · 2025-09-24
# What does this PR do?

Adds native support for the **MiniCPM3** architecture from OpenBMB
([`openbmb/MiniCPM3-4B`](https://huggingface.co/openbmb/MiniCPM3-4B)).

MiniCPM3 combines:

- **Multi-head Latent Attention (MLA)** from DeepSeek-V2, with the original cos/sin rotary embedding convention preserved on the rotary part of the query/key heads.
- A standard SwiGLU MLP (no MoE).
- Three scalar scaling factors:
  - `scale_emb` — multiplier on input embeddings.
  - `scale_depth / sqrt(num_hidden_layers)` — multiplier on residual connections.
  - `hidden_size / dim_model_base` — divisor on hidden states before the LM head.

When `scale_depth` / `dim_model_base` are not provided, they default to no-op values (`sqrt(num_hidden_layers)` and `hidden_size` respectively).

Implemented as a single `modular_minicpm3.py` deriving from `LlamaConfig` / `LlamaModel`, with a custom `MiniCPM3Attention` for MLA. The model reuses the upstream `LlamaTokenizer` (no new tokenizer files); `minicpm3` is added to `MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS` so the auto loader falls back to `TokenizersBackend`. `AutoConfig` / `AutoModel` / `AutoModelForCausalLM` / `AutoModelForSequenceClassification` mappings, docs, and tests are wired up.

Fixes #41115.

A second, more recent PR (#45613) covers the same model. This PR (#41116) is the original; the implementations differ on the RoPE convention (this one preserves the original cos/sin RoPE; #45613 uses DeepSeek-V2's complex RoPE) and on a few config defaults.

AI assistance was used during the rebase/refactor; the changes have been reviewed and tested locally end-to-end (see follow-up comment for the example run on `openbmb/MiniCPM3-4B`).

## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [x] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case. — #41115
- [x] Did you make sure to update the documentation with your changes?
- [x] Did you write any new necessary tests?

## Who can review?

@ArthurZucker @Cyrilvallez

## [c3327964440] Rocketknight1 (MEMBER) · issue_comment · 2025-09-24
cc @arthurzucker for text models

## [c3354435354] bzantium (CONTRIBUTOR) · issue_comment · 2025-10-01
@ArthurZucker please check this out! I've tested with MiniCPM3-4B and it worked well.

## [c3386562919] bzantium (CONTRIBUTOR) · issue_comment · 2025-10-09
@ArthurZucker I've tested MiniCPM3-4B and our in-house model for the case `q_lora_rank is None`. Please review this when you are available and let me know if more needed! thanks a lot :) I hope this can be merged because we are going to open-weights one of models using this modeling in the near future.

## [c4341492677] bzantium (CONTRIBUTOR) · issue_comment · 2026-04-29
End-to-end check on `openbmb/MiniCPM3-4B`:

```python
import torch
from transformers import AutoTokenizer, MiniCPM3ForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM3-4B")
[… 21 lines of code omitted]

## [c4341533009] bzantium (CONTRIBUTOR) · issue_comment · 2026-04-29
@aliyevaladddin Sorry I missed your email earlier. I just rebased this PR, would be happy to add you as coauthor since your message got me to come back to it.

## [c4341578646] bzantium (CONTRIBUTOR) · issue_comment · 2026-04-29
@ArthurZucker @Cyrilvallez when you have a moment, would appreciate a review on this. Thanks!

## [c4349488106] aliyevaladddin (CONTRIBUTOR) · issue_comment · reply to [c4341578646] · 2026-04-30
Thanks for the rebase, @bzantium! I'm honored to be added as a co-author. I'm really excited to see MiniCPM3 support getting closer to being merged. Let me know if there's anything else I can help with regarding testing or further updates!

## [c4708373261] vasqu (COLLABORATOR) · issue_comment · 2026-06-15
Will try to take a look today :hugs:

## [c3416130288] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/minicpm3.md:1 · 2026-06-15
```suggestion

## [c3416133221] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/minicpm3.md:16 · 2026-06-15
Pretty sure the wording was slightly updated so please sync with main and recheck

## [c3416139723] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/minicpm3.md:40 · 2026-06-15
please use automodelforcausallm - its more user friendly and oftentimes reveals if we missed some mapping

## [c3416142629] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/__init__.py:1 · 2026-06-15
```suggestion
# Copyright 2026 The HuggingFace Team. All rights reserved.
```

## [c3416144098] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:1 · 2026-06-15
```suggestion
# Copyright 2026 The OpenBMB Team and the HuggingFace Inc. team. All rights reserved.
```

## [c3416162995] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:70 · 2026-06-15
Is this actually used in any of the models then if we default to hidden size == no effect? If its not really used, we could remove it imo

## [c3416173406] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:127 · 2026-06-15
```suggestion
    vocab_size: int = 73448
    hidden_size: int = 2560
    intermediate_size: int = 6400
    num_hidden_layers: int = 62
    num_attention_heads: int = 40
[… 13 lines of code omitted]
```
we can remove a few things here since they are already defined in llama with the same values

## [c3416177147] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:122 · 2026-06-15
Ig we need this because of the remote configs?

## [c3416186490] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:142 · 2026-06-15
Seems like we dont overwrite in e.g. dsv3. Any reason we need it here?

## [c3416194813] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:160 · 2026-06-15
We can still inherit from some model like dsv3 or dsv2 no? There should be some big similarities

You can delete attributes with `del self.attr` if you dont want to inherit or override/add by just declaring like usual `self.attr = ..`

## [c3416200777] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:189 · 2026-06-15
Is this actually something that happens. Would like to avoid if possible but no worries if not

## [c3416209413] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:307 · 2026-06-15
Why not directly calculate the scale as attributes? We dont really need to do this each time and let's add a small comment higlighting the diff

## [c3416211796] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:334 · 2026-06-15
same here to highlight

## [c3416214177] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:417 · 2026-06-15
should be a direct config attribute imo

## [c3416215070] vasqu (COLLABORATOR) · review_comment on tests/models/minicpm3/test_modeling_minicpm3.py:1 · 2026-06-15
```suggestion
# Copyright 2026 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.
```

## [c3416217924] vasqu (COLLABORATOR) · review_comment on tests/models/minicpm3/test_modeling_minicpm3.py:53 · 2026-06-15
```suggestion
```
shouldnt be needed, it mostly happens on moe models

## [c3416227917] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:160 · reply to [c3416194813] · 2026-06-15
Also noticing doesnt this match dsv2 exactly? Why not inherit from there?

## [c3416231155] vasqu (COLLABORATOR) · review_comment on tests/models/minicpm3/test_modeling_minicpm3.py:93 · 2026-06-15
please check minimax m2 for proper integration tests: we want actual values (I can adjust for our CI)

## [c4500596710] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-15
Some initial comments from my side 🫡

## [c3425519952] bzantium (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/minicpm3.md:1 · reply to [c3416130288] · 2026-06-17
Done, bumped to 2026.

## [c4512289898] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520015] bzantium (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/minicpm3.md:16 · reply to [c3416133221] · 2026-06-17
Rechecked against main: the doc-builder note and the 'released on / added on' line already match the current wording, so I just bumped the copyright year.

## [c4512289957] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520061] bzantium (CONTRIBUTOR) · review_comment on docs/source/en/model_doc/minicpm3.md:40 · reply to [c3416139723] · 2026-06-17
Done, switched the usage example to AutoModelForCausalLM.

## [c4512290080] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520123] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/__init__.py:1 · reply to [c3416142629] · 2026-06-17
Done.

## [c4512290294] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520198] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:1 · reply to [c3416144098] · 2026-06-17
Done.

## [c4512290463] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520270] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:70 · reply to [c3416162995] · 2026-06-17
These are not identity operations for the released checkpoint: openbmb/MiniCPM3-4B sets scale_emb=12, scale_depth=1.4, dim_model_base=256, so the input to the LM head is divided by hidden_size / dim_model_base = 2560 / 256 = 10. The defaults only collapse to identity when left unset (for example a randomly initialised tiny test model), so I kept dim_model_base and exposed the logit scaling as a config property (see below).

## [c4512290532] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520340] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:127 · reply to [c3416173406] · 2026-06-17
Done, removed the fields that match LlamaConfig defaults and kept only the ones that differ plus the MLA and scaling extras (also dropped the now unused RopeParameters import).

## [c4512290608] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520402] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:122 · reply to [c3416177147] · 2026-06-17
Same as DeepseekV2Config: in MLA the per head dim used by RoPE is the rotary part (qk_rope_head_dim), so head_dim has to be set here for the rope and cache shapes. Kept it, and it is now consistent with the inherited DeepseekV2Attention.

## [c4512290791] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520453] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:142 · reply to [c3416186490] · 2026-06-17
Removed the override so the standard Llama divisibility check applies, matching dsv2 and dsv3. The released config (2560 % 40 == 0) and the test configs pass, so it was not needed.

## [c4512290946] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520495] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:160 · reply to [c3416194813] · 2026-06-17
Good point. The __init__ is indeed identical to DeepseekV2Attention and forward only differs in the RoPE call: MiniCPM3 keeps the original cos/sin convention (apply_rotary_pos_emb) whereas DeepSeek-V2 uses the complex interleaved apply_rotary_emb. I now inherit from DeepseekV2Attention and override only forward, which drops the duplicated __init__. Verified the attention output is bit identical before and after the change. The rotary embedding stays Llama based (cos/sin), and the decoder layer cannot be inherited from dsv2 since MiniCPM3 adds the scale_depth residual scaling.

## [c4512291027] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520563] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:160 · reply to [c3416194813] · 2026-06-17
Right, the module construction matches dsv2 exactly. Done as discussed in the other thread: MiniCPM3Attention now inherits DeepseekV2Attention and overrides only forward for the cos/sin RoPE.

## [c4512291104] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520608] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:189 · reply to [c3416200777] · 2026-06-17
The released checkpoint uses q_lora_rank=768, so the None branch is dead for the real weights. It now comes for free from the inherited DeepseekV2Attention, so it is no longer extra code here.

## [c4512291165] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520654] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:307 · reply to [c3416209413] · 2026-06-17
Done, precomputed as self.residual_scale in the decoder layer __init__ and reused for both residual branches, with a comment noting the diff from Llama.

## [c4512291227] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520716] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:334 · reply to [c3416211796] · 2026-06-17
Added comments highlighting the embedding, residual and logit scalings versus Llama, and both residual branches now use the precomputed self.residual_scale.

## [c4512291298] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520777] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:417 · reply to [c3416214177] · 2026-06-17
Done, exposed it as a MiniCPM3Config.logits_scaling property (hidden_size / dim_model_base); the forward now divides by self.config.logits_scaling.

## [c4512291365] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520849] bzantium (CONTRIBUTOR) · review_comment on tests/models/minicpm3/test_modeling_minicpm3.py:1 · reply to [c3416215070] · 2026-06-17
Done.

## [c4512291445] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520901] bzantium (CONTRIBUTOR) · review_comment on tests/models/minicpm3/test_modeling_minicpm3.py:53 · reply to [c3416217924] · 2026-06-17
Done, removed test_all_params_have_gradient = False. MiniCPM3 is dense, so the MoE oriented flag is not needed and all params get gradients.

## [c4512291616] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3425520964] bzantium (CONTRIBUTOR) · review_comment on tests/models/minicpm3/test_modeling_minicpm3.py:93 · reply to [c3416231155] · 2026-06-17
Reworked the integration tests to the value based Expectations pattern (a logits slice plus exact generated text), following MiniMax-M2. The expected values still need a reference run on the target CI hardware, so I left them as Expectations placeholders that skip until filled. Would appreciate your help adjusting those for CI as you offered.

## [c4512291800] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-17 (no text)

## [c3430105708] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:70 · reply to [c3416162995] · 2026-06-17
But the default values should match to those of `openbmb/MiniCPM3-4B` (the ckpt in auto docs). Im fine with defaulting to a no op if it is explicitly set to None

## [c3430110333] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:98 · 2026-06-17
```suggestion
```
should be the same in llama no?

## [c3430136725] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:194 · 2026-06-17
So the forward is essentially dsv3 with the non interleave, correct? Could we add a small comment here to highlight?

## [c3430144111] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:225 · 2026-06-17
```suggestion
```
shouldnt be needed, no? Modular should also inherit this properly I think

## [c3430156020] vasqu (COLLABORATOR) · review_comment on tests/models/minicpm3/test_modeling_minicpm3.py:93 · reply to [c3416231155] · 2026-06-17
Yep no worries, will adjust for our CI

## [c3430160434] vasqu (COLLABORATOR) · review_comment on tests/models/minicpm3/test_modeling_minicpm3.py:101 · 2026-06-17
Can you at least add some values based on your hw for now?

## [c4517962977] vasqu (COLLABORATOR) · review APPROVED · 2026-06-17
Just some small comments but overall lgtm. Let me update the CI tomorrow then after you fixed the last comments 🫡

## [c4733345646] vasqu (COLLABORATOR) · issue_comment · 2026-06-17
We also need to sync with main and fix CI please

## [c4520832414] aliyevaladddin (CONTRIBUTOR) · review APPROVED · 2026-06-18 (no text)

## [c3432474839] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:70 · reply to [c3416162995] · 2026-06-18
Good point — updated the class defaults to match `openbmb/MiniCPM3-4B` (`scale_emb=12`, `scale_depth=1.4`, `dim_model_base=256`). The no-op behaviour is now only triggered when a field is explicitly set to `None` (e.g. a tiny randomly-initialised test model), as you suggested.

## [c4520888291] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-18 (no text)

## [c3432474915] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:98 · reply to [c3430110333] · 2026-06-18
Right, `keys_to_ignore_at_inference = ["past_key_values"]` is already inherited from `LlamaConfig`. Removed.

## [c4520888362] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-18 (no text)

## [c3432474975] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:194 · reply to [c3430136725] · 2026-06-18
Correct. Added a comment at the rotary call noting the forward is the same MLA path as DeepSeek-V2/V3, except MiniCPM3 keeps the standard (non-interleaved) cos/sin rotary instead of DeepSeek's complex/interleaved variant.

## [c4520888460] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-18 (no text)

## [c3432475047] bzantium (CONTRIBUTOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:225 · reply to [c3430144111] · 2026-06-18
You're right, modular renames the inherited `self_attn` to `MiniCPM3Attention` automatically, so the explicit assignment was redundant. Removed; the generated `modeling_minicpm3.py` is unchanged.

## [c4520888656] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-18 (no text)

## [c3432475143] bzantium (CONTRIBUTOR) · review_comment on tests/models/minicpm3/test_modeling_minicpm3.py:101 · reply to [c3430160434] · 2026-06-18
Ran `openbmb/MiniCPM3-4B` on an A100 (bf16) and filled in the `Expectations`:
- logits slice: `[0.765625, 3.640625, -0.189453125, -0.8359375, -0.8359375]`
- generation: `"My favourite condiment is \n[A]. ketchup \n[B]. mustard \n[C"`

Both `test_minicpm3_4b_logits` and `test_minicpm3_4b_generation` pass under `("cuda", 8)`. Feel free to adjust/add per-hardware entries for your CI.

## [c4520888749] bzantium (CONTRIBUTOR) · review COMMENTED · 2026-06-18 (no text)

## [c4737151957] bzantium (CONTRIBUTOR) · issue_comment · reply to [c4733345646] · 2026-06-18
@vasqu 

The `security-gate / security-check` failure comes from the "Enforce allowlist" step. The only offending file is `docs/source/en/_toctree.yml`, which the gate rejects because `.yml` is not allowed outside `tests/` (only `.py`, `.txt`, `.md` are permitted there). That `_toctree.yml` entry is required to register MiniCPM3 in the docs navigation, so it cannot be dropped.

This gate only runs for fork PRs, and the failure comment notes that a maintainer can manually approve CI when a finding is a false positive. Could you approve CI when you get a chance? All 11 other changed files pass the allowlist. Thanks!

## [c4527204396] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-18 (no text)

## [c3437764112] vasqu (COLLABORATOR) · review_comment on src/transformers/models/minicpm3/modular_minicpm3.py:141 · 2026-06-18
Can we inherit from gemma3 for example?

## [c4527338470] vasqu (COLLABORATOR) · review APPROVED · 2026-06-18
LGTM, the logits differ quite a bit from 8 to 8.6 so just wanna make sure if you can check that I didnt break anything

## [c4768142608] vasqu (COLLABORATOR) · issue_comment · reply to [c4737151957] · 2026-06-22
gentle ping @bzantium

## [c4775608769] bzantium (CONTRIBUTOR) · issue_comment · reply to [c4768142608] · 2026-06-23
@vasqu Thanks for the careful check! I verified on an A100 (compute capability 8.0, same hardware the `("cuda", 8)` reference came from) using the current branch, which includes your modular cleanup. Nothing is broken.

**bf16 reproduction (A100, cc 8.0):**
```
slice : [0.765625, 3.640625, -0.189453, -0.835938, -0.835938]
ref   : [0.765625, 3.640625, -0.189453125, -0.8359375, -0.8359375]
diff  : [0, 0, 0, 0, 0]   # exact match
```
So your refactor (inheriting `Gemma3TextScaledWordEmbedding` and dropping the duplicated `MiniCPM3Model.forward`) is numerically a no op. That matches the code too: the new embedding base is identical to the old one, and the removed `forward` was a verbatim copy of the inherited `LlamaModel.forward`.

**On the 8 vs 8.6 gap:** both buckets run bf16 (the config `torch_dtype` is `bfloat16`, and the test loads with `dtype="auto"`), so the difference is purely cross GPU bf16 variation, not a dtype or code change. These particular logits sit near zero and the model is 62 layers deep, so bf16 rounding accumulates and the low magnitude entries wobble. To put a number on it, I also ran fp32 as a ground truth on the same A100:
```
fp32  : [0.535951, 3.610042, -0.174161, -1.174419, -1.174420]
```
The bf16 vs fp32 gap reaches about 0.34, which is larger than the 8.0 vs 8.6 gap of about 0.08. Most importantly, the argmax token is 11213 in bf16 and fp32 alike, so the actual prediction is stable across precision and only the near zero logit tail moves.

Net: your `(8, 6)` bucket is a legitimate per hardware entry and the existing `(8)` reference still reproduces exactly. No regression. Good to go from my side.

## [c4550477688] aliyevaladddin (CONTRIBUTOR) · review APPROVED · 2026-06-23 (no text)

## [c4778324317] vasqu (COLLABORATOR) · issue_comment · 2026-06-23
run-slow: minicpm3

## [c4778326400] vasqu (COLLABORATOR) · issue_comment · 2026-06-23
Just sanity checking but should be good to go then :saluting_face:

## [c4779112022] vasqu (COLLABORATOR) · issue_comment · 2026-06-23
Nice thanks for sticking through and gz on the merge :hugs:

## [c4779704077] aliyevaladddin (CONTRIBUTOR) · issue_comment · 2026-06-23
Finaly!

---
outcome: state=closed, merged=True
