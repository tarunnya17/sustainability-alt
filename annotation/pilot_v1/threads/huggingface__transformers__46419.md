# #46419 [pull_request] Port ESMC and ESMFold2 to Transformers
https://github.com/huggingface/transformers/pull/46419  ·  opened by Rocketknight1 on 2026-06-04

## [body] Rocketknight1 (MEMBER) · opening post · 2026-06-04
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=46419)

This is just about ready at this point! I've deferred the experimental ESMFold2 models and the SAEs to a follow-up PR so we can launch this as soon as possible. It's got a lot of very verbose Claude comments and docstrings too, but those are forgivable and we can always clean them up later :sweat_smile: 

The main thing for reviewers to know is that there are two model classes here, ESMC (a masked protein LM) and ESMFold2 (a protein folding model that uses ESMC as a backbone model). ESMC is mostly just a 'normal' masked LM, but I wasn't able to find exact modular matches for a lot of methods. Still, it should be easy enough to review. ESMFold2 is much more challenging and is not like any other model in the codebase, even the original ESMFold. I have some tests confirming that we get the same output as BioHub's original code, so I think reviewers should probably just trust that for correctness and focus on the API, although you can try going through all of the diffusion code if you want!

[collapsed block '🤖 mlinter review state', 7 lines omitted]

## [c3461050844] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/esmc.md:32 · 2026-06-23
```suggestion
- [`biohub/ESMC-300M`](https://huggingface.co/biohub/ESMC-300M)
- [`biohub/ESMC-600M`](https://huggingface.co/biohub/ESMC-600M)
- [`biohub/ESMC-6B`](https://huggingface.co/biohub/ESMC-6B)
```
nit

## [c3461053413] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/esmc.md:38 · 2026-06-23
I guess we don't have any auto model for this?

## [c3461065249] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/hub_kernels.py:139 · 2026-06-23
Would it be possible to port to kernels community? Lemme nudge internally if you need help check the kernels channel `#kernels`

## [c3461073455] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/__init__.py:23 · 2026-06-23
claued going crazy? We shouldnt need all these linter ignore stuff

## [c3461076093] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/configuration_esmc.py:18 · 2026-06-23
ok last mention since it might happen more often

## [c3461080600] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/configuration_esmc.py:38 · 2026-06-23
I'd rather we use intermedate size like in llama and other models

## [c3461095888] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/configuration_esmc.py:36 · 2026-06-23
to remove, we use rope parameters https://github.com/huggingface/transformers/blob/892be11b82cf77acf99ec751467f3b5e54e717ca/src/transformers/models/llama/configuration_llama.py#L80

## [c3461098521] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/configuration_esmc.py:36 · reply to [c3461095888] · 2026-06-23
If you need a different theta, you can set default_theta = 20_000 for example

## [c3461101439] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/configuration_esmc.py:68 · 2026-06-23
let's avoid attribute map if possible

## [c3461106824] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:71 · 2026-06-23
we forgot to guard the auto tokenizer then no?

## [c3461122775] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:113 · 2026-06-23
it's a bit weird because the special tokens are also somewhat included here

## [c3461127668] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:67 · 2026-06-23
should be tokenizers backend

## [c3461141306] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:191 · 2026-06-23
This is weird, imo this should directly be backed into tokenizer_config.json so the tokenizer resolves these as special tokens, e.g. see llama where you should have entires like "bos_token": "..." --> resolves into tokenizer.bos_token and tokenizer.bos_token_id

## [c3461144938] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:203 · 2026-06-23
would also check if there were really no native alternatives

## [c3461152191] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:49 · 2026-06-23
we dont really use that anymore

## [c3461153570] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:56 · 2026-06-23
please no let's go with llama!

## [c3461156278] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:85 · 2026-06-23
just do this for config post init if necessary imo

## [c3461165206] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:90 · 2026-06-23
This is LlamaMLP but W1 weights need to be unfused (W1 -> gate, up projections)

We can use the conversion mapping for this

## [c3461171852] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:140 · 2026-06-23
ternaries please, they are more modular friendly

## [c3461178419] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:110 · 2026-06-23
Imo we should follow nomic bert -> it will need slight adjustments to config / init but this is essentially what you do here + the additional q/k norm

## [c3461181753] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:137 · 2026-06-23
should be unfused (I think nomic bert or some also do this, definitely findable in conversion mapping)

## [c3461182893] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:152 · 2026-06-23
```suggestion
```

## [c3461183831] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:153 · 2026-06-23
typing transformers kwargs

## [c3461187111] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:197 · 2026-06-23
forward sould look the same as nomic bert then + the additional q/k norm

## [c3461192169] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:205 · 2026-06-23
Can be inherited from llama (just needs renames)

## [c3461193739] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:228 · 2026-06-23
hidden_states

## [c3461215836] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:257 · 2026-06-23
```suggestion
        residual = hidden_states
        attention_output, _ = self.self_attn(
            self.input_layernorm(x),
            attention_mask,
            position_embeddings=position_embeddings,
[… 10 lines of code omitted]
```
let's follow llama more closely

## [c3461221453] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:260 · 2026-06-23
to remove, we should move the below directly to the model as in llama. this module is not really necessary

## [c3461223206] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:266 · 2026-06-23
```suggestion
        self.layers = nn.ModuleList([ESMCLayer(config) for _ in range(config.n_layers)])
```

## [c3461226197] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:225 · 2026-06-23
```suggestion
    base_model_prefix = "model"
```
can likely be inherited from llama for most flags tbh

## [c3461230305] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:336 · 2026-06-23
should be solvable when we inherit from llama for the layer instead, the issue is that it uses nn.Module instead of GradientcheckpointLayer

## [c3461232164] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:349 · 2026-06-23
wont be needed when rope is refactored

## [c3461234065] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:358 · 2026-06-23
can be inherited from llama / nomic bert probably

## [c3461236057] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:370 · 2026-06-23
only needed because the name is embed not embed_tokens

## [c3461239017] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:333 · 2026-06-23
output recorder are missing (should be identical to llama / nomic bert)

## [c3461242820] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:373 · 2026-06-23
wrong decorators, we probably can use nomic bert 1:1 for this forward

## [c3461247428] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:423 · 2026-06-23
to remove imo, why was this needed?

## [c3461249161] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:426 · 2026-06-23
note that this will require right padding to work correctly

## [c3461252477] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:430 · 2026-06-23
goes for flex attention as well then (for FA check is_flash_attention_requested)

## [c3461256854] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:430 · reply to [c3461252477] · 2026-06-23
still confused why we need this in the first place and cannot describe it via the attention mask

## [c3461260833] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:488 · 2026-06-23
looks fairly standard, no similar module?

## [c3461262079] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:516 · 2026-06-23
bert to copy maybe?

## [c3461266481] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:606 · 2026-06-23
```suggestion
        self.dropout = nn.Dropout(config.classifier_dropout)
```
iiuc only dropout is different

## [c3461279021] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:45 · 2026-06-23
I would drop the msa prefix and just use the normal namings no need for attribute mapping as well

we already have a subconfig so it's clear it's msa only

## [c3461280119] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:31 · 2026-06-23
no autodocs?

## [c3461283420] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:53 · 2026-06-23
```suggestion
    num_coda_layers: int | None = 2
```
nit

## [c3461285451] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:60 · 2026-06-23
same here imo

## [c3461291539] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:81 · 2026-06-23
```suggestion
    sliding_window: int | None = 128
```

## [c3461292621] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:82 · 2026-06-23
intermediate size

## [c3461295756] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:68 · 2026-06-23
rope parameters please instead all of this weird one letter stuff 

Same for above lets use something more explicit for d_

## [c3461298301] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:92 · 2026-06-23
weird why not save it here directly

## [c3461301498] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:107 · 2026-06-23
last mention let's please use something along hidden size

## [c3461304383] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:126 · 2026-06-23
same here

## [c3461308866] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:198 · 2026-06-23
should there be something akin to diffusion gemma with its own generation config?

## [c3461312884] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:196 · 2026-06-23
imo no, we should add a new implementation then if there is really something new to be added

## [c3461323391] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:56 · 2026-06-23
maybe autocast? also used in rope modules to force fp32

## [c3461327347] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:79 · 2026-06-23
this is the llama mlp

## [c3461334958] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:105 · 2026-06-23
clip mlp tbh with sigmoid activation but its a bit awkward so probably better to leave as is but please more descriptive namings

## [c3461344208] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:124 · 2026-06-23
weigh, bias -> should this be a parameter instead? a bit weird because we dont really support persistent buffers

## [c3461346021] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:138 · 2026-06-23
llama mlp

## [c3461347973] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:165 · 2026-06-23
just properly register intemediate size in the config

## [c3461352801] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:174 · 2026-06-23
same here, it really doesnt make much sense to have multiple implementations for the same swiglu mlp op

## [c3461365129] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:251 · 2026-06-23
this just means that the sin/cos were incorrectly prepared - we should have per layer type rope or something where we properly adjust the hidden size

e.g. see gemma3 or laguna

## [c3461366700] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:259 · 2026-06-23
let's use nanochat norm module

## [c3461372249] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:285 · 2026-06-23
why do we need this as its own function?

## [c3461387982] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:387 · 2026-06-23
this is way over the top imo, we should make the attention mask with a custom and function atp imo

## [c3461392905] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:346 · 2026-06-23
varlen should not be done this way, please check qwen 2 vl for a possible version without exploding the calculations like here

## [c3461394948] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:413 · 2026-06-23
own module imo

## [c3461395886] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:417 · 2026-06-23
to remove imo

## [c3461402679] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:453 · 2026-06-23
this should be a proper rope module tbh where we only allow the default implementation (see ernie 4.5 moe vl)

Why do we mark the compile disable?

## [c3461406974] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:542 · 2026-06-23
why not config values and validated in validate architecture?

## [c3461411908] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:618 · 2026-06-23
too much going at once imo

## [c3461416618] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:760 · 2026-06-23
yea it's really hard to read for me because it assumes I know what all the abbreviations mean

## [c3461428740] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:855 · 2026-06-23
to refactor with proper interface usage and we should split the projections to q k v

imo its not worth it to force FA on this model? There is a lot of implicitly incompatible stuff going on like the bias

## [c3461431083] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:887 · 2026-06-23
swiglu mlp

## [c3461435609] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:899 · 2026-06-23
probably not super necessary as standalone module? just 2 set of layers done one of the other

## [c3461439115] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1108 · 2026-06-23
way too much going on / parameters

## [c3461443565] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1421 · 2026-06-23
torch finfo min

## [c3461445397] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1440 · 2026-06-23
why do we have this module when this alone suffices?

## [c3461450572] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1546 · 2026-06-23
looks very similar calls 3 times in a row?

## [c3461452955] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1573 · 2026-06-23
no sequential please

## [c3461455277] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1634 · 2026-06-23
config

## [c3461461601] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1649 · 2026-06-23
these ops dont really seem einsum worthy to me; can we not use normal matmuls with transpose or similar

## [c3461464402] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1714 · 2026-06-23
no private arg

## [c3461467790] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1777 · 2026-06-23
yea similar feeling that it's not really necessary as own module

## [c3461470507] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1969 · 2026-06-23
wasnt there something similar already?

## [c3461471422] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2012 · 2026-06-23
config

## [c3461473364] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1638 · 2026-06-23
way too big forward

## [c3461476960] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2250 · 2026-06-23
I would avoid focusing on this for now; it really doesnt seem to compatible

## [c3461477514] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2285 · 2026-06-23
config

## [c3461481736] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2767 · 2026-06-23
same way too much going on at once in one forward, it's hard

## [c4554870649] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-23
Deep dive, I think esmc is alright. Unsure about esmfold2 it seems very much all over the place where I cannot really read the flow too easily and a lot of stuff is going at once

## [c3475395495] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/configuration_esmc.py:38 · reply to [c3461080600] · 2026-06-25
Yep, switched to it! We use the old `expansion_ratio` to compute a default value if it's not provided, to keep compatibility with the existing checkpoints

## [c3475397471] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/configuration_esmc.py:68 · reply to [c3461101439] · 2026-06-25
The existing checkpoints use the old names! I can get rid of this but we have to regenerate everything

## [c3475504779] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:71 · reply to [c3461106824] · 2026-06-25
Guarded!

## [c3475656860] Rocketknight1 (MEMBER) · review_comment on src/transformers/integrations/hub_kernels.py:139 · reply to [c3461065249] · 2026-06-25
Yeah, this was a placeholder location while I was working on the PR! We should definitely move this before merging

## [c3475658155] Rocketknight1 (MEMBER) · review_comment on docs/source/en/model_doc/esmc.md:38 · reply to [c3461053413] · 2026-06-25
Replaced with AutoModel! (ESMC is fairly normal, it's just a masked LM)

## [c3475660462] Rocketknight1 (MEMBER) · review_comment on docs/source/en/model_doc/esmc.md:32 · reply to [c3461050844] · 2026-06-25
Done!

## [c3475663484] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:67 · reply to [c3461127668] · 2026-06-25
Fixed! Cleaned up the special tokens too.

## [c3475664797] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:203 · reply to [c3461144938] · 2026-06-25
Dropped these

## [c3475665512] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:49 · reply to [c3461152191] · 2026-06-25
Also dropped!

## [c3475666132] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:56 · reply to [c3461153570] · 2026-06-25
Imported!

## [c3475666856] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:85 · reply to [c3461156278] · 2026-06-25
Yeah, moved it to post_init

## [c3475671598] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:153 · reply to [c3461183831] · 2026-06-25
Fixed!

## [c3475707267] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:205 · reply to [c3461192169] · 2026-06-25
I think this doesn't actually work because of the scaling factor! Llama doesn't have that

## [c3475801163] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:113 · reply to [c3461122775] · 2026-06-25
Should be fixed!

## [c3475820478] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:191 · reply to [c3461141306] · 2026-06-25
Not sure here because a lot of this comes from standard biology notation - this means the tokenizer vocab and unique special tokens like the chain break are kinda 'fixed' and very unlikely to change in other models. So keeping them in the code makes sense, even though we shouldn't do that in most text tokenizers?

## [c3475824108] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:90 · reply to [c3461165206] · 2026-06-25
Done!

## [c3475825540] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:140 · reply to [c3461171852] · 2026-06-25
Done!

## [c3475827980] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:110 · reply to [c3461178419] · 2026-06-25
I was able to do this with the Attention layer but not the others because each one had a little bit of weirdness somewhere

## [c3475829251] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:137 · reply to [c3461181753] · 2026-06-25
Unfused it!

## [c3475844342] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:197 · reply to [c3461187111] · 2026-06-25
Yeah, I still need to override the forward for the QK norm but I was able to import most of the init with modular

## [c3475847350] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:228 · reply to [c3461193739] · 2026-06-25
Done!

## [c3475851472] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:260 · reply to [c3461221453] · 2026-06-25
Done!

## [c3475852569] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:266 · reply to [c3461223206] · 2026-06-25
Done!

## [c3475877574] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:336 · reply to [c3461230305] · 2026-06-25
Yep, inheriting from Llama now!

## [c3475878683] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:349 · reply to [c3461232164] · 2026-06-25
Gone!

## [c3475966013] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:358 · reply to [c3461234065] · 2026-06-25
I wasn't able to get this working! Other MLMs have embedding norms that we don't, or they use things like token_type_ids. I couldn't find a single clean match

## [c3476006401] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:370 · reply to [c3461236057] · 2026-06-25
Fixed!

## [c4572167898] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c3476303218] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/__init__.py:23 · reply to [c3461073455] · 2026-06-25
Removed!

## [c4573321070] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c3476303640] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/configuration_esmc.py:18 · reply to [c3461076093] · 2026-06-25
Also removed!

## [c4573321559] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c3476306168] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/configuration_esmc.py:36 · reply to [c3461095888] · 2026-06-25
Done!

## [c4573324515] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c3476428974] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:333 · reply to [c3461239017] · 2026-06-25
Eventually got this working when I realized I could override the last hidden state! In now

## [c4573480239] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c3476435923] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:426 · reply to [c3461249161] · 2026-06-25
It's an MLM, so right-padding is expected, right? It doesn't generate.

## [c4573491465] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c3476460949] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:373 · reply to [c3461242820] · 2026-06-25
All handled with modular now!

## [c4573533875] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c3476501137] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:257 · reply to [c3461215836] · 2026-06-25
Done!

## [c4573580265] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c3476513080] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:488 · reply to [c3461260833] · 2026-06-25
I couldn't get a clean mapping to BERT or ESM! ESMC does a weird thing where they always use GELU here, even though `config.hidden_act` is `silu`. However they use real GELU and not BERT's manual gelu-by-hand.

## [c4573592893] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c3476521668] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:516 · reply to [c3461262079] · 2026-06-25
The gelu thing is blocking here too!

## [c4573601659] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c3476524115] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:606 · reply to [c3461266481] · 2026-06-25
Done!

## [c4573604165] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-25 (no text)

## [c3480014734] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:191 · reply to [c3461141306] · 2026-06-26
I think we went past each other :D I meant that if you make the token as https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/tokenizer_config.json#L2052 then it should auto resolve into `tokenizer.chain_break_token` and `tokenizer.chain_break_token_id` without adding them as manualy property

## [c4577806476] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-26 (no text)

## [c3480031210] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:426 · reply to [c3461249161] · 2026-06-26
Yep should be fine then, was just asking because there were cases with left padding and assuming right padding (as here with arange pos ids without any shifts)

## [c4577826837] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-26 (no text)

## [c3480048238] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/esmc.md:43 · 2026-06-26
Imo if you want to you can use hf options and show each variant instead, wdyt?

## [c3480054350] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/configuration_esmc.py:83 · 2026-06-26
I think this is alreadythe default could be removed I think
```suggestion
```

## [c3480057781] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/configuration_esmc.py:68 · reply to [c3461101439] · 2026-06-26
Yea dw, it would have been nice to avoid but it's also fine this way imo

## [c3480063370] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/configuration_esmc.py:109 · 2026-06-26
the attributes are fairly standard, would it make sense to move this to modular and inherit from e.g. llama config? You can always override / add (`attr = ...`) or delete unnecessary attributes (`attr = AttributeError()`)

## [c3480074198] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:131 · 2026-06-26
a bit weird, iirc wasnt bert using cls / sep, this is kind of a mix here. but if that's already their convention it's nothing to change, just curious

## [c3480077380] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:172 · 2026-06-26
yea like mentioned before can be baked into the json I think

## [c3480084982] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:50 · 2026-06-26
Ok you need to explain this to me why this is needed. I actually don't get this as we don't really change anything?

## [c3480087539] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:50 · reply to [c3480084982] · 2026-06-26
Otherwise I'd rather avoid overriding the private torch module fn and do it our public functions where necessary

## [c3480090766] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:75 · 2026-06-26
```suggestion
        self.q_norm = nn.LayerNorm(config.hidden_size, bias=False) if config.qk_layernorm else nn.Identity()
        self.k_norm = nn.LayerNorm(config.hidden_size, bias=False) if config.qk_layernorm else nn.Identity()
```
nit but should be the more canonical naming

## [c3480093557] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:69 · 2026-06-26
```suggestion
    """Multi-head self-attention with QK-LayerNorm and RoPE."""
```
a bit too bloated imo

## [c3480094533] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:89 · 2026-06-26
```suggestion
```

## [c3480098791] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:101 · 2026-06-26
Do they do nope? Otherwise no need for this iff

## [c3480107515] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:152 · 2026-06-26
```suggestion
class ESMCLayer(LlamaDecoderLayer):
    def __init__(self, config: ESMCConfig):
        super().__init__(config)
        # layernorm instead of rms norm
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
[… 23 lines of code omitted]
```

## [c3480122667] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:170 · 2026-06-26
```suggestion
class ESMCPreTrainedModel(NomicBertPreTrainedModel):
    _no_split_modules = ["ESMCLayer"]
    # ``inv_freq`` / ``original_inv_freq`` are non-persistent rotary buffers; ``_extra_state``
    # keys come from the published checkpoint's fused TransformerEngine layout.
    _keys_to_ignore_on_load_unexpected = [r"\._extra_state$", r"\.inv_freq$", r"\.original_inv_freq$"]
    
    def _init_weights(self, module):
        raise AttributeError("Not overriden, simple super")
```
might be easy to inherit?

## [c3480129404] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:245 · 2026-06-26
```suggestion
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = ESMCRotaryEmbedding(config)
        self.layers = nn.ModuleList([ESMCLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, bias=False)
        self.gradient_checkpointing = False
        self.post_init()
```

## [c3480130909] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:250 · 2026-06-26
```suggestion
    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
```

## [c3480141330] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:226 · 2026-06-26
do we need the `sequence_id > 0 .any()`? This makes torch compile and export less compatible

## [c3480142411] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:307 · 2026-06-26
this is nicely done

## [c3480144588] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:248 · 2026-06-26
would still call it attention mask tbh

## [c3480146259] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:248 · reply to [c3480144588] · 2026-06-26
especially since sdpa will use a bool mask 😬

## [c3480157985] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:488 · reply to [c3461260833] · 2026-06-26
Then let's restructure a bit: It is a CLIP mlp with an additional layer norm in between, can we make it follow the naming a bit closer to that then?

## [c3480174952] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:516 · reply to [c3461262079] · 2026-06-26
should be unrelated no? If you do modular here it only cares about the namings but not the underlying implementation (unless you didn't override then it will try to create its version based on the original parent's)

## [c3480181173] vasqu (COLLABORATOR) · review_comment on tests/models/esmc/test_modeling_esmc.py:204 · 2026-06-26
can you check nomic bert for integration tests --> would be nice on logits

## [c3480184388] vasqu (COLLABORATOR) · review_comment on tests/models/esmc/test_tokenization_esmc.py:23 · 2026-06-26
TokenizerTesterMixin?

## [c4577847844] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-26
Part2 for esmc, I think that portion looks solid now 🫡 only minor details

## [c3500372186] Rocketknight1 (MEMBER) · review_comment on docs/source/en/model_doc/esmc.md:43 · reply to [c3480048238] · 2026-06-30
Done!

## [c4602290354] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500372939] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/configuration_esmc.py:83 · reply to [c3480054350] · 2026-06-30
Done!

## [c4602291228] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500376758] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/configuration_esmc.py:109 · reply to [c3480063370] · 2026-06-30
Yeah, inherited from LlamaConfig in `modular` and overrode to restore the ESMC default values

## [c4602295674] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500389581] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:172 · reply to [c3480077380] · 2026-06-30
Still not sure about this one! This is very unlikely to vary between models, so I think we can keep the chain break token as a constant, right?

## [c4602310769] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500394390] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:50 · reply to [c3480084982] · 2026-06-30
Dropped this one - it was necessary in the case where `inv_freq` was being cast to `bfloat16`, but this was me being stupid; once I added it to `keep_in_fp32_strict` then that wasn't a problem anymore. Deleted the `_apply` override.

## [c4602317174] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500395198] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:75 · reply to [c3480090766] · 2026-06-30
Done!

## [c4602318221] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500396149] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:69 · reply to [c3480093557] · 2026-06-30
did a little Claude Cleanup here

## [c4602319406] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500396777] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:89 · reply to [c3480094533] · 2026-06-30
Same!

## [c4602320517] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500398575] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:101 · reply to [c3480098791] · 2026-06-30
Dropped!

## [c4602322760] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500419696] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:170 · reply to [c3480122667] · 2026-06-30
Yup, reparented this one!

## [c4602353278] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500420169] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:152 · reply to [c3480107515] · 2026-06-30
Reparented this one!

## [c4602353852] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500423212] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:250 · reply to [c3480130909] · 2026-06-30
Added!

## [c4602358082] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500429078] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:226 · reply to [c3480141330] · 2026-06-30
Yeah, dropped this one. It wasn't working properly regardless. It was there to handle a thing that ESMC does, where you can pass multiple protein chains in the same input sequence, sort of like sequences that are already internally packed. I just disabled flashattn + multi-chain for now, and pass all of that through eager/SDPA.

## [c4602365677] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500431477] Rocketknight1 (MEMBER) · review_comment on tests/models/esmc/test_tokenization_esmc.py:23 · reply to [c3480184388] · 2026-06-30
Done!

## [c4602369445] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500478708] Rocketknight1 (MEMBER) · review_comment on tests/models/esmc/test_modeling_esmc.py:204 · reply to [c3480181173] · 2026-06-30
Integration tests in!

## [c4602428496] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500479743] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:248 · reply to [c3480144588] · 2026-06-30
Done!

## [c4602429695] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500523621] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:131 · reply to [c3480074198] · 2026-06-30
Yeah, this is the convention in the model so I left it alone!

## [c4602487063] Rocketknight1 (MEMBER) · review COMMENTED · 2026-06-30 (no text)

## [c3500756750] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:226 · reply to [c3480141330] · 2026-06-30
I mean you could also do this via position ids (and the fa kwargs instead), no need for an additional kwarg for this. 

Check out the data collator for flattening. The idea is to define packed sequences via the position ids and ideally the cu_seq, max_seq kwargs

## [c4602774060] vasqu (COLLABORATOR) · review COMMENTED · 2026-06-30 (no text)

## [c4858069293] Rocketknight1 (MEMBER) · issue_comment · reply to [c4602774060] · 2026-07-01
@vasqu writing this in one big comment rather than breaking it up to every individual location, because there was a lot of repeated stuff! I just did a big pass for ESMFold2, now that ESMC is mostly okay.

I dropped the flashattention path for now, inlined a bunch of methods, broke RoPE out into a single class, tried to merge as many SwiGLUs as I could (not all were possible, some have little embedded norms). A lot of stuff got renamed and moved to configs instead of just being constants, and I tried to pull out private methods from the big long forward methods.

There's a few tricky bits still:

1) DiffusionModule still has massive signatures with 20 args. Want me to bundle that into one big object we just pass around and access attributes on?
2) Renamed a lot of things and broke up a lot of `nn.Sequential()` and single-letter vars. This required lots of weight renaming and `attribute_map`. We'll need to regenerate the checkpoint anyway (the existing checkpoints don't contain the ESMC backbone weights), so we could rename everything in the checkpoints and then delete the rename lines before merging. Marked those with TODO.
3) Some bits I left untouched. In particular, some single-letter names were kept because they're kind of standard, like splitting a quaternion into `r, i, j, k`, and some `q` and `v` keys. I can rename those if you want! Other stuff like `plddt` comes from AlphaFold and [people doing protein folding know what it means](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/plddt-understanding-local-confidence/) even if it's quite weird for most ML engineers!
4) Some other bits I left include an `F.rms_norm()` call with no affine scaling and so no weights. You suggested nanochat there but I'm not sure why we need a module at all! It works fine as one functional line.

Overall I tried to make it more readable, but it's a very complex model, so there's only so much I can do! :sweat_smile:

## [c4859132709] vasqu (COLLABORATOR) · issue_comment · reply to [c4858069293] · 2026-07-01
@Rocketknight1 Thanks a lot! Yes the model is super complex so I don't expect perfect stuff. I think the esmc model should be fairly good now. I will try to take a look at esmfold2 in the coming days.

Re 
1. Yea that might be more reasonable, having too long signatures is really hard to read imo - similar to how we pass configs around and only access the portions we need. Just a suggestion and I'm not 100% sure it's a good one, would not put it high prio
2. Gotcha, sounds good. Yea we have a few models that have a long list of renames as well dw.
3. Yea, hard to catch the abbreviations sometimes. Sometimes it's me quickly glancing, sometimes my missing background for biology etc. It's not too bad if there really is meaning; I just would avoid too many where possible
4. This is mainly for inheritance with modular + potentially applying kernels on them. Won't be possible if we rely on an F. portion of torch

## [c4866130157] Rocketknight1 (MEMBER) · issue_comment · 2026-07-02
On 4 I'm unsure - we're not doing modular for ESMFold2 because the code is so weird compared to other models. I'd leave that one for a future refactor, if there are kernels that turn out to really justify it, or if we get a lot of similar folding models in the codebase.

1 is done, though, and I managed to drop some dead args in the process!

## [c3519992669] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:124 · reply to [c3461344208] · 2026-07-03
This is a tricky one because they do need to be saved but they shouldn't be trainable. We could make them `parameter(requires_grad=False)` maybe?

## [c4625928072] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-03 (no text)

## [c3540990648] Cyrilvallez (MEMBER) · review_comment on docs/source/en/model_doc/esmfold2.md:39 · 2026-07-08
No need for `eval()`, and `device_map` is much preferred and faster than `cuda()`
```suggestion
model = ESMFold2Model.from_pretrained("biohub/ESMFold2", dtype=torch.bfloat16, device_map=0)
```

## [c3540992910] Cyrilvallez (MEMBER) · review_comment on docs/source/en/model_doc/esmfold2.md:43 · 2026-07-08
It's an example, quite weird to save to file here
```suggestion
print(pdb_string)
```

## [c3540994354] Cyrilvallez (MEMBER) · review_comment on docs/source/en/model_doc/esmfold2.md:66 · 2026-07-08
```suggestion
    "biohub/ESMFold2", dtype=torch.bfloat16, device_map="cuda", use_kernels=True
)
```

## [c3540998540] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/__init__.py:22 · 2026-07-08
Let's not skip rules that do not need to be skipped
```suggestion
from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure

if TYPE_CHECKING:
    from .configuration_esmfold2 import *
    from .modeling_esmfold2 import *
```

## [c3541215080] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:34 · 2026-07-08
Dangerous TODO haha, let's try not to forget if we control the format in the end

## [c3541221191] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:81 · reply to [c3461291539] · 2026-07-08
Oh yeah it's a big one, it's absolutely need to exist, i.e. be here or in the `attribute_map`, otherwise mask and cache won't be correct!!

## [c3541223508] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:59 · 2026-07-08
Only 2 fields, would probably be nicer inside another one of the configs, the most relevant one

## [c3541225377] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:115 · 2026-07-08
Only 1 field, and the map is useless here -> let's merge into another config

## [c3541227204] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:135 · 2026-07-08
Instead of having a subconfig just for the attention, let's merge both configs!

## [c3541232287] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:358 · 2026-07-08
As noted above, in most cases it does not really make sense to have subconfigs here, they can live in the main config

## [c3541237057] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:59 · 2026-07-08
Very expensive, let's simply have the weight to `_keep_in_fp_32_strict` so that the weight itself stays in fp32, instead of upcasting EVERY forward, which is quite expensive

## [c3541237865] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:60 · 2026-07-08
Same for bias

## [c3541240321] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:64 · 2026-07-08
Let's call it a MLP no?

## [c3541259165] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:96 · 2026-07-08
Why not reuse the Norm class we just defined above here, instead of redoing the same?

Otherwise, same comment, if `s_scale` needs to be upcasted EVERY forward, then we just want to always keep it in fp32 -> add to `_keep_in_fp32_strict`

## [c3541268953] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:124 · reply to [c3461344208] · 2026-07-08
Nop, having them as persistent buffers is totally fine and the preferred way in such a case! 🤗

## [c3541278309] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:119 · 2026-07-08
Should not need to mess up with device here, it's not such a good practice in general as it may collide with accelerate hooks
```suggestion
        # embedding is full precision.
        t = t_hat.to(dtype=self.w.dtype).reshape(-1)
```

## [c3541284263] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:138 · reply to [c3461346021] · 2026-07-08
More like Phi3 iirc, it also has the merged weight and the split, could prob be inherited

## [c3541288443] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:230 · 2026-07-08
The fact that we have a few Copied from, and the MLPs look like existing MLPs make me believe that we could still benefit from modular a bit for the modeling - I'm really not a fan of introducing new Copied from, it's more something that we want to drop completely when the time is right

## [c3541291959] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:250 · 2026-07-08
This is part of the config, not modeling!

## [c3541303804] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:266 · 2026-07-08
I don't get it, do we have padding in the MIDDLE of the sequence? Otherwise, we already have primitive to create sliding masks

## [c3541310913] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:337 · 2026-07-08
```suggestion
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
```

## [c3541316274] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:333 · 2026-07-08
What is `attention_params`? Will the mask be different on every layer due to it? Otherwise we REALLY want to create the mask once and for all inside the model, not on every layer

## [c3541319269] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:413 · reply to [c3461394948] · 2026-07-08
Or no module at all since it's single line, and directly where it's called as a one-liner

## [c3541321401] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:358 · 2026-07-08
Can we call it a Layer instead, as we usually do?

## [c3541324635] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:435 · 2026-07-08
Why do we force bf16 here? It will potentially upcast all later operations using it

## [c3541327740] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:457 · 2026-07-08
This module looks like the bad architecture we have from the past, and it's probably useless -> let's have this directly inside the 2 modules that uses it, otherwise it's kinda weird and redundant
Also, you declare a rotary_emb that's never used

## [c3541341350] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:542 · 2026-07-08
Big sign that `atom_transformer` should not exist as per my comment above, it's super weird to do this given that the rotary is not used inside the atom_transformer -> let's move it all here directly

## [c3541345803] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:579 · 2026-07-08
layer_cache is a very confusing name IMO, as it relates strongly to kv cache but is completely different - let's call it something else

## [c3541347120] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:632 · 2026-07-08
Why have this? Completely useless, it's the exact same function

## [c3541352212] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:552 · 2026-07-08
Not sure if it really helps to have a helper just for this, but alright if it's called very often

## [c3541359341] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:758 · 2026-07-08
Should probably use proper mask and attention interfaces here

## [c3541363130] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:779 · 2026-07-08
comment is unreadable imo
```suggestion
        self.adaln = ESMFold2AdaptiveLayerNorm(d_model, d_cond)
```

## [c3541376558] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:930 · 2026-07-08
same as before, device should not be needed
```suggestion
        # Noise embedding
        t = t_hat.to(dtype=torch.float32).reshape(-1)
```

## [c3541381132] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:970 · 2026-07-08
nit: new lines and comments are really useless here and make it ugly

```suggestion

        self.conditioning = ESMFold2DiffusionConditioning(config)
        self.atom_encoder = ESMFold2AtomEncoder(config, structure_prediction=True)
        self.atom_decoder = ESMFold2AtomDecoder(config)
        self.s_to_token = nn.Linear(c_token, c_token, bias=False)
        self.token_transformer = ESMFold2DiffusionTransformer(config)
        self.s_step_norm = ESMFold2LayerNorm(c_token)
        self.token_norm = ESMFold2LayerNorm(c_token)
```

## [c3541383987] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1071 · 2026-07-08
To remove

## [c3541388314] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1112 · 2026-07-08
damned

## [c3541400792] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1152 · 2026-07-08
Could we have more descriptive names here? 😭 It's really unreadable, and `denominator` instead of `denom` or `determinant` instead of `det` is really not a waste

## [c3541401443] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1157 · 2026-07-08
```suggestion
```

## [c3541406521] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1158 · 2026-07-08
Are we sure about this?

## [c3541414693] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1267 · 2026-07-08
Would be better to prepare upstream, to reuse it in every layer

## [c3541417508] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1270 · 2026-07-08
Can we expand the `einsum` easily here instead?

## [c3541418825] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1302 · 2026-07-08
Useless, let's remove this module

## [c3541428753] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1427 · 2026-07-08
We are messing way too much with dtype everywhere, I'm pretty sure we end up upcasting/downcasting a lot of unwanted ops... In general we do not allow loading several parts in different dtypes, so the comment is not true and we could remove

## [c3541435148] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1473 · 2026-07-08
Will we ever do that?

## [c3541440423] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1462 · 2026-07-08
Why do we have all those complicated attributes and global class attrs, just for a few einsum notations? Then when we check the forward, we have no idea which op is being performed anymore... Let's remove!
And let's even remove the einsums in general in favor of simpler ops if we can haha

## [c3541442753] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1471 · 2026-07-08
Let's be explicit rather than using an hardcoded global anyway
```suggestion
        # Default chunked for memory on long sequences (set_chunk_size(None) disables).
        self._chunk_size: int | None = 64

```

## [c3541452631] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1525 · 2026-07-08
This module is useless as well, let's remove

## [c3541454791] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1539 · 2026-07-08
Same as before for all this

## [c3541456258] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1564 · 2026-07-08
probably never really used either?

## [c3541459043] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1594 · 2026-07-08
Same as before for the Transformer -> we do not want to have a module for this, layers should be directly in the model using them without additional wrapper

## [c3541460525] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1653 · 2026-07-08
Would be nice to not have einsum

## [c3541464578] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1790 · 2026-07-08
Let's not have global such as this, either hardcoded in modules or config (we do not want to allow changing global vars anyway, would be very bad practice)

## [c3541467010] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1933 · 2026-07-08
Flagging some more einsums

## [c3541468926] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1531 · 2026-07-08
If we detach everything, might as well never compute grads on this function

## [c3541472502] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2100 · 2026-07-08
Comment is unreadable imo

## [c3541475811] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2236 · 2026-07-08
Let's keep this directly inside the module that wraps it

## [c3541476480] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2236 · 2026-07-08
If we detach, let's not compute grad

## [c3541478235] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2341 · 2026-07-08
Once again, let's be very careful about all the dtype shenanigans

## [c3541481523] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2533 · 2026-07-08
Hahaha, are you trying to compete with Trainer? 😆😆

## [c3541489006] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2758 · 2026-07-08
Therse 2 should appear before the Model, as they are used inside!

## [c4650624032] Cyrilvallez (MEMBER) · review COMMENTED · 2026-07-08
ALright, verry sorry for the huge amount of comment haha, it's a biggy

## [c4937396501] Rocketknight1 (MEMBER) · issue_comment · 2026-07-10
@cyrilvallez review mostly addressed by now! A lot of the comments became outdated as a result, so I'll just address it here in one big block rather than individual comments that might have disappeared:

- config flattened, no more subconfigs
- Weight renames removed, we do that all in the conversion script now. I rebuilt `Rocketknight1/ESMFold2-merged-temp` with the new conversion script and we can make it the official checkpoint after merging this PR
- Inlined loads of wrappers or modules that were only used once
- Sliding-window mask is built once per forward instead of per layer
- Routed things through the attention interface where possible
- Less device juggling, tensors are built in the right place with fewer `.to()` calls
- Dropped lots of globals like `_DEFAULT_CHUNK_SIZE` and `_CONFIDENCE_EPS`, mostly by inlining values
- Lots and lots of renames, especially spelling out abbreviations like `det` and `denom`
- No more `copied from`
- Rearranging done, doc linter stuff fixed
- Using `no_grad()` to avoid grad computation instead of computing with grad and `detach()`
- Also merged a patch from the BioHub team to the ipTM calculations, this was a bug in their fork too
 - Fixed the casting of the backbone (the casting was right but the comment was very stale and wrong)
 
There's some stuff I didn't change, though:
- Kept the custom `_swa_window_mask_function`, since their behaviour differs a bit from ours depending on padding (they count the number of unpadded tokens, our functions just use the index)
- I mostly kept the einsums, because the non-einsum versions used a lot more memory. Einsums for complex contractions really are just better, unless they map to something torch's compiler recognizes and can optimize (like self-attention). I can edit these for readability but the matmul/bmm versions are imo less readable, as well as using more FLOPs/memory

And the biggest thing I'm wary of changing, the dtype/casting behaviour: The way they ran the original model was to have a frozen ESMC backbone (they used a separate ESMC checkpoint with fp32 norm weights) cast to `bf16` with `to()`. This casts **all** weights to bf16, including norms. Then they ran it in an `autocast` context, which will *upcast* for ops like norms. A bf16 norm layer in an autocast context will upcast both weights and computations to fp32! Therefore, the only way to actually match their output is to downcast weights to `bf16` and then upcast them at runtime. The only alternative would be to move this to the conversion script, downconvert weights to bf16 then upconvert to fp32, then save as fp32. Or we could just add `autocast()` back in where they used it in the fork and let it handle things so I stop doing this manually, which gets rid of a lot of the manual weight casting. WDYT?

## [c4981983505] Rocketknight1 (MEMBER) · issue_comment · 2026-07-15
Some more fixes, caught some missing docstrings, and realized I hadn't committed the ESMFold2 converter script, just the ESMC one. So that's in now!

## [c3603575258] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/auto/modeling_auto.py:155 · 2026-07-17
```suggestion
        ("esmc", "EsmcModel"),
```

## [c3603576452] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/auto/modeling_auto.py:155 · reply to [c3603575258] · 2026-07-17
let's push for camel casing as much as possible

## [c3603590821] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:136 · 2026-07-17
let's do camel casing please

## [c3603593887] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:150 · 2026-07-17
the qk layernorm if else: are there any released models that don't use qk layernorm?

## [c3603596108] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:215 · 2026-07-17
cool

## [c3603614924] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:1 · 2026-07-17
nice, overall camel case. modeling_granite.py has residual scaling, could be worth re-using

## [c3603661679] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:130 · 2026-07-17
Nope. you are not supposed to add anything here.
see https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama.py#L135-L135. 
The TokenizersBackend already adds all special tokens.

## [c3603679309] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:146 · 2026-07-17
this is not normal you should not have to do this. you need to use `extra_special_tokens` which will use https://github.com/huggingface/transformers/blob/fix-inkling-feature-extractor/src/transformers/tokenization_utils_base.py#L1268-L1320

## [c3603680804] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:173 · 2026-07-17
```suggestion
```

## [c3603696399] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:183 · 2026-07-17
```suggestion
```

## [c3603704248] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:196 · reply to [c3461312884] · 2026-07-17
strong strong rejection of deffered follow up. to remove

## [c3603708679] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:148 · 2026-07-17
is this used? if not to remove entirely

## [c3603711065] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:162 · 2026-07-17
same here. explicit for the reviewer / reader if any released model leverage this if bother true and false version exist

## [c3603717145] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:51 · 2026-07-17
strong push back here! keep in fp32 needs to be used.

## [c3603722514] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:124 · reply to [c3461344208] · 2026-07-17
just give them weight and bias name please ::)

## [c3603723617] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:112 · 2026-07-17
what is c?

## [c3603729095] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:138 · reply to [c3461346021] · 2026-07-17
looks very very classic. should be gate_up_proj, down_proj. @Rocketknight1 you'll find them in all MoEs.

## [c3603730901] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:230 · reply to [c3541288443] · 2026-07-17
yep. +1 here :)

## [c3603733064] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:194 · 2026-07-17
this looks standard as well

## [c3603734521] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:250 · reply to [c3541291959] · 2026-07-17
absolutetly.

## [c3603736077] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:266 · reply to [c3541303804] · 2026-07-17
+ 1 here please answer and adress

## [c3603740238] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:317 · 2026-07-17
i think we need to use standard attention

## [c3603741603] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:332 · 2026-07-17
appart from sigmoid here this is all very very standard

## [c3603742899] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:303 · 2026-07-17
why is this different from any llama?

## [c3603746305] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:353 · 2026-07-17
why attention params? its not something common in transformers. use normal norations

## [c3603752544] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:463 · 2026-07-17
define all in advance in the config please

## [c3603759597] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:501 · 2026-07-17
this is a bit hectic

## [c3603766086] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:555 · 2026-07-17
cache needs at least a class

## [c3603768468] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:578 · 2026-07-17
too many codepath, are they used, are they all needed?

## [c3603770483] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:619 · 2026-07-17
its only called once no?

## [c3603772817] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:659 · 2026-07-17
again, is this ever not used?

## [c3603785576] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:771 · 2026-07-17
it does not make sense to pass config and stuff from config that are not layer dependant

## [c3603793637] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:834 · 2026-07-17
why define c_z in the first place haha

## [c3603797365] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:891 · 2026-07-17
why are there so many cases

## [c3603801061] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:936 · 2026-07-17
looks like this is the second time we expand / interleave

## [c3604004459] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1700 · 2026-07-17
nn module it?

## [c3604005286] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1741 · 2026-07-17
to remove

## [c3604010026] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2152 · 2026-07-17
again let's get as lean as possible model is already huge

## [c3604010807] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2180 · 2026-07-17
same

## [c3604012417] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2203 · 2026-07-17
tokenizer does that in general

## [c3604020419] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2533 · reply to [c3541481523] · 2026-07-17
tldr generation config ?

## [c4723172886] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-07-17
wow its already a lot better!

## [c3604116963] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:150 · reply to [c3603593887] · 2026-07-17
Pretty sure it's only decoder models and we would carry a cache in that case (forward) + its layernorm instead of rms norm

So no real benefit to inherit from elsewhere

## [c4723790934] vasqu (COLLABORATOR) · review COMMENTED · 2026-07-17 (no text)

## [c3623400423] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/auto/modeling_auto.py:155 · reply to [c3603575258] · 2026-07-21
Done!

## [c4746190070] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623401130] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:136 · reply to [c3603590821] · 2026-07-21
Done!

## [c4746190842] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623542233] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:150 · reply to [c3603593887] · 2026-07-21
Yeah, dropped the flag and branch since all the checkpoints use qk_norm

## [c4746361194] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623550162] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:1 · reply to [c3603614924] · 2026-07-21
I took a look but they're different enough that it's hard to reuse! Granite multiplies by a scaling factor and ESM divides, so I'd have to change the value of all the attributes to `1/original_value` (and also this would introduce small float errors)

## [c4746370258] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623552876] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:130 · reply to [c3603661679] · 2026-07-21
Removed!

## [c4746373319] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623554448] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:146 · reply to [c3603679309] · 2026-07-21
Using `extra_special_tokens` now!

## [c4746375292] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623555547] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:173 · reply to [c3603680804] · 2026-07-21
Done!

## [c4746376588] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623556294] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:183 · reply to [c3603696399] · 2026-07-21
Done!

## [c4746377448] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623593938] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:148 · reply to [c3603708679] · 2026-07-21
Dropped the experimental branch stuff and the msa_encoder branch!

## [c4746421273] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623597109] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:162 · reply to [c3603711065] · 2026-07-21
Yeah, we always build this now

## [c4746425040] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623604524] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:112 · reply to [c3603723617] · 2026-07-21
They're kinda-but-not-really a weight and bias! It's part of the fourier transform, so I just called these 'frequencies', 'phases' and 'embedding_dim'.

## [c4746434658] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623613692] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:51 · reply to [c3603717145] · 2026-07-21
We don't want to keep the weights in fp32 here! Because of the awkward fp16-fp32 dance, we need to downconvert them but compute in fp32 to match the original code. I talked about this in Slack a bit, but just putting weights in the keep in fp32 list doesn't give us what we want here.

## [c4746446273] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623626993] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:194 · reply to [c3603733064] · 2026-07-21
Dropped `copied_from` since we're trying to deprecate it, and did the renames to make the modules more standard but the amount of inheritance in the whole file is still very small so I didn't make a `modular`!

## [c4746463356] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623711480] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:353 · reply to [c3603746305] · 2026-07-21
Done!

## [c4746565147] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623720485] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:303 · reply to [c3603742899] · 2026-07-21
There's a bunch of conditioning that doesn't really match to an existing model, like 3D rope over atom coords, the pair attention bias and so on. I tried to match to the standard attention interface but there's a lot we can't inherit precisely!

## [c4746575524] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623722151] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:463 · reply to [c3603752544] · 2026-07-21
Done!

## [c4746577584] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623726014] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:501 · reply to [c3603759597] · 2026-07-21
Tried to simplify this a bit, cut a little dead code and moved some bits out into methods

## [c4746582317] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623729679] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:555 · reply to [c3603766086] · 2026-07-21
Moved to a cache class. Note that this is definitely **not** a normal generation Cache though, so we don't want to inherit from that. It's much weirder

## [c4746586749] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623736221] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:578 · reply to [c3603768468] · 2026-07-21
Removed the `pred_coords` branch, the `d_pair > 0` branch and the `msa_encoder_enabled` branch!

## [c4746593842] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623740264] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:619 · reply to [c3603770483] · 2026-07-21
Yeah, I took this out to make the method a little more readable but we can inline if you really want to!

## [c4746598550] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623741068] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:659 · reply to [c3603772817] · 2026-07-21
Removed!

## [c4746599570] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623744803] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:771 · reply to [c3603785576] · 2026-07-21
Fixed! We just pass config now

## [c4746603941] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623747631] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:834 · reply to [c3603793637] · 2026-07-21
Yeah, cleaned all of this up! `c_z` is just a local variable now, no `self.c_z`

## [c4746607514] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623752822] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:891 · reply to [c3603797365] · 2026-07-21
A lot of these were dead or from experimental branches, I cleaned them out!

## [c4746613120] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623757334] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:936 · reply to [c3603801061] · 2026-07-21
Yep, correct! Cut this down to one expansion and reused it

## [c4746617976] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623759144] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1700 · reply to [c3604004459] · 2026-07-21
Done! It's `EsmFold2ConfidenceInputEmbedder`

## [c4746620091] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623760947] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1741 · reply to [c3604005286] · 2026-07-21
Done! Inlined two of these

## [c4746622033] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623764311] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2152 · reply to [c3604010026] · 2026-07-21
Done, we assume `config.lm_encoder_enabled` is always true

## [c4746625534] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c3623774061] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2203 · reply to [c3604012417] · 2026-07-21
This one is tricky, we can't actually use the tokenizer for BOS/EOS wrapping here because we're operating on the output of ESMC, not raw input. I think what we're doing here is probably the best option!

## [c4746636160] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-21 (no text)

## [c5036635139] Rocketknight1 (MEMBER) · issue_comment · 2026-07-21
cc @arthurzucker everything should be addressed by now! I did a lot of cleanup, but there's a couple I didn't do - I left the fp32 dance alone, and the bit at the end can't be a generation config because this isn't really like text generation at all, but I tried to move stuff into the config where possible to clean up the giant argument list

## [c3633595654] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/esmfold2.md:39 · 2026-07-22
```suggestion
model = EsmFold2Model.from_pretrained("biohub/ESMFold2", device_map="auto")
```
dtype is auto per default so might not be needed? Also would use device map auto except there is a good reason not to

## [c3633629656] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/esmfold2.md:41 · 2026-07-22
that is a bit weird, what's the normal forward?

## [c3633634332] vasqu (COLLABORATOR) · review_comment on docs/source/en/model_doc/esmfold2.md:73 · 2026-07-22
🔥

## [c3633637144] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/hub_kernels.py:139 · reply to [c3461065249] · 2026-07-22
Rebump before we forget, let me know if you need help with kernels community repo

## [c3633658726] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:109 · 2026-07-22
this is a bit weird no? Can it not be handled with extra /additional special tokens or similar?

## [c3633681283] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:94 · 2026-07-22
Ok I probably asked that already but do we really need these attribute mappings?

## [c3633684121] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:94 · reply to [c3633681283] · 2026-07-22
They dont really follow transformers namings (or are so old they no longer follow modern conventions)

## [c3633688651] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:110 · 2026-07-22
```suggestion
```
nit: if values overlap with the parent you can remove them in modular and just let them be inherited

## [c3633696867] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:132 · 2026-07-22
Then I'd rather set num kv heads directly (not None) above and use validate architecture override to confirm this

## [c3633704924] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:238 · 2026-07-22
```suggestion
    _keys_to_ignore_on_load_unexpected = ["extra_state", "inv_freq", r"original_inv_freq"]
```
nit

## [c3633708102] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:238 · 2026-07-22
For the init we could at least inherit from somewhere maybe?

## [c3633719044] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:307 · 2026-07-22
Maybe we want to extend bidirectional mask with `block_sequence_ids` kwarg as well? I didnt see the usage initially but it makes sense here

## [c3633730551] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:360 · 2026-07-22
Pretty sure its already abstracted within pretrained model
```suggestion
```

## [c3633740215] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:399 · 2026-07-22
self.loss_function maybe?

## [c3633743036] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:430 · 2026-07-22
Let's drop these claude comments, same for above

## [c3633755636] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/modular_esmc.py:425 · 2026-07-22
```suggestion
        super().__init__(self, config)
        self.esmc = EsmcModel(config)
        self.classifier = EsmcClassificationHead(config)
```

## [c3633769746] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:172 · 2026-07-22
Feels bloated: Imo it would make sense to split this into multiple sub configs, no?

This feels too flattened into one config but we should structure it a bit, then we also dont really need these prefixes like "structure_"

## [c3633773177] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:282 · 2026-07-22
Yea you even show great structure here and have an implicit substructure so would imo feel like the natural move

## [c3633782505] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/convert_esmfold2_checkpoint.py:36 · 2026-07-22
some are identical, could we not do get with default to the same key?

## [c3633792515] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/protein_utils.py:413 · 2026-07-22
nit: n -> number_of_

## [c3633800759] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/protein_utils.py:1 · 2026-07-22
That's a lot of constants. Not on you but I'm wondering why they are not part of the config and or weights in a way?

## [c3633811858] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:44 · 2026-07-22
autodoc and docstring for the attributes are missing, feels like we should inherit from `ModelOutput` instead to define our own type here

## [c3633817943] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:44 · reply to [c3633811858] · 2026-07-22
Or is this inputs? It still feels a bit off

## [c3633820716] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:61 · 2026-07-22
This is weird, yea I know we wanted to reduce the kwargs but it feels off

## [c3633831473] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:91 · 2026-07-22
This cries for own generate process now? We could hugely simplify of course but splitting the modeling and the generation steps could probably make this much more readable

## [c3633839080] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:104 · 2026-07-22
Have to agree with Cyril, we can simply add it to `_keep_in_fp_32_strict` and cast the input / output instead of this override

## [c3633848178] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:116 · 2026-07-22
This is essentially LlamaMLP, can we try to fit the same naming? gate, up, down?

## [c3633850665] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:132 · 2026-07-22
Why not config values directly?

## [c3633852306] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:96 · reply to [c3541259165] · 2026-07-22
Rebump

## [c3633856109] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:161 · 2026-07-22
```suggestion
```

## [c3633859007] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:116 · reply to [c3633848178] · 2026-07-22
EsmFold2SwiGLU even exists, just a matter of fused vs unfused weights

## [c3633876889] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:230 · reply to [c3541288443] · 2026-07-22
+1 on the modular part. there are a lot of existin parts at least. A bit torn re copied from because sometimes it makes it hard to keep track of relations: I don't want them for whole classes but on the attention eager part it can be useful. Either way, we should modularize or deprecate the old style at some point 😬 

We should also separate the modeling from the generation ideally. Seeing a lot of hints for generation

## [c3633885554] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:194 · reply to [c3603733064] · 2026-07-22
Can we not make the function name and parameters identical? It's still "customized"

## [c3633887928] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:259 · reply to [c3461366700] · 2026-07-22
Couldve been modular as well for example 👀 ?

## [c3633901002] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:285 · 2026-07-22
If we had subconfigs and same namings we can just pass it easily based on the subconfig instead and access the same names instead of this

So `passed_config = config.diffusion_config if structure_prediction else config.atom_config; ... attr = config.hidden_size`

## [c3633914624] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:293 · 2026-07-22
Are there really any fully padded rows and in those cases instead of modifying the mask, wouldn't it make more sense to use torch.nan_to_num after the attn to ensure this? This makes the mask creation pretty expensive

## [c3633926239] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:361 · 2026-07-22
Then the mask wasnt properly created tbh, we should use the create mask functions exactly because of this to avoid manually taking care of the padding

## [c3633932980] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:303 · reply to [c3603742899] · 2026-07-22
This is all fairly standard attention - it's SWA with qk norm and RoPE. It doesn't really matter that it is 3d rope

Also the rope could be combined into one. Let me try a quick rewrite to show what I mean

## [c3633951804] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:377 · 2026-07-22
```suggestion
    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
[… 28 lines of code omitted]
```

## [c3633955350] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:390 · 2026-07-22
like mentioned the subconfig would make this way easier

## [c3633957135] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:398 · 2026-07-22
Why is the silu not part of the ada linear?

## [c3633960950] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:409 · 2026-07-22
hidden states instead of x please

## [c3633963805] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:407 · 2026-07-22
Could we move the F.rms_norm to weightless module like nanochat rms norm?

## [c3633969904] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:425 · 2026-07-22
We only use the hidden size and the number of heads, why not directly have a head dim attribute?

## [c3633972913] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:435 · reply to [c3541324635] · 2026-07-22
Yea this is a weird cast tbh 👀

## [c3633980874] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:486 · 2026-07-22
```suggestion
        batch_size, _, hidden_dim = atom_features.shape
        
        idx = atom_to_token_idx
        if atom_mask is not None:
            idx = idx.masked_fill(~atom_mask, n_tokens)
[… 16 lines of code omitted]
```

## [c3633986611] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:519 · 2026-07-22
could be also subtely added to the subconfig output_dim?

## [c3633990672] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:558 · 2026-07-22
Yea no wonder we manually do padding when we dont add the mask here??

## [c3633993848] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:549 · 2026-07-22
Should be in the forward as option to interleave there maybe?

## [c3633999052] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:561 · 2026-07-22
Yea no this is not worth saving the few kwargs - position_embeddings, attention_mask

## [c3634009570] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:619 · reply to [c3603770483] · 2026-07-22
lets inline if its only once

## [c3634017005] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:855 · reply to [c3461428740] · 2026-07-22
Rebum: this is still very much representable in our normal attention ways

You can also now pass position bias to the interface so no need to do anything with your mask if you properly prepared with create attn mask

## [c3634020072] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:762 · 2026-07-22
hidden states

## [c3634031254] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:865 · 2026-07-22
Yea I feel like if we have a generate that handles the cache would be nicer so the forwards dont have to care about the if / else

## [c3634034316] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:896 · 2026-07-22
why is it not a float in its config already?

## [c3634038245] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:980 · 2026-07-22
That feels very generaty

## [c3634044617] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1158 · reply to [c3541406521] · 2026-07-22
It just tells me that we likely need some separation between forward and generate, even if its a custom generate tbh

## [c3634047558] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1267 · reply to [c3541414693] · 2026-07-22
Rebump

## [c3634056175] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1282 · 2026-07-22
ClipMLP?

## [c3634065388] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1353 · 2026-07-22
I'd rather we not hide the einsum equation then. It's already harder to read einsums if not used to it but having to figure out what it picked is a nono from me

## [c3634070351] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1384 · 2026-07-22
What if we just simplify chunk size to full sequence size in those cases instead of all the else ifs?

## [c3634071841] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1400 · 2026-07-22
Yea here it seems even weirder, why do we have this?

## [c3634073815] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1411 · 2026-07-22
Same principle we can set to full length == chunk size to have only one path

## [c3634074657] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1564 · reply to [c3541456258] · 2026-07-22
Bump

## [c3634077543] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1450 · 2026-07-22
Ok this is really weird, why isnt it a config value instead of some weird hierarchical passing of some value?

## [c3634081578] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1458 · 2026-07-22
Inference vs training? Still very weird imo - gradient checkpointing layer?

## [c3634084844] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1516 · 2026-07-22
same here, let's simplify into one path

## [c3634091718] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1554 · 2026-07-22
This feel very attention ish. Could we try to write it into a similar flow at least?

## [c3634094837] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1697 · 2026-07-22
why not in config but a priv attribute?

## [c3634098678] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1448 · 2026-07-22
?

## [c3634103501] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1614 · 2026-07-22
it feels like we repeat basically everything, there has to be a more uniform way

## [c3634107564] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2021 · 2026-07-22
Why not one module and set to identity if not final block

## [c3634109518] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2083 · 2026-07-22
Doubt 😬 
```suggestion
```

## [c3634116130] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2402 · 2026-07-22
Doesnt that kinda also hint towards a generation loop?

## [c3634118277] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2174 · 2026-07-22
maybe one module?

## [c4758517756] vasqu (COLLABORATOR) · review COMMENTED · 2026-07-22
My brain is fried, sorry for the delays 😭

## [c3667349641] Rocketknight1 (MEMBER) · review_comment on docs/source/en/model_doc/esmfold2.md:39 · reply to [c3633595654] · 2026-07-28
Done!

## [c4799578732] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667357980] Rocketknight1 (MEMBER) · review_comment on docs/source/en/model_doc/esmfold2.md:41 · reply to [c3633629656] · 2026-07-28
Basically, forward does one step, diffusion is done in the generation mixin, all of that is wrapped by `infer_protein_as_pdb`. That's the main entry point users want, even if it's not a single forward!

## [c4799589273] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667362815] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:109 · reply to [c3633658726] · 2026-07-28
Yep, fixed and this is gone now!

## [c4799595763] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667365230] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:94 · reply to [c3633681283] · 2026-07-28
Yeah, this is handled in the conversion script now

## [c4799598642] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667368417] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:110 · reply to [c3633688651] · 2026-07-28
Done!

## [c4799602544] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667370457] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:132 · reply to [c3633696867] · 2026-07-28
Yeah, handled in validate_architecture now

## [c4799604973] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667373678] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:238 · reply to [c3633704924] · 2026-07-28
Cleaned this up!

## [c4799608925] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667378420] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:238 · reply to [c3633708102] · 2026-07-28
I tried but couldn't get this working, it's not exactly the same as Llama or other models I checked, so I'd have to inherit and then override multiple attributes. I just left it.

## [c4799614722] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667382917] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:307 · reply to [c3633719044] · 2026-07-28
I'm afraid to touch the actual repo masking_utils at this point in the PR! I'm tempted just to leave this one for now

## [c4799620101] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667388734] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:360 · reply to [c3633730551] · 2026-07-28
The lm head is actually dense->layernorm->decoder so the base class methods get this wrong

## [c4799627049] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667389814] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:399 · reply to [c3633740215] · 2026-07-28
Done!

## [c4799628249] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667390938] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:430 · reply to [c3633743036] · 2026-07-28
Gone!

## [c4799629616] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667392911] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/modular_esmc.py:425 · reply to [c3633755636] · 2026-07-28
Done!

## [c4799631955] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667397726] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:172 · reply to [c3633769746] · 2026-07-28
Yeah, this is back to six subconfigs now

## [c4799637575] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667400136] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:282 · reply to [c3633773177] · 2026-07-28
everything is now subconfigs

i have been staring at so many subconfigs for days

## [c4799640380] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667402990] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/convert_esmfold2_checkpoint.py:36 · reply to [c3633782505] · 2026-07-28
Cleaned up!

## [c4799644119] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667404566] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/protein_utils.py:413 · reply to [c3633792515] · 2026-07-28
Done!

## [c4799645874] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667419209] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/protein_utils.py:1 · reply to [c3633800759] · 2026-07-28
These values are weird, they basically encode physics. They say things like "oh yeah this amino acid has a positive charge of +1" or something. The model needs those to fold proteins, but they're just constants, they're not going to change unless we destroy the laws of the universe, so putting them in config doesn't make sense either.

There isn't really an equivalent to this in LLMs because everything in LLMs is flexible, the tokenizer vocab is learnable, they don't have any hardcoded knowledge. But protein folding models do!

## [c4799663908] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667430520] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:44 · reply to [c3633811858] · 2026-07-28
Yeah, the actual outputs are ModelOutput now, but this one is weird and I left it alone as a dataclass, since it's all consumed downstream and we need all of its output fields. Little weird but protein folding is always weird :sweat_smile:

## [c4799677195] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667438293] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:61 · reply to [c3633820716] · 2026-07-28
That class is gone! The balance was tricky, there's just a huge amount of arguments here, so either we need massive signatures, or we need occasional whales of dataclasses floating around in the code. So this problem still occurs sometimes and it's kind of hard to stop lol

## [c4799686573] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667446438] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:91 · reply to [c3633831473] · 2026-07-28
Yeah, did a massive refactor and dragged it all out into generation_esmfold2.py. The forward method now just does one diffusion step, everything else goes into generation. Cache class is gone. Everything is gone. The generation code prepares the initial state, then loops through forward to iteratively fold a protein

## [c4799696945] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667448959] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:104 · reply to [c3633839080] · 2026-07-28
Done!

## [c4799700119] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667450835] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:116 · reply to [c3633848178] · 2026-07-28
Yeah, everything is now EsmFold2SwiGLU

## [c4799702379] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667452121] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:132 · reply to [c3633850665] · 2026-07-28
Done!

## [c4799704014] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667458169] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:161 · reply to [c3633856109] · 2026-07-28
Removed!

## [c4799711592] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667470741] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:285 · reply to [c3633901002] · 2026-07-28
The subconfig part is done! I didn't do modular for esmfold2, though, there's just so little of it that's actually importable, maybe 5-10% of the file. It would remove a few helper functions but the modular file would be almost as big as a raw modeling file.

## [c4799727695] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667472912] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:293 · reply to [c3633914624] · 2026-07-28
Yes, done! There are some genuine fully-padded rows, awkwardly, but we handle them with nan_to_num now.

## [c4799730821] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667486807] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:361 · reply to [c3633926239] · 2026-07-28
Done! And stopped recreating this once per layer.

## [c4799747650] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667495335] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:377 · reply to [c3633951804] · 2026-07-28
Yeah, this was overcomplicated. Used your suggestion!

## [c4799758017] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667497039] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:390 · reply to [c3633955350] · 2026-07-28
subconfigs in!

## [c4799760038] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667499665] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:398 · reply to [c3633957135] · 2026-07-28
Done I think!

## [c4799763184] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667500814] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:409 · reply to [c3633960950] · 2026-07-28
Renamed!

## [c4799764448] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667502025] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:407 · reply to [c3633963805] · 2026-07-28
Done, it's `ESMFold2RMSNorm`

## [c4799765962] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667503958] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:425 · reply to [c3633969904] · 2026-07-28
Yep, done!

## [c4799768712] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667507682] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:486 · reply to [c3633980874] · 2026-07-28
Weird cast fixed, refactored and renamed this bit!

## [c4799773645] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667508599] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:519 · reply to [c3633986611] · 2026-07-28
Done!

## [c4799774747] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667510981] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:558 · reply to [c3633990672] · 2026-07-28
Fixed, a lot of this manual padding work is gone now

## [c4799777504] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667595123] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:549 · reply to [c3633993848] · 2026-07-28
Moved! This is done once per fold now, not once per step

## [c4799882492] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667597439] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:561 · reply to [c3633999052] · 2026-07-28
The missing mask fixed almost all of this

## [c4799885514] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667600198] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:762 · reply to [c3634020072] · 2026-07-28
Renamed!

## [c4799888994] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667609305] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:762 · reply to [c3634020072] · 2026-07-28
Also passed position bias as suggested, replaced all the non-standard attention stuff

## [c4799900412] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667611600] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:865 · reply to [c3634031254] · 2026-07-28
Done! The refactor is that `prepare_conditioning` does everything once and then the sampler passes it into each step

## [c4799903598] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667613277] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:896 · reply to [c3634034316] · 2026-07-28
Fixed!

## [c4799905908] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667619644] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1282 · reply to [c3634056175] · 2026-07-28
ClipMLP doesn't work because the shapes don't line up! I left that one

## [c4799915185] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667624689] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1353 · reply to [c3634065388] · 2026-07-28
Yeah, done, so `_triangular_contract` writes things out now

## [c4799922193] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667626528] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1384 · reply to [c3634070351] · 2026-07-28
Yeah, `chunk_size = self.chunk_size or seq_len` now

## [c4799924780] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667634264] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1400 · reply to [c3634071841] · 2026-07-28
Moved chunk_size to the config, but we do need to keep it because the triangle multiplication is O(n^3), so without some chunking method everyone will OOM. It's a bit awkward, but at least there's no more set_chunk_size everywhere

## [c4799936072] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667635152] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1411 · reply to [c3634073815] · 2026-07-28
Done!

## [c4799937334] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667637167] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1450 · reply to [c3634077543] · 2026-07-28
Done! No more weird hierarchical passing, all a config var

## [c4799940249] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667639078] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1458 · reply to [c3634081578] · 2026-07-28
Yep, `GradientCheckpointingLayer` now

## [c4799942926] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667639901] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1516 · reply to [c3634084844] · 2026-07-28
Done!

## [c4799944132] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667644622] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1697 · reply to [c3634094837] · 2026-07-28
Yep, moved to config!

## [c4799950469] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667651535] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1554 · reply to [c3634091718] · 2026-07-28
Restructured to make it more attention-y, but it doesn't fit exactly, but there's a lot of weirdness that makes it not quite standard attention.

## [c4799958975] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667654887] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1614 · reply to [c3634103501] · 2026-07-28
Yeah, wrote one helper and cleaned up a lot of `repeat_interleave` sites

## [c4799963264] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667666804] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2021 · reply to [c3634107564] · 2026-07-28
I'm not sure identity works because these are residuals, so we'd really want zeros to make it a no-op. I left this one alone, it's not major (since none of this is a data-dependent conditional or anything)

## [c4799978223] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667667853] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2083 · reply to [c3634109518] · 2026-07-28
Quietly got rid of it :sweat_smile:

## [c4799979493] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667669155] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2402 · reply to [c3634116130] · 2026-07-28
Yep, all factored out to the generation file

## [c4799981092] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c3667669999] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2174 · reply to [c3634118277] · 2026-07-28
Done! `EsmFold2Parcae`

## [c4799982315] Rocketknight1 (MEMBER) · review COMMENTED · 2026-07-28 (no text)

## [c5107329688] Rocketknight1 (MEMBER) · issue_comment · reply to [c4758517756] · 2026-07-28
@vasqu finished handling everything! Big refactors, moved a lot out to a `generation` file. The forward is a lot cleaner at this point. I moved some stuff to the conversion which means I need to rebuild the repos, including ESMC. I'll do that locally and we can move them to BioHub later. I also still need to move the kernel to `kernels-community`, I'll work on that while I wait for a re-review!

## [c3705639500] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/hub_kernels.py:140 · 2026-08-03
Do we need to update to kernels community?

## [c3705646614] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:128 · 2026-08-03
Could this not be handled with the super call?

## [c3705647867] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:131 · 2026-08-03
and this after the super call then?

## [c3705659612] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:53 · 2026-08-03
I would rather remove expansion ratio and directly infere intermediate size on conversion / post init

Also lets remove the ffn_, it's old legacy naming imo

## [c3705663322] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:56 · 2026-08-03
Could we at least partially use rope parameters?

## [c3705669879] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:65 · 2026-08-03
Let's please set them here tho! The we can ensure in the parent via validate architecture that the values match but we should not set them elsewhere but in the config itself

## [c3705683479] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:102 · 2026-08-03
Why do we still have token_ prefix? Can we not simplify to remove them?

## [c3705690236] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:102 · reply to [c3705683479] · 2026-08-03
num_blocks -> num hidden layers (keep consistent naming please)

transition_multiplier is not needed, same as expansion factor -> bake into intermediate size

## [c3705695134] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:147 · 2026-08-03
s, p -> expand please

## [c3705705704] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:150 · 2026-08-03
A lot of these attributes do sound like as if they were a generation config tbh. We have a generation mixin now, any chance we could refactor these out for that?

## [c3705713865] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:297 · 2026-08-03
Are we repeatin values from the children here? Any reason we have to and they need to be at top level as well?

## [c3705723474] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:396 · 2026-08-03
yea we should avoid setting attr to child configs in the parent

## [c3705732329] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:40 · 2026-08-03
```suggestion
        attention_mask: Tensor,
```

## [c3705734766] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:108 · 2026-08-03
why shorten?

## [c3705736326] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:112 · 2026-08-03
can keep as kwargs and add typed dict instead maybe?

## [c3705737984] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:112 · reply to [c3705736326] · 2026-08-03
at least normal kwargs seems more appropriate imo

## [c3705742716] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:100 · 2026-08-03
Why are we detaching? Should we maybe just decorate with inference mode instead?

## [c3705753764] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:133 · 2026-08-03
checking the model in a bit but maybe not make this nested: structure head living beside diffusion module or is it needed because of the forward?

## [c3705759629] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:149 · 2026-08-03
Same question here essentially, is it needed in the forward why not let it live in the generation mixin?

## [c3705765253] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:161 · 2026-08-03
let's make the loop use non single letters x -> hidden states (there is more but you get the gist)

## [c3705767537] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:191 · 2026-08-03
ok might happen more often but let's keep simple kwargs please

## [c3705770980] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:195 · 2026-08-03
prepare_protein_features should do this no? as option when a device is passed?

## [c3705776183] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:212 · 2026-08-03
why? i dont see why this needs to be a separate fn

## [c3705781396] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/protein_utils.py:353 · 2026-08-03
Can we have some reference links to where these come from. otherwise its hard to follow and see hardcoded values

## [c3705793358] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/protein_utils.py:557 · 2026-08-03
any reason we need the numpy / cpu always?

## [c3705810562] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:99 · 2026-08-03
Do we really need a module for this? Could still use normal nn layernorm or too many casts?

## [c3705813072] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:80 · 2026-08-03
What is c base? lets avoid single letters

## [c3705820028] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:38 · 2026-08-03
I know we mentioned to do this but are we kind of just avoiding the issue of long signatures and move it to an object - probably better but it doesn't seem that much better?

## [c3705824357] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:108 · 2026-08-03
Please use the same implementation as in nanochat then, no?

## [c3705825461] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:108 · reply to [c3705824357] · 2026-08-03
Also why torch.nn.Module?

## [c3705828807] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:88 · 2026-08-03
Something really wrong is going on if we need 2 masks

## [c3705843402] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:125 · 2026-08-03
Ok lets be honest: the chunk has no numerical implications. It only partially saves on peak memory if anything

Can we not drop the chunking here? Also I assume this module lives somewhere where attention is applied in its parent -> opportunity to follow the llama structure maybe Edit: likely not in this case but the chunking is unnecessary

## [c3705856413] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:153 · 2026-08-03
Why do we even have a separaten layernorm modul if we use layernorm inconsistently anyways? 

Either we use a separate module and allow everything to be done there (optional dtype casting at the end) or we always use F.

## [c3705866497] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:144 · 2026-08-03
why are we passing the config and not the hidden size as in every other layernorm? Also why do we have to go that deep into the config? Should we have passed a subconfig earlier?

## [c3705893416] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:158 · 2026-08-03
```suggestion
        residual = F.layer_norm(residual.float(), (self.hidden_size,), None, None, self.eps)
        
        hidden_states = self.norm(hidden_states)
        gate = torch.sigmoid(self.gate_proj(hidden_states))
        hidden_states = self.shift_proj(hidden_states)
        
        return (gate * residual + hidden_states).to(hidden_states.dtype)
```
small renaming, because the naming is a bit confusing

## [c3705905602] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:189 · 2026-08-03
Lets please follow llama mlp, it's exactly this with unfused weights

## [c3705914937] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:263 · 2026-08-03
confusing naming, are there more attention modules why not keep it a simple attention or if its specific to a module something along AtomAttention

## [c3705919970] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:255 · 2026-08-03
Tbh we have quite a few copies, it would be worth it to potentially use modular no?

## [c3705924824] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:300 · 2026-08-03
Pretty sure nomic bert is similar

## [c3705926253] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:272 · 2026-08-03
hidden size

## [c3705935904] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:336 · 2026-08-03
Wait what? How, when? The sequence if fucked then either way no? Like it makes no sense to have complete padding no?

## [c3705943379] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:338 · 2026-08-03
No. let's not hide activation for a linear

## [c3705958557] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:351 · 2026-08-03
why pass the whole config instead of the few atom values that are needed?

## [c3705959587] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:362 · 2026-08-03
kwargs?

## [c3705961210] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:363 · 2026-08-03
atom cond --> residual?

## [c3705969222] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:407 · 2026-08-03
can be buffered no?

## [c3705976838] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:466 · 2026-08-03
a lot of d_, n_ -> lets use modern naming please -> hidden_size, num_hidden_layers (this happens a lot)

## [c3705983476] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:489 · 2026-08-03
batch size, seq_len

## [c3705988737] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:516 · 2026-08-03
why not use swa creation directly and fixup the swa size in the config?

## [c3705991303] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:481 · 2026-08-03
Why does this live here if it isnt called here in the forward?

## [c3705999007] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:663 · 2026-08-03
could be moved outside this module no? Not really attn relevant

## [c3706002122] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:671 · 2026-08-03
same as in the other attention module no?

## [c3706007040] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:690 · 2026-08-03
This last gating is also not 100% relevant to the attn

Imo this module is very similar to the other one which we may be able to reuse instead

## [c3706015372] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:654 · 2026-08-03
Why do we have a lot of this live here but not used within the module?

## [c3706019934] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:654 · reply to [c3706015372] · 2026-08-03
Check inkling out: position bias is a kwarg now that you can use directly

## [c3706023305] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:693 · 2026-08-03
This pattern occurs a lot very likely we can refactor here too

## [c3706027849] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:720 · 2026-08-03
blocks <-> layers; we confuse namings a lot and no longer have consistent namings for same attr/vars which makes it hard to follow

## [c3706032416] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:724 · 2026-08-03
We could pass this as kwarg instead of the attn mask modification + let's move things out of the attn module where its not used?

## [c3706036805] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:727 · 2026-08-03
why was it named token_acts in the first place then?

## [c3706037734] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:740 · 2026-08-03
...

## [c3706045370] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:801 · 2026-08-03
Yea this shouldnt live here when not called here

This is a general pattern - we add functionality that is called at some other time which makes readability hard

## [c3706059951] vasqu (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2192 · 2026-08-03
given we have the full signature, why do we even keep the input dataclasses (their purpose is not really super worth it then if we cannot avoid the signature either way?)

## [c4846135497] vasqu (COLLABORATOR) · review COMMENTED · 2026-08-03
Ok I'm stopping a bit early. My issue is that parts are still confusing
1. The naming is super mixed up -> sometimes we use hidden size, sometimes d_
2. We have some weird modules where I don't see the purpose, e.g. layernorm usage with module, F., etc. but no consistency 
3. A lot of functionality is baked into a module but it's called in its parent modules or somewhere entirely else which makes it super hard to keep track of the actual forward flow
4. We have dataclasses for inputs but in the end they still find their way into the signature at parts
5. Abbreviations everywhere, s_, c_
6. A lot of modules feel very similar. I'm pretty sure there is a lot of potential

## [c3736670096] Rocketknight1 (MEMBER) · review_comment on src/transformers/integrations/hub_kernels.py:140 · reply to [c3705639500] · 2026-08-07
Kernel is being moved to biohub, I'm on it!

## [c4883986247] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736684712] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:131 · reply to [c3705647867] · 2026-08-07
Done, moved it after the super call!

## [c4884004564] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736687523] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:53 · reply to [c3705659612] · 2026-08-07
Done, we infer it at conversion time now, ffn prefixes also dropped

## [c4884007994] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736696522] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:56 · reply to [c3705663322] · 2026-08-07
This got messy and I gave up! I don't think it's worth it. ESM uses two ropes with different bases and denominators, and `position_ids` doesn't work like you'd expect because we have 3 continuous co-ordinate channels in 3D space instead. This is not like standard LLM RoPE in a lot of ways!

## [c4884018475] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736907925] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:102 · reply to [c3705683479] · 2026-08-07
Yep, removed the prefix, switched blocks-> name, baked the expansion factor into intermediate size

## [c4884278425] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736913189] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:147 · reply to [c3705695134] · 2026-08-07
All done. I chased these down and there should be no single-letter vars left outside of a few cases like the quaternions

## [c4884284411] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736918334] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:150 · reply to [c3705705704] · 2026-08-07
We don't want a generation config, though, since the model can't really generate! I tried to group all of the 'sampler settings' together onto `EsmFold2StructureHeadConfig` instead, which at least keeps them in one place and stops them wandering around the configs

## [c4884290415] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736926093] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:297 · reply to [c3705713865] · 2026-08-07
No, these are different! Vars like `hidden_size` exist on subconfigs, and apply to that module. The `hidden_size` in the MSA is different from the one in diffusion. We're not replicating values pointlessly

## [c4884299065] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736927092] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:396 · reply to [c3705723474] · 2026-08-07
Got rid of it!

## [c4884300274] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736927784] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:40 · reply to [c3705732329] · 2026-08-07
Done!

## [c4884301044] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736932920] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:108 · reply to [c3705734766] · 2026-08-07
These are names from the [mmCIF](https://en.wikipedia.org/wiki/Macromolecular_Crystallographic_Information_File) file format, so we can't change them without breaking output. We could rename the vars internally but we'd have to write to those output names eventually.

## [c4884307099] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736963591] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:112 · reply to [c3705736326] · 2026-08-07
I left this one alone - we can move it to kwargs if you want, but I feel like a big explicit signature here is fine. I experimented with typeddict but it was only used once so it just made the PR even bigger!

## [c4884342766] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736964513] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:100 · reply to [c3705742716] · 2026-08-07
Yeah, decorated instead!

## [c4884343826] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736967302] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:133 · reply to [c3705753764] · 2026-08-07
Yeah, I flattened this out, and made the `structure_head` the diffusion module

## [c4884346766] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736969933] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:149 · reply to [c3705759629] · 2026-08-07
Moved a lot of this to the mixin!

## [c4884350082] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736971337] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:161 · reply to [c3705765253] · 2026-08-07
Got rid of all the single-letter vars

## [c4884351733] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736972429] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:191 · reply to [c3705767537] · 2026-08-07
Done!

## [c4884353058] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736974482] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:195 · reply to [c3705770980] · 2026-08-07
Yeah, the method can set the device now!

## [c4884355564] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736976290] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/generation_esmfold2.py:212 · reply to [c3705776183] · 2026-08-07
Yeah, this was kind of an empty wrapper. Got rid it of it and just called `protein_utils`!

## [c4884357685] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736981666] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/protein_utils.py:353 · reply to [c3705781396] · 2026-08-07
A lot of this is hardcoded values about the physical structure of proteins. It's not config (it never changes unless physics changes, and then we have bigger problems :sweat_smile:), but there isn't really a 'citation' for it anywhere. This is just the shape that those molecules have!

## [c4884364105] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3736988319] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/protein_utils.py:557 · reply to [c3705793358] · 2026-08-07
It's because there are no more tensor ops after this step. Instead, the `output_to_pdb` method walks the array and does single-element Python operations to make the final output. If we kept them as `torch.Tensor` on GPU then that would be `.item()` and a GPU sync every time. Numpy is much cleaner + faster for this bit!

## [c4884372015] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737034919] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:99 · reply to [c3705810562] · 2026-08-07
Yeah, we need the casts here! The model is very sensitive to small errors in the port, and we need to upcast inputs and downcast outputs like this. Just using standard layernorm with `_keep_in_fp32_modules_strict` doesn't work because that will only upcast the weights, and the output dtype can wander.

## [c4884429977] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737036109] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:80 · reply to [c3705813072] · 2026-08-07
Got rid of those single-letter names!

## [c4884431467] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737040545] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:38 · reply to [c3705820028] · 2026-08-07
There just a lot of args being passed here! We can group them into dataclasses or leave them as single args, I think both are fine, but there isn't really a clean way to avoid it.

## [c4884437325] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737042547] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:108 · reply to [c3705824357] · 2026-08-07
Yep! I went back to modular and inherited from `NanoChatRMSNorm`

## [c4884439761] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737056922] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:125 · reply to [c3705843402] · 2026-08-07
It has small numerical implications, unfortunately! Split matmuls have slightly different outputs. I tried changing this but I started to see errors climbing and I got scared

## [c4884457111] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737072906] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:153 · reply to [c3705856413] · 2026-08-07
We almost always use `EsmFold2LayerNorm`! The only place we use `F.layer_norm` is in this `AdaptiveLayerNorm` module. It's totally different from standard layer norm. It computes `F.layer_norm` internally in one line but also does a lot of other stuff, so we can't merge the two classes. I think what we're doing here is fine

## [c4884476482] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737077055] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:144 · reply to [c3705866497] · 2026-08-07
Yeah, fixed! I went through the config passing and tried to pass subconfigs where possible, and also only passed `hidden_size` etc. when a layer only needed a single value

## [c4884481658] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737085571] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:158 · reply to [c3705893416] · 2026-08-07
Partially applied this one! `residual` isn't correct though, so I changed that bit

## [c4884492659] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737088902] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:189 · reply to [c3705905602] · 2026-08-07
This actually matches `Phi3MLP`, so I inherited from that in the modular!

## [c4884497180] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737093301] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:263 · reply to [c3705914937] · 2026-08-07
Yeah, renamed to `EsmFold2AtomAttention`. And there's also `EsmFold2AttentionPairBias` so we can't just call it `Attention` :sweat_smile:

## [c4884503436] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737095460] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:255 · reply to [c3705919970] · 2026-08-07
Yeah, I gave up and went back to modular.

## [c4884506356] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737101253] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:300 · reply to [c3705924824] · 2026-08-07
It's close but I couldn't inherit, nomic doesn't have `q_norm` or `k_norm` or the sigmoid on the attention output, etc. They're different enough that I'd have to override `init()` and `forward()` and then I've just written a whole new class anyway

## [c4884514257] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737103243] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:272 · reply to [c3705926253] · 2026-08-07
Done!

## [c4884516778] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737120655] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:336 · reply to [c3705935904] · 2026-08-07
This is another weird artifact of operating in 3D space - we're operating on **atoms which can be in arbitrary places, not nice 1D `position_ids`** The sliding window operates over space, not position_ids, so a single atom can have 0 other tokens in its attention radius. This generally only happens with padding tokens, so the `nan_to_num` guard just makes sure that padding tokens don't create `NaN`. The values will be fucked, but they're fucked on padding tokens so they don't damage the rest of the output

## [c4884539353] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737122682] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:338 · reply to [c3705943379] · 2026-08-07
Fixed!

## [c4884541963] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737124337] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:351 · reply to [c3705958557] · 2026-08-07
Basically passing it just for `self.config._attn_implementation` here!

## [c4884543928] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737126860] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:362 · reply to [c3705959587] · 2026-08-07
I kept these! There's only 4 tensors and they're all required, so I'm not sure we want to put them in `**kwargs`

## [c4884546806] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737130032] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:363 · reply to [c3705961210] · 2026-08-07
It's not really the residual! I renamed to `atom_conditioning`

## [c4884550479] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737170456] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:407 · reply to [c3705969222] · 2026-08-07
This got awkward for me because building on CPU vs GPU shifted the output significantly, but the init didn't always know what device it was going to be placed on. I could maybe make buffering work but I got scared of the output changes and left it because the performance impact is tiny

## [c4884599357] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737171964] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:466 · reply to [c3705976838] · 2026-08-07
Yeah, went through and tried to catch every instance of that!

## [c4884601187] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737172988] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:489 · reply to [c3705983476] · 2026-08-07
Done!

## [c4884602586] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737180795] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:516 · reply to [c3705988737] · 2026-08-07
The way they handle sliding attention radius is a bit different from ours - we could maybe get this to work but it requires breaking all existing configs and changing everything in the convert stage. I left it alone!

## [c4884613061] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737185583] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:481 · reply to [c3705991303] · 2026-08-07
Basically because the weights are here and we want to compute it once per fold, rather than repeatedly recomputing it!

## [c4884618672] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737186659] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:663 · reply to [c3705999007] · 2026-08-07
Yeah, moved to `EsmFold2DiffusionLayer`

## [c4884619889] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737191297] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:671 · reply to [c3706002122] · 2026-08-07
It's similar but there's a lot of changes, enough to require overridden `init` and `forward()`, so I can't just inherit/copy it

## [c4884626960] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737193949] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:690 · reply to [c3706007040] · 2026-08-07
Wasn't able to reuse it in the end, but I did move the gate out to `EsmFold2DiffusionLayer`!

## [c4884630896] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737212207] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:654 · reply to [c3706015372] · 2026-08-07
Yeah, moved adaLN and the output gate out of the attention. The problem with using `position_bias` like inkling is that `position_bias` is not learned differently in each layer for us, so we can fuse the mask with the position bias once at the start. If we don't do that then we have to do it in every layer, hundreds of times per fold, which is a bit messy.

## [c4884658971] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737215672] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:720 · reply to [c3706027849] · 2026-08-07
Done!

## [c4884664734] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737217267] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:727 · reply to [c3706036805] · 2026-08-07
This came from the fork code! Gone now

## [c4884667404] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737221263] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:724 · reply to [c3706032416] · 2026-08-07
I tried moving things out of the attention module, but I'm keeping the attention mask modification here to avoid repeating it inside the loop

## [c4884673605] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737223246] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:740 · reply to [c3706037734] · 2026-08-07
Did more naming fixes here :sweat:

## [c4884676944] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737613061] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:801 · reply to [c3706045370] · 2026-08-07
I moved a lot of this, but kept `prepare_conditioning`, `embed_atoms`, `build_attention_mask` and `compute_pair_bias` since those are actually on the model where their weights live, but they're called from the parent because the parent caches the result once and reuses it over and over.

## [c4885171874] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737619413] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:2192 · reply to [c3706059951] · 2026-08-07
In this case, we pass this block of args around together to a couple of downstream layers, so I kept the dataclass. In some other cases I dropped it when it was really only being used once

## [c4885179413] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737912005] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:128 · reply to [c3705646614] · 2026-08-07
I wasn't sure here, since I wasn't too familiar with the deep tokenizer code. Claude insists that this is correct for a tokenizer whose vocabulary is built in Python, not loaded - you set `self._tokenizer` and then call the `super()`, which has alternate paths when `self._tokenizer is None` is false. I wasn't certain but it seems to work!

## [c4885535721] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3737993369] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/configuration_esmfold2.py:65 · reply to [c3705669879] · 2026-08-07
This should mostly be resolved by other refactors. I tried to avoid `validate_architecture` when everything is in one config though, in that case there's no point in allowing illegal combinations, we just derive the expected value and don't let the config set it at all

## [c4885631976] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3738018179] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:88 · reply to [c3705828807] · 2026-08-07
It's awkward! The way the code works is that the 4D mask is the "sliding window" in 3D space, it marks which atoms are in range for other atoms to see + attend to. The 2D mask just marks which tokens are padding. Their setup has padding tokens get redirected to a sentinel token and it's quite hard to refactor that, so I really want to just leave this bit alone!

## [c4885661562] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c3738089878] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:693 · reply to [c3706023305] · 2026-08-07
I refactored to make the two attention layers more similar (bits of them really were doing the same thing two different ways), but we can't merge them. They differ on a bunch of stuff, qk-norm, q_proj bias, different rope etc.

## [c4885746842] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-07 (no text)

## [c5220937037] Rocketknight1 (MEMBER) · issue_comment · reply to [c4846135497] · 2026-08-07
Review done! cc @ArthurZucker @Cyrilvallez since @vasqu is out. The remaining work is moving the kernel/repos to Biohub and regenerating everything with the new conversion script, I'm working on that.

Also, if you're doing another review, please bear in mind that **it's not an LLM, even if it looks like one**! There's a lot of stuff that looks like we should be able to inherit or refactor it, but it's all a bit different. We need a load of protein molecule constants, everything has continuous float positions in 3D space instead of 1D `position_ids`, etc. There have been like 5 review rounds at this point and a lot of the proposed refactors weren't really possible, and I'd really like to merge this soon :sweat_smile: !

## [c5254529643] Rocketknight1 (MEMBER) · issue_comment · reply to [c4723172886] · 2026-08-11
Just finished a performance/correctness pass. We're faster than the original code (about 15%) and correct to within tiny numerical error. All protein folds look good at this stage, so we should be ready for final reviews soon once people are happy with code quality cc @Cyrilvallez @ArthurZucker

## [c5315552928] Rocketknight1 (MEMBER) · issue_comment · 2026-08-17
`mlinter` pass done! I accepted most but there are a few I didn't accept:

1) It wanted the dataclasses like `AtomInputs` to inherit from `ModelOutput` but this doesn't fit. `ModelOutput` always has one primary output, and everything else has to be optional. Every field in `AtomInputs` is mandatory. They're really an internal dataclass for passing arguments together rather than a `ModelOutput`.
2) TRF034 says that `EsmcLayer` doesn't inherit `GradientCheckpointingLayer` but I think this is a bug in `mlinter`. It's firing on the `modular` file, but that class inherits from `LlamaDecoderLayer` and it correctly inherits from `GradientCheckpointingLayer` in the `modeling` file. So I don't think there's anything for us to fix here (cc @tarekziade)

## [c3803055289] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:339 · 2026-08-18
init its not input dependant

## [c3803058787] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:354 · 2026-08-18
same

## [c3803062480] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:356 · 2026-08-18
same if you take all, no slicing

## [c3803069526] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:508 · 2026-08-18
to remove

## [c3803079693] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:572 · 2026-08-18
`self.pair_bias_proj(self.pair_norm(pair_states))` don't belong in this module

## [c3803090775] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:653 · 2026-08-18
remove that layer entirely no?

## [c3803091860] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:653 · reply to [c3803090775] · 2026-08-18
will look a lot more like the normal attention?

## [c3803093656] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:699 · 2026-08-18
why not 0 an 1

## [c3803094142] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:715 · 2026-08-18
smae

## [c3803103946] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:907 · 2026-08-18
again this does not need einsum at all

## [c3803108430] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:970 · 2026-08-18
lot of casting, maybe keep in fp32? or something?

## [c3803113006] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:1020 · 2026-08-18
these do look like simple matmult

## [c3803146173] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:1149 · 2026-08-18
same?

## [c3803148758] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:1344 · 2026-08-18
why can they be none

## [c3803152185] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:1657 · 2026-08-18
mmm we try to avoid this as much as possible

## [c3803155980] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:1814 · 2026-08-18
why not a module

## [c3803158319] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:1972 · 2026-08-18
is this generate ish?

## [c3803160097] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:2143 · 2026-08-18
this is kinda insane

## [c3803171824] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/protein_utils.py:1 · 2026-08-18
we should put all of these on the hub via converter imo. Put them in any file. Hardcoded knowledge -> not maintained, not code

## [c3803173268] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/protein_utils.py:1 · reply to [c3803171824] · 2026-08-18
just like MEL features audio

## [c4959805051] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-08-18
Already much much cleaner

## [c5327042511] tarekziade (COLLABORATOR) · issue_comment · reply to [c5315552928] · 2026-08-18
> `mlinter` pass done! I accepted most but there are a few I didn't accept:
> 
>     1. It wanted the dataclasses like `AtomInputs` to inherit from `ModelOutput` but this doesn't fit. `ModelOutput` always has one primary output, and everything else has to be optional. Every field in `AtomInputs` is mandatory. They're really an internal dataclass for passing arguments together rather than a `ModelOutput`.
> 
>     2. TRF034 says that `EsmcLayer` doesn't inherit `GradientCheckpointingLayer` but I think this is a bug in `mlinter`. It's firing on the `modular` file, but that class inherits from `LlamaDecoderLayer` and it correctly inherits from `GradientCheckpointingLayer` in the `modeling` file. So I don't think there's anything for us to fix here (cc @tarekziade)

Yeah those are two mlinter bugs, thanks for pointing out. 

https://github.com/huggingface/transformers-mlinter/issues/36
https://github.com/huggingface/transformers-mlinter/issues/37

## [c3805526192] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/protein_utils.py:1 · reply to [c3803171824] · 2026-08-18
Hmmn, I don't think mel features works! With mel features we recompute them in `__init__`, but I don't think we have anything like these protein features. These are really physical constants, and they're never going to change. We can't recompute them, so we'd just have to store an identical `protein_constants.json` file or something in every ESMFold2 checkpoint, and make a feature extractor etc. Are you sure that's the right solution? ESMFold1 just keeps them in the library and it's never been a problem

## [c4962824620] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807139201] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:339 · reply to [c3803055289] · 2026-08-18
I tried this but this area of the file is very dangerous! The model is sensitive to these values, and they compute differently on CPU and GPU. So if we compute them on CPU in init and then move to GPU with `to()`, they're different than they are in the original code and this actually causes divergence. A lot of the code here is hard to refactor because of reasons like this :sweat_smile:

## [c4964835981] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807141081] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:354 · reply to [c3803058787] · 2026-08-18
I was able to clean up this bit, since the numerics don't change on CPU/GPU!

## [c4964838190] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807142062] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:356 · reply to [c3803062480] · 2026-08-18
Done!

## [c4964839226] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807142831] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:508 · reply to [c3803069526] · 2026-08-18
Done!

## [c4964840146] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807144093] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:572 · reply to [c3803079693] · 2026-08-18
Done, moved to `EsmFold2DiffusionLayer`

## [c4964841773] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807146313] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:653 · reply to [c3803090775] · 2026-08-18
Done! I used more standard naming too

## [c4964844287] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807147956] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:699 · reply to [c3803093656] · 2026-08-18
Done! I have to reupload the checkpoints after this but I updated the conversion script

## [c4964846143] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807148565] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:715 · reply to [c3803094142] · 2026-08-18
Done

## [c4964846742] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807150626] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:907 · reply to [c3803103946] · 2026-08-18
Removed that one! (But I kept the triangle multiplication)

## [c4964848999] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807154162] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:970 · reply to [c3803108430] · 2026-08-18
Yep, removed almost all the casts here by setting things to the right dtype earlier!

## [c4964852925] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807164215] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:1020 · reply to [c3803113006] · 2026-08-18
This is the trimul! It's the most expensive op in the whole thing, bmm is slower, and we have a kernel for it. I'm very scared of changing it because it will probably break model numerics too.

## [c4964864527] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807165696] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:1149 · reply to [c3803146173] · 2026-08-18
Got rid of this one!

## [c4964866252] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807166996] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:1344 · reply to [c3803148758] · 2026-08-18
Got rid of those conditionals

## [c4964867648] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807170270] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:1657 · reply to [c3803152185] · 2026-08-18
I renamed it! We still do have one final layer without the MSA modules, that's just how the model works, but I tried to at least have clearer variable names

## [c4964871443] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807171727] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:1814 · reply to [c3803155980] · 2026-08-18
Moved this to `EsmFold2InputEmbedder`!

## [c4964873147] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807175910] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:1972 · reply to [c3803158319] · 2026-08-18
It's not! There's stuff that looks like generation here but iterative protein folding diffusion is very different and we can't fit it into standard generation methods/classes.

## [c4964878345] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c3807178627] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:2143 · reply to [c3803160097] · 2026-08-18
I grouped some of it! It's still quite long, but a bit less insane :sweat_smile:

## [c4964882369] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-18 (no text)

## [c5332933787] Rocketknight1 (MEMBER) · issue_comment · 2026-08-18
@arthurzucker I think everything is done! There's some red CI because of the linter issues mentioned above, so those should get fixed soon. Also slow tests will fail until I reconvert + reupload all the checkpoints, I'll do that tomorrow

## [c3819750350] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:150 · 2026-08-20
again to remove please

## [c3819769639] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/protein_utils.py:1 · reply to [c3803171824] · 2026-08-20
The thing is that we are setting a precedent with this 😉 Not a single model has this. That's the bid issue at hand! 

Yes we compute Mel in init but we do save them. This is the same! If these are feature extractor, at best we can put them in feature extractor.py, and have #fmt skip to make it minimal.
The issue is in the breaking of the contract of file names!

## [c3819790730] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:187 · 2026-08-20
I think we have a decorator to not inherit a decorator!

## [c3819792736] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:230 · 2026-08-20
this part is inheritable

## [c3819812634] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:234 · 2026-08-20
Close to AfmoeAttention ! but simpler so makes sense. I think you can easily inherit from qwen3moe or a model that is vision only as there is no cache + make the oproj and gate a single module? might bee too much so just inherit for the init in that case!!

## [c3819844521] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:339 · reply to [c3803055289] · 2026-08-20
In that case should we not make them persistant buffers instead?

## [c3819846138] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:339 · reply to [c3803055289] · 2026-08-20
with a keep in fp32 strict at load time?

## [c3819849921] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:412 · 2026-08-20
```suggestion
        # CODEPATH: xxx uses with / YYY uses without
        if structure_prediction:
            self.coords_linear = nn.Linear(6, atom_config.hidden_size, bias=False)
```

## [c3819854626] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:480 · 2026-08-20
in a way, because this happens independently of anything it begs the question of why even put it in this module!

## [c3819863243] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:526 · 2026-08-20
same! Init van def inherit! 
Also, if we want 100% easy inheritance the oproj should be a module that inside has the gate and sigmoid. 
Its a small nit

## [c3819881064] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1190 · 2026-08-20
same about the # Codepath

## [c3819883370] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1163 · 2026-08-20
in general we explicit all config attrs

## [c3819889078] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1667 · 2026-08-20
again on the codepath

## [c4980470657] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-08-20
A few small things here and there to keep pushing for standards. But this is mergeable. Ty for the huge effort 🤗

## [c3822029286] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:150 · reply to [c3819750350] · 2026-08-20
Done!

## [c4983283091] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-20 (no text)

## [c3823027788] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:234 · reply to [c3819812634] · 2026-08-20
Yeah, I tried! The problem is the config doesn't quite match, so I wasn't able to actually inherit anything without breaking a lot of structure and adding lots of config attributes. In the end I think it's simpler to leave it

## [c4984588866] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-20 (no text)

## [c3823031035] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:230 · reply to [c3819792736] · 2026-08-20
See below, I tried but I don't think we can cleanly inherit in this class without adding a lot of lines elsewhere

## [c4984592649] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-20 (no text)

## [c3823035971] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:412 · reply to [c3819849921] · 2026-08-20
I investigated this one and this was actually a stale codepath from the experimental model! It's removed.

## [c4984598808] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-20 (no text)

## [c3823040317] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modular_esmfold2.py:480 · reply to [c3819854626] · 2026-08-20
Yeah, moved this! It cleaned things up a lot but I had to regenerate the checkpoints

## [c4984604177] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-20 (no text)

## [c3823044730] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1190 · reply to [c3819881064] · 2026-08-20
This was another case that was only here to support the old experimental codepath, so it's been dropped!

## [c4984610182] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-20 (no text)

## [c3823045811] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1163 · reply to [c3819883370] · 2026-08-20
Yes, done!

## [c4984611642] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-20 (no text)

## [c3823047331] Rocketknight1 (MEMBER) · review_comment on src/transformers/models/esmfold2/modeling_esmfold2.py:1667 · reply to [c3819889078] · 2026-08-20
Yeah, this one has been added! The tl;dr is this applies to the final layer only

## [c4984613730] Rocketknight1 (MEMBER) · review COMMENTED · 2026-08-20 (no text)

## [c5359307091] Rocketknight1 (MEMBER) · issue_comment · 2026-08-20
Comments mostly addressed! The one big thing is that I left `protein_utils` - we could do a giant refactor to upload it into Hub JSON, but I don't think it's worth it, plus we already have precedent, and that precedent is ESMFold1! Science models generally need precomputed consts like this to work and we've put them in the codebase before, so I don't want to break the model up for that refactor this close to launch. I merged/fixed just about everything else though

## [c5359482585] Rocketknight1 (MEMBER) · issue_comment · 2026-08-20
Also, tests currently point at my repos - I'll make a small PR updating that once they get their proper locations! But merging now to make sure we have support in the next release

## [c3991806252] RHMTshaikh (NONE) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:12 · 2026-09-11
new one

## [c5181693319] RHMTshaikh (NONE) · review COMMENTED · 2026-09-11 (no text)

## [c3991828841] RHMTshaikh (NONE) · review_comment on src/transformers/models/esmc/tokenization_esmc.py:12 · reply to [c3991806252] · 2026-09-11
self reply

## [c5181723608] RHMTshaikh (NONE) · review COMMENTED · 2026-09-11 (no text)

---
outcome: state=closed, merged=True
