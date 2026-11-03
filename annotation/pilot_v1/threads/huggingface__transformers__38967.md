# #38967 [pull_request] Add ModernBERT Decoder Models - ModernBERT, but trained with CLM!
https://github.com/huggingface/transformers/pull/38967  ·  opened by orionw on 2025-06-22

## [body] orionw (CONTRIBUTOR) · opening post · 2025-06-22
This PR adds the new ModernBERTDecoder architecture: it's ModernBERT but trained from scratch as a CLM! This replaces my attempt in #35946 to fit them all in one class. Instead I'll have a script to convert them from encoders to decoders and vice versa by renaming the class and head name (`head` vs `lm_head`)

The models/paper are ready to go, but we wanted to have this up before we do the official release so we can have all the release artifacts ready. Thanks again for all your help on the last PR and for suggesting this approach!

FAQs:

Q: Why don't we just add this to ModernBERT?
A: Because we discussed in #35946 that it was too complicated to add the causal attention combined with everything else. This keeps it simpler and uses the standard attention functions

Q: How can BERT be a decoder? 
A: It's really the "ModernBERT" architecture trained from scratch via CLM. It's not MLM at all. We could rename the architecture to whatever we decide to name it in the paper, if that's preferable. But I thought this was the easiest to understand since they share the same everything else.

## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [x] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [x] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [x] Did you write any new necessary tests?


## Who can review?
@Rocketknight1 @gante @ArthurZucker

## [c2994269018] orionw (CONTRIBUTOR) · issue_comment · 2025-06-22
Hi @Rocketknight1 @gante, this is the new PR after our last conversation. I think it's much simpler as it's just a generic decoder with re-used ModernBERT parts. Thanks again for the help!

## [c2996786681] Rocketknight1 (MEMBER) · issue_comment · 2025-06-23
cc @arthurzucker if you think this is appropriate for the library! I'm uncertain - I still think a smaller `ForCausalLM` wrapper that doesn't replace ModernBERT attention layers might be better, like `RobertaForCausalLM`.

## [c2996813844] orionw (CONTRIBUTOR) · issue_comment · reply to [c2996786681] · 2025-06-23
Just to clarify though this does use normal causal attention like any other decoder, unlike ModernBERT (encoder). 

I'm a bit confused though, in the last PR ((https://github.com/huggingface/transformers/pull/35946) the attention layers were what was deemed too complicated (dealing with local vs global and causal vs non-casual at the same time). e.g. the last PR  was the `ForCausalLM` wrapper you mention, but it can't really be much "smaller" due to the complexity of handling that. 

EDIT, to quote the last comment from @Rocketknight1 in #35946

> Adding large tokenizer / internal layer changes to an MLM can create bugs, reduce performance and make it harder to maintain the core MLM functionality, which we don't want! Running MLMs as CLMs is a very niche thing to do.

We can't change ModernBERT to have a `ForCausalLM` without changing the internal attention layers. See @ArthurZucker's [comment on my last PR in the attention layer](https://github.com/huggingface/transformers/pull/35946/files#r2033255809) where he said to make a new class, which is this PR.

## [c2998030535] orionw (CONTRIBUTOR) · issue_comment · 2025-06-23
Adding one more comment here to reinforce this - this isn't for a simple continued pre-train on top of ModernBERT. 

This is a **brand new from scratch decoder model suite (17m to 1B sizes) trained on 2T of open-data** but that uses the same architecture as ModernBERT in terms of shapes.  It is essentially a new decoder-LM for the library, it **also** just happens to be the same architecture which is great for comparisons.

## [c2166766734] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:285 · 2025-06-25
it should not be that complicated to pass the sliding window cc @Cyrilvallez

## [c2166768703] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:366 · 2025-06-25
There should be 0 code path here. Causal or not causal only depends on the mask, which you are suppose to only create and compute once.

## [c2166771187] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:420 · 2025-06-25
_supports_flex_attention and _supports_sdpa are here for this for the PreTrainedModel

## [c2166772139] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:435 · 2025-06-25
we should split the weights to support loras!

## [c2166772680] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:448 · 2025-06-25
we should put this in the model this way it is re-used by each layer

## [c2166773336] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:477 · 2025-06-25
why do this on every decoder layer?

## [c2166774226] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:499 · 2025-06-25
let's use the `Cache` api that we have in transformers ! this is very restrictive

## [c2166774459] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:516 · 2025-06-25
no longer needed

## [c2166774960] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:529 · 2025-06-25
no needed either!

## [c2166775490] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:555 · 2025-06-25
don't think this is relevant

## [c2166777386] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:577 · 2025-06-25
we have config.layer_type standardized now for this!

## [c2166777923] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:731 · 2025-06-25
not needed

## [c2166778931] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:906 · 2025-06-25
should not be needed if you use the new attention function checkout llama please!

## [c2166784380] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:946 · 2025-06-25
we use GradientChekpointingLayer now@

## [c2958235744] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-06-25
Thanks for the PR!

## [c3012928954] jbdel (NONE) · issue_comment · reply to [c2998030535] · 2025-06-27
Super interested in this PR as well. Sending cheers to @orionw to boost your motivation. 
JB

## [c3013841401] orionw (CONTRIBUTOR) · issue_comment · reply to [c3012928954] · 2025-06-27
Appreciate the support @jbdel! :) 

And thanks @ArthurZucker, sorry I didn't notice the new setup. It definitely looks a lot cleaner with those changes!

I got everything you suggested, so I think it's good to go! Please let me know if there's anything else.

## [c3028291707] orionw (CONTRIBUTOR) · issue_comment · reply to [c2958235744] · 2025-07-02
Hi @ArthurZucker, just a ping if you have a sec :)

## [c3036376197] ArthurZucker (COLLABORATOR) · issue_comment · 2025-07-04
Sorry, reviewing again!

## [c3044916556] orionw (CONTRIBUTOR) · issue_comment · reply to [c3036376197] · 2025-07-07
Thanks @ArthurZucker, but unfortunately I don't see the review. Do you mind double checking?

## [c2192898460] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:1 · 2025-07-08
```suggestion
# Copyright 2025 Answer.AI, LightOn, and contributors, and the HuggingFace Inc. team. All rights reserved.
```

## [c2195934184] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:309 · 2025-07-09
in general we always use dropout but set to 0 the default dropout value, this is simpler

## [c2195935572] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:306 · 2025-07-09
```suggestion
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
```
cf what we have for gemma when there are hybrid attention models!

## [c2195937426] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:381 · 2025-07-09
```suggestion
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
[… 40 lines of code omitted]
```

## [c2195939081] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:401 · 2025-07-09
I don't think this is the way we usually go, in general we recommend using model.forward's compile and using static cache, or something like that. Compile is not supported on mps for example (not well supported)

## [c2195940171] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:399 · 2025-07-09
you are missing the can record output tuples from #39120

## [c2195940724] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:534 · 2025-07-09
are there any models where this i None?

## [c2195941039] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:559 · 2025-07-09
```suggestion
```

## [c2195941460] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:484 · 2025-07-09
```suggestion
    @check_model_inputs
    @auto_docstring
```

## [c2195941729] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:569 · 2025-07-09
```suggestion
```

## [c2195942896] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:617 · 2025-07-09
why are we doing this again? we can just always have sliding window, it should not matter whether this layer is sliding or not, as you only get the sliding window mask at the end of the day

## [c2195943103] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:626 · 2025-07-09
```suggestion
```

## [c2195943987] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:621 · 2025-07-09
you should always create both mapping as they are gonna be used

## [c2195944364] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:666 · 2025-07-09
```suggestion
        hidden_states = self.final_norm(hidden_states)
```

## [c2195944601] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:671 · 2025-07-09
```suggestion
```

## [c2195945108] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:709 · 2025-07-09
inheritance should allow you not to have to write that

## [c2195946258] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:656 · 2025-07-09
same here, not sure this forward is any different from classic models like llama. As I said we don't want to configure compile from the config in general!

## [c2195952691] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:260 · 2025-07-09
this attention is very very standard, so let's make sure it is as close to other as possible. the only diff you hvae is the name of your WO basically, the rest is the same as gemm2, or qwen2, all of there are hybrid

## [c2195953110] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:442 · 2025-07-09
we no longer use this!

## [c2195953619] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:779 · 2025-07-09
same comment about inheritance here!

## [c2195954328] ArthurZucker (COLLABORATOR) · review_comment on tests/models/modernbert_decoder/test_modeling_modernbert_decoder.py:1 · 2025-07-09
Nice thanks!!!

## [c2998254582] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-07-09
Thanks, there are still a few things to adress: 
- from the look of it the main different with say gemma2 is that it does not have sofcap, or with qwen2 I am not sure BUT let's make sure it's standardized!
- have a look at modular Gemma2 for example! 

once this is fixed we can merge! Let's also disable the compilation it is not a good idea in the scope of transformers, and it is something that users can granurally do themselves!

## [c2196339486] orionw (CONTRIBUTOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:709 · reply to [c2195945108] · 2025-07-10
I had to keep the `set_decoder` and `get_decoder` because they aren't inherited (and are needed for the tied weights), but I removed the others.

## [c3003616240] orionw (CONTRIBUTOR) · review COMMENTED · reply to [c2998254582] · 2025-07-10
Thanks so much @ArthurZucker! And for the pointer to the new `@check_model_inputs` decorator.

It is very basic, similar to Gemma3 without the softcap, exactly. I updated it to follow Gemma3 for the sliding windows and otherwise keep Llama's new updates.

Also I love that this PR has dropped 500+ lines of code due to all your recent refactors - amazing!

I think it's ready for a re-review when you get a chance.

## [c3057882775] orionw (CONTRIBUTOR) · issue_comment · reply to [c2998254582] · 2025-07-10
@ArthurZucker I merged in main but it seems like the recent lfm2 merge broke some things unrelated to this. Otherwise all the tests pass.

> FAILED tests/models/lfm2/test_modeling_lfm2.py::Lfm2ModelTest::test_sample_generate - AttributeError: 'Lfm2HybridConvCache' object has no attribute '_seen_tokens'

## [c3059445827] orionw (CONTRIBUTOR) · issue_comment · reply to [c2998254582] · 2025-07-10
Merged in the recent fixes from main, everything passes. Ready @ArthurZucker!

## [c2206850903] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:348 · 2025-07-15
```suggestion
        if layer_idx == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
```

## [c2206851589] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py:411 · 2025-07-15
perfect

## [c3019334049] ArthurZucker (COLLABORATOR) · review APPROVED · 2025-07-15
Perfect thanks a lot for your patience, I am sorry this took so long an kudos! 🤗 Merging !

## [c3073674096] orionw (CONTRIBUTOR) · issue_comment · reply to [c3019334049] · 2025-07-15
Thank you so much @ArthurZucker!! Appreciate your patience as I learn the new stuff :)

---
outcome: state=closed, merged=True
