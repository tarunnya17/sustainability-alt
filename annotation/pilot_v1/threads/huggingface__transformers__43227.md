# #43227 [pull_request] GptOss experts implementation
https://github.com/huggingface/transformers/pull/43227  ·  opened by IlyasMoutawwakil on 2026-01-12

## [body] IlyasMoutawwakil (MEMBER) · opening post · 2026-01-12
# What does this PR do?





[wip] Fixes #43193 @vasqu @mattteochen
[image]


## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [ ] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [ ] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [ ] Did you write any new necessary tests?


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

## [c3743126083] IlyasMoutawwakil (MEMBER) · issue_comment · 2026-01-13
run-slow: gpt_oss

## [c2687454861] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/models/gpt_oss/modeling_gpt_oss.py:78 · 2026-01-13
i avoided transposing the weights and made minimal changes that don't impact how computation is done in eager mode because the slow tests are already failing on main from very long so it's hard to debug.

## [c3657081580] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-13 (no text)

## [c2690318118] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/models/gpt_oss/modeling_gpt_oss.py:78 · reply to [c2687454861] · 2026-01-14
@ArthurZucker keeping the weights like this also avoids problems with megablocks expectations

## [c3660596340] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-14 (no text)

## [c2690443848] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/integrations/moe.py:264 · 2026-01-14
this only happens in tests that load, save and then reload the model on cpu, 
the problem is that `safetensors` adds metadata that's only 8 bytes aligned, causing the model weights to start from an offset of 8 bytes.

## [c3660758436] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-14 (no text)

## [c2690447724] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/integrations/moe.py:264 · reply to [c2690443848] · 2026-01-14
the expert weights not being 16 bytes aligned shouldn't be an issue on cpu though because that constraint only applies to cuda kernels using TMA https://github.com/pytorch/pytorch/issues/172440

## [c3660763006] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-14 (no text)

## [c2690816910] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/gpt_oss/modeling_gpt_oss.py:74 · 2026-01-14
this is the only thing I don't like! 
We could just put it like above ⬆️

## [c2690817869] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/gpt_oss/modeling_gpt_oss.py:67 · 2026-01-14
```suggestion
@use_experts_implementation(transposed_experts=True)
```

## [c2690820370] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/gpt_oss/modeling_gpt_oss.py:87 · 2026-01-14
This is fine, makes it more readable!

## [c3661198429] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-01-14
Looks good! Let's agree on transposed vs not transposed (shape ccould help?)

## [c2690968795] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/moe.py:264 · reply to [c2690443848] · 2026-01-14
Maybe adding the link to the issue? Would be nice for users I think if they encounter this (which shouldn't really happen but you never know)

## [c2690989586] vasqu (COLLABORATOR) · review_comment on src/transformers/models/gpt_oss/modeling_gpt_oss.py:78 · reply to [c2687454861] · 2026-01-14
Do we need to init with `torch.empty`? Just thinking about the case where we would not use `from_pretrained`, these weights would be random (I think)

## [c2691008641] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/moe.py:214 · 2026-01-14
A lot of these functions essentially only either change the order or add a transpose. Could we somehow solve this more elegantly with `functools.partial` or similarly? Just not a fan of all these if-else's

## [c3661383456] vasqu (COLLABORATOR) · review COMMENTED · 2026-01-14
Hmm, so we no longer transpose during weight conversion but do it manually? Not against it but we will need to solve the loading/reverse issue either way, not necessarily here 

Left a few comments but they are mostly aesthetic / nits

## [c3750548254] IlyasMoutawwakil (MEMBER) · issue_comment · reply to [c3661383456] · 2026-01-14
> Not against it but we will need to solve the loading/reverse issue either way, not necessarily here

yes and the transposition can be achieved and works fine (all tests passà with the temporary weight renaming trick, however the biggest problem here is that megablocks expect the transposed format of expert weights so it might make more sense to transpose the other MoEs to be compatible with megablocks (if it's worth the speedup)

## [c2691315017] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/models/gpt_oss/modeling_gpt_oss.py:78 · reply to [c2687454861] · 2026-01-14
not really needed, but I thought that using empty initializer will uncover any uninitialized weights since the real initialization should be done in `_init_weights` iiuc, maybe @Cyrilvallez

## [c3661813490] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-14 (no text)

## [c3750677450] vasqu (COLLABORATOR) · issue_comment · reply to [c3750548254] · 2026-01-14
> the biggest problem here is that megablocks expect the transposed format of expert weights

Argh, yea this will be hard to solve otherwise (would need to rewrite that kernel). Maybe we transpose the other way around for the other models at some point - depends on how much improvement that kernel provides 

On another note, is the transpose expensive on runtime? I would expect slight perf regression having to call transpose here and there (in `grouped_mm` implementation, I see a few).

## [c3753266184] IlyasMoutawwakil (MEMBER) · issue_comment · reply to [c3750677450] · 2026-01-15
> On another note, is the transpose expensive on runtime?

no, the grouped_mm work on strided tensors so the transpose is just a view (no op).

## [c2693492279] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/integrations/moe.py:264 · reply to [c2690443848] · 2026-01-15
@vasqu good news i was able to force 16 bytes alignment using the weight loader.

## [c3664510942] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-15 (no text)

## [c2694061682] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/conversion_mapping.py:348 · reply to [c3750677450] · 2026-01-15
it's seems that the order of operations matters when both quantization conversions and weight conversions are defined. e.g. with gpt_oss if I want to apply the Force16ByteAlignment conversion, it has to be placed after the dequantization otherwise the loader is confused (see output below). @vasqu @ArthurZucker 

```python
device = torch.device("cuda:1")
model_id = "openai/gpt-oss-20b"
model = GptOssForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    dtype=torch.bfloat16,
    quantization_config=Mxfp4Config(dequantize=True),
).eval()
```
results in:
```
Loading weights: 100%|█████████████████████████████████████████████| 363/363 [00:00<00:00, 1003.90it/s, Materializing param=model.norm.weight]
GptOssForCausalLM LOAD REPORT from: openai/gpt-oss-20b
Key                                                   | Status     | 
------------------------------------------------------+------------+-
model.layers.{0...23}.mlp.experts.down_proj_scales    | UNEXPECTED | 
[… 9 lines of code omitted]
```

## [c3665217258] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-15 (no text)

## [c2694974317] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/gpt_oss/modeling_gpt_oss.py:78 · reply to [c2687454861] · 2026-01-15
Will not really help to uncover, but yes it's preferred to have `empty` rather than `zeros` etc (though now that we really init on meta device, it does not really change anything, more for the convention)

## [c3666352059] Cyrilvallez (MEMBER) · review COMMENTED · 2026-01-15 (no text)

## [c2695043382] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/conversion_mapping.py:348 · reply to [c2694061682] · 2026-01-15
also @Cyrilvallez

## [c3666438908] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-15 (no text)

## [c2695181880] Cyrilvallez (MEMBER) · review_comment on src/transformers/conversion_mapping.py:348 · reply to [c2694061682] · 2026-01-15
Yes! Since `get_weight_conversions` is only to DEQUANTIZE it makes sense that it's first. However, as discussed offline, for now it's not possible to match 1 param with 2 converter (i.e. 1 dequant converter, and 1 from the hardcoded mapping). So it means that any model with a mapping registered cannot be dequantized 😭 So in theory you're right it should come first, but as it does not work anyway currently it does not make a difference 😭 Let's still keep this change adding a comment on all that please!

## [c3666615750] Cyrilvallez (MEMBER) · review COMMENTED · 2026-01-15 (no text)

## [c2697477054] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/conversion_mapping.py:348 · reply to [c2694061682] · 2026-01-16
thanks @Cyrilvallez added a comment mostly rewording your explanation !

## [c3669536947] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-16 (no text)

## [c2698696123] vasqu (COLLABORATOR) · review_comment on tests/models/gpt_oss/test_modeling_gpt_oss.py:124 · 2026-01-16
I think it's still important to have the test running just to be sure, so I added a kwarg for cases where we do not modify keys

## [c2698697573] vasqu (COLLABORATOR) · review_comment on tests/test_modeling_common.py:4515 · 2026-01-16
This is what I meant

## [c2698700540] vasqu (COLLABORATOR) · review_comment on src/transformers/models/gpt_oss/modular_gpt_oss.py:64 · 2026-01-16
Nice, the additional bias is a good addition!

## [c2698703690] vasqu (COLLABORATOR) · review_comment on src/transformers/conversion_mapping.py:366 · 2026-01-16
```suggestion
        # NOTE: Since get_weight_conversions() only serves to dequantize, we need to put them first in the list.
        # However, for now it's not possible to match 1 param with 2 converters (i.e. 1 dequantization converter
        # and 1 model-specific converter). Which means that if a model that has model-specific conversions and is being
        # dequantized, the model-specific conversion that has patterns matching the dequantization patterns will be ignored.
```

## [c2698709029] vasqu (COLLABORATOR) · review_comment on src/transformers/core_model_loading.py:445 · 2026-01-16
Wdyt about giving an explicit use case? I.e. `torch._grouped_mm` - might come off weird otherwise what's behind this converter

## [c2698712503] vasqu (COLLABORATOR) · review_comment on src/transformers/core_model_loading.py:455 · 2026-01-16
I copied the fix on transpose; what I'm unsure about is whether we always clone or not (in the original version I saw it being cloned either way, might be wrong)

## [c2698715879] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/moe.py:111 · 2026-01-16
Trusting you on this, I always feel a bit unsafe with `view` xD

## [c2698717901] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/moe.py:122 · 2026-01-16
This is from e.g. deepseek v2? Or at least before it was fixed

## [c2698724086] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/moe.py:251 · 2026-01-16
Did it change memory requirements? But honestly I rather have a torch compile friendly version either way

## [c2698725543] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/moe.py:257 · 2026-01-16
Any reason to shorten this, let's keep it as fullname if possible

## [c2698729279] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/moe.py:337 · 2026-01-16
This makes it a bit hidden, can we move this a bit outside?

## [c3671104319] vasqu (COLLABORATOR) · review COMMENTED · 2026-01-16
I added some changes to the conversion + test, so please take another look there 

Most comments are more nits about style / fmi

## [c2698740473] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/core_model_loading.py:455 · reply to [c2698712503] · 2026-01-16
ah yes it's a typo, only clone if non-aligned

## [c3671157579] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-16 (no text)

## [c2698745415] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/integrations/moe.py:122 · reply to [c2698717901] · 2026-01-16
yep exactly

## [c3671162862] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-16 (no text)

## [c2698753741] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/integrations/moe.py:251 · reply to [c2698724086] · 2026-01-16
no it's the same memory footprint, it's also sometimes slower because the torch.unique overhead.
passing all experts seems to be the intended way, the kernel takes care of loading active experts to memory.

## [c3671172075] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-16 (no text)

## [c2698756996] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/integrations/moe.py:257 · reply to [c2698725543] · 2026-01-16
because grouped_mm names it offs, no opinion about it, reverting to full name

## [c3671175848] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-16 (no text)

## [c2698757914] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/integrations/moe.py:337 · reply to [c2698729279] · 2026-01-16
makes sense ! moving

## [c3671176874] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-16 (no text)

## [c2698759698] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/models/gpt_oss/modular_gpt_oss.py:64 · reply to [c2698700540] · 2026-01-16
yeah better be explicit than rely on getattrs

## [c3671178969] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-16 (no text)

## [c2698762834] IlyasMoutawwakil (MEMBER) · review_comment on tests/models/gpt_oss/test_modeling_gpt_oss.py:124 · reply to [c2698696123] · 2026-01-16
makes sense ! thanks

## [c3671182548] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-16 (no text)

## [c2698825751] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/core_model_loading.py:445 · reply to [c2698709029] · 2026-01-16
added

## [c3671252131] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-16 (no text)

## [c3671399941] vasqu (COLLABORATOR) · review APPROVED · 2026-01-16
LGTM, I guess the outputs slightly change now with the interface's `grouped_mm/batched_mm` or even the eager rewrite?

## [c3760587085] IlyasMoutawwakil (MEMBER) · issue_comment · 2026-01-16
> LGTM, I guess the outputs slightly change now with the interface's grouped_mm/batched_mm or even the eager rewrite?

the eager for training is the same, I kept everything as is when i realized that the slow tests are brokens (see https://github.com/huggingface/transformers/pull/43246), I removed the batched implementation from there as it is redundant, so users will have to set experts impl to batched to get the same eval output as before.

## [c3760934496] vasqu (COLLABORATOR) · issue_comment · 2026-01-16
Probably merging next week, currently don't have the permission to merge with red CI :smile:

## [c2703790287] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/integrations/moe.py:111 · reply to [c2698715879] · 2026-01-19
> When possible, the returned tensor will be a view of input

you are right ! they should be equivalent if tensor mem is contiguous (when possible) so reverting to reshapes !

## [c3676938341] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-19 (no text)

## [c2704139226] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/conversion_mapping.py:367 · 2026-01-19
I am not sure it makes a difference at all no? Because the operation are ordered by length of collected tensors I thing.

## [c2704140056] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/core_model_loading.py:447 · 2026-01-19
very nice 🫡

## [c3677353021] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-01-19
Great work! LGTM, lets make sure quantized tests pass but let's go!

## [c2704398322] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/conversion_mapping.py:367 · reply to [c2704139226] · 2026-01-19
Yeah now that I'm digging deeper into the weight loader I think the reason I got the error above is because it's not possible to cascade converters (i.e., applying model-specific conversions on top of tensors created by the dequantization conversions). Not because you can't match one tensor with two converters (that's a valid limitation, but not the one happening here in gpt oss).

I added $ to the end of Force16BytesAlignement source pattern to fix my error without changing this order. Basically making sure that the sources of the mxfp dequant converter and Force16BytesAlignement are exclusive.
I will revert the line change and make this comment instead:
```py
        # NOTE: Since get_weight_conversions() only serve to dequantize, we normally want to apply them first.
        # However, for now it's not possible to cascade converters (i.e., applying model-specific conversions on top
        # of tensors created by the dequantization conversions)
        # This means that if a model has model-specific conversions and is being dequantized, the model-specific conversion
        # that relies on tensors created by dequantization conversions will not be applied.
        # GptOss example: with Mxfp4Config(dequantize=True), Force16BytesAlignment converters are ignored because the tensors
        # "mlp.experts.gate_up_proj$" and "mlp.experts.down_proj$" are only created after dequantization conversions are applied.
        weight_conversions.extend(hf_quantizer.get_weight_conversions())
```

## [c3677659532] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-01-19 (no text)

## [c3769680124] IlyasMoutawwakil (MEMBER) · issue_comment · 2026-01-19
Should be good to merge

## [c3770010852] vasqu (COLLABORATOR) · issue_comment · 2026-01-19
#43353 leads me to believe that a simple weight converter won't be enough. Somehow under the concurrent workers we hit the same 16byte alignment issue on our CI with (qwen 3 omni moe).

I'll take a proper look tomorrow but it might make sense to add the fallback regardless sadly. Just wanted to put it out before we hastly merge

## [c3771483139] IlyasMoutawwakil (MEMBER) · issue_comment · reply to [c3770010852] · 2026-01-20
@vasqu I think what you are dealing with might be the cpu+safetensors memory misalignement issue i have seen with gptoss ; there are two kinds of misalignment issues i have encountered so far:
- on cuda: this only happens if the strided tensor has shapes that are not divisible by 8 (for fp16/bf16) or 4 (for fp32), otherwise the cuda allocator will allocate an aligned tensor. That's the issue I was solving in the first experts PR, because I was testing on the dgx so tests were automatically running on cuda.
- on cpu: Safetensors adds an 8 bytes-aligned metadata on top of the file ([see this issue from which the 8 bytes were decided](https://github.com/huggingface/safetensors/issues/231)),  so when reloaded on cpu the metadata can "misalign" the tensors in the memmaped file. This is what the weight converter defined here solves.

I would suggest you try the weight converter defined here because the ci runner uses cpu (I had to set TRANSFORMERS_TEST_DEVICE=cpu to reproduce the gpt_oss issue locally)

## [c3773490474] IlyasMoutawwakil (MEMBER) · issue_comment · reply to [c3770010852] · 2026-01-20
To sum up the conversation I had with @vasqu internally:

> All moes that have a converter that creates new tensors, for example all the ones that merge their experts using cat/stack, will create 16-bytes aligned tensors on cpu. This is because they will use the allocator to create *new* merged tensors, and [the allocator will return a 16-bytes aligned memory](https://stackoverflow.com/questions/70692795/why-is-malloc-16-byte-aligned). 
If any MoE fails because of 16 bytes misalignment, it's gonna be because it's not allocating its tensors ; for example using safetensors memmaped tensors (which are only 8 bytes aligned). Those are the ones that will need the Force16BytesAlignement converter.

## [c2709294210] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/conversion_mapping.py:367 · reply to [c2704139226] · 2026-01-20
cool, as we talked offline let's add more comments about how to handle the weight converter

## [c3683309887] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-01-20 (no text)

## [c3691493635] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-01-22 (no text)

---
outcome: state=closed, merged=True
