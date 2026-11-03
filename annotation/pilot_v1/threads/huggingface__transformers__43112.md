# #43112 [pull_request] SINQ quantization strategy integration (adapted for Transformers V5)
https://github.com/huggingface/transformers/pull/43112  ·  opened by ChiaraBoretti on 2026-01-05

## [body] ChiaraBoretti (CONTRIBUTOR) · opening post · 2026-01-05
# What does this PR do?
**Summary**
This PR introduces an integration of the [SINQ](https://huggingface.co/papers/2509.22944) quantization method into the Hugging Face Transformers library. It follows the pattern of existing quantization integrations such as HQQ, AWQ, and others.
The goal is to enable users to configure and apply SINQ quantization directly through model configuration parameters, which can then be passed via the ```from_pretrained()``` function within the Transformers framework.

**Example Usage**
The pipeline then will be the following:
```bash
cfg = SinqConfig(
    nbits=4,
    group_size=64,
    tiling_mode="1D",
    method="sinq",     
[… 54 lines of code omitted]
```

## [c2661776926] SunMarc (MEMBER) · review_comment on docs/source/en/quantization/sinq.md:57 · 2026-01-05
It should be fine to remove this as we should already have this information when calling `from_pretrained`

## [c2661779405] SunMarc (MEMBER) · review_comment on docs/source/en/quantization/sinq.md:59 · 2026-01-05
same for this one, the user can just pass `device_map` in `from_pretrained`.

## [c2661781229] SunMarc (MEMBER) · review_comment on docs/source/en/quantization/sinq.md:83 · 2026-01-05
`torch_dtype` is deprecated in favor of `dtype` 
```suggestion
    dtype=torch.bfloat16
```

## [c2661784275] SunMarc (MEMBER) · review_comment on docs/source/en/quantization/sinq.md:103 · 2026-01-05
```suggestion
model.push_to_hub("HF_Hub_username/qwen3-1.7B-sinq-4bit")
```

## [c2661785915] SunMarc (MEMBER) · review_comment on docs/source/en/quantization/sinq.md:119 · 2026-01-05
```suggestion
# Push to the Hub
qmodel.push_to_hub("HF_Hub_username/qwen3-1.7B-sinq-4bit")
```

## [c2661790054] SunMarc (MEMBER) · review_comment on docs/source/en/quantization/sinq.md:100 · 2026-01-05
instead of asking users to use this, we can move this code into hf_quantizer

## [c2661792953] SunMarc (MEMBER) · review_comment on docs/source/en/quantization/sinq.md:182 · 2026-01-05
it doesn't work with `device_map` ?

## [c2661793985] SunMarc (MEMBER) · review_comment on doctest_list.txt:1 · 2026-01-05
to remove

## [c2661795228] SunMarc (MEMBER) · review_comment on src/transformers/integrations/sinq.py:1 · 2026-01-05
add license 
```suggestion
```

## [c2661802441] SunMarc (MEMBER) · review_comment on src/transformers/integrations/sinq.py:64 · 2026-01-05
is it really needed ? In which cases it happens ?

## [c2661806002] SunMarc (MEMBER) · review_comment on src/transformers/integrations/sinq.py:72 · 2026-01-05
same for this one, I don't think we need this. It means that `param_needs_quantization` was not correctly coded.

## [c2661894226] SunMarc (MEMBER) · review_comment on src/transformers/integrations/sinq.py:83 · 2026-01-05
you can just use the device of the tensor as it should be loaded on the correct one, same for the dtype.

## [c2661907830] SunMarc (MEMBER) · review_comment on src/transformers/integrations/sinq.py:110 · 2026-01-05
Is there really no other way to make this work ? I see that you took a lot of inspiration from hqq code but it's a nightmare to actually maintain it. Right now, even our hqq integration is broken due to v5 and I don't know if we will ever fix it. Part of the reason is that the api is too complicated / hard to maintain. Is it possible to think of a simpler api like we have for fp8/mxfp4 where we just replace the linear beforehand and then we just correctly set the params/buffer here ?

## [c2661915945] SunMarc (MEMBER) · review_comment on src/transformers/integrations/sinq.py:169 · 2026-01-05
probably not needed same as above

## [c2661928678] SunMarc (MEMBER) · review_comment on src/transformers/integrations/sinq.py:159 · 2026-01-05
If there is indeed only one value, we should just return that as it means that we didn't collected sinq parameters as expected. We should also remove the warning. This is what we do for finegrained_fp8

## [c2661929681] SunMarc (MEMBER) · review_comment on src/transformers/integrations/sinq.py:191 · 2026-01-05
same as above, we shouldn't need this

## [c2661937279] SunMarc (MEMBER) · review_comment on src/transformers/integrations/sinq.py:225 · 2026-01-05
Same as above, let's see if there is a way to simplify this.

## [c2661938199] SunMarc (MEMBER) · review_comment on src/transformers/integrations/sinq.py:199 · 2026-01-05
no needed potentially !

## [c2661943322] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:145 · 2026-01-05
we don't use this anymore 
```suggestion
```

## [c2661945788] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:151 · 2026-01-05
not needed 
```suggestion
```

## [c2661956734] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:169 · 2026-01-05
We don't do unsafe serialization anymore. Until sinq supports it, let's not enable serialization.

## [c2661959786] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:181 · 2026-01-05
let's do this check in `update_dtype` method instead

## [c2661962069] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:192 · 2026-01-05
I prefer not to do that if possible. As I said somewhere, you can directly use the device of each loaded tensors

## [c2661972261] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:207 · 2026-01-05
let's simplify this also, this is way too complex to just check values inside the `device_map`.

## [c2661976710] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:216 · 2026-01-05
let's not do that and just check if the torch_dtype passed is correct or not.

## [c2661978841] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:220 · 2026-01-05
we don't to care about BC for `update_torch_dtype`, let's just use `update_dtype`

## [c2661980870] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:225 · 2026-01-05
remove

## [c2661984160] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:46 · 2026-01-05
let's import the relevant lib in the corresponding method and not buckle everything here

## [c2661987604] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:36 · 2026-01-05
this should happen in validate_environment. Can you create a `is_sinq_available` method like the other quant methods ?

## [c2661991804] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:253 · 2026-01-05
We already have a utility function called `should_convert_module` . You can just reuse this

## [c2661996921] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:264 · 2026-01-05
we don't need a getattr + lower() , `cfg` should always have a `method` and the value is either `sinq` or `asinq`. Also just use `self.quantization_config`

## [c2661998974] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:270 · 2026-01-05
why ?

## [c2662004532] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:315 · 2026-01-05
we only need to pass target_patterns=".weight"

## [c2662007145] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:339 · 2026-01-05
simplify

## [c2662011176] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:367 · 2026-01-05
simplify

## [c2662027553] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:378 · 2026-01-05
don't think this is needed

## [c2662027556] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:371 · 2026-01-05
I think it will be better to leave methods that requires calibration to sinq lib. We prefer not to support these methods and usually ask users to use directly the original lib e.g. AutoAWQ or or AutoRound

## [c2662045652] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:374 · 2026-01-05
I don't understand why some some arch, it is preferable to do all at once. It should be the same from what I see in the code. If we have to do all at once, it's might better to ask users to use sinq lib to quantize.

## [c2662047360] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:405 · 2026-01-05
we can simplify a bunch of things

## [c2662053228] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:418 · 2026-01-05
as said before, i prefer to redirect users to sinq lib if they want to calibrate models

## [c2662080162] SunMarc (MEMBER) · review_comment on src/transformers/utils/quantization_config.py:2146 · 2026-01-05
let's remove this

## [c2662082338] SunMarc (MEMBER) · review_comment on src/transformers/utils/quantization_config.py:2158 · 2026-01-05
let's not upcast here directly and check in the post init

## [c2662083903] SunMarc (MEMBER) · review_comment on src/transformers/utils/quantization_config.py:2161 · 2026-01-05
let's leave it as `None`

## [c2662087048] SunMarc (MEMBER) · review_comment on src/transformers/utils/quantization_config.py:2218 · 2026-01-05
`to_dict` should work fine no ?

## [c2662087889] SunMarc (MEMBER) · review_comment on src/transformers/utils/quantization_config.py:2192 · 2026-01-05
not neeedd

## [c2662089758] SunMarc (MEMBER) · review_comment on src/transformers/utils/quantization_config.py:2234 · 2026-01-05
not sure what we are doing here

## [c2662090565] SunMarc (MEMBER) · review_comment on test_preparation/pipelines_torch_test_list.txt:1 · 2026-01-05
remove

## [c3627171791] SunMarc (MEMBER) · review COMMENTED · 2026-01-05
Thanks for the work, really appreciate to see that you adapted your work to fit the v5 !  Left a bunch of comments. I see that sinq api is quite similar to hqq and it would be great if we can improve a bit the api so that the integration is simpler and easier to maintain in general ! We had a lot of issues in our hqq integration and I prefer that we don't go through that again ;) The best would be to take inspiration from other quants methods like bnb, fp8, mxfp4 or torchao. Also, I would be great to reduce the complexity of this PR by simplifying as much as possible. You will see that the other integrations are much more easy to go through which makes the maintenance and usability in the future much better. Feel free to ask any questions, I'm really eager to merge this PR !

## [c3723112048] ChiaraBoretti (CONTRIBUTOR) · issue_comment · reply to [c3627171791] · 2026-01-08
Hi @SunMarc,

Thank you very much for taking the time to review my code and for all your helpful comments. I’ve just pushed an updated version that incorporates several of the changes you suggested.
That said, I still have a few questions, and I ran into some difficulties while trying to implement a couple of your recommendations. I was hoping you might be able to help me with those points.
I’ll reply directly to your specific comments with my questions. I’m keen to further improve the code and, with your guidance, get this pull request ready to be merged.

Thanks again for your support, and I really appreciate your help!

## [c2671726314] ChiaraBoretti (CONTRIBUTOR) · review_comment on docs/source/en/quantization/sinq.md:100 · reply to [c2661790054] · 2026-01-08
I agree that your approach would be better overall. I also tried to implement your suggestion, but I ran into a few issues along the way.
In particular, I moved the relevant lines in quantizer_sinq.py to the beginning of the file so that my custom io_patch is applied as soon as the quantizer module is imported. While this works correctly for saving models, I’m encountering problems when loading already Sinq-quantized models.

At the moment, I’m working on a workaround where, during installation of the sinq package via pip, an autopatch is applied that automatically calls patch_hf_pretrained_io(). However, before going too far in that direction, I wanted to ask whether you might have any hints or suggestions on how to better implement your original approach.

## [c3638704025] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-01-08 (no text)

## [c2671752528] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/integrations/sinq.py:110 · reply to [c2661907830] · 2026-01-08
Honestly, I tried to adapt my code to follow the FP8-style approach, but I wasn’t able to make it work successfully. My goal is to avoid modifying the original SINQ codebase, and the current implementation is the only one I’ve found that works reliably across all architectures (both pure LLMs and multimodal models) without touching the official SINQ code.
This is the part where I feel your help is really needed. If you have any suggestions or alternative ideas in mind, I’d be more than happy to try them out.

Thanks again for your time and support!

## [c3638732661] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-01-08 (no text)

## [c2671761616] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:371 · reply to [c2662027556] · 2026-01-08
I’m not sure I fully understand your suggestion. Are you proposing to remove the asinq option and instead ask users to rely on the official SINQ repository if they want to use the calibrated variant of SINQ?
I just want to make sure I’m interpreting your point correctly before moving forward.

## [c3638742879] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-01-08 (no text)

## [c2671847022] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:374 · reply to [c2662045652] · 2026-01-08
This is mainly because the current SINQ implementation works very well for pure LLM architectures, where parameter-level conversion is stable and behaves as expected. However, SINQ has not yet been fully tested on multimodal models.
I ran several experiments across different multimodal model families. While quantization of the linear layers generally works, I encountered issues with parameter-level conversion for certain architectures. For this reason, I adopted a hybrid approach: for pure LLMs, I rely on parameter-level conversion, which is well tested and robust; for more “risky” architectures (particularly multimodal models) I instead fall back to an all-at-once conversion strategy, which has proven to be more reliable in practice.

I arrived at this solution after testing models from the Gemma family. As I understand it, the root cause is that during ```from_pretrained()```, Transformers still calls the model’s ```_init_weights``` method on modules it considers missing or uninitialized. In models like Gemma3, specifically within its SigLIP component, ```_init_weights``` performs operations such as:

```init.xavier_uniform_(module.q_proj.weight)```

At that point, however, ```q_proj``` has already been replaced with SINQLinear. Since SINQLinear does not expose a ```.weight``` attribute, the initialization step fails.

## [c3638833734] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-01-08 (no text)

## [c2686531824] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:374 · reply to [c2662045652] · 2026-01-13
> I arrived at this solution after testing models from the Gemma family. As I understand it, the root cause is that during from_pretrained(), Transformers still calls the model’s _init_weights method on modules it considers missing or uninitialized. In models like Gemma3, specifically within its SigLIP component, _init_weights performs operations such as:
> 
> init.xavier_uniform_(module.q_proj.weight)
> 
> At that point, however, q_proj has already been replaced with SINQLinear. Since SINQLinear does not expose a .weight attribute, the initialization step fails.
> 

You can skip the initialization for a specific parameter or even modules by setting `_is_hf_initialized` attribute ! We usually set it when we load the weights (e.g check the other quant methods) but we can also do it directly when we replace the modules if this is simpler/better.

## [c3655914479] SunMarc (MEMBER) · review COMMENTED · 2026-01-13 (no text)

## [c2686535288] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:371 · reply to [c2662027556] · 2026-01-13
Yeah that's right. It will be better ! Maybe in the future it could make sense to add it, but for now let's remove this !

## [c3655919181] SunMarc (MEMBER) · review COMMENTED · 2026-01-13 (no text)

## [c2686588240] SunMarc (MEMBER) · review_comment on src/transformers/integrations/sinq.py:110 · reply to [c2661907830] · 2026-01-13
A few things that could be nice to simplify if possible: 
- Can't we replace the linear with the `SINQLinear` before instead of here ? 
- Do we really have to pass the linear layer in `SINQLinear` ? We can maybe recreate a `SINQLinear` impl that is a bit more simpler ? 
- When does the quantization happens ? When the SINQLinear is created ? Maybe we can try to have a bit more flexibility here and create a quantize() method.

## [c3655993124] SunMarc (MEMBER) · review COMMENTED · 2026-01-13 (no text)

## [c3744535844] SunMarc (MEMBER) · issue_comment · 2026-01-13
Thanks for taking the time to go through the comments, answered to a couple of them !

## [c2690627434] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:374 · reply to [c2662045652] · 2026-01-14
Thank you for the suggestion about using `_is_hf_initialized`! After further investigation, I believe that the hybrid approach is not always required for multimodal models. In my understanding, whether it is needed depends largely on how `_init_weights` is implemented for a given architecture.
In the case of Gemma3, I ran into issues due to differences between how the language model and the vision encoder (SigLIP) handle weight initialization.
The language model in Gemma3 defines `_init_weights` in a module-level way, checking individual submodules:
```
if isinstance(module, nn.Linear):
    ...
```
and with param-level conversion:
- Each nn.Linear is replaced by SINQLinear
[… 24 lines of code omitted]
```

## [c3660976130] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-01-14 (no text)

## [c2695068775] SunMarc (MEMBER) · review_comment on docs/source/en/quantization/sinq.md:100 · reply to [c2661790054] · 2026-01-15
Can you add it in is_serializable method, so that it is only called when we want to actually save the model ?

## [c3666470593] SunMarc (MEMBER) · review COMMENTED · 2026-01-15 (no text)

## [c2695126162] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:374 · reply to [c2662045652] · 2026-01-15
Oh okay I see the issue. Thanks for digging into this, a lot of things changed in the v5 and initialization was one big issue that still needs some minor fixes to work with quantization. But basically, if i understood well, for module.q_proj.weight is not available for `SINQLinear` unlike a classic `Linear`. Usually, `_is_hf_initialized` is initialized at the level of the parameters and we skip the initialization if the param has this param. But since here, we can't even access it.

```python
def zeros_(tensor: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return TORCH_INIT_FUNCTIONS["zeros_"](tensor)
    return tensor
[… 9 lines of code omitted]

## [c3666544337] SunMarc (MEMBER) · review COMMENTED · 2026-01-15 (no text)

## [c3755762772] ChiaraBoretti (CONTRIBUTOR) · issue_comment · reply to [c3666544337] · 2026-01-15
Hi @SunMarc,

I’ve just pushed an updated version of the code with the following changes:

- Removal of asinq. I removed the asinq implementation and added an explicit error message when a user tries to set `method='asinq'`. The message directs users to the official SINQ repository for running the calibrated version of SINQ.

- Architecture-specific handling. I answered your comment regarding handling particular architectures with an all-in-one approach. For now, I’ve moved away from the all-in-one strategy. As mentioned in my answer, I believe the issue is model-specific rather than a general multimodal architecture problem. Based on this, I opted to explicitly specify the vision components in `modules_to_not_convert` for models that raise errors due to their `init_weights` implementation.

- Updated quantization flow. I reworked the quantization flow by modifying how SINQLinear is implemented, following your suggestion (I hope to understand them correctly). The new flow is: Create an empty SINQLinear -> Load the weights -> Explicitly call the weights quantization via a dedicated `quantize()` function instead of performing quantization when the SINQLinear is created (in the `__init__` of the class).
As a result, also the SinqQuantize and SinqDeserialize classes are now much simpler.

Please let me know if these changes are moving the PR in the right direction for approval, and if there’s anything else I should address.

Thanks!

## [c3755820990] SunMarc (MEMBER) · issue_comment · reply to [c3755762772] · 2026-01-15
Thanks a lot for this ! 

> 
Architecture-specific handling. I answered your comment regarding handling particular architectures with an all-in-one approach. For now, I’ve moved away from the all-in-one strategy. As mentioned in my answer, I believe the issue is model-specific rather than a general multimodal architecture problem. Based on this, I opted to explicitly specify the vision components in modules_to_not_convert for models that raise errors due to their init_weights implementation.

Answered in the thread ! But this is a separate issue from this PR and it should be quite simple to fix. 

> Updated quantization flow. I reworked the quantization flow by modifying how SINQLinear is implemented, following your suggestion (I hope to understand them correctly). The new flow is: Create an empty SINQLinear -> Load the weights -> Explicitly call the weights quantization via a dedicated quantize() function instead of performing quantization when the SINQLinear is created (in the __init__ of the class).
> As a result, also the SinqQuantize and SinqDeserialize classes are now much simpler.

I had a quick look at it looks much better. The flow also makes more sense now.

Let me know if you want a new review but the simpler the code looks like, the better it is !

## [c2695178293] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:283 · 2026-01-15
put that in a replace function like the other methods

## [c3666611234] SunMarc (MEMBER) · review COMMENTED · 2026-01-15 (no text)

## [c2695179986] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:312 · 2026-01-15
put that in validate_environment

## [c3666613563] SunMarc (MEMBER) · review COMMENTED · 2026-01-15 (no text)

## [c2695180275] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:329 · 2026-01-15
remove

## [c3666613956] SunMarc (MEMBER) · review COMMENTED · 2026-01-15 (no text)

## [c2695181348] SunMarc (MEMBER) · review_comment on test_preparation/pipelines_torch_test_list.txt:7 · 2026-01-15
lot of txt files to detele

## [c3666615119] SunMarc (MEMBER) · review COMMENTED · 2026-01-15 (no text)

## [c2704079355] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:283 · reply to [c2695178293] · 2026-01-19
Did it! I added a ```replace_with_sinq_linear()``` function in sinq.py

## [c3677284546] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-01-19 (no text)

## [c2704080178] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:312 · reply to [c2695179986] · 2026-01-19
moved to validate_environment()

## [c3677285530] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-01-19 (no text)

## [c2704081798] ChiaraBoretti (CONTRIBUTOR) · review_comment on test_preparation/pipelines_torch_test_list.txt:7 · reply to [c2695181348] · 2026-01-19
I deleted all these txt files

## [c3677287500] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-01-19 (no text)

## [c2704083188] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:329 · reply to [c2695180275] · 2026-01-19
Thanks! I removed this function since it is no more used in the code.

## [c3677289111] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-01-19 (no text)

## [c3767503100] ChiaraBoretti (CONTRIBUTOR) · issue_comment · reply to [c3666615119] · 2026-01-19
Thanks, @SunMarc, for your comments!

I’ve submitted a new version of the code, could you please take a look and let me know if there’s anything else I should change? I think that your suggestions helped me in making it simpler and cleaner.

Thanks again!

## [c3817657380] philippebich (NONE) · issue_comment · reply to [c3666615119] · 2026-01-29
Hi @SunMarc ! Any news?

## [c2742197986] SunMarc (MEMBER) · review_comment on docs/source/en/quantization/sinq.md:59 · reply to [c2661779405] · 2026-01-29
can't be not put the device here ?

## [c2742230848] SunMarc (MEMBER) · review_comment on src/transformers/integrations/sinq.py:158 · 2026-01-29
add a comment above to explain this path

## [c2742235863] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:96 · 2026-01-29
the safe_serialization=False option was removed. So just return True

## [c2742242653] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:134 · 2026-01-29
don't do that here, there is a method for that called update_torch_dtype I think. You won't need to check for str as it will be a torch.dtype

## [c2742250318] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:143 · 2026-01-29
we don't need `_flatten_device_map`. Just simplify to use the same check as the other methods.

## [c2742272532] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:283 · 2026-01-29
You can use the value from device_map actually instead of using the one passed in quantization_config. So it should be easy to remove. Note that `_process_model_before_weight_loading` is actually performed under the with torch.device("meta") context manager, so all the tensors are on meta device actually

## [c2742276382] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:261 · 2026-01-29
Can't we put that in `is_serializable` instead of here ? Also in the docs, you are still asking the user to do it

## [c2742278433] SunMarc (MEMBER) · review_comment on src/transformers/utils/quantization_config.py:2140 · 2026-01-29
to remove

## [c2742282620] SunMarc (MEMBER) · review_comment on src/transformers/utils/quantization_config.py:2174 · 2026-01-29
don't convert it here. Otherwise, we won't be able to add good default like the lm_head that shouldn't be quantized

## [c2742283692] SunMarc (MEMBER) · review_comment on src/transformers/utils/quantization_config.py:2196 · 2026-01-29
not needed

## [c2742289164] SunMarc (MEMBER) · review_comment on tests/quantization/sinq/test_sinq.py:180 · 2026-01-29
we have specific decorator for this. use those instead of this one

## [c2742302797] SunMarc (MEMBER) · review_comment on tests/quantization/sinq/test_sinq.py:20 · 2026-01-29
those tests are good but it would be easier for us to try to respect the same structure as the other quantization tests. It will help us better maintain it

## [c2742308891] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:216 · 2026-01-29
this is where the good default are added

## [c3723401015] SunMarc (MEMBER) · review COMMENTED · 2026-01-29
Thanks, just a few more fixes and we should be good to merge. Thanks for spending lots of time on this PR ! Sorry I took a break hence I didn't review the PR since then.

## [c2745672660] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:283 · reply to [c2742272532] · 2026-01-30
I’m still a bit confused about this point. I tried removing the device specification from SinqConfig and instead relying directly on the device_map argument provided to from_pretrained(). However, this only works correctly when device_map=None.
If I set device_map to a CUDA device, the full-precision model gets loaded onto the GPU first, which is exactly what I want to avoid. My goal is for only the final quantized model to be loaded onto the user-specified GPU.

This is the change I introduced, (notice that with this I cannot give the user the possibility to specify where to locate the final quantized model):
```python
        # Get device from device_map
        if isinstance(device_map, dict):
            device_str = next(iter(device_map.values()), "cuda:0")
        else:
            device_str = "cuda:0"
[… 14 lines of code omitted]
```

## [c3727579305] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-01-30 (no text)

## [c2745682438] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:261 · reply to [c2742276382] · 2026-01-30
I tried putting it in ```is_serializable```, but it doesn't work on the first save. The issue is that ```is_serializable()``` is called from within ```save_pretrained()```, so by that point we're already executing the unpatched method. The patch only takes effect for subsequent saves. Moving it to ```_process_model_after_weight_loading``` ensures the patch is applied immediately after model loading, before any save attempt.

## [c3727589798] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-01-30 (no text)

## [c2758886448] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:261 · reply to [c2742276382] · 2026-02-03
sounds good then !

## [c3745047727] SunMarc (MEMBER) · review COMMENTED · 2026-02-03 (no text)

## [c2758909024] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:283 · reply to [c2742272532] · 2026-02-03
> If I set device_map to a CUDA device, the full-precision model gets loaded onto the GPU first, which is exactly what I want to avoid. My goal is for only the final quantized model to be loaded onto the user-specified GPU.

Indeed, the weights are loaded to the GPU but as soon as they are loaded, they should be converted thanks to your `SinqQuantize` ops, thus decreasing the memory. Since we are doing this async and with threads, it should not load all the weights then quantize but rather, load some weights -> quantize and so on. We tested with different methods like bnb and there was no issues. Maybe check on your side that `module.quantize(weight_tensor)` decreases the memory when `weight_tensor` is on a CUDA. Let me know if this is clear !

## [c3745073016] SunMarc (MEMBER) · review COMMENTED · 2026-02-03 (no text)

## [c2759271649] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:283 · reply to [c2742272532] · 2026-02-03
Yes, this makes more sense now, thanks!
However, I also tried the bitsandbytes (bnb) quantization strategy using the following simple script:
```python
qmodel_name = "Qwen/Qwen3-4B"
from transformers import BitsAndBytesConfig

cfg = BitsAndBytesConfig(load_in_4bit=True)
qtok = AutoTokenizer.from_pretrained(qmodel_name)
[… 14 lines of code omitted]
```

## [c3745490496] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-02-03 (no text)

## [c2759647192] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:283 · reply to [c2742272532] · 2026-02-03
For bnb, this is normal, we coded it this way and kept it this way due to prior limitation. For SINQ, I don't think this should happen.

## [c3745940646] SunMarc (MEMBER) · review COMMENTED · 2026-02-03 (no text)

## [c2759649532] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:283 · reply to [c2742272532] · 2026-02-03
see what we have in bnb 

```python
 def update_device_map(self, device_map):
        if device_map is None:
            if torch.cuda.is_available():
                device_map = {"": torch.cuda.current_device()}
[… 14 lines of code omitted]

## [c3745943351] SunMarc (MEMBER) · review COMMENTED · 2026-02-03 (no text)

## [c2759836824] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:283 · reply to [c2742272532] · 2026-02-03
Ah, okay, thanks that clears things up.

Is this behavior also expected when using fine-grained FP8 quantization? I noticed that in the code there are some checks on the values of ```device_map```, but nothing that explicitly sets it when it is None.

With this type of quantization as well, when I explicitly specify ```device_map```, it looks like the full-precision model is first loaded onto the GPU (~8 GB), very similar to what happens with SINQ.
Is this the intended behavior, or am I misunderstanding how fine-grained FP8 loading is supposed to work?

Thanks again!

## [c3746156389] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-02-03 (no text)

## [c2769381191] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:283 · reply to [c2742272532] · 2026-02-05
Oh you need to remove the cache allocator most probably as it makes you think that you are using more memory. You need to check for cuda allocated memory and not reserved memory. This is what I get with 

```
  ┌────────────────────────┬─────────┐
  │         Metric         │  Value  │
  ├────────────────────────┼─────────┤
  │ Model memory footprint │ 2.42 GB │
[… 45 lines of code omitted]

## [c3757365610] SunMarc (MEMBER) · review COMMENTED · 2026-02-05 (no text)

## [c2781606497] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:283 · reply to [c2742272532] · 2026-02-09
Ah ok now I get it! thanks. I checked the allocated memory after applying the Sinq quantization, and it’s around 2.7 GB, consistent with what you observed using bnb.

I’ve pushed a new version of the code where I removed the device specification from the SinqConfig and now rely on device_map instead. I also applied the other minor adjustments you suggested.
Could you please take a look and let me know if everything seems good to go, or if I may have missed anything?

Thanks a lot for your support!
Chiara

## [c3772235900] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-02-09 (no text)

## [c2788387445] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:53 · 2026-02-10
We don't have the safe_serialization flag anymore 
```suggestion
    def is_serializable(self) -> bool:
```

## [c2788402216] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:140 · 2026-02-10
not needed, just the `isinstance(module, SINQLinear)` is enough 
```suggestion
```

## [c2788445225] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:211 · 2026-02-10
```suggestion
```

## [c2788454524] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_sinq.py:226 · 2026-02-10
If you do that here, you should just update the `device_map` as we do for bnb

```python
def update_device_map(self, device_map):
        if device_map is None:
            if torch.cuda.is_available():
                device_map = {"": torch.cuda.current_device()}
```

## [c2788467631] SunMarc (MEMBER) · review_comment on tests/quantization/sinq/test_sinq.py:492 · 2026-02-10
Please only keep the integration tests that looks like the other quantization tests. I will do a refactor later on so that all the tests looks the same but for now, i prefer to keep the test short / same as the others

## [c3779435088] SunMarc (MEMBER) · review APPROVED · 2026-02-10
Thanks a lot ! This is ready to be merged. Just a few things to remove/improve + fix the CI. You need to run make style + check-repo or fix-repo

## [c2788894741] ChiaraBoretti (CONTRIBUTOR) · review_comment on tests/quantization/sinq/test_sinq.py:492 · reply to [c2788467631] · 2026-02-10
Thanks for the feedback. Just to make sure I understand correctly, would you like me to remove the internal quantizer tests and the module-level tests, and keep only the Config tests and the end-to-end integration tests? Or do you prefer that I merge the quantizer validation and module conversion tests in the SinqIntegration one, like in ```finegrained_fp8```?

## [c3780024912] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-02-10 (no text)

## [c2792526812] SunMarc (MEMBER) · review_comment on tests/quantization/sinq/test_sinq.py:492 · reply to [c2788467631] · 2026-02-11
What would be nice is to just have something like `FineGrainedFP8ConfigTest` and `FP8QuantizerTest`. So just the config tests + end-to-end integrations for now

## [c3784000542] SunMarc (MEMBER) · review COMMENTED · 2026-02-11 (no text)

## [c2793834438] ChiaraBoretti (CONTRIBUTOR) · review_comment on tests/quantization/sinq/test_sinq.py:492 · reply to [c2788467631] · 2026-02-11
Ok, thanks! I modified the tests script. Hope now it is correct.

## [c3785429264] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-02-11 (no text)

## [c3885102026] ChiaraBoretti (CONTRIBUTOR) · issue_comment · reply to [c3784000542] · 2026-02-11
Hi @SunMarc 
I pushed the latest version of the code with the modification you suggested. Then I run make style, make quality and make repo-consistency. Does the code now seems ready to be merged? Do I have to run other checks?
Thanks, 
Chiara

## [c3885385423] SunMarc (MEMBER) · issue_comment · 2026-02-11
The CI is still red, can you try to check why ? Also there are some conflicts

## [c3889733717] ChiaraBoretti (CONTRIBUTOR) · issue_comment · reply to [c3885385423] · 2026-02-12
Hi @SunMarc,

I’ve resolved the merge conflicts and rerun the test suite. At the moment, the two remaining failing tests appear to be related, with the root issue coming from test_torch. From what I can tell, the failure is tied to the Bark model tests rather than my integration code:

```bash
FAILED tests/models/bark/test_modeling_bark.py::BarkFineModelTest::test_inputs_embeds
NotImplementedError: Module [ModuleList] is missing the required "forward" function
```

Would you like me to investigate and fix this as well, or should it be handled by the maintainers of the Bark model implementation?

Thanks!

## [c3890302435] SunMarc (MEMBER) · issue_comment · 2026-02-12
No, this is fine ! Thanks for all your hardwork and keeping up with me. I'll ask @MekkCyber for a final review to see if I missed anything.

## [c2811604365] MekkCyber (CONTRIBUTOR) · review_comment on docs/source/en/quantization/selecting.md:64 · 2026-02-16
```suggestion
| Super-fast but high-quality quantization process, no calibration data needed.              | Accuracy can degrade significantly at bit widths <=2-bit.                     |
```

## [c2811605534] MekkCyber (CONTRIBUTOR) · review_comment on docs/source/en/quantization/selecting.md:66 · 2026-02-16
```suggestion
| Supports wide range of bit widths (8, 4, 3, 2 bits).              |                                                                            |
```

## [c2811615261] MekkCyber (CONTRIBUTOR) · review_comment on src/transformers/integrations/sinq.py:2 · 2026-02-16
```suggestion
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
```

## [c2811639378] MekkCyber (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:1 · 2026-02-16
```suggestion
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
```

## [c2811646444] MekkCyber (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:91 · 2026-02-16
do we need cuda for `sinq` in general or just the `gemlite` backend ?

## [c2811650264] MekkCyber (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:107 · 2026-02-16
```suggestion
            raise ValueError(
                "You are using `method='asinq'` in the quantization config. Right now the calibrated version of SINQ"
                " is not supported in Hugging Face, please refer and use the official SINQ repository "
                "`to quantize a model with this method. "
            )
```

## [c2811657558] MekkCyber (CONTRIBUTOR) · review_comment on tests/quantization/sinq/test_sinq.py:1 · 2026-02-16
```suggestion
# Copyright 2026 The HuggingFace Team. All rights reserved.
```

## [c2811665518] MekkCyber (CONTRIBUTOR) · review_comment on tests/quantization/sinq/test_sinq.py:100 · 2026-02-16
maybe we can add an expected output slow test with a small model to make sure we don't generate gibberish

## [c3807710648] MekkCyber (CONTRIBUTOR) · review COMMENTED · reply to [c3889733717] · 2026-02-16
Thank you for the contribution @ChiaraBoretti ! it looks great 🔥 ! Just have some few nits, mostly typos and tests and we are good to go

## [c2811835745] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:91 · reply to [c2811646444] · 2026-02-16
CUDA is only strictly required when using the GemLite backend. The core sinq algorithm can also run on the CPU, although this results in significantly slower inference and longer quantization times compared to GPU execution.

## [c3807960311] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-02-16 (no text)

## [c2811849846] ChiaraBoretti (CONTRIBUTOR) · review_comment on tests/quantization/sinq/test_sinq.py:100 · reply to [c2811665518] · 2026-02-16
Yes, that makes sense. Would you recommend adding a new test class with a small model definition and just a ```test_quantized_model``` function (like the one for the Qwen model test) to verify that the quantized model produces the expected output?

## [c3807974589] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-02-16 (no text)

## [c2811859126] MekkCyber (CONTRIBUTOR) · review_comment on tests/quantization/sinq/test_sinq.py:100 · reply to [c2811665518] · 2026-02-16
nw we can just add it to the same `SinqTest` class

## [c3807984228] MekkCyber (CONTRIBUTOR) · review COMMENTED · 2026-02-16 (no text)

## [c2811861855] MekkCyber (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:91 · reply to [c2811646444] · 2026-02-16
but in the validation we are erroring out if no gpu is available, is that intended ?

## [c3807987324] MekkCyber (CONTRIBUTOR) · review COMMENTED · 2026-02-16 (no text)

## [c2812130485] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:91 · reply to [c2811646444] · 2026-02-16
You are right, raising an error may be too conservative here. I can replace it with a warning if you agree that this would be the better approach.

## [c3808295798] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-02-16 (no text)

## [c2812144599] MekkCyber (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:91 · reply to [c2811646444] · 2026-02-16
yes that sounds better! let's do that, and update the tests, and maybe keep the error if gemlite is available and the dispatching to gemlite is done automatically in the library

## [c3808310925] MekkCyber (CONTRIBUTOR) · review COMMENTED · 2026-02-16 (no text)

## [c2812470150] ChiaraBoretti (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_sinq.py:91 · reply to [c2811646444] · 2026-02-16
Thank you for your suggestions. I have fixed the typos, added the requested test, and replaced the error with a warning. Please let me know if any further changes are needed.

Thanks!

## [c3808675711] ChiaraBoretti (CONTRIBUTOR) · review COMMENTED · 2026-02-16 (no text)

## [c2812497354] MekkCyber (CONTRIBUTOR) · review_comment on tests/quantization/sinq/test_sinq.py:196 · 2026-02-16
```suggestion
```

## [c3808706775] MekkCyber (CONTRIBUTOR) · review APPROVED · 2026-02-16
Very cool! thanks a lot for iterating on this!

## [c2812501819] MekkCyber (CONTRIBUTOR) · review_comment on tests/quantization/sinq/test_sinq.py:196 · reply to [c2812497354] · 2026-02-16
removing since we don't need it in our test suite

## [c3808711404] MekkCyber (CONTRIBUTOR) · review COMMENTED · 2026-02-16 (no text)

## [c3908630045] MekkCyber (CONTRIBUTOR) · issue_comment · 2026-02-16
@bot /style

---
outcome: state=closed, merged=True
