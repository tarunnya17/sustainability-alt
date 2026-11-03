# #40176 [pull_request] [`Mxfp4`] Add a way to save with a quantization method
https://github.com/huggingface/transformers/pull/40176  ·  opened by ArthurZucker on 2025-08-14

## [body] ArthurZucker (COLLABORATOR) · opening post · 2025-08-14
# What does this PR do?
Allows saving gpt_oss after it was trained. You can also save a mxfp4 model. 

```python
import torch
import gc
from transformers import Mxfp4Config, GptOssForCausalLM, AutoTokenizer
model_name = "hf-internal-testing/gpt-oss-20b-bf16"
[… 17 lines of code omitted]

## [c3191064852] ArthurZucker (COLLABORATOR) · issue_comment · 2025-08-15
run-slow: mxfp4

## [c2280123212] winglian (COLLABORATOR) · review_comment on src/transformers/quantizers/quantizer_mxfp4.py:222 · 2025-08-16
stray debugging/print?

## [c3125498848] winglian (COLLABORATOR) · review COMMENTED · 2025-08-16 (no text)

## [c2280123518] winglian (COLLABORATOR) · review_comment on tests/quantization/mxfp4/test_mxfp4.py:507 · 2025-08-16
did you want to substitute back `tmp` from `with tempfile.TemporaryDirectory() as tmp:` into `/fsx/Arthur/mxfp4`?

## [c3125499182] winglian (COLLABORATOR) · review COMMENTED · 2025-08-16 (no text)

## [c2281669124] ArthurZucker (COLLABORATOR) · review_comment on tests/quantization/mxfp4/test_mxfp4.py:507 · reply to [c2280123518] · 2025-08-18
yes I am just testing locally to make sure I see all the weights~

## [c3127328890] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-08-18 (no text)

## [c2281672018] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/quantizers/quantizer_mxfp4.py:222 · reply to [c2280123212] · 2025-08-18
yes, not ready yet :wink:

## [c3197448771] ArthurZucker (COLLABORATOR) · issue_comment · 2025-08-18
run-slow: gpt_oss, mxfp4

## [c3200813651] ArthurZucker (COLLABORATOR) · issue_comment · 2025-08-19
run-slow: gpt_oss, mxfp4

## [c2285314476] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/integrations/mxfp4.py:53 · 2025-08-19
1. we need the torch version
2. swizzle is done at loading time already so duplicating fails

## [c2285315677] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/integrations/mxfp4.py:128 · 2025-08-19
this is just a cleanup, you are not using the same function in the conversion script, with this you can

## [c2285316033] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/integrations/mxfp4.py:280 · 2025-08-19
```suggestion
```

## [c2285317969] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/integrations/mxfp4.py:374 · 2025-08-19
for proj in xxxxx but you only do sometyhing if proj in param_name, but param name is either gate or down

## [c2285319079] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/integrations/mxfp4.py:374 · 2025-08-19
because we do this on load

## [c2285319777] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:3925 · 2025-08-19
this does not cost much

## [c2285320072] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:3926 · 2025-08-19
```suggestion
```

## [c2285321049] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:4915 · 2025-08-19
this is a must, we cannot keep bloating the code when the quantizer has an api for itselfe

## [c2285321904] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/quantizers/quantizer_mxfp4.py:65 · 2025-08-19
avoids many imports

## [c2285327442] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/quantizers/quantizer_mxfp4.py:206 · 2025-08-19
this branch is for quantizing so loading a bf16 -> mxfp4 not tested and might fail why? 
because on load you do the reverse operation, but only on load. 
I do this here for saving, but if you read a model we need another branch, will add that later. 

cc @MekkCyber and @SunMarc this is the main issue I had to deal with: because we do some rehsape on load, instead of having done it in the conversion. Might be for vllm compatibility, but tldr it's a little bit messed up

## [c2285327994] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/quantizers/quantizer_mxfp4.py:404 · 2025-08-19
maybe we did not have time

## [c3127333360] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-08-19 (no text)

## [c2285331420] SunMarc (MEMBER) · review_comment on src/transformers/integrations/mxfp4.py:280 · 2025-08-19
```suggestion
```

## [c2285334734] SunMarc (MEMBER) · review_comment on src/transformers/integrations/mxfp4.py:59 · 2025-08-19
it might change the value inside to make the computation faster for the targeted hardware. This is the bottleneck when it comes to saving the model. Either quantize the model without swizzle or find a way to undo this.

## [c2285349024] SunMarc (MEMBER) · review_comment on src/transformers/integrations/mxfp4.py:373 · 2025-08-19
to remove ?

## [c2285353508] SunMarc (MEMBER) · review_comment on src/transformers/modeling_utils.py:872 · 2025-08-19
:<

## [c2285360569] SunMarc (MEMBER) · review_comment on src/transformers/modeling_utils.py:5044 · 2025-08-19
why is this needed ? we shouldn't touch this 
```suggestion
```

## [c2285364440] SunMarc (MEMBER) · review_comment on src/transformers/models/gpt_oss/convert_gpt_oss_weights_to_hf.py:143 · 2025-08-19
let's not change this. the checkpoint will be different if we change this

## [c2285387625] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_mxfp4.py:201 · 2025-08-19
can you quickly explain the transformation you are doing here ?

## [c2285388081] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_mxfp4.py:206 · 2025-08-19
same

## [c2285411275] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_mxfp4.py:203 · 2025-08-19
ok i see what you are trying to do here but you are thinking that users will only go through that path for saving.

## [c2285413266] SunMarc (MEMBER) · review_comment on src/transformers/quantizers/quantizer_mxfp4.py:205 · 2025-08-19
can you quickly explain why we do that ?

## [c2285419780] SunMarc (MEMBER) · review_comment on src/transformers/modeling_utils.py:3925 · 2025-08-19
i think it is best if we allow users to pass a `quantization_config` in save_pretrained, this will make things a bit messy in my opinion.

## [c3132429349] SunMarc (MEMBER) · review COMMENTED · 2025-08-19
Thanks for adding this ! This looks quite good. I was thinking it would be better if we can do the following instead of allowing users to quantize the model in `save_pretrained` as this will add more complexity. 
```
model = GptOssForCausalLM.from_pretrained(
    model_name,
    quantization_config=  Mxfp4Config(swizzle=False)
)
model.save_pretrained(...)
```
If the user didn't set `swizzle=False` when quantizing the model for saving, we can just raise an error for that. WDYT ? 

BTW, right now if a user try to quantize the model with the following way, we can't use it at all as the weights are not swizzled.

## [c2285464241] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:5044 · reply to [c2285360569] · 2025-08-19
Had to hardcode, will chekc how to remove now!

## [c3132617914] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-08-19 (no text)

## [c2285465078] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/gpt_oss/convert_gpt_oss_weights_to_hf.py:143 · reply to [c2285364440] · 2025-08-19
no no it was done right after this

## [c3132619096] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-08-19 (no text)

## [c2285465574] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/quantizers/quantizer_mxfp4.py:201 · reply to [c2285387625] · 2025-08-19
just the reverse of loading

## [c3132619742] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-08-19 (no text)

## [c2285466910] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/quantizers/quantizer_mxfp4.py:203 · reply to [c2285411275] · 2025-08-19
I can add another one for people loading bf16 to mxfp4 tho should be easy

## [c3132621600] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-08-19 (no text)

## [c2285469868] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:3925 · reply to [c2285419780] · 2025-08-19
I am not sure I understand here, it is what we are allowing with this PR: `quantization_config` is passed, allowing the init of the hf_quantizer

## [c3132625654] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-08-19 (no text)

## [c2288284704] SunMarc (MEMBER) · review_comment on src/transformers/models/gpt_oss/convert_gpt_oss_weights_to_hf.py:143 · reply to [c2285364440] · 2025-08-20
oh indeed

## [c3136781097] SunMarc (MEMBER) · review COMMENTED · 2025-08-20 (no text)

## [c2288285342] SunMarc (MEMBER) · review_comment on src/transformers/modeling_utils.py:5044 · reply to [c2285360569] · 2025-08-20
i'll check and fix

## [c3136782349] SunMarc (MEMBER) · review COMMENTED · 2025-08-20 (no text)

## [c2288286615] SunMarc (MEMBER) · review_comment on src/transformers/modeling_utils.py:3925 · reply to [c2285419780] · 2025-08-20
discussed internally

## [c3136784470] SunMarc (MEMBER) · review COMMENTED · 2025-08-20 (no text)

## [c3210377641] SunMarc (MEMBER) · issue_comment · 2025-08-21
run-slow: gpt_oss, mxfp4

## [c3140516862] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-08-21
As discussed offline, we really need a way to save_pretrained without having to use this swizzle setting, let's think about how to cover all cases and simplify please!

## [c2291469408] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:4124 · 2025-08-21
```suggestion
```

## [c3141292797] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-08-21
LGTM thanks for iterating

## [c3141353253] SunMarc (MEMBER) · review APPROVED · 2025-08-21
;)

## [c3211227122] SunMarc (MEMBER) · issue_comment · 2025-08-21
run-slow: gpt_oss, mxfp4

---
outcome: state=closed, merged=True
