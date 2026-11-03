# #43872 [issue] bitsandbytes incompatibility: TypeError: Int8Params.__new__() got an unexpected keyword argument '_is_hf_initialized'
https://github.com/huggingface/transformers/issues/43872  ·  opened by edmcman on 2026-02-09

## [body] edmcman (NONE) · opening post · 2026-02-09
### System Info

Well that's interesting...
```
ed@banana /tmp [127]> uvx --with huggingface-hub,ipython,transformers,torch,bitsandbytes,accelerate transformers env
Traceback (most recent call last):
  File "/home/ed/.cache/uv/archive-v0/93vgXQSK-rYWDyEU_W9k9/bin/transformers", line 6, in <module>
    from transformers.cli.transformers import main
  File "/home/ed/.cache/uv/archive-v0/93vgXQSK-rYWDyEU_W9k9/lib/python3.11/site-packages/transformers/cli/transformers.py", line 22, in <module>
    from transformers.cli.serve import Serve
  File "/home/ed/.cache/uv/archive-v0/93vgXQSK-rYWDyEU_W9k9/lib/python3.11/site-packages/transformers/cli/serve.py", line 360, in <module>
    class Serve:
  File "/home/ed/.cache/uv/archive-v0/93vgXQSK-rYWDyEU_W9k9/lib/python3.11/site-packages/transformers/cli/serve.py", line 588, in Serve
    validator: TypeAdapter,
               ^^^^^^^^^^^
NameError: name 'TypeAdapter' is not defined
```

I guess that's a bug too?

Here are the major versions:
torch 2.10.0+cu128 transformers 5.1.0 bitsandbytes 0.49.1 accelerate 1.12.0

### Who can help?

@sunmarc @MekkCyber 

### Information

- [ ] The official example scripts
- [x] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [x] My own task or dataset (give details below)

### Reproduction

1. Run with `uvx --with huggingface-hub,ipython,transformers,torch,bitsandbytes,accelerate python bug.py`
``` python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
MODEL_ID = "ejschwartz/decaf-v1-22b"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "3GiB", "cpu": "64GiB"},  # tighter GPU cap than before
)
```
2. Receive exception
```
Traceback (most recent call last):
  File "/tmp/bug.py", line 17, in <module>
    model = AutoModelForCausalLM.from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/ed/.cache/uv/archive-v0/93vgXQSK-rYWDyEU_W9k9/lib/python3.11/site-packages/transformers/models/auto/auto_factory.py", line 374, in from_pretrained
[… 31 lines of code omitted]
```

### Expected behavior

No exception

## [c3883744962] Cyrilvallez (MEMBER) · issue_comment · 2026-02-11
Hey @SunMarc! Ha yes, I remember that the bnb quantizer was first removing the flag before adding it back indeed! However I must say in general I'm unsure if bnb works with offloading?? I.e. I don't think accelerate is correctly serializing everything to reload correctly?

## [c3884435779] SunMarc (MEMBER) · issue_comment · reply to [c3883744962] · 2026-02-11
>  However I must say in general I'm unsure if bnb works with offloading?

It should work but I don't recommend offloading to CPU in general as it is super slow

## [c3884664623] edmcman (NONE) · issue_comment · reply to [c3884435779] · 2026-02-11
@Cyrilvallez The [documentation says that offloading is possible with 8-bit quantization](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes#offloading).  

But part of the reason I opened this issue is that the overall support for offloading with bnb is very unclear.  It seems like offloading with 4-bit is not supported, but I don't see that explicitly stated in the documentation.  And I've seen an exception when using 4-bit quants - I think it is [this one](https://github.com/huggingface/transformers/blob/b52b6631a26fd362e49968af14ed77ff359006a0/src/transformers/quantizers/quantizer_bnb_4bit.py#L72) - that at first glance suggests that offloading with 4-bit quants should be possible.  After closer inspection, the `int8` references probably indicate that message should only be shown for 8-bit quants?

The larger picture of what I'm trying to do is make two large models at least *usable* for people with consumer GPUs like mine.  I've had very good luck with the 22B model using llama.cpp and Q4_K_M quants.  With partial offloading using my 8GiB GPU, I get about 6 tokens per second, which is fine for my purposes.  The 32B model is a qwen3 model with a classification head on it, which llama.cpp/GGML does not support.  This is why I am interested in understanding whether transformers/bitsandbytes could possibly help.

Obviously in an ideal world, everyone would have access to enough VRAM, but they don't.  IMHO if CPU offloading makes sense anywhere, it's with 4-bit quants, as llama.cpp/Ollama and related projects are showing.

@SunMarc When you say CPU offloading is super slow, do you mean 100% offloading, or partial offloading?  For the 22B quantized model on llama.cpp, I found that 100% offloading is completely unusable, but partial offloading works really well.

## [c3884680022] edmcman (NONE) · issue_comment · 2026-02-11
Also, now I've gotten a little confused about my two issues.  This issue #43872 seems like a transformers v5 bug.  #43873 was supposed to be the more general "offloading in bnb is not working" issue when avoiding the `_is_hf_initialized` problem.

## [c3909735919] Cyrilvallez (MEMBER) · issue_comment · reply to [c3884435779] · 2026-02-16
cc @SunMarc, we should make sure the flag is not passed to the constructor when reinstantiating params with bnb

## [c4054717762] edmcman (NONE) · issue_comment · 2026-03-13
Not stale

On Fri, Mar 13, 2026 at 4:09 AM github-actions[bot] -
***@***.***
***@***.***> wrote:
>
> github-actions[bot] left a comment (huggingface/transformers#43872)
>
> This issue has been automatically marked as stale because it has not had recent activity. If you think this still needs to be addressed please comment on this thread.
>
> Please note that issues that do not follow the contributing guidelines are likely to be ignored.
>
> —
> Reply to this email directly, view it on GitHub, or unsubscribe.
> You are receiving this because you authored the thread.Message ID: ***@***.***>

## [c4063624606] dorpxam (NONE) · issue_comment · reply to [c4054717762] · 2026-03-15
@edmcman I confirm that it's a bug. 4.57.6 no problem, 5.0.0 problem. Same configuration, same models.

## EDIT (Monkey Patch)

```python
import bitsandbytes as bnb

original_new = bnb.nn.Int8Params.__new__

def patched_new(cls, data=None, requires_grad=False, has_fp16_weights=False, **kwargs):
    kwargs.pop('_is_hf_initialized', None)
    return original_new(cls, data, requires_grad, has_fp16_weights, **kwargs)

bnb.nn.Int8Params.__new__ = patched_new
```

Just before any import ...

## [c4075336338] codybarrett108-pixel (NONE) · issue_comment · reply to [c4063624606] · 2026-03-17
Thanks for reporting this I'll take a look

On Sun, Mar 15, 2026, 1:39 PM Max Prod ***@***.***> wrote:

> *dorpxam* left a comment (huggingface/transformers#43872)
> <https://github.com/huggingface/transformers/issues/43872#issuecomment-4063624606>
>
> @edmcman <https://github.com/edmcman> I confirm that it's a bug. 4.57.6
> no problem, 5.0.0 problem. Same configuration, same models.
>
> File "C:\anaconda3\envs\ltx2caption\Lib\site-packages\transformers\modeling_utils.py", line 4129, in from_pretrained    accelerate_dispatch(  File "C:\anaconda3\envs\ltx2caption\Lib\site-packages\transformers\integrations\accelerate.py", line 353, in accelerate_dispatch    dispatch_model(model, **device_map_kwargs)  File "C:\anaconda3\envs\ltx2caption\Lib\site-packages\accelerate\big_modeling.py", line 426, in dispatch_model    attach_align_device_hook_on_blocks(  File "C:\anaconda3\envs\ltx2caption\Lib\site-packages\accelerate\hooks.py", line 676, in attach_align_device_hook_on_blocks    attach_align_device_hook_on_blocks(  File "C:\anaconda3\envs\ltx2caption\Lib\site-packages\accelerate\hooks.py", line 676, in attach_align_device_hook_on_blocks    attach_align_device_hook_on_blocks(  File "C:\anaconda3\envs\ltx2caption\Lib\site-packages\accelerate\hooks.py", line 676, in attach_align_device_hook_on_blocks    attach_align_device_hook_on_blocks(  [Previous line repeated 1 more time]  File "C:\anaconda3\envs\ltx2caption\Lib\site-packages\accelerate\hooks.py", line 639, in attach_align_device_hook_on_blocks    attach_align_device_hook(  File "C:\anaconda3\envs\ltx2caption\Lib\site-packages\accelerate\hooks.py", line 530, in attach_align_device_hook    attach_align_device_hook(  File "C:\anaconda3\envs\ltx2caption\Lib\site-packages\accelerate\hooks.py", line 530, in attach_align_device_hook    attach_align_device_hook(  File "C:\anaconda3\envs\ltx2caption\Lib\site-packages\accelerate\hooks.py", line 521, in attach_align_device_hook    add_hook_to_module(module, hook, append=True)  File "C:\anaconda3\envs\ltx2caption\Lib\site-packages\accelerate\hooks.py", line 166, in add_hook_to_module    module = hook.init_hook(module)             ^^^^^^^^^^^^^^^^^^^^^^  File "C:\anaconda3\envs\ltx2caption\Lib\site-packages\accelerate\hooks.py", line 313, in init_hook    set_module_tensor_to_device(module, name, "meta")  File "C:\anaconda3\envs\ltx2caption\Lib\site-packages\accelerate\utils\modeling.py", line 363, in set_module_tensor_to_device    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^TypeError: Int8Params.__new__() got an unexpected keyword argument '_is_hf_initialized'
>
> return BitsAndBytesConfig(
>     load_in_8bit=True,
>     llm_int8_enable_fp32_cpu_offload=True,
>     llm_int8_threshold=0.0, # 6.0 default
> )
>
> With that kind of problem, I can't work with two wonderful models :
>
>    - Step3-VL-10B : works with transformers 4.57.6 (latest 4.x)
>    - Music Flamingo : work with transformers 5.x
>
> The problem is that I need to offload part of the Step3-VL-10B, 8bit quant
> is not enough.
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/huggingface/transformers/issues/43872#issuecomment-4063624606>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/B7HX4LXAOGJFKS7JP3NFA3L4Q32G7AVCNFSM6AAAAACURFCIXCVHI2DSMVQWIX3LMV43OSLTON2WKQ3PNVWWK3TUHM2DANRTGYZDINRQGY>
> .
> You are receiving this because you are subscribed to this thread.Message
> ID: ***@***.***>
>

## [c4075823916] SunMarc (MEMBER) · issue_comment · 2026-03-17
Fixed on accelerate side, please install the version from main for now until I release a new version

---
outcome: state=closed
