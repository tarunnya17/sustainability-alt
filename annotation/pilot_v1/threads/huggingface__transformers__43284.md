# #43284 [issue] Customize Quantization-Friendly Backward Compatibility
https://github.com/huggingface/transformers/issues/43284  ·  opened by yiliu30 on 2026-01-14

## [body] yiliu30 (CONTRIBUTOR) · opening post · 2026-01-14
### Feature request

Hi guys,
It’s great to see Transformers moving to V5 with a modular design and improved performance! We’ve started adapting it with the RC branch, but noticed that some of the changes are not very friendly for quantization tools.
Here’s the context:
Given a BF16 model, quantization tools typically detect `torch.nn.Linear` layers and then quantize them to the target dtypes. However, for some MoE models like DeepSeek, the experts are concatenated into a large tensor. 

https://github.com/huggingface/transformers/blob/e8ae373133be1eff2254c5dd71fcd628445cb4a4/src/transformers/models/mixtral/modular_mixtral.py#L144-L146

This introduces two potential issues:

- Detection Issue
The quantization tool can’t detect the individual linear layers and requires maintaining a lot of patching code to enable proper quantization.

- Memory Issue
Since the expert concatenation logic happens at the beginning of model loading, it materializes all weights in RAM. This means the peak memory usage exceeds the model size. Previously (before V5), we could quantize layer by layer and offload the quantized layers to disk, so peak memory was only slightly higher than the size of a single decoding layer.

I’m wondering if you have any solution to make these changes more backward-compatible or if there’s a way to provide an interface to disable this behavior?
Thanks!

### Motivation

Better backward compatible

### Your contribution

Instead of concatenating all experts at load time, use lazy loading or chunked concatenation?

## [c3754713629] Rocketknight1 (MEMBER) · issue_comment · 2026-01-15
cc @MekkCyber for quants and @cyrilvallez @IlyasMoutawwakil for the MoE refactor

## [c3755487459] IlyasMoutawwakil (MEMBER) · issue_comment · reply to [body] · 2026-01-15
Unfortunately it's only convenient during the quantization process to see MoEs as a chain of num_experts linear layers. During inference / training it is an extremely inefficient way to model it.

Also it can be argued that quantization should increase the arithmetic intensity of blocks, but it can't do so if the operations are being performed sequentially with small per-expert matrices. I think the new format opens the path for quantization to be even more beneficial for MoEs than before.

> Detection Issue
The quantization tool can’t detect the individual linear layers and requires maintaining a lot of patching code to enable proper quantization.

The standardization of the MoE code in transformers v5 means maintaining even less patching code because it's basically the same MoE block across multiple architectures.

> Memory Issue
Since the expert concatenation logic happens at the beginning of model loading, it materializes all weights in RAM. This means the peak memory usage exceeds the model size. Previously (before V5), we could quantize layer by layer and offload the quantized layers to disk, so peak memory was only slightly higher than the size of a single decoding layer.

You should still be able to quantize experts one by one by looping over the experts dimension if that's what you mean. Why is ram an issue here ?

## [c3782649821] yiliu30 (CONTRIBUTOR) · issue_comment · reply to [c3755487459] · 2026-01-22
Hi @IlyasMoutawwakil, thanks for the explanation! Let me clarify the RAM usage concern here.

If we concatenate all experts’ `gate`/`up` weights into a single `gate_up_proj` with shape
`[self.num_experts, 2 * self.intermediate_dim, self.hidden_dim]`. 
Wouldn’t this materialize all the experts immediately? That could lead to a significant increase in RAM usage.
I ran a quick test using a 6‑layer DeepSeek‑R1 setup to illustrate this behavior. It takes ~49 GB of RAM, with the first six layers being moved into RAM.

```bash
[Memory] before model load Current RAM usage: 638.5MB
...
[Memory] after model load Current RAM usage: 49004.6MB
```
Previously (before v5), the initialization stage didn’t materialize all the weights, right?
And we can load layer and quantize it one by one:

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype="auto")
for decoding_layer in model.model.layers:
[… 59 lines of code omitted]

## [c3795829000] ITcarrot (NONE) · issue_comment · 2026-01-25
Using raw `nn.Parameter` for fused experts currently breaks compatibility with PEFT (LoRA) and quantization tools. To prevent this fragmentation, I've proposed a standardized `BatchLinear` module in #43472 .

Integrating this abstraction into Qwen3-MoE may allow immediate support for quantization pipelines. This also make it easier to implement fast Moe de-quant in downstream libs.

## [c3796437061] ITcarrot (NONE) · issue_comment · reply to [c3782649821] · 2026-01-25
The RAM usage in @yiliu30 cases is likely because quantization library is NOT quantizing new Moe Experts AT ALL. Here's a code snippet to demonstrate this problem:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
[… 15 lines of code omitted]
```
In transformer v4, it prints `Memory: 15.09 GB`, which it properly quantized. But in transformer v5, it prints `Memory: 55.60 GB`, showing quantizing is not working correctly.

## [c3797516794] yiliu30 (CONTRIBUTOR) · issue_comment · reply to [c3796437061] · 2026-01-26
Hi @ITcarrot, I think we’re actually dealing with two issues here:
1) Detecting experts in the new MoE module — this causes the new MoE experts to remain unquantized.
2) High RAM usage during model initialization — this happens even before any quantization starts, so it’s not specific to the quantization workflow.

For 2, the RAM issue I’m referring to occurs during the model loading stage. Even without running quantization, loading the full model consumes a large amount of memory.
Here is the code I used to reproduce the QWen example. The RAM usage during model loading is even larger than the BF16 model (~60GB). I’m not sure whether the original expert weights are still being kept in memory after the concatenation step, which might explain the peak. @IlyasMoutawwakil 
```bash
[Memory] before model load Current RAM usage: 294.68MB, 0.29 GB
Loading weights: 100%|█| 531/531 [00:08<00:00, 61.93it/s, Materializing param=model.
[Memory] after model load Current RAM usage: 110931.2MB, 108.33 GB
```

```python
import psutil
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
[… 48 lines of code omitted]

## [c3800572233] IlyasMoutawwakil (MEMBER) · issue_comment · 2026-01-26
I think what's happening is that in v4, the RAM usage is low after model init because tensors are still memmaped from disk through Safetensors. The tensors are only materialized after the first inference pass (you can add a third ram measurement in your code to check). In v5 we materialize them during weight loading to perform conversions on them.

Here's how you can achieve similar RAM usage using Accelerate's disk offloading:
```py
import psutil
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.import_utils import clear_import_cache
[… 42 lines of code omitted]
```

```
(.transformers) (base) ilyas@hf-dgx-01:~/transformers$ CUDA_VISIBLE_DEVICES="" python script.py 
[Memory] before model load Current RAM usage: 772.19MB, 0.75 GB
use_kernel_func_from_hub is not available in the installed kernels version. Please upgrade kernels to use this feature.
use_kernel_func_from_hub is not available in the installed kernels version. Please upgrade kernels to use this feature.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████| 531/531 [02:32<00:00,  3.47it/s, Materializing param=model.norm.weight]
Some parameters are on the meta device because they were offloaded to the cpu and disk.
[Memory] after model load Current RAM usage: 28860.89MB, 28.18 GB
[Memory] after inference Current RAM usage: 29314.91MB, 28.63 GB
```

## [c3805176749] IlyasMoutawwakil (MEMBER) · issue_comment · reply to [c3797516794] · 2026-01-27
@yiliu30 please tell me if this solves your issue, it is also easier to customize the forward pass of the experts module using the `ExpertsInterface` which will allow you to define an implementation that loops over all experts for example.
https://github.com/huggingface/transformers/blob/4854dbf9da4086731256496cf4a8e4ea45d4d54e/src/transformers/integrations/moe.py#L288

## [c3808575393] yiliu30 (CONTRIBUTOR) · issue_comment · reply to [c3805176749] · 2026-01-28
Hi @IlyasMoutawwakil, thanks for the information!
I understand that Accelerate’s disk‑offloading approach can help mitigate the RAM issue, but it still requires more RAM than before and also needs additional disk space, correct?
Regarding the `ExpertsInterface`: if I understand correctly, it provides an interface to register a custom forward implementation for the MoE module. The issue I mentioned earlier is caused by the modeling definition—we’re expecting an MoE module that uses a standard `torch.nn.Linear`.
I understand that performance is also important. What do you think about defining the module using `MoeMLP` module, and then performing a post‑processing step (concatenating the expert weights) after loading the weights? I take the Qwen3 as example, and below is the initial design.

Please review 1) the `Qwen3MoeExperts` module definition where we use `Qwen3MoeMLP` instead of `gate_up_proj`.
2) And the `post_process_for_grouped_mm` function to repack weights into a kernel‑friendly layout. 

This pattern is common in vLLM/SGLang, and different kernels may require different packing formats. 
With this approach, we keep flexibility for downstream quantization tools and preserve runtime performance. I’m not entirely sure whether it will slow down initialization, though.
Let me know what you think.

```python
import torch
from torch import nn
from transformers.models.deepseek_v3.modeling_deepseek_v3 import ACT2FN
[… 103 lines of code omitted]

## [c3810093206] IlyasMoutawwakil (MEMBER) · issue_comment · reply to [c3808575393] · 2026-01-28
> I understand that performance is also important. 

Its actually performance and simplicity of the modeling code. Unfortunately we can't bloat the modeling code like that 🥲 you might have noticed that the experts implementation is light decorator that lives outside of the modeling code. Also in your proposition we can't switch between experts implementations without changing the weights layout, the current process does everything at load time in parallel threads (I/O optimized).
I'm pinning @SunMarc @MekkCyber and @ArthurZucker to see if they have any inputs on this matter.

## [c3810450361] ArthurZucker (COLLABORATOR) · issue_comment · 2026-01-28
Hey! 
1. Applying weight conversion operations HAS to be done before the model is loaded. This allows for any deserialization, dequantization / quantization operations to happend. The "contract" is that we first modify the model (replace nn.Linears, replace xxxExperts layers) then model.state_dict() gives the missing keys
2. You can have a look at https://github.com/huggingface/transformers/blame/be87564082c7d0fc1c6508652b4490ecc89ad9ff/src/transformers/integrations/finegrained_fp8.py it shows how you can interface with the weight converter with Conversion ops: https://github.com/huggingface/transformers/blame/be87564082c7d0fc1c6508652b4490ecc89ad9ff/src/transformers/integrations/finegrained_fp8.py#L715C21-L715C34. 

We are here to help make sure transformers allows you to easily quantize, so let's try to find a solution! 

One thing is you might want to disable the weight fusion -> we might need to expose a 'skip_conversion' for example. Or you can just replace the weight conversion ops specifically when using certain quantization methods as long as the model is prepared before hand!

## [c3810862539] yiliu30 (CONTRIBUTOR) · issue_comment · reply to [c3810093206] · 2026-01-28
Hi @IlyasMoutawwakil @ArthurZucker, thanks for the detailed background. I understand that the Conversion op can be used to preprocess weights before they are bound to the model.

>  can't bloat the modeling code like that

The example above is only meant to demonstrate the idea. For a production‑level implementation, I believe we can decouple this logic from the modeling code.

> everything at load time in parallel threads (I/O optimized)

Could you point me to the relevant code? Thanks!

> One thing is you might want to disable the weight fusion 

Yes—exactly. That’s what we want, and currently we disable it using the `register_checkpoint_conversion_mapping` function. For example, we disable it for DeepSeek as shown below:

```python
register_checkpoint_conversion_mapping("deepseek_v3", [], overwrite=True)
```

It would be great to have native support in Transformers for this, similar to the `skip_conversion` option you mentioned.

Regarding MoE: if we disable weight fusion, there should be a corresponding modeling definition for the unfused weights, right? Otherwise, the weights can’t be loaded correctly. Currently, we are handling this by patching the MoE module. Here is the full implementation for DeepSeek, https://github.com/intel/auto-round/pull/1359/changes#diff-4de8b1c0f3fa626c8f72877f1c6253c3c6ea8c94c161524e18821847072d7ee8

While it works, it would be much nicer to rely on native support in Transformers. New model evolves so quickly, and maintaining patches becomes a pretty heavy technical burden over time.
Thanks again for the guidance!

## [c3811597625] ArthurZucker (COLLABORATOR) · issue_comment · reply to [c3810093206] · 2026-01-28
cc @IlyasMoutawwakil as we talked about the potential `fuse_qkv` flag, which would be fairly similar in feature to this but for moes. 

1. 🟢 for a `from_pretrained` flag that does not force you to overwrite the mappin
2. 🟡 we can probably try to expose functions and a good API to actually standardize monkey patching / patching for quantization but also for fusing merging etc. I'm not against it

## [c3814685917] yiliu30 (CONTRIBUTOR) · issue_comment · reply to [c3811597625] · 2026-01-29
> cc [@IlyasMoutawwakil](https://github.com/IlyasMoutawwakil) as we talked about the potential `fuse_qkv` flag, which would be fairly similar in feature to this but for moes.
> 
> 1. 🟢 for a `from_pretrained` flag that does not force you to overwrite the mappin
> 2. 🟡 we can probably try to expose functions and a good API to actually standardize monkey patching / patching for quantization but also for fusing merging etc. I'm not against it

Awesome! Thanks @ArthurZucker — both options work great for us!

## [c3841373456] Cyrilvallez (MEMBER) · issue_comment · 2026-02-03
Hey! In general you should not witness any increased RAM usage compared to v4, or very slightly. Theoretical max memory usage during loading is `model_size + size_of_layers_needed_for_biggest_conversion`, or here `model_size + experts_on_1_layer` as described [here](https://huggingface.co/docs/transformers/v5.0.0/en/weightconverter#fast-and-efficient-model-loading). Note that cpu ram consumption is very tricky to measure, it would be much easier on gpu.
Offloading is a separate topic, and if you use it indeed before it would simply keep everything on disk initially before 1st forward, while now it needs to materialize on cpu + resave to disk.
If you are sure of your memory consumption, I'd be happy to look into it as it's not supposed to happen at all

## [c3841925509] yiliu30 (CONTRIBUTOR) · issue_comment · reply to [c3841373456] · 2026-02-03
> model_size + size_of_layers_needed_for_biggest_conversion

Hi @Cyrilvallez, for the quantization use case.
Since we are handling the decoding layer one by one, I’d argue that the maximum memory usage is `decoding_layer_size` for the model part.

Given the flow here:
```
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype="auto")
#>>>>>>>>>>> very little memory is consumed, since the weights are still mem‑mapped from disk. <<<<<<
for decoding_layer in model.model.layers:
    materialize_layer(decoding_layer) # <- move weights to RAM
    quantized_layer = quantize(decoding_layer)
    offload_to_cpu(quantized_layer )
    del decoding_layer
```

## [c3842214527] Cyrilvallez (MEMBER) · issue_comment · 2026-02-03
Humm, when loading everything is materialized to cpu in turn, there is no mem-mapped or i don't know what which would keep it on disk only. This was the case previously as well. As for the quantization, why do you need the full layer? Usually you would only need separate Linear layers

## [c3853166982] yiliu30 (CONTRIBUTOR) · issue_comment · reply to [c3842214527] · 2026-02-05
Hi @Cyrilvallez  Yes, I agree the memory usage if we need to load the whole model into DRAM.

>why do you need the full layer? Usually you would only need separate Linear layers

It’s because some quantization algorithms require the context of the entire decoding layer to work effectively:

- AWQ: Needs the architectural info to map the 'smooth' layers (like LayerNorm) to their corresponding weights.  https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/awq/mappings.py
- AutoRound: Needs to calculate the reconstruction loss across the whole block to properly tune the gradients.

[image]

https://aclanthology.org/2024.findings-emnlp.662/

## [c3857186007] IlyasMoutawwakil (MEMBER) · issue_comment · reply to [c3853166982] · 2026-02-06
@Cyrilvallez in v4 cpu tensors were not "kept on disk only" but rather lazily loaded ; if you measure ram before first inference, it shows very low usage, once a first pass is performed, it shows the same usage as current transformers. So basically this seems like an unintended (or maybe intended idk) feature inherited from safetensors 😅 it's a side effect of lazy tensor loading through memory mapping. This only happens if you're loading on cpu though and happens without enabling disk offloading which i only suggest as an alternative to this behaviour. you can run the snippet suggested by @yiliu30 and add a memory measurement after inference as well to verify this.

## [c3861607918] Cyrilvallez (MEMBER) · issue_comment · 2026-02-06
Humm I'm very surprised as we used to have the `param = param[...]` explicitly supposed to materialize the slice from the safetensors opened file, independently of the device (cpu or not)... But taking your word for it!

---
outcome: state=closed
