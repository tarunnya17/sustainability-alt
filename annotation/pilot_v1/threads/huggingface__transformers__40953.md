# #40953 [issue] gpt-oss-120b inference failed running on 16 GPUs, single node and with tp_plan="auto"
https://github.com/huggingface/transformers/issues/40953  ·  opened by yuanhangsu1986 on 2025-09-17

## [body] yuanhangsu1986 (NONE) · opening post · 2025-09-17
### System Info

- `transformers` version: 4.57.0.dev0
- Platform: Linux-5.4.0-80-generic-x86_64-with-glibc2.35
- Python version: 3.10.12
- Huggingface_hub version: 0.34.5
- Safetensors version: 0.5.3
- Accelerate version: 1.6.0
- Accelerate config:    not found
- DeepSpeed version: not installed
- PyTorch version (accelerator?): 2.6.0+cu124 (CUDA)
- Tensorflow version (GPU?): not installed (NA)
- Flax version (CPU?/GPU?/TPU?): not installed (NA)
- Jax version: not installed
- JaxLib version: not installed
- Using distributed or parallel set-up in script?: <fill in>
- Using GPU in script?: <fill in>
- GPU type: Tesla V100-SXM3-32GB-H

### Who can help?

@SunMarc @zucchini-nlp @vasqu @ArthurZucker @Cyrilvallez 
Running inference with TP with gpt-oss-120b model on a node with 16 GPUs.
Got the following error in the cross attention layer:
```
[rank0]:   File "/usr/local/lib/python3.10/dist-packages/transformers/models/gpt_oss/modeling_gpt_oss.py", line 314, in forward                                                                                                                 [rank0]:     key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
[rank0]: RuntimeError: shape '[1, 89, -1, 64]' is invalid for input of size 2848
```

It seems the projected tensor shape, which is 2848, is less than the minimum allowed by the hidden_shape. This happens due to the TP partitioning of the projection matrix.

I'm wandering if changing the self.head_dim from 64 to 32 is a possible solution to this?
In other words, pseudo-code:
```
k_proj = self.k_proj(hidden_states)
if torch.numel(k_proj) < math.prod((*input_shape, 1, self.head_dim)):
    head_dim = self.head_dim / math.prod((*input_shape, 1, self.head_dim)) * torch.numel(k_proj)
...
```

One thing to note that is the GPUs we used is v100, which is why we need that many GPUs to run gpt-oss-120b.
[Here ](https://huggingface.co/openai/gpt-oss-20b/discussions/61) is the official fix for running gpt-oss on older GPUs.

### Information

- [x] The official example scripts
- [ ] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [x] My own task or dataset (give details below)

### Reproduction

To reproduce:
- python code
gpt-oss-120b.py
```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
quantization_config = Mxfp4Config(dequantize=False)
[… 22 lines of code omitted]
```

- Launch job with torchrun
```
torchrun --nproc-per-node 16 gpt-oss-120b.py
```

### Expected behavior

Should run without error

## [c3311635988] Cyrilvallez (MEMBER) · issue_comment · 2025-09-19
Hey! Indeed, as `num_key_value_heads` is 8, you won't be able to shard on more than 8 gpus with TP. Changing the head_dim would not work, as then computation of the attention is not equivalent anymore.

If you want to keep using TP, what you could do is change the sharding of q/k/v to colwise_rep, and o to rowwise_rep. It will be less efficient as the gpus will need to sync more and each of them will perform attention on all heads, but you should not have any shape issue anymore as the outputs will be gathered!

## [c3311640557] Cyrilvallez (MEMBER) · issue_comment · 2025-09-19
Note that it's not a bug, simply how sharding works and its limitations. You would not be able to shard on only 3 gpus either (it would need to be a multiple)

## [c3312422293] yuanhangsu1986 (NONE) · issue_comment · reply to [c3311635988] · 2025-09-19
> Hey! Indeed, as `num_key_value_heads` is 8, you won't be able to shard on more than 8 gpus with TP. Changing the head_dim would not work, as then computation of the attention is not equivalent anymore.
> 
> If you want to keep using TP, what you could do is change the sharding of q/k/v to colwise_rep, and o to rowwise_rep. It will be less efficient as the gpus will need to sync more and each of them will perform attention on all heads, but you should not have any shape issue anymore as the outputs will be gathered!

Ah, makes sense.

I have a follow up question. Since 16 V100s wouldn't work for gpt-oss-120b with TP (at least no stratightforward solution), and 8 V100s will get OOM, do you have any suggestions for making the model work on V100? Have we tried that before or you may have any idea in how to approach this?

Many thanks!

## [c3312426060] yuanhangsu1986 (NONE) · issue_comment · reply to [c3311640557] · 2025-09-19
> Note that it's not a bug, simply how sharding works and its limitations. You would not be able to shard on only 3 gpus either (it would need to be a multiple)

Exactly, my bad.

## [c3323000463] Cyrilvallez (MEMBER) · issue_comment · 2025-09-23
Humm, given that the model is already quantized, I'm not sure if you can quantize further (I don't remember if all the potential weights are quantized already or not, you can check that). If you only OOM at runtime though (i.e. not at loading time), you could try kv cache offloading! `model.generate(..., cache_implementation="offloaded")` which will offload the cache to cpu

---
outcome: state=closed
