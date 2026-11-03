# #38403 [issue] Qwen2.5-VL-7B-Instruct model keys are different between a saved model and downloaded one
https://github.com/huggingface/transformers/issues/38403  ·  opened by harshit2997 on 2025-05-27

## [body] harshit2997 (NONE) · opening post · 2025-05-27
### System Info

- `transformers` version: 4.52.3
- `bitsandbytes` version: 0.45.5
- `vLLM` version: 0.8.5.post1
- Platform: Linux-5.15.0-1045-azure-x86_64-with-glibc2.35
- Python version: 3.10.16
- Huggingface_hub version: 0.32.1
- Safetensors version: 0.5.3
- Accelerate version: 1.7.0
- Accelerate config:    not found
- DeepSpeed version: not installed
- PyTorch version (GPU?): 2.6.0+cu124 (True)
- Tensorflow version (GPU?): not installed (NA)
- Flax version (CPU?/GPU?/TPU?): not installed (NA)
- Jax version: not installed
- JaxLib version: not installed
- Using distributed or parallel set-up in script?: no
- Using GPU in script?: yes
- GPU type: NVIDIA A100 80GB PCIe

### Who can help?

I have been trying to get fast inference on Qwen-2.5-VL-7B-Instruct through vLLM and have been successful. But to get more throughput, I tried saving a bitsandbytes 4 bit quantized model locally and loading it into vLLM. It fails with key mismatch. To root cause, I tried doing the same thing with just loading a transformers model, saving it locally, and then loading it into vLLM. It failed again. Both times the error message was:

```
ERROR 05-27 12:50:03 [core.py:396]   File "/home/aiscuser/.conda/envs/slm/lib/python3.10/site-packages/vllm/model_executor/models/qwen2.py", line 405, in load_weights
ERROR 05-27 12:50:03 [core.py:396]     param = params_dict[name]
ERROR 05-27 12:50:03 [core.py:396] KeyError: 'language_model.embed_tokens.weight'
```

Investigating more, when I download the model using `huggingface-cli` and check in the cache, the `model.safetensors.index.json` file has different keys as compared to the `model.safetensors.index.json` of the locally saved model (even unquantized).

Some lines from `model.safetensors.index.json` of the model from cache:
```
    "lm_head.weight": "model-00005-of-00005.safetensors",
    "model.embed_tokens.weight": "model-00001-of-00005.safetensors",
    "model.layers.0.input_layernorm.weight": "model-00001-of-00005.safetensors",
    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00005.safetensors",
    "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00005.safetensors",
    "model.layers.0.mlp.up_proj.weight": "model-00001-of-00005.safetensors",
    "model.layers.0.post_attention_layernorm.weight": "model-00001-of-00005.safetensors",
    "visual.blocks.0.attn.proj.bias": "model-00001-of-00005.safetensors",
    "visual.blocks.0.attn.proj.weight": "model-00001-of-00005.safetensors",
    "visual.blocks.0.attn.qkv.bias": "model-00001-of-00005.safetensors",
    "visual.blocks.0.attn.qkv.weight": "model-00001-of-00005.safetensors",
    "visual.blocks.0.mlp.down_proj.bias": "model-00001-of-00005.safetensors",
    "visual.blocks.0.mlp.down_proj.weight": "model-00001-of-00005.safetensors",
```

Corresponding lines from `model.safetensors.index.json` of the locally saved model:
```
    "lm_head.weight": "model-00004-of-00004.safetensors",
    "model.language_model.embed_tokens.weight": "model-00001-of-00004.safetensors",
    "model.language_model.layers.0.input_layernorm.weight": "model-00001-of-00004.safetensors",
    "model.language_model.layers.0.mlp.down_proj.weight": "model-00001-of-00004.safetensors",
    "model.language_model.layers.0.mlp.gate_proj.weight": "model-00001-of-00004.safetensors",
    "model.language_model.layers.0.mlp.up_proj.weight": "model-00001-of-00004.safetensors",
    "model.language_model.layers.0.post_attention_layernorm.weight": "model-00001-of-00004.safetensors",
    "model.visual.blocks.0.attn.proj.bias": "model-00001-of-00004.safetensors",
    "model.visual.blocks.0.attn.proj.weight": "model-00001-of-00004.safetensors",
    "model.visual.blocks.0.attn.qkv.bias": "model-00001-of-00004.safetensors",
    "model.visual.blocks.0.attn.qkv.weight": "model-00001-of-00004.safetensors",
    "model.visual.blocks.0.mlp.down_proj.bias": "model-00001-of-00004.safetensors",
    "model.visual.blocks.0.mlp.down_proj.weight": "model-00001-of-00004.safetensors",
```

This is also probably why vLLM can load the model when I just give it the HF model name but cannot load it from the locally saved model. But my question is why is this happening? Why is the locally saved model different?

Code used to locally save the unquantized model:

```
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
[… 15 lines of code omitted]

Code used to save the quantized model:
```
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda:0"
[… 22 lines of code omitted]
```

Code used to load the model into vLLM
```
import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

model_path = "/home/aiscuser/qwen7b"
[… 11 lines of code omitted]
```

### Information

- [ ] The official example scripts
- [x] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [x] My own task or dataset (give details below)

### Reproduction

Steps to repro:

1. Run the script to save the model locally. Just running the one for unquantized should be fine.
2. Now run the script to try loading the model into vLLM.

### Expected behavior

Model keys should be same between downloaded and locally saved model. Subsequently, vLLM should load the model correctly.

## [c2912545846] zucchini-nlp (MEMBER) · issue_comment · 2025-05-27
Fixed by https://github.com/huggingface/transformers/pull/38385 in `main` and we'll be doing a patch release soon. Closing as resolved

---
outcome: state=closed
