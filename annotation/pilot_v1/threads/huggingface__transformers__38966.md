# #38966 [issue] Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models
https://github.com/huggingface/transformers/issues/38966  ·  opened by AshAnand34 on 2025-06-21

## [body] AshAnand34 (CONTRIBUTOR) · opening post · 2025-06-21
### Model description

Nemotron-H is a family of hybrid Mamba-Transformer models developed by NVIDIA that combines the efficiency of Mamba layers with the accuracy of Transformer architecture. The models come in two sizes:
- 8B parameter model
- 56B/47B parameter models (with a compressed 47B version using MiniPuzzle compression)

**Key Features:**
- **Hybrid Architecture:** Replaces majority of self-attention layers with Mamba layers for constant computation and memory per token
- **Superior Performance:** Up to 3x faster inference compared to similarly-sized state-of-the-art Transformer models
- **Competitive Accuracy:** Achieves better or on-par accuracy compared to Qwen-2.5-7B/72B and Llama-3.1-8B/70B
- **FP8 Training:** Introduces FP8-based training recipe achieving on-par results with BF16 training
- **Compression Technique:** MiniPuzzle compression reduces 56B model to 47B while maintaining accuracy and improving inference speed by 20%

**Technical Innovations:**
- Constant computation and memory requirements per generated token
- Novel compression via pruning and distillation technique (MiniPuzzle)
- FP8 training recipe for efficient training
- Hybrid Mamba-Transformer architecture optimization

### Open source status

- [x] The model implementation is available
- [x] The model weights are available

### Provide useful links for the implementation

HF model: https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/tree/main
arXiv paper: https://arxiv.org/abs/2504.03624

I would definitely love to integrate this into Transformers.

## [c3053557025] Ruhaan838 (NONE) · issue_comment · 2025-07-09
Are you working on this model, or can I contribute to it?

## [c3053609893] AshAnand34 (CONTRIBUTOR) · issue_comment · reply to [c3053557025] · 2025-07-09
> Are you working on this model, or can I contribute to it? 

I will be working on this model.

## [c3057166726] Ruhaan838 (NONE) · issue_comment · 2025-07-10
I have one thing to tell you: the original author of the paper has already included this model in the Transformer Library, so you don't need to add it again. 

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer  = AutoTokenizer.from_pretrained("nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("nvidia/Nemotron-H-8B-Base-8K", torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

prompt = "When was NVIDIA founded?"

outputs = model.generate(**tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device))
print(tokenizer.decode(outputs[0]))
```

## [c3057704714] AshAnand34 (CONTRIBUTOR) · issue_comment · reply to [c3057166726] · 2025-07-10
> I have one thing to tell you: the original author of the paper has already included this model in the Transformer Library, so you don't need to add it again. 
> 
> ```python
> import torch
> from transformers import AutoTokenizer, AutoModelForCausalLM
> 
> # Load the tokenizer and model
> tokenizer  = AutoTokenizer.from_pretrained("nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True)
> model = AutoModelForCausalLM.from_pretrained("nvidia/Nemotron-H-8B-Base-8K", torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
> 
> prompt = "When was NVIDIA founded?"
> 
> outputs = model.generate(**tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device))
> print(tokenizer.decode(outputs[0]))
> ```
> 

Oh really? I did not realize that. Thanks for telling me about it.

## [c3605447012] 1himan (CONTRIBUTOR) · issue_comment · reply to [c3057704714] · 2025-12-03
@AshAnand34, I think you should close this issue.

---
outcome: state=closed
