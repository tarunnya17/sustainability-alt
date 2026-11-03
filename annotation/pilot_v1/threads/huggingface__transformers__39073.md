# #39073 [issue] Inefficient default GELU implementation in GPT2
https://github.com/huggingface/transformers/issues/39073  ·  opened by null-pointer-access on 2025-06-27

## [body] null-pointer-access (CONTRIBUTOR) · opening post · 2025-06-27
While profiling the HuggingFace GPT2 model, I found that the default GELU backend used is NewGELUActivation, which is inefficient in most cases. Instead of using a fused CUDA kernel, NewGELUActivation executes multiple separate PyTorch-level operators, leading to unnecessary kernel launches and memory overhead.

```python
# activations.py:L46
class NewGELUActivation(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
```

Is there a reason why NewGELUActivation is still used as the default for GPT2, rather than switching to nn.functional.gelu or another fused alternative?

I’d be happy to share profiler traces or help test a patch if helpful.

## [c3012406495] ArthurZucker (COLLABORATOR) · issue_comment · 2025-06-27
No real reasons AFAIK, do you want to open a PR to fix? Having test to make sure results are the same would also be welcome! 🤗

## [c3012406892] ArthurZucker (COLLABORATOR) · issue_comment · 2025-06-27
GPT2 is one of the oldest models!

## [c3177580183] null-pointer-access (CONTRIBUTOR) · issue_comment · reply to [c3012406495] · 2025-08-12
> No real reasons AFAIK, do you want to open a PR to fix? Having test to make sure results are the same would also be welcome! 🤗

Hi @ArthurZucker ! This is fixed in #40059 . Would you like to have a look at this?

---
outcome: state=closed
