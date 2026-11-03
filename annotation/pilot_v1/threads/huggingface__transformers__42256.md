# #42256 [pull_request] Integrate Core AutoAWQ Inference Components into Transformers
https://github.com/huggingface/transformers/pull/42256  ·  opened by fanqiNO1 on 2025-11-18

## [body] fanqiNO1 (CONTRIBUTOR) · opening post · 2025-11-18
# What does this PR do?

Since the `AutoAWQ` repository has been archived, this PR proposes integrating its essential inference-related components into `transformers` to ensure continued support and maintenance.

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


## `AutoAWQ` Module Analysis

| Module               | Description                                      | Integration Plan |
|----------------------|--------------------------------------------------|------------------|
| `awq.evaluation`     | Model evaluation utilities                       | Not integrated — out of scope for inference. |
| `awq.models`         | Model wrappers (e.g., `AutoAWQForCausalLM`)      | Not integrated — redundant with existing `transformers` model classes. |
| `awq.modules.fused`  | Fused attention, MLP, MoE, RMSNorm               | Integrated with simplifications. **Depends on `autoawq-kernels`.** |
| `awq.modules.linear` | Quantized linear layers (`WQLinear_*`)           | This module is required; however, it remains to be decided whether to integrate it directly or retain a dependency on `AutoAWQ`. Additionally, newly introduced models may bring compatibility issues—for instance ([Qwen3 support](https://github.com/casper-hansen/AutoAWQ/pull/751)). **Depends on `autoawq-kernels`.** |
| `awq.modules.triton` | Fallback `gemm` in Triton                        | Likely unnecessary if `autoawq-kernels` is available; excluded for now. |
| `awq.modules.act`    | `ScaledActivation`                               | Integrated — lightweight and self-contained. |
| `awq.quantize`       | Quantization logic                               | Not integrated — focus is on inference only. |
| `awq.utils`          | Utilities used by fused/linear modules           | Partially integrated — only relevant helpers included. |

`AutoAWQ-Kernels` consists of the following four components:

- awq_ext: Kernels for `WQLinear_GEMM` in `awq.modules.linear`.
- awq_v2_ext: Kernels for `WQLinear_GEMVFast` in `awq.modules.linear`. (We do not need this component, as we do not use `WQLinear_GEMVFast`; however, support could potentially be added in the future.)
- exl_ext: Kernels for `WQLinear_Exllama` in `awq.modules.linear`.
- exlv2_ext: Kernels for `WQLinear_ExllamaV2` in `awq.modules.linear`.

## Design Proposal

To maintain backward compatibility and minimize disruption:

```bash
transformers/integrations/
├── awq/
│   ├── awq_integration.py   # Replaces current AWQ integration logic
│   ├── autoawq_fused.py     # Fused modules (attention, MLP, etc.)
│   ├── autoawq_linear.py    # Quantized linear layers
│   ├── autoawq_utils.py     # Utilities + ScaledActivation
│   └── init.py          # Preserves existing transformers.integrations.awq API
```

- All existing usage of `transformers.integrations.awq` remains functional.
- New components are internal implementation details and not exposed directly to users.

@SunMarc @MekkCyber

## [c3547273607] MekkCyber (CONTRIBUTOR) · issue_comment · reply to [body] · 2025-11-18
Hi @fanqiNO1!  

Thanks a lot for taking the time to work on this!  

From my side, the repository structure you suggest makes sense — I don’t have any strong opinions as long as the API exposed from `awq.py` previously remains unchanged.  

Just an FYI: we’re currently refactoring the quantization API, but since AWQ will be inference-only, I don’t expect any friction for you.  

For `autoawq_kernels`, I’d suggest adding them to `kernels-community`. That repo is used to build kernels on the hub and host prebuilt shared libraries for different Torch/CUDA versions and capabilities — which makes it easier to use than building kernels from source.  

Regarding the quantized linear layers, I agree it’s better to keep them in `transformers`, since we want to minimize the dependency on AutoAWQ. That said, I’ll let @SunMarc make the final call, as he has more experience than me with AWQ.  

For anything related to kernels or quantization, feel free to reach out — I can help you build the kernels and add them to the community repo.

## [c3547294980] fanqiNO1 (CONTRIBUTOR) · issue_comment · reply to [c3547273607] · 2025-11-18
> Hi @fanqiNO1!
> 
> Thanks a lot for taking the time to work on this!
> 
> From my side, the repository structure you suggest makes sense — I don’t have any strong opinions as long as the API exposed from `awq.py` previously remains unchanged.
> 
> Just an FYI: we’re currently refactoring the quantization API, but since AWQ will be inference-only, I don’t expect any friction for you.
> 
> For `autoawq_kernels`, I’d suggest adding them to `kernels-community`. That repo is used to build kernels on the hub and host prebuilt shared libraries for different Torch/CUDA versions and capabilities — which makes it easier to use than building kernels from source.
> 
> Regarding the quantized linear layers, I agree it’s better to keep them in `transformers`, since we want to minimize the dependency on AutoAWQ. That said, I’ll let @SunMarc make the final call, as he has more experience than me with AWQ.
> 
> For anything related to kernels or quantization, feel free to reach out — I can help you build the kernels and add them to the community repo.

Thank you very much for your reply! I will migrate `AutoAWQ-Kernels` to `kernels-community` next, and I might need your help again at that time~ Thanks!

## [c3548758580] SunMarc (MEMBER) · issue_comment · 2025-11-18
Thanks for this nice design proposal ! If there are too much code to just put in a awq.py file, feel free to divide it as you proposed ! As for the Quantized Linear, let's maybe focus first on `GEMM` as this is the most used version. The other are most likely not that used and we can always integrate them later if needed / ask the community to do it ! 
GEMM rely on three implementation (awq_ext, triton or naive version). We can probably try to benchmark awq_ext and triton to see what is worth keeping and we can put in `kernels-community`. Hopefully the triton version is good as it is easier to deal with triton kernels. We can probably discard the naive one. 

```python
        if awq_ext is not None:
            FP16_MATMUL_HEURISTIC_CONDITION = x.shape[0] * x.shape[1] >= 1024

            if FP16_MATMUL_HEURISTIC_CONDITION:
[… 27 lines of code omitted]

## [c3559026690] SunMarc (MEMBER) · issue_comment · reply to [c3547294980] · 2025-11-20
Just to let you know @fanqiNO1, there is a creator of gptq-model who wants to maintain the awq inference backend. We are still thinking what could be the best solution.

---
outcome: state=open, merged=False
