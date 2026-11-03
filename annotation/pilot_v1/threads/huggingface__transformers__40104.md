# #40104 [issue] router_logits not getting populated on  output_router_logits=True for GptOssForCausalLM
https://github.com/huggingface/transformers/issues/40104  ·  opened by qrdlgit on 2025-08-12

## [body] qrdlgit (NONE) · opening post · 2025-08-12
### System Info

On kaggle - https://www.kaggle.com/code/kaggleqrdl/router-logits-visualization?scriptVersionId=255623848 see https://cookbook.openai.com/articles/gpt-oss/run-transformers and https://github.com/huggingface/transformers/pull/39940

- `transformers` version: 4.56.0.dev0
- Platform: Linux-6.6.56+-x86_64-with-glibc2.35
- Python version: 3.11.13
- Huggingface_hub version: 0.34.4
- Safetensors version: 0.5.3
- Accelerate version: 1.8.1
- Accelerate config: 	not found
- DeepSpeed version: not installed
- PyTorch version (accelerator?): 2.8.0+cu128 (CUDA)
- Tensorflow version (GPU?): 2.18.0 (True)
- Flax version (CPU?/GPU?/TPU?): 0.10.6 (gpu)
- Jax version: 0.5.2
- JaxLib version: 0.5.1
- Using distributed or parallel set-up in script?: yes
- Using GPU in script?: yes
- GPU type: Tesla T4

### Who can help?

@SunMarc @MekkCyber (tagging as they are in the kernels pr for this model)

### Information

- [x] The official example scripts
- [x] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [ ] My own task or dataset (give details below)

### Reproduction

When I pass   output_router_logits=True to out=model(.., the out.router_logits is none.   I believe the issue is a combination of having kernels installed and being used and https://github.com/huggingface/transformers/blob/913c0a8c334afcbb42d8ee74567ff9ed0344b11d/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L394C67-L394C68  

Any brief comments on the underlying technical solution to this problem would be greatly appreciated!  I have a workaround of sorts in  https://www.kaggle.com/code/kaggleqrdl/router-logits-visualization?scriptVersionId=255623848  and comments would be helpful given the tight timeline.  Also note the code for https://github.com/huggingface/transformers/blob/913c0a8c334afcbb42d8ee74567ff9ed0344b11d/src/transformers/models/gpt_oss/modeling_gpt_oss.py#L685 

I nulled that out as we're not training for this task.

### Expected behavior

out.router_logits should be populated via the OutputRecorder when kernels is used

## [c3180287166] qrdlgit (NONE) · issue_comment · 2025-08-12
I've switched to using hooks, so less of an issue.

## [c3184845843] ArthurZucker (COLLABORATOR) · issue_comment · 2025-08-13
Sorry for the delay, and yes your hack works well, cc @SunMarc if you can check when you come back from vacations!

---
outcome: state=closed
