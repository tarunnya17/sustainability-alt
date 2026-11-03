# #39136 [issue] bf16_full_eval=True moves model to device before FSDP application and causes cuda OOM
https://github.com/huggingface/transformers/issues/39136  ·  opened by jlu-figma on 2025-06-30

## [body] jlu-figma (NONE) · opening post · 2025-06-30
### System Info

Hi! Just sharing an interaction I found between accelerate's FSDP v2 code and `bf16_full_eval=True` in trainer.

1. Setting `bf16_full_eval=True` causes us to move the model to device over [here](https://github.com/huggingface/transformers/blob/b922b22ec2e458978dbd89038ad4b47885b34195/src/transformers/trainer.py#L2137-L2146).
2. During FSDP preparation, `accelerate` creates empty tensors for all of the optimizer states over [here](https://github.com/huggingface/accelerate/blob/a16d2bb3c1c2ac8029842c8baf2d03388baf09c7/src/accelerate/accelerator.py#L1493-L1500)
3. When the model is very large, e.g. a Qwen 32B model on an A100-80GB instance, this will cause an oom.

The solution I found for the time being is to simply set `bf16_full_eval=False`, but it would be great to support `bf16_full_eval` without oom.

### Who can help?

_No response_

### Reproduction

1. Load Qwen 2.5-32B on 8xA100-80GB instance.
2. Set `bf16_full_eval=True` in your training args (I also set `bf16=False` because FSDP does an internal upcast to float32 when that is set to true)
3. Before creating the `Trainer`, set the environment variable `FSDP_VERSION` to `"2"`. This will trigger `accelerate` to use FSDP v2 instead of FSDP v1.

### Expected behavior

It should at least make it to the training step before OOMing out.

## [c3045124687] ArthurZucker (COLLABORATOR) · issue_comment · 2025-07-07
cc @SunMarc !

## [c3045546753] SunMarc (MEMBER) · issue_comment · 2025-07-07
cc @S1ro1 for fsdpv2

## [c3183766228] S1ro1 (CONTRIBUTOR) · issue_comment · reply to [body] · 2025-08-13
Hi @jlu-figma sorry for late reply, must have missed it. We already have this changed in latest accelerate, now we only create tensors with 1 element, shouldn't OOM anymore. See [here](https://github.com/huggingface/accelerate/blob/23cf4ef8a3b58f016f63eeb158b4aa2c3e79fe6f/src/accelerate/accelerator.py#L1675), was added in this [PR](https://github.com/huggingface/accelerate/pull/3682/files#diff-2d7515874eaecac2687c7fc1a9c720be53f802bf14b4c3dcebe14ad443d075dcR1610)

## [c3279172623] ArthurZucker (COLLABORATOR) · issue_comment · 2025-09-11
Closing as fixed!

---
outcome: state=closed
