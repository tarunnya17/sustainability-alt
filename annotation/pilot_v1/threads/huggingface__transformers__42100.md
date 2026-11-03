# #42100 [issue] Qwen3Moe models have bug during generation since they calculate unnecessary load_balancing_loss
https://github.com/huggingface/transformers/issues/42100  ·  opened by zhangchong25 on 2025-11-07

## [body] zhangchong25 (NONE) · opening post · 2025-11-07
### System Info

Qwen3Moe models will calculate load_balancing_loss during evaluation, which will cause bug during generation on the 2nd (and later) steps. 

The problem can be handled by modifying 

```python
if output_router_logits:
    aux_loss = load_balancing_loss_func(
    ...
```

to

```python
if output_router_logits and self.training:
    aux_loss = load_balancing_loss_func(
    ...
```

### Who can help?

_No response_

### Information

- [ ] The official example scripts
- [ ] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [ ] My own task or dataset (give details below)

### Reproduction

Perform model.generate() using a Qwen3MoeForCausalLM model. 

### Expected behavior

Generate successfully.

## [c3513478559] diegoakel (CONTRIBUTOR) · issue_comment · 2025-11-10
cc @Rocketknight1 I only did it for the Qwen models but the pattern is the same on other MoE, should I add the fix here already?

I think there was a bug on the test for qwen3-moe, it was using eval instead of train for training tasks.

## [c3517521606] Rocketknight1 (MEMBER) · issue_comment · 2025-11-11
Hey, just following up on this - `aux_loss` should only be calculated when `output_router_logits` is `True`, but that defaults to `False` for all the `Qwen` models I checked. Is `model.generate()` setting that flag somehow? I'd like to understand how this bug is happening before we start fixing it.

## [c3528275779] diegoakel (CONTRIBUTOR) · issue_comment · 2025-11-13
yeah I was trying to replicate the bug now with `model.eval()` mode and the only way I managed to get a not None `aux_loss` was by directly passing`output_router_logits=True` on the model kwargs for the forward pass

## [c3532908145] Rocketknight1 (MEMBER) · issue_comment · 2025-11-14
In that case, I'm not sure this is a bug! Generating with `output_router_logits=True` (rather than generating without it and then doing a single forward pass with it at the end) is a strange thing to do, so it's not clear that we need to support it

## [c3606077211] zhangchong25 (NONE) · issue_comment · reply to [c3532908145] · 2025-12-03
> In that case, I'm not sure this is a bug! Generating with `output_router_logits=True` (rather than generating without it and then doing a single forward pass with it at the end) is a strange thing to do, so it's not clear that we need to support it

@Rocketknight1 
Hi Matt,
Thank you for noticing this issue. 
Router logits in generation is needed when doing interpretation studies. Please consider to support it for flexible use.

---
outcome: state=closed
