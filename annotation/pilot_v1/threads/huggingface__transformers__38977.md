# #38977 [issue] LMHead is processing redundant tokens in prefill
https://github.com/huggingface/transformers/issues/38977  ·  opened by null-pointer-access on 2025-06-23

## [body] null-pointer-access (CONTRIBUTOR) · opening post · 2025-06-23
While using `GPT2LMHeadModel.generate()` and compare its performance with vLLM, I noticed a significant inefficiency in the `forward()` implementation of many huggingface models. For example, in the `GPT2LMHeadModel.forward`, `self.lm_head` is applied to all token hidden states, even when called from the `generate()` method, where only the logits of the last token are needed for next-token prediction. This computes logits over the entire sequence and can introduce significant overhead.

```py
# src/transformers/models/gpt2/modeling_gpt2.py, line 1233
lm_logits = self.lm_head(hidden_states)
```

Suggested Fix: add a conditional branch in forward() to slice the hidden states before computing logits if it’s a generation step.

## [c2995563862] zucchini-nlp (MEMBER) · issue_comment · reply to [body] · 2025-06-23
Hey @null-pointer-access !

We have a similar way to drop unnecessary tokens in generation in most recent models (e.g. Llama) though the change wasn't propagated to all language models in the library. Would you like to open a PR to add smth similar to below code snippet in GPT models?

https://github.com/huggingface/transformers/blob/2166b6b4ff09f6dd3867ab982f262f66482aa968/src/transformers/models/llama/modeling_llama.py#L565-L567

## [c3001430117] null-pointer-access (CONTRIBUTOR) · issue_comment · reply to [c2995563862] · 2025-06-24
> Hey [@null-pointer-access](https://github.com/null-pointer-access) !
> 
> We have a similar way to drop unnecessary tokens in generation in most recent models (e.g. Llama) though the change wasn't propagated to all language models in the library. Would you like to open a PR to add smth similar to below code snippet in GPT models?
> 
> [transformers/src/transformers/models/llama/modeling_llama.py](https://github.com/huggingface/transformers/blob/2166b6b4ff09f6dd3867ab982f262f66482aa968/src/transformers/models/llama/modeling_llama.py#L565-L567)
> 
> Lines 565 to 567 in [2166b6b](/huggingface/transformers/commit/2166b6b4ff09f6dd3867ab982f262f66482aa968)
> 
>  # Only compute necessary logits, and do not upcast them to float if we are not computing the loss 
>  slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep 
>  logits = self.lm_head(hidden_states[:, slice_indices, :])

Sure! I'll submit a PR soon

## [c3001493046] null-pointer-access (CONTRIBUTOR) · issue_comment · 2025-06-24
Fixed in #39016

---
outcome: state=closed
