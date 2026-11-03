# #39682 [issue] Accelerate beam search decoding via tree attention
https://github.com/huggingface/transformers/issues/39682  ·  opened by transcend-0 on 2025-07-25

## [body] transcend-0 (NONE) · opening post · 2025-07-25
### Feature request

Provide an optional `tree_attention` flag in `model.generate()` that enables Tree-Attention decoding for beam search.
Tree-Attention reuses the common prefixes among beams by organizing them into a prefix tree, reducing redundant computations and improving inference speed.

### Motivation

When `num_beams>1`, the current beam search recomputes for every beam separately even when most tokens are shared.
Tree-Attention fuses these shared prefixes into a single tree-structured batch, cutting both FLOPs and KV Cache.
This method is theoretically lossless and has a significant speedup [1].

[image]

[1] Yao et al. DeFT: Flash Tree-attention with IO-Awareness for Efficient Tree-search-based LLM Inference. In *ICLR 2024 Workshop*.

### Your contribution

I have simply implemented it on `transformers 4.41–4.45` and released it at [BeamSD](https://github.com/transcend-0/BeamSD). I’m happy to port this feature to the latest `transformers` version. 

To achieve the minimum modification, we could only add code immediately before and after the line
`model_outputs = self(**model_inputs, return_dict=True)`
(`line 4079` of `generation/utils.py` in `transformers 4.53.3`).

The idea is:

1. Before it: Convert the batched beam `model_inputs` into a single-batch tree.
2. After it: Convert the returned `model_outputs` back to the original layout so that the rest of the generate loop remains unchanged.

May I open a Pull Request for this idea? I’m happy to contribute the code and the doc.
@gante

## [c3151183896] transcend-0 (NONE) · issue_comment · 2025-08-04
@gante @zucchini-nlp

## [c3151449133] gante (CONTRIBUTOR) · issue_comment · reply to [c3151183896] · 2025-08-04
Hey @transcend-0! 👋 Thank you for opening this issue!

TL;DR I’d like to first invite you to add the proposed method as a custom generate method on the Hub ([docs](https://huggingface.co/docs/transformers/generation_strategies#custom-decoding-methods)). This will make the method immediately available to all `transformers` users, with all models, using the well known `generate` interface.

_____________________
Full reply:

We are in a debloating phase of our library as a whole, and we want to improve the quality of frequently used tools. Fewer features with better tests, documentation, and optimizations = better UX and DevX.

For `generate`, each new decoding method (/kv cache/...) implies having a new set of tests with all compatible models, forward maintenance for years, and often a few more flags in `generate` or `GenerationConfig` -- a heavy price for our users and our team to pay. As such, existing infrequently-used decoding methods are being moved to the Hub, and new decoding methods should be added to the Hub first. Then, if the method gets high usage, we will port it into `transformers` (ideally with your help 🤗). We hope that, by following this two-step process, the library improves its quality for everyone, while also lowering the barrier for experimentation. 

I'm here to answer any questions and to help you add your method to the Hub!

cc @manueldeprada

## [c3152848721] transcend-0 (NONE) · issue_comment · reply to [c3151449133] · 2025-08-05
@gante Thank you very much! I'll try the Hub to add new decoding methods 🤗

---
outcome: state=open
