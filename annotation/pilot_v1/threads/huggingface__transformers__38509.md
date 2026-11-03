# #38509 [issue] SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference
https://github.com/huggingface/transformers/issues/38509  ·  opened by aryanchauhan31 on 2025-05-31

## [body] aryanchauhan31 (CONTRIBUTOR) · opening post · 2025-05-31
### Feature request

I’d like to request the addition of **SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference** to the 🤗 Transformers library.

**Paper**: [https://arxiv.org/abs/2410.04417](https://arxiv.org/abs/2410.04417)  
**Authors**: Yuan Zhang, Chun-Kai Fan, Junpeng Ma, et al.  
**Code**: No official repo yet (as of submission date)

SparseVLM is a training-free, text-guided visual token sparsification method. It prunes redundant image tokens layer-wise using self-attention weights and introduces token recycling to compress pruned information. It works with existing VLMs like BLIP, Flamingo, and VideoBLIP, reducing FLOPs and latency by up to 60% while preserving accuracy.

### Motivation

Vision-language models like BLIP, Flamingo, and others can be computationally heavy at inference time due to dense visual tokens. SparseVLM addresses this by pruning visual tokens **without retraining** or adding parameters. It offers an efficient, plug-and-play way to speed up inference for image and video tasks.

Integrating this into 🤗 Transformers could make existing VLMs much more usable in real-time applications and low-resource environments. It also fits perfectly with Hugging Face's ongoing work on efficient inference (e.g., bitsandbytes, quantization, MobileLLM).

### Your contribution

Yes! I’d be happy to contribute an initial implementation of SparseVLM, including the visual token selection mechanism, sparsification logic, and a wrapper for models like BLIP or ViT.

Once the authors release a reference implementation or weights, I can also help align it with their code for accuracy.

## [c2931225267] zucchini-nlp (MEMBER) · issue_comment · reply to [body] · 2025-06-02
Hey @aryanchauhan31 !

Thanks for sharing, compressing number of tokens in vision LLMs is a nice way to get higher performance. I think the strategy in SparseVLM is similar to the recent work in KV cache compression techniques (for ex we got a request for LagKV https://github.com/huggingface/transformers/issues/38312)

We recently added a custom generation code support from the hub ([docs](https://github.com/huggingface/transformers/pull/38055#issuecomment-2900553588)), to give users and researchers more freedom to share their generation techniques in a transformers-compatible way. I think SparseVLM can be implemented as a new type of cache and hosted on the hub. That way we don't get stuck in a long review cycle and can get vibe check from the community. If the feature becomes popular, we will add it in the core library 🤗 

cc @gante as well who added the custom generation support

## [c2936205654] aryanchauhan31 (CONTRIBUTOR) · issue_comment · reply to [c2931225267] · 2025-06-03
@zucchini-nlp Awesome, I’m totally up for this! I’ll go ahead and build SparseVLM as a custom generation strategy on the Hub and share benchmarks soon. Thanks for pointing me in the right direction 🙌
Also Do i keep this as a feature request or close this ?

## [c2936280583] zucchini-nlp (MEMBER) · issue_comment · reply to [c2936205654] · 2025-06-03
@aryanchauhan31 feel free to keep it open, in case you need assistance when shipping as a custom generation method 🤗

## [c2936299520] aryanchauhan31 (CONTRIBUTOR) · issue_comment · reply to [c2936280583] · 2025-06-03
@zucchini-nlp Thanks I'll start working on it :)

---
outcome: state=open
