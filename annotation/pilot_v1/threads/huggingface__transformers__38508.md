# #38508 [issue] Model Request: SLaM (Sparse Latent Mixer) – Multimodal Flamingo Alternative
https://github.com/huggingface/transformers/issues/38508  ·  opened by aryanchauhan31 on 2025-05-31

## [body] aryanchauhan31 (CONTRIBUTOR) · opening post · 2025-05-31
### Model name
SLaM (Sparse Latent Mixer)

### Paper link
https://arxiv.org/abs/2405.12321

### Description
I came across the recent paper on **SLaM: Sparse Latent Mixer** and found it super exciting — it's a really promising direction for efficient multimodal modeling. SLaM builds on Flamingo-style architectures but introduces sparsity and latent tokens to make the cross-modal interactions much more efficient (and faster too).

It seems like a great candidate for integration into 🤗 Transformers, especially since there’s no open-source implementation yet.
Also, it introduces some novel ideas — like sparse latent mixing and cross-modal token selectors — that aren’t directly represented in existing models in the library.

Let me know if this sounds good — happy to get started!

---
outcome: state=closed
