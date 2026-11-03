# #38680 [issue] Add support for Orthogonal Residual Updates
https://github.com/huggingface/transformers/issues/38680  ·  opened by BootsofLagrangian on 2025-06-09

## [body] BootsofLagrangian (NONE) · opening post · 2025-06-09
### Feature request

This is a feature request to implement "Orthogonal Residual Update," a novel mechanism for training deep neural networks proposed in the paper "[Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks](https://arxiv.org/abs/2505.11881)".

The core idea is to modify the standard additive residual connection, which is pivotal in modern architectures like Transformers and ResNets. Instead of a standard update, $$x_{n+1} = x_{n} + f( σ(x_{n}))$$, the proposed method decomposes the module's output $$f(σ(x_{n} ))$$ into two components: one parallel to the input stream $$x_{n} $$ and one orthogonal to it. The update is then performed using only the orthogonal component.

This method has been shown to improve generalization accuracy and training stability across diverse architectures (including Vision Transformers and ResNetV2) and datasets. The implementation is computationally efficient, adding only a minimal number of FLOPs ($$O(sd)$$ in a Transformer block) compared to the main modules.

Paper: https://arxiv.org/abs/2505.11881
Original Code: https://github.com/BootsofLagrangian/ortho-residual

### Motivation

The motivation for this proposal is to address a potential inefficiency in standard residual connections. In current models, a module's learned transformation can predominantly scale or modulate the magnitude of the existing feature stream, potentially underutilizing the module's capacity for learning entirely new, complex features. This can lead to representational redundancy.

By explicitly encouraging modules to contribute novel information (i.e., new directional components orthogonal to the existing representation), the Orthogonal Residual Update mechanism provides a path toward more efficient and stable training, as well as better generalization. For instance, a ViT-B model with this update achieved a +4.3%p top-1 accuracy gain on ImageNet-1k.

Given that the principle of a linear residual stream is a core feature in most transformers models, integrating this mechanism as a configurable option could provide a significant and low-cost benefit to the community for a wide range of models and tasks.

### Your contribution

Yes, I can help by submitting a Issue. I have read the `CONTRIBUTING.MD` guide.

My proposed contribution would include:

- Implementing the core Orthogonal Residual Update logic in a reusable way that can be applied to different layers (e.g., Attention and MLP blocks).
- Adding a first concrete implementation with the OrthoViT model, including `OrthoViTConfig`, `OrthoViTModel`, and `OrthoViTForImageClassification`.
- Providing comprehensive tests for the new model and the orthogonal update logic to ensure correctness and integration with the library's testing framework.
 - Adding the necessary documentation for the new model, following the library's standards.
 - An OrthoViT model example : https://huggingface.co/BootsofLagrangian/ortho-vit-b-imagenet1k-hf

## [c2955949169] Rocketknight1 (MEMBER) · issue_comment · 2025-06-09
This is interesting, but it will be quite a large PR and I think we'd need to see more widespread use of this technique in the field before we add it to the `Trainer`! cc @sunmarc

---
outcome: state=open
