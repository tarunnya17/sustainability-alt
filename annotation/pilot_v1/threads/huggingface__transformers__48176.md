# #48176 [issue] DisentangledFlash: fused Triton attention for DeBERTa-v2/v3
https://github.com/huggingface/transformers/issues/48176  ·  opened by delyan-boychev on 2026-08-21

## [body] delyan-boychev (NONE) · opening post · 2026-08-21
### Feature request

I’ve been working on **[DisentangledFlash](https://github.com/delyan-boychev/disentangled-flash)**, an open-source Triton implementation of DeBERTa-v2/v3 disentangled self-attention.

DeBERTa’s Content-to-Position (C2P) and Position-to-Content (P2C) terms make its attention incompatible with the standard optimized SDPA/FlashAttention path. As a result, the current DeBERTa implementation still relies on a dense eager attention path and is substantially slower than the optimized attention available to architectures that can use SDPA directly.

DisentangledFlash preserves the original DeBERTa attention formulation while evaluating it with a tiled, streaming Triton kernel that avoids materializing the full `[B, H, L, L]` attention matrix. On the full 12-layer encoder, the current implementation achieves a **1.75x geometric-mean FP16 speedup** over Transformers, with up to **1.93x speedup** in the tested configurations, while also reducing peak GPU memory at longer sequence lengths.

I’d like to discuss adding an optimized DeBERTa attention path to Transformers and what the preferred integration point would be. I’m happy to prepare the corresponding PR.

**Code:** https://github.com/delyan-boychev/disentangled-flash  
**PyPI:** https://pypi.org/project/disentangled-flash/

**Relevant papers:**
- DeBERTa: https://arxiv.org/abs/2006.03654
- FlashAttention: https://arxiv.org/abs/2205.14135

### Motivation

The main issue is that DeBERTa cannot directly benefit from the optimized attention backends that are now standard for many Transformer architectures.

Its attention score contains additional Content-to-Position (C2P) and Position-to-Content (P2C) terms on top of the usual content-to-content score. Because of this, simply replacing the current implementation with SDPA or FlashAttention would not preserve the original DeBERTa attention semantics.

The current Transformers implementation therefore computes the dense attention scores, adds the disentangled relative-position contribution, applies the mask, materializes the softmax probabilities, and then multiplies by `V`. This becomes increasingly expensive for larger batches and longer sequences, both in latency and intermediate GPU memory.

Related issues / prior work:

- [#28005 – Add SDPA support to more architectures](https://github.com/huggingface/transformers/issues/28005)
- [#18239 – TF2 DeBERTaV2 runs super slow on TPUs](https://github.com/huggingface/transformers/issues/18239)
- [#20815 – Cannot export DeBERTa to TorchScript](https://github.com/huggingface/transformers/issues/20815)
- [#27734 – DeBERTa TorchScript export PR](https://github.com/huggingface/transformers/pull/27734)
- [#18199 – Exported DeBERTa ONNX model is incorrect](https://github.com/huggingface/transformers/issues/18199)
- [#18237 – ONNX runtime error for DeBERTa-v3](https://github.com/huggingface/transformers/issues/18237)
- [#24194 – mDeBERTa ONNX export error](https://github.com/huggingface/transformers/issues/24194)
- [#35332 – `DisentangledSelfAttention` mixed-precision dtype issue](https://github.com/huggingface/transformers/issues/35332)
- [#35162 – DeBERTa attention `masked_fill` issue](https://github.com/huggingface/transformers/issues/35162)
- [#17152 – DeBERTa relative attention TPU/XLA compilation issue](https://github.com/huggingface/transformers/issues/17152)
- [#24194 - mDeBERTa-v3 uses the same DeBERTaV2 implementation and has historically had backend/export limitations](https://github.com/huggingface/transformers/issues/24194)

DeBERTa-v2/v3 disentangled attention remains in active use in modern systems including [GLiNER](https://huggingface.co/urchade/gliner_base)/[GLiNER2](https://huggingface.co/fastino/gliner2-base-v1) information extraction, [Meta Prompt Guard](https://huggingface.co/meta-llama/Prompt-Guard-86M), multilingual NLI systems, LLM-Blender pairwise ranking, and OpenAssistant reward models. Several of these downstream releases collectively receive millions of downloads per month on Hugging Face.

DisentangledFlash addresses this by evaluating the same C2P/P2C formulation using a tiled streaming Triton kernel with online softmax. QK computation, relative-score lookup, masking, softmax, and PV accumulation are performed without materializing the full `[B, H, L, L]` attention score/probability tensor.

I benchmarked the complete 12-layer DeBERTa encoder using eager inference on an NVIDIA RTX 6000 Ada:

| Precision | Geometric-mean speedup vs. Transformers |
|---|---:|
| FP16 | **1.75x** |
| strict FP32 | **1.56x** |

Representative FP16 results:

| Batch | Sequence length | Transformers | DisentangledFlash | Speedup |
|---:|---:|---:|---:|---:|
| 8 | 512 | 21.94 ms | 12.03 ms | **1.82x** |
| 16 | 512 | 52.37 ms | 27.19 ms | **1.93x** |
| 32 | 512 | 108.11 ms | 57.84 ms | **1.87x** |

At batch size 32 and sequence length 512, peak allocated GPU memory decreases from approximately:

- FP16: **2.09 GB -> 1.55 GB**
- FP32: **4.12 GB -> 3.09 GB**

The benchmark covers batch sizes `1, 8, 16, 32` and sequence lengths `16, 32, 64, 128, 256, 384, 512`.

The implementation has also been validated directly against the Transformers reference path:

| Precision | Level | Max absolute error | Mean absolute error |
|---|---|---:|---:|
| FP16 | Attention | **7.63e-6** | **2.48e-7** |
| FP16 | Full encoder | **1.56e-2** | **1.12e-3** |
| FP32 | Attention | **4.89e-9** | **1.79e-10** |
| FP32 | Full encoder | **7.57e-6** | **6.26e-7** |

I additionally tested `microsoft/deberta-v2-xlarge-mnli`. The original Transformers model and the same checkpoint using DisentangledFlash achieved full task-level parity, including identical final MNLI predictions.

The current public implementation is focused on inference. I’m also working on a corresponding Triton backward/training kernel so that the same approach can eventually accelerate DeBERTa training as well.

### Your contribution

The implementation is already available at:

https://github.com/delyan-boychev/disentangled-flash

and is published on PyPI:

https://pypi.org/project/disentangled-flash/

I can prepare the PR and adapt the implementation to whichever integration approach the Transformers maintainers prefer.

If the low-level Triton kernel would be better hosted through `kernels-community`, I’m also happy to adapt and contribute it there together with the necessary Transformers-side integration.

I’m also currently working on the backward/training Triton kernel and would be happy to align that work with the upstream integration direction.

## [c5464108183] greekera1000 (NONE) · issue_comment · 2026-08-29
**Independent testing on NVIDIA T4 (Turing)**

I ran the repo's validation and benchmark scripts, plus an end-to-end parity check, on an **NVIDIA Tesla T4 (Turing, compute 7.5)** — a different architecture from the RTX 6000 Ada the original numbers were measured on. Summary: the kernel is numerically sound on the real DeBERTa path and task predictions match HF exactly, the speedup is very hardware/shape dependent on Turing, and I hit one NaN edge case. Environment: Kaggle T4, current `transformers`, Triton FP16 unless noted.

**1. Correctness — faithful on the bidirectional path**

From `validation.validate_cuda`, the Triton op matches the reference for the `none` mask (standard bidirectional DeBERTa attention):

```
backend  dtype   max abs err   mean abs err
triton   fp16    1.53e-5       7.83e-7
triton   fp32    7.45e-9       1.38e-9
triton   bf16    1.22e-4       5.62e-6
```

The FP16 figures line up with the ~7.63e-6 claim; FP32 is essentially exact.

**2. End-to-end parity on `microsoft/deberta-v2-xlarge-mnli`** (batch 8, bucket 64, FP16)

```
predictions identical:       True
logits max abs error:        0.0061
probabilities max abs error: 0.0012
hidden max abs error:        0.1641
hidden mean abs error:       0.0042
```

The last-hidden-state drift (~0.16 max) accumulates through the stack but does **not** affect task output — logits differ by ≤0.006 and **every predicted label is identical** to the HF reference.

**3. Speed on T4 is strongly shape/model dependent**

```
model                    batch  seq  reference  triton    speedup
deberta-v2-xlarge-mnli   8      64   67.35 ms   55.96 ms  1.20x
deberta-v3-base          32     512  514 ms     614 ms    0.85x
```

It helps at short sequences on xlarge but regresses at long sequences on base — no 1.75x
anywhere on Turing, but not a uniform slowdown either. Memory was ~15% lower at batch 32 /
seq 512. Autotuning cold-start was also heavy (~20-30s per new shape).

**4. Possible bug: NaN for `left` directional masking at L ≥ 64**

In `validation.validate_cuda`, every Triton `both left` case at sequence length **≥ 64**
returns NaN — across fp16, bf16 **and** fp32. The PyTorch backend does not NaN on the same
inputs:

```
triton fp16 eager D=64 B=1 L=64  both left  max=nan mean=nan
triton fp16 eager D=64 B=1 L=127 both left  max=nan mean=nan
...  (same for bf16 and fp32, all L >= 64)
```

Looks like the classic fully-masked-row FlashAttention case (all-masked block → running-max
= -inf → exp(-inf − (-inf)) = NaN). It's on the `left` directional mode, which a
bidirectional DeBERTa encoder doesn't exercise, so it likely doesn't affect normal inference
— but it's a latent bug for causal/directional use. Also worth noting the final "Maximum
observed errors" summary reports clean finite values and silently omits the NaN rows, so the
suite currently passes despite producing NaNs; flagging non-finite outputs there would help.

Happy to share full logs or run more shapes if useful.

## [c5467837593] delyan-boychev (NONE) · issue_comment · reply to [c5464108183] · 2026-08-30
@greekera1000 Thanks a lot for testing! The auto-tuning soon will be enabled only for unsupported gpu models and there will be a a tuned bucketed kernel for this. The speed-up is logically mainly on longer sequences, but also the memory is a really the advantage here. Working on it and soon I will release the backward+tuned configurations. If you want to contribute with tuning config, when I set this up - I will ping you to run tuning on the T4.

## [c5468085262] greekera1000 (NONE) · issue_comment · reply to [c5467837593] · 2026-08-30
@delyan-boychev Sounds good — ping me whenever it's ready and I'll run the tuning on the T4.

Makes sense re: long sequences — what I saw was the untuned kernel on Turing, so tuned configs should flip that. Memory win was solid either way.

One thing in case it got buried: the Triton kernel NaNs on the left directional-mask cases at seq ≥ 64 (fp16/bf16/fp32; PyTorch backend is fine on the same inputs). Looks like a fully-masked-row edge case. Doesn't touch the normal bidirectional path, but the validation summary drops the NaN rows so the suite still passes — might be worth a non-finite check there.

## [c5468103929] delyan-boychev (NONE) · issue_comment · reply to [c5468085262] · 2026-08-30
@greekera1000 
Thanks again for testing this on T4 and for the detailed report.

Small clarification on the NaN case: `left` in the validator means **left padding**, not causal/left-directional attention, and `both` means both DeBERTa positional terms $\ c2p + p2c $. But the bug itself is real: with enough left padding, a valid query can hit a fully masked key tile first, causing the online-softmax update to encounter the \$-\infty - (-\infty)$ case. You’re also right that the current summary can hide NaNs, so I’ll fix the finite checks and add a regression test for this.

On performance, the lower T4 speedup is plausible. The published 1.75x is the geometric mean from RTX 6000 Ada, not an architecture-independent number. T4/Turing has a different balance of register capacity, occupancy, Tensor Core throughput, and memory bandwidth, and the current Triton tile/autotune choices were mainly exercised on Ada (I am planning building a CUTLASS kernel for this). This kernel also keeps more relative-attention state live than standard FlashAttention, so those differences can hurt more on Turing.

That explains why it can still win at B8/L64 (~1.20x) but lose at B32/L512 (~0.85x): the benefit from avoiding the large attention materialization no longer outweighs the kernel execution/occupancy cost at that shape.

Really useful report — especially the NaN case and the T4 numbers. Thanks again.

## [c5720506477] delyan-boychev (NONE) · issue_comment · 2026-09-17
A quick scope update after reviewing the current Transformers Hub-kernel API and recent kernel integrations:

I propose keeping the Transformers-side change deliberately small:

```python
from ...integrations import use_kernel_forward_from_hub

@use_kernel_forward_from_hub("DebertaV2Encoder")
class DebertaV2Encoder(nn.Module):
    ...
```

The optimized implementation would live in a separate Hub kernel repository and provide a stateless, inference-only `DebertaV2Encoder` layer with the exact existing forward signature. It would operate on the already-instantiated HF encoder/attention modules, preserve checkpoint keys, and fall back to the original HF forward for unsupported modes or inputs (training/gradients, `output_attentions=True`, custom `relative_pos`, `query_states`, unsupported masks/dtypes/shapes/devices).

The proposed first Transformers PR would contain only:

- the import and decorator above;
- a small kernel-registration test;
- no Triton source;
- no DeBERTa config fields;
- no module replacement or weight conversion;
- no default kernel mapping initially.

Users could opt in explicitly through `transformers.KernelConfig`. A default inference mapping could be proposed separately after the Hub kernel has broader GPU coverage and trusted-publisher hosting.

The encoder boundary is intentional: `DebertaV2Encoder.forward()` currently expands the 2-D mask and constructs dense relative positions before invoking each attention layer, so an attention-only hook would lose important memory and latency benefits.

Would this scoped encoder-level Hub-kernel hook be acceptable? If so, I will prepare the external kernel repository and the minimal Transformers PR.

AI-assistance disclosure: I used Codex to review the current Hub-kernel APIs, contribution requirements, and comparable Transformers PRs; I have reviewed and am responsible for the proposed scope.

## [c5729275954] Rocketknight1 (MEMBER) · issue_comment · 2026-09-18
Hmmn, I'm not sure we want this, I'm sorry! DeBERTa has generally been replaced in production by newer MLMs like ModernBERT, so we don't want to put a huge amount of effort into reviewing kernel optimizations for it, especially when the performance is a bit ambiguous

## [c5729361029] delyan-boychev (NONE) · issue_comment · 2026-09-18
First, the performance on long-context and relative position-sensitive tasks remains higher on DeBERTa compared to ModernBERT: https://aclanthology.org/2025.ijcnlp-long.164.pdf; it is applied in models like [GLINER2](https://github.com/fastino-ai/GLiNER2). Second, the speedup performance is definitely not ambiguous; it is simply that tuning will be the first step before using the kernel, and not tuning during inference: https://github.com/delyan-boychev/disentangled-flash. The memory reduction is around 70%, and the packed speedup on some real tasks with padding is really high; on MNLI - 10x faster. Also, on the standard setup, it's up to 7x faster for longer seqs. Check this out: https://huggingface.co/spaces/kernels-community/README/discussions/31

## [c5955416174] qgallouedec (MEMBER) · issue_comment · reply to [c5729361029] · 2026-10-02
Hey @delyan-boychev, thanks for the detailed numbers and the follow-up. We're going to stick with @Rocketknight1's call for now. Even a one-line hook on `DebertaV2Encoder` turns its full forward signature into a contract with an external kernel, and with no other requests for this so far (plus the mixed T4 results above) that's hard to justify for DeBERTa right now.

Users aren't blocked in the meantime: `optimize_deberta(model.deberta)` from your package already works on stock transformers. If the kernel lands in kernels-community with broader GPU coverage and people start asking for it here, we're happy to revisit the hook. Closing for now. Thanks 🤗

---
outcome: state=closed
