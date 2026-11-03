# #48163 [pull_request] Prefer the cuDNN SDPA backend on Hopper
https://github.com/huggingface/transformers/pull/48163  ·  opened by qgallouedec on 2026-08-20

## [body] qgallouedec (MEMBER) · opening post · 2026-08-20
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=48163) [[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=48163)

> **Edit:** originally gated to `capability >= 9` ("Hopper and newer") based on H100 numbers alone.
> After benchmarking six GPUs across five architectures, cuDNN turns out to win on Hopper and nowhere
> else: it is slower on Ampere (0.82x on A100, 0.93x on A10G) and on Blackwell (0.88x on RTX PRO 6000),
> and a wash on Ada (1.00x on L40S). The gate is now `== 9`, Hopper only. Full table in
> [this comment](https://github.com/huggingface/transformers/pull/48163#issuecomment-5361546353);
> the text below has been updated to match.

`torch.nn.functional.scaled_dot_product_attention` picks its backend by a fixed priority that puts FlashAttention first. On Hopper the cuDNN attention kernels are consistently **1.7–2.3× faster** for training shapes, at the same numerical accuracy. This adds a backend *preference* (not a restriction) on Hopper, so shapes cuDNN cannot serve fall back exactly as before.

Confirmed that the default really is FlashAttention — profiling Qwen3-0.6B at 16K on `main` shows
`pytorch_flash::flash_bwd_...` / `flash_fwd_kernel` taking 335 ms of a 677 ms step.

### Measured

One GPU, pure torch, no transformers involved: forward+backward through
`scaled_dot_product_attention` once per backend (script at the end of this section).

```
NVIDIA H100 80GB HBM3, torch 2.13.0+cu130

speed, bf16 causal forward+backward, 32 heads x 128
     seq               flash               cudnn  speedup
[… 17 lines of code omitted]

cuDNN is ahead at every sequence length and at every head shape shipped by the models people are training
today, from 1.34x (GLM-5.2, 64 heads x 192) to 1.83x. The gradients land at the same distance from an fp64
reference, so this is a free change numerically.

[image]

Confirmed that the default really is FlashAttention: profiling Qwen3-0.6B at 16K on `main` shows `pytorch_flash::flash_bwd_...` / `flash_fwd_kernel` taking 335 ms of a 677 ms step.

[collapsed block '<code>repro_cudnn_sdpa.py</code>', 86 lines omitted]

### Implementation

wrapping the single `scaled_dot_product_attention` call in `sdpa_attention_forward`.

Design notes, each of which I measured the hard way:
- **The `q_length == 1` gate is load-bearing.** Expressing a preference costs ~18 us because it saves and restores global backend flags. That is nothing next to a training attention call, but a decode-shaped call is only ~32 us, so wrapping it ungated made 64-token generation 13% slower (2872 ms vs 2542 ms). With the gate, generation is back to baseline.
- **Wrapping the SDPA call itself, rather than the training step, is what makes activation checkpointing work.** Recompute re-enters `sdpa_attention_forward` and therefore re-selects the same backend. Wrapping at a coarser granularity that does not cover the backward gives `CheckpointError: Recomputed values ... have different metadata`.
- `set_priority=True` expresses an *order*, so shapes cuDNN cannot serve fall through to flash/efficient/math (verified with `head_dim=72`, which cuDNN refuses).
- Gated to CUDA SM90+, because that is where I have measurements.

### End-to-end effect (steady-state median step time, gradient checkpointing, Qwen3-0.6B unless noted)

[image]

| config | `main` | this PR | |
|---|---|---|---|
| generation, 64 tokens | 2542 ms | 2511 ms | unchanged (gated out) |
| training, seq 512 x batch 16 | 377 ms | 375 ms | no change — attention is a small share of the step |
| training, seq 16384 x batch 1 | 1129 ms | **924 ms** | 1.22x |
| training, Qwen3-8B, seq 1,048,576, 8xH100 + context parallelism | 612 s | **364 s** | **1.68x** |

Losses match to 5 decimal places in every case (e.g. 11.42171 vs 11.42174). The gain grows with how much of the step is attention, and there is no configuration in which this is slower.

### Caveats / questions for maintainers

- Benchmarks are H100 only. If you want this on Ampere or Blackwell, that needs numbers I do not have; the gate is one line to widen.
- There is an open torch issue about cuDNN SDPA + context parallelism strides https://github.com/pytorch/pytorch/issues/176915; I did not hit it in extensive CP runs (Qwen3-8B / 30B-A3B at 1M–4M tokens, torch 2.13), but it is worth knowing about.
- If a global preference feels too broad, the same change scoped to `is_causal and attention_mask is None` would still capture nearly all of the training win.

### If you would rather not take this

I'd understand if you decide not to move forward: it changes a default for everyone

Nothing is blocked either way, users can select the backend themselves, but it's not super clean:

```python
class FastSFTTrainer(SFTTrainer):
    def training_step(self, *args, **kwargs):
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION], set_priority=True):
            return super().training_step(*args, **kwargs)
```

It has to wrap forward *and* backward, since checkpoint recompute must pick the same backend. This is what I
used for every long-context run here.

So the question is really whether the default should do this for people who will never read a benchmark. I
think yes, but happy to close it and document the workaround instead.

## [c5361546353] qgallouedec (MEMBER) · issue_comment · 2026-08-20
Update: I ran the same benchmark on other GPUs, and it changed the gate.

cuDNN wins on Hopper (H100 and H200 alike). It is *slower* on Ampere and on Blackwell, and a wash on Ada. My original gate was `capability[0] >= 9`, which includes Blackwell, so as written this PR would have made Blackwell about 12% slower. Now gated to Hopper only (`== 9`), with the measurements in the docstring.

| GPU | arch | 8K | 32K | 64K | 128K |
|---|---|---|---|---|---|
| A100 | Ampere, sm80 | 0.85x | 0.83x | 0.82x | 0.82x |
| A10G | Ampere, sm86 | 0.91x | 0.93x | 0.93x | 0.93x |
| L40S | Ada, sm89 | 0.96x | 1.00x | 1.00x | 1.00x |
| H100 | Hopper, sm90 | 2.29x | 1.81x | 1.84x | 1.72x |
| H200 | Hopper, sm90 | 2.13x | 1.78x | 1.79x | 1.73x |
| RTX PRO 6000 | Blackwell, sm120 | 1.10x | 0.87x | 0.88x | 0.88x |

Same torch build (2.13.0+cu130) and the gradients match an fp64 reference to the same relative error everywhere.

H100 ran on our cluster and the rest on HF Jobs, so the two Hopper points are not infrastructure matched, but they agree within 7%.

sm100 (B200) I had no access to, so it is excluded.

The benchmark script lives in this PR under Measured: pure torch, one GPU, no transformers needed, so any card I did not cover is a couple of minutes to check.

[image]

## [c3894512510] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/sdpa_attention.py:32 · 2026-08-31
I'm a bit surprised that blackwell is losing quite a bit as well, potentially a big reason why this is still seen as quite experimental. 

Another thing I'm concerned about is the shipped cuda version you tested these on: Cudnn implementations may vary would be my initial intuition

## [c3894527731] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/sdpa_attention.py:43 · 2026-08-31
Could we not cache the cuda availability (`torch.cuda.is_available()`) into `_prefer_cudnn_sdpa`?

## [c3894532002] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/sdpa_attention.py:20 · 2026-08-31
Honestly we could keep them public as well

## [c5066552669] vasqu (COLLABORATOR) · review COMMENTED · 2026-08-31
I'm not too opposed to this because in a sense it's a free perf gain. I have some initial questions first tho

Still a bit surprised about the blackwell results ngl but hopper shouldnt be changing anymore so these should be consistent improvements

## [c5485303387] qgallouedec (MEMBER) · issue_comment · 2026-08-31
Thanks for the review. Answering the CUDA version question turned up something that says this PR should not land 🫠

[image]
PyTorch already does exactly this.

[`check_prefer_cudnn_attention()` in sdp_utils.cpp](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp#L78-L99) sets the same backend order this PR sets, gated on cudnn_version > 91500 and sm90/sm100, and that code is already in 2.10. What changes at 2.11 is the bundled cuDNN crossing the threshold. So on recent torch the default is already cuDNN, and asking again only adds the context manager's overhead.

The builds where this PR does change something are the ones PyTorch excludes on purpose. That threshold comes from pytorch/pytorch#138522 making cuDNN opt-in after the 2.5 bugs (pytorch/pytorch#139298 among them), and the comment above it points at pytorch/pytorch#169849, a seq_len not divisible by 128 fix that only landed in cuDNN 9.15.1. Forcing it there overrides a deliberate guard rather than winning anything. Closing.

For context: I only saw a win because our cluster puts the system CUDA on `LD_LIBRARY_PATH`, which shadows the cuDNN inside the torch wheel. Torch falls back to flash, so on my machine the default really was flash and forcing cuDNN really was 1.8x. Same node with `unset LD_LIBRARY_PATH` and the default is already cuDNN. The benchmark was right, the conclusion was not: it compared two forced backends and never the default. see https://huggingface.slack.com/archives/C0BF837HWLX/p1788212991953029

## [c5485582188] vasqu (COLLABORATOR) · issue_comment · 2026-08-31
Woah I did expect something re cuda but that's super interesting and TIL as well. 

A bit sad but it's something very good having it documented somewhere and on how to setup on our clusters, appreciate it.

---
outcome: state=closed, merged=False
