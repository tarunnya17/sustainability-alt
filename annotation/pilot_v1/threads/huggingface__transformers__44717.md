# #44717 [issue] Support packed sequences for linear attention models (i.e. Qwen3.5)
https://github.com/huggingface/transformers/issues/44717  ·  opened by kirawi on 2026-03-14

## [body] kirawi (NONE) · opening post · 2026-03-14
### Feature request

Currently, packing does not seem supported for text-based datasets (https://github.com/unslothai/unsloth/issues/4160). It would be good to support this.

### Motivation

Without packing, my training runs are approximately 3-5x more expensive with the dataset that I'd like to use, and also suffer from overhead on very short sequences.

### Your contribution

I cannot help; I have no experience with deep learning.

## [c4071314931] vasqu (COLLABORATOR) · issue_comment · 2026-03-16
Cannot promise anything, but I had it on my todo list for quite a while. Note that you will need the fla package then (fast path).

Should be fairly similar to flash attention based on the fla api.

## [c4075268933] michaelanderson01826-glitch (NONE) · issue_comment · reply to [c4071314931] · 2026-03-17
Maybe we can improve error handling here

On Mon, Mar 16, 2026, 11:34 PM Anton Vlasjuk ***@***.***>
wrote:

> *vasqu* left a comment (huggingface/transformers#44717)
> <https://github.com/huggingface/transformers/issues/44717#issuecomment-4071314931>
>
> Cannot promise anything, but I had it on my todo list for quite a while.
> Note that you will need the fla package then (fast path).
>
> Should be fairly similar to flash attention based on the fla api.
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/huggingface/transformers/issues/44717#issuecomment-4071314931>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/B7HTEEIZM2TYSXRBFPOEHL34RCFRTAVCNFSM6AAAAACWSE7QE2VHI2DSMVQWIX3LMV43OSLTON2WKQ3PNVWWK3TUHM2DANZRGMYTIOJTGE>
> .
> You are receiving this because you are subscribed to this thread.Message
> ID: ***@***.***>
>

## [c4085456190] sdharani91 (CONTRIBUTOR) · issue_comment · 2026-03-18
Hi, I’d like to work on this if that sounds reasonable. I dug into the Qwen3.5 codepath a bit and wanted to sanity-check the implementation direction.

From what I can see, packed inputs already work for the full-attention layers because create_causal_mask(...) infers packed boundaries from reset position_ids. The mismatch seems to come from the linear-attention layers, where the conv / recurrent state continues across packed sequence boundaries.

My current understanding is that there are really two cases to handle:

Fast path:
Qwen3.5’s linear-attention block uses both the conv kernel and the FLA gated-delta kernels.
It looks like the conv side can use seq_idx, while the FLA gated-delta kernels expose cu_seqlens-style varlen inputs, similar to Flash Attention.
So for the fast path, my plan would be to thread packed-boundary metadata down to the linear-attention block and pass:
seq_idx to the conv kernel
cu_seqlens to the chunked / recurrent gated-delta kernels

Slow path:
The current torch fallback implementations do not seem to expose the same packed-sequence boundary API.
So for correctness, my tentative plan is to detect packed boundaries once from position_ids and, in the linear-attention fallback only, run the packed subsequences independently and concatenate the outputs back together.
That would reset both the conv history and recurrent state at packed boundaries, at the cost of some slowdown in the fallback path.

Does that sound like the right split?
In particular, I want to confirm whether:

1. handling both fast and slow paths is desirable here, or
2. the expectation is to fix only the fast path for now and leave fallback behavior unchanged.

## [c4086202158] sdharani91 (CONTRIBUTOR) · issue_comment · 2026-03-18
Update: I was able to reproduce the packed-sequence mismatch on both the torch fallback and fast (FLA) paths.
For the fast path, Qwen3.5 is using:
causal_conv1d_fn
fla.ops.gated_delta_rule.chunk
fla.ops.gated_delta_rule.fused_recurrent

Using a small Qwen3_5ForCausalLM, I compared padded vs packed inputs (with reset position_ids). Before the fix, they diverged on the fast path (allclose=False, max abs diff ~8e-3).
I prototyped a minimal fast-path-only fix for packed prefill (no cache):

pass seq_idx to the conv path
pass cu_seqlens to the FLA kernels

With this, padded vs packed now match:
2-segment and multi-segment repros pass (max abs diff ~6e-8)
standard (unpacked) forward is unchanged

I haven’t addressed the slow fallback path yet—this is scoped to the fast linear-attention path first. If this direction looks good, I can clean things up and open a PR.

## [c4087937920] vasqu (COLLABORATOR) · issue_comment · 2026-03-19
We have a PR for olmo hybrid over here #44836

It should look similar at least: Meaning kwargs are passed downstream (i.e. `cu_seqlens` are passed to the kernels that can handle padding free application)

## [c4099277908] sdharani91 (CONTRIBUTOR) · issue_comment · reply to [c4087937920] · 2026-03-20
Hi @vasqu , @Rocketknight1  - I see that my PR was closed. Is this still something I could help with, or would you prefer that I not continue on this issue?

I was able to reproduce the Qwen3.5 packed-sequence mismatch on the linear-attention fast path and had a prototype fix that made the fast-path packed-vs-padded repro pass. That PR was intentionally scoped to the fast path only.

What is the expected direction here:
fast path only first, with a direct regression test
both fast and fallback paths in the same PR
or something else

## [c4111032748] Rocketknight1 (MEMBER) · issue_comment · reply to [c4099277908] · 2026-03-23
@sdharani91 sorry, I've been closing PRs that looked very "code-agenty", but that one might be too aggressive. Maybe the right approach is to wait for #44836 to be merged and then inherit the functions via modular to reduce code bloat? cc @vasqu

## [c4111895914] sdharani91 (CONTRIBUTOR) · issue_comment · reply to [c4111032748] · 2026-03-23
@Rocketknight1  - can you please expand on that suggestion? Aren't the changes in [#44836 ](https://github.com/huggingface/transformers/pull/44836) in files specific to OlmoHybrid?

## [c4122332367] onder-prosus (NONE) · issue_comment · 2026-03-25
I hit this while fine-tuning Qwen3.5 with TRL (SFTTrainer + packing=True + flash_attention_2 + gradient checkpointing).

Got it working with a runtime monkey-patch — three things needed:
1. Pass `cu_seqlens` to fla's `chunk_gated_delta_rule` (it already accepts the param, just never gets it)
2. Replace the hardcoded `seq_idx=None` in `causal_conv1d_fn` with real packed boundary indices
3. Reject 3D `position_ids` in `_is_packed_sequence()` to avoid the illegal memory access from #44910

One thing worth noting: if you store the packing metadata in `threading.local()`, gradient checkpointing breaks — the recomputed forward runs after cleanup. Module globals work.

Tested on 0.8B, 2B, 4B, all pass.

## [c4126945251] vasqu (COLLABORATOR) · issue_comment · reply to [c4122332367] · 2026-03-25
Please see my comments on that PR to only allow passing the kwargs - not manually unpadding and padding again. Other than that, I'm open to have a PR on this - not sure if olmo hybrid can be reused in a 1:1 manner @onder-prosus

## [c4127964068] sdharani91 (CONTRIBUTOR) · issue_comment · reply to [c4126945251] · 2026-03-25
@vasqu @onder-prosus  @Rocketknight1 - could you help review this PR in that case https://github.com/huggingface/transformers/pull/44867

## [c4128025232] vasqu (COLLABORATOR) · issue_comment · reply to [c4127964068] · 2026-03-25
@sdharani91 it looks like it manually prepares these kwargs, we should move to a pure data collator version as mentioned in the olmo hybrid PR as well

## [c4138194471] sdharani91 (CONTRIBUTOR) · issue_comment · reply to [c4128025232] · 2026-03-26
@vasqu I have created a draft PR using the info from data collator: https://github.com/huggingface/transformers/pull/45034/changes#diff-6064941ca492d13e60e6c551ee54b967d803c2fdc75dc0751676563eb615ae63.  Does this align with your suggestion?

## [c4529489388] ExtReMLapin (NONE) · issue_comment · 2026-05-24
Just met the issue, working test : 

```
from types import SimpleNamespace

import torch

[… 64 lines of code omitted]

## [c4543928830] vasqu (COLLABORATOR) · issue_comment · 2026-05-26
Closing as the same manner as for #46093 and as it was added by #45034

---
outcome: state=closed
