# #45184 [pull_request] [CB] [Major] Add CPU request offloading
https://github.com/huggingface/transformers/pull/45184  ·  opened by remi-or on 2026-04-02

## [body] remi-or (COLLABORATOR) · opening post · 2026-04-02
# Summary

This PR adds CPU offloading to continuous batching. It's in raft until perf and test status are reported.

When the GPU KV cache is full and a request must be evicted, we check if there is enough VRAM to copy the request's KV cache  to a pre-allocated pinned CPU buffer instead of discarding them. Otherwise we use the legacy "soft reset" path that forces a full re-prefill on restore. This avoids redundant compute when cache pressure is temporary.

Main additions:

  - New OffloadingManager class (offloading_manager.py, ~300 lines) owns a static CPU swap pool of pinned tensors, performs GPU↔CPU block copies, and falls back to soft reset when the pool is full.
  - Two new ContinuousBatchingConfig parameters: cpu_offload_space (GiB, default 0 = disabled) and cpu_offload_space_safety_threshold (fraction of system RAM cap, default 0.8).
  - The scheduler prioritizes restoring CPU-offloaded requests over prefilling fresh ones (_get_waiting_candidates), since a CPU→GPU copy is cheaper than a full prefill.
  - When prefix sharing is on, restored blocks are marked incomplete so mark_shareable_blocks_as_complete re-hashes and deduplicates them against existing shared blocks, avoiding permanent memory waste.
  - Compatible with async batching (the update_batch guard skips PENDING requests that were offloaded between scheduling and update), sliding window models (per-group block counts are tracked and the rolling-buffer arithmetic is
  preserved), and forking.

## Performances 

As expected, no major difference for generation without offloading: 

| Arguments                                                       | Main (tok/s) | Current (tok/s) | Diff (%) |
|-----------------------------------------------------------------|--------------|-----------------|----------|
| --samples 10                                                    | 854.03       | 853.33          | -0.1%    |
| --samples 20 --num-blocks 20                                    | 508.44       | 508.62          | +0.0%    |
| --samples 50                                                    | 3543.77      | 3623.23         | +2.2%    |
| --samples 100                                                   | 5324.99      | 5277.82         | -0.9%    |
| --samples 100 --attn flash_attention_2                          | 3647.27      | 3649.99         | +0.1%    |
| --samples 100 --attn sdpa                                       | 1027.38      | 1035.21         | +0.8%    |
| --samples 500 --no-use-async                                    | 6609.49      | 6587.37         | -0.3%    |
| --samples 500 --use-async                                       | 7903.19      | 7913.05         | +0.1%    |
| --samples 32 --max-new-tokens 2048 --use-async                  | 2039.83      | 2039.52         | -0.0%    |
| --samples 32 --max-new-tokens 2048 --use-async --block-table 32 | 2653.75      | 2654.51         | +0.0%    |
| --samples 500 --add-prefix --compile                            | 8521.59      | 8686.75         | +1.9%    |
| --samples 50 --num-return-sequences 8 --do-sample               | 916.43       | 912.93          | -0.4%    |
| --samples 100 --num-return-sequences 4 --do-sample              | 1808.55      | 1801.04         | -0.4%    |

## Tests 

The following tests pass:
```
tests/generation/test_continuous_batching.py 
tests/cli/test_serve.py
tests/generation/test_paged_attention.py
```

## [c3135821011] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/continuous_api.py:346 · 2026-04-24
```suggestion
        self.offloading_manager.free_requests(cancelled_states)
```
api wise isolate code ?

## [c3135822852] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/continuous_api.py:367 · 2026-04-24
makes sense sinse here you restore in a bulk

## [c3135824426] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:1 · 2026-04-24
```suggestion
# Copyright 2026 The HuggingFace Inc. team
```

## [c3135836043] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:18 · 2026-04-24
well if you have a shitty cpu that can still help make everything fit!

## [c3135836823] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:21 · 2026-04-24
nice

## [c3135845144] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:86 · 2026-04-24
ahh smart so this is a ref to the viewed cache, which you then internally just have to use for copy_ ?

## [c3135852702] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:157 · 2026-04-24
I don't see this or the offloading manager being gated for this capability (but I mean for mps you have shared memory and IDK if neuron tpu etc have the same API)

## [c3135866861] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:159 · 2026-04-24
Makes sense to copy in a bulk after we just compute which pages we need to copy imo!

## [c3135893536] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:240 · 2026-04-24
```suggestion
gpu_k.index_copy_(0, gpu_ids, torch.index_select(cpu_k, 0, cpu_ids).to(device, non_blocking=True))
```
less small DMA transfer schedules ?

## [c3135937536] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:261 · 2026-04-24
should release gil maybe?

## [c3135941856] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:293 · 2026-04-24
ah ok, we should also avoif for loop copy_

## [c4168305874] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-04-24
Nice! is this benched as well?

## [c3144840616] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/continuous_api.py:346 · reply to [c3135821011] · 2026-04-27
Not sure I understand sorry!

## [c4178110288] remi-or (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144846803] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:18 · reply to [c3135836043] · 2026-04-27
Agreed, changed the doc.

## [c4178116343] remi-or (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144847115] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:21 · reply to [c3135836823] · 2026-04-27
Thanks!

## [c4178116718] remi-or (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144848993] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:86 · reply to [c3135845144] · 2026-04-27
Yup! There is a comment explaining this, you think we need more doc?

## [c4178118705] remi-or (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144852427] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:157 · reply to [c3135852702] · 2026-04-27
Not sure what you mean: this is not a no-op only if there is already a compute stream around.

## [c4178122241] remi-or (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144854476] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:159 · reply to [c3135866861] · 2026-04-27
That's what happen I think

## [c4178124322] remi-or (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144860325] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:261 · reply to [c3135937536] · 2026-04-27
This will be done in the asynchronous version of CPU offloading.

## [c4178130223] remi-or (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144865200] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:293 · reply to [c3135941856] · 2026-04-27
They are different tensors, so it's not really avoidable... I have tried consolidating them into one tensor and found torch has trouble allocating this much memory at once / it does not speed up the process. But it was a while ago, are you ok with a TODO and we will try it out later?

## [c4178134841] remi-or (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144867885] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:240 · reply to [c3135893536] · 2026-04-27
different tensors, so not really do-able. Resolving this and leaving the discussion open on the comment below

## [c4178137459] remi-or (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144915041] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/continuous_api.py:346 · reply to [c3135821011] · 2026-04-27
I mean can you wrap the looping on the cancelled states inside the free requests api? re my comments about memory transfer, you are better of with the full list

## [c4178192954] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144916060] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:157 · reply to [c3135852702] · 2026-04-27
I just mean that torch.cuda will error out for mps!

## [c4178194403] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144916638] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:86 · reply to [c3135845144] · 2026-04-27
No no I was just saying its cool!

## [c4178195076] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144918031] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:159 · reply to [c3135866861] · 2026-04-27
I don't think so no, each copy_ is a DMA / copy request, not async unless requested no?

## [c4178196559] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144918850] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:293 · reply to [c3135941856] · 2026-04-27
yep for sure!!

## [c4178197389] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144929831] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:86 · reply to [c3135845144] · 2026-04-27
Haha thx

## [c4178209042] remi-or (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3144949319] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:157 · reply to [c3135852702] · 2026-04-27
No I'm pretty sure we are safe because the ternary will only evaluate the CUDA part if there is a compute stream that's not None, and that will not happen on MPS.

## [c4178231149] remi-or (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3145028629] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:159 · reply to [c3135866861] · 2026-04-27
Ha yes I thought this was about restore. For the async part, I think it's ok because there is no async mode yet. As for the "bulk" part, currently we only offload one request at a time, so we group as much as possible but it's still the cache for one request only.

## [c4178317265] remi-or (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3145580084] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/continuous_api.py:346 · reply to [c3135821011] · 2026-04-27
Does not change much, this is a CPU only operation. Not even torch intervenes. Added a note about this.

## [c4178958076] remi-or (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

## [c3145800133] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/offloading_manager.py:157 · reply to [c3135852702] · 2026-04-27
ok!

## [c4179217565] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-04-27 (no text)

---
outcome: state=closed, merged=True
