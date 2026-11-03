# #47272 [issue] Avoid repeated Cohere ASR encoder projection during cached generation
https://github.com/huggingface/transformers/issues/47272  ·  opened by AliOsm on 2026-07-12

## [body] AliOsm (CONTRIBUTOR) · opening post · 2026-07-12
**AI-assisted coordination disclosure:** This proposal was developed with AI assistance after three days of implementation, testing, benchmarking, and human review in our [Cohere Arabic batch-inference project](https://github.com/AliOsm/cohere-transcribe-arabic-batch-inference). I am opening this issue first to follow the repository's agentic-contribution policy and will wait for maintainer approval before opening a PR.

## Summary

`CohereAsrDecoder.forward()` currently applies its encoder-to-decoder projection on every autoregressive decoder call. During cached generation, each decoder layer stores its encoder K/V states after prefill, so later steps do not consume the projected encoder values; they only need source-shape and mask metadata.

Would you accept a scoped Cohere ASR change that applies the projection only while at least one cross-attention cache layer is uninitialized?

## Proposed behavior

- Preserve the current projection for training, direct forwards, `use_cache=False`, fresh caches, and partially initialized caches.
- Once every `EncoderDecoderCache.is_updated[layer_idx]` is true, skip the shared projection.
- Use the cached first-layer K tensor as zero-copy mask metadata so source length and projected/KV dtype remain consistent under mixed precision.
- Add no public API and no mutable module-level memoization.

This uses the existing cache lifecycle, so independent calls, beam cache reordering, external `encoder_outputs`, compilation, and device maps do not share retained state.

## Evidence

On the current tiny test model, five generated tokens invoke `decoder.proj` five times today. All cross-attention layers report initialized immediately after prefill. With deliberately different encoder/decoder widths (24 -> 16), skipping the projection on a cached step was bit-exact (`max_abs_diff=0.0`).

Our production implementation has used equivalent one-projection-per-decode behavior in [AliOsm/cohere-transcribe-arabic-batch-inference](https://github.com/AliOsm/cohere-transcribe-arabic-batch-inference/blob/5035d8ef5787bde147681d6760122a00f8a21303/transcribe.py#L1168-L1192).

Matched 500-clip ablation (RTX 3060 12 GB, BF16, batch 24, 5,032.699 seconds Arabic audio, `max_new_tokens=445`):

| Configuration | Wall | Generation | Outputs changed |
|---|---:|---:|---:|
| Stock repeated projection | 61.208 s | 38.677 s | baseline |
| One projection per decode | 59.603 s | 37.034 s | 0 / 500 |
| Single-run change | -2.62% | -4.25% | none |

This table is a matched single-run production observation of the deployed wrapper, not a claim of statistically stable speedup. The proposed PR will include repeated measurements of the exact upstream diff.

## Validation plan

Add targeted projection-call-count tests for greedy and beam generation, cache-disabled and partial-cache coverage, external encoder outputs, padded eager/SDPA parity, mixed precision, compilation, and training gradients. Run the complete Cohere ASR model suite plus repository consistency checks.

I searched open and closed issues/PRs before filing and found no existing implementation. The open Cohere ASR PRs associated with #46894 address an unrelated training-loss bug. #47271 only cites this optimization's production result.

cc @eustlb @Cyrilvallez

## [c4950697949] AliOsm (CONTRIBUTOR) · issue_comment · 2026-07-12
The exact upstream implementation is now complete and available for review before PR creation:

- Branch: [`AliOsm:optimize-cohere-asr-encoder-projection`](https://github.com/AliOsm/transformers/tree/optimize-cohere-asr-encoder-projection)
- Commit: [`52d19f812c`](https://github.com/AliOsm/transformers/commit/52d19f812c7af98902f21ea4fa6c653dc228109c)

## Exact-diff benchmark

I benchmarked upstream baseline `63f32a8782` against the exact proposed commit on the same 500 clips / 5,032.699 seconds of Arabic audio. Each measurement used a fresh process and model load, primed the same cached inputs, and ran the same 24-sample warmup. Hardware and inference settings were held constant: RTX 3060 12 GB, BF16, batch 24, SDPA, `max_new_tokens=445`.

Across four complete runs per condition:

| Exact source | Generation runs (s) | Median | Range |
|---|---|---:|---:|
| Upstream baseline | 36.762, 38.083, 37.239, 36.858 | 37.049 s | 36.762–38.083 s |
| Proposed cache-aware projection | 36.270, 35.719, 35.792, 35.734 | 35.763 s | 35.719–36.270 s |
| Change | | **-3.47% / 1.036x** | |

The fourth pair controlled for run-order and thermal effects with approximately 60 seconds idle cooldown before each fresh process. It measured `36.858 s -> 35.734 s` generation (`-3.05%`) and `61.304 s -> 60.236 s` wall time (`-1.74%`); decode differed by only 10.9 ms.

Instrumentation on the timed workload counted **3,224 -> 21** encoder-projection calls (`-99.35%`), one remaining call for each of the 21 generation batches.

All raw 500-row hypothesis files were byte-identical across every run: 0/500 outputs changed, lexical WER remained 23.2034766%, and CER remained 12.9864281%.

## Validation completed

```text
make style
  4/4 checks passed

make check-repo
  24/24 checks passed

PYTHONPATH=src pytest -q tests/models/cohere_asr/test_modeling_cohere_asr.py
  128 passed, 131 skipped
```

The targeted tests cover greedy and beam projection counts, `use_cache=False`, deliberately mismatched encoder/decoder widths, padded inputs, partially initialized cross-attention caches, bit-exact cached-step parity, preservation of raw public encoder outputs, and projection gradients. The inherited suite also passed full-graph compiled generation, cache continuation, CPU/disk offload, save/load, and attention backend coverage. An independent final review found no correctness or compatibility issues and manually checked static-cache and zero-layer behavior.

The PR description is ready with the required AI-generated disclosure, production-project link, duplicate-work differentiation, exact benchmark, and validation report. May I open the PR?

## [c5009589797] varneeth01 (NONE) · issue_comment · 2026-07-18
I reviewed the proposal and the cache lifecycle described here. Using the existing EncoderDecoderCache state to avoid repeated projections while preserving behavior for fresh and partial caches seems like a reasonable design. The validation and benchmarking are very helpful.

Looking forward to the PR.

---
outcome: state=closed
