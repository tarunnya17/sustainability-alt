# #46247 [issue] Faster preprocessing when using new Parakeet TDT please
https://github.com/huggingface/transformers/issues/46247  ·  opened by BBC-Esq on 2026-05-27

## [body] BBC-Esq (NONE) · opening post · 2026-05-27
### Feature request

Hi! Thanks for adding native Parakeet support in #44171 — really nice not having to pull in all of NeMo just to run these models. I've been benchmarking and noticed a speed gap I think has a clean fix, but wanted to file an issue before attempting a PR.

## What I noticed

Same weights (`nvidia/parakeet-tdt-0.6b-v3`), same audio, same GPU (RTX 4090, bf16):

- **NeMo**: ~8s for my test set
- **transformers `ParakeetForTDT`**: ~47–50s

About 6x slower, which surprised me since the model is identical.

## Why I think this happens

The model forward pass looks comparable between the two. The big difference seems to be preprocessing.

`src/transformers/models/parakeet/feature_extraction_parakeet.py` uses **librosa** for the STFT and mel filterbank. Librosa is a pure-Python wrapper over numpy/scipy and runs CPU-only — so each chunk has to leave the GPU pipeline, get processed on CPU, then come back as a tensor.

NeMo's `AudioToMelSpectrogramPreprocessor` is a `torch.nn.Module` using `torch.stft` + `torch.matmul`. It runs on GPU via cuDNN FFT, or on CPU via MKL FFT — and the audio never leaves the device. For chunked workloads where you call the feature extractor over and over, those roundtrips add up.

## The precedent

`src/transformers/models/whisper/feature_extraction_whisper.py` already solves this exact problem! It has two methods:

- `_torch_extract_fbank_features` — uses `torch.stft`, runs on whatever `device` you pass
- `_np_extract_fbank_features` — the librosa fallback

`__call__` picks between them based on whether `device=` is passed. The same pattern should map cleanly onto Parakeet: keep the librosa code as a fallback, add a `_torch_extract_fbank_features` method, and accept an optional `device` argument.

## On PR #44171

I saw the maintainers noted the initial implementation prioritized **correctness over speed**, which totally makes sense for a first landing. Not saying this is a bug — just wondering if a follow-up speed PR would be welcome now that correctness is in.

## Things I'd be careful about in a PR

1. **Numerical parity** — add a test comparing both paths within tolerance, like Whisper does
2. **Mel filterbank** — register as a non-persistent buffer so it lives on the right device
3. **Backwards compatibility** — calls without `device=` should behave exactly as today
4. **Decoder loop** — I noticed the TDT decoder is also pure Python (vs NeMo's CUDA kernels). That's a bigger lift and out of scope for this issue, so mel-spec alone won't fully close the 6x gap — but should be a meaningful chunk.

## Ask

Would the maintainers be open in principle to this kind of change before I try writing a PR? Happy to share more detailed benchmarks or isolate just the feature-extraction step if useful.

Thanks! 🙏

### Motivation

Just really like this model and want it to be as good as the nemo-toolkit version, just without that heavy dependency.

### Your contribution

I'd probably be willing to help out on a PR.

## [c4561376961] Shanmuk4622 (NONE) · issue_comment · 2026-05-28
This looks like a very meaningful optimization opportunity, especially given the existing Whisper implementation already supports both NumPy/librosa and torch-based feature extraction paths.

I’m currently reviewing:

* `feature_extraction_whisper.py`
* `feature_extraction_parakeet.py`

to better understand how the torch-based STFT/mel pipeline could be adapted for Parakeet while preserving numerical parity and backward compatibility.

The proposed approach of:

* keeping librosa as fallback
* adding a torch-based GPU path
* selecting based on `device=`

seems very reasonable to me.

I’d be interested in exploring a PR for this after studying the current preprocessing flow and benchmarking behavior a bit more.

## [c4562291987] Shanmuk4622 (NONE) · issue_comment · 2026-05-28
I’ve been working on a prototype implementation for GPU-accelerated preprocessing in `ParakeetFeatureExtractor`.

Current implementation:

* moves the full preprocessing pipeline onto the target device
* removes unnecessary CPU↔GPU synchronization
* adds mel-filter caching
* preserves numerical parity with the CPU implementation
* includes CUDA parity/integration tests

Benchmarks on GTX 1650 currently show:

* ~3.45x speedup for single 30s clips
* ~5.19x speedup for batches of 5 clips

I’m doing final validation and cleanup now and plan to open a PR shortly if this direction looks acceptable.

## [c4563734393] zucchini-nlp (MEMBER) · issue_comment · 2026-05-28
Looks like the same issue as in https://github.com/huggingface/transformers/pull/45134, and it is being covered by a larger refactor in https://github.com/huggingface/transformers/pull/44394

cc @eustlb

## [c4564495104] BBC-Esq (NONE) · issue_comment · 2026-05-28
If it's a true duplicate no worries about closing or it might be something that deserves a specific pull request, up to the maintainer.

## [c4567320087] BBC-Esq (NONE) · issue_comment · reply to [c4563734393] · 2026-05-28
Thanks @zucchini-nlp — that's great news, and #45134 looks like a near-exact match for what I was proposing here: `torch.stft`/mel with a `device=` selector and librosa kept as the fallback, with the nice bonus that it also speeds up the CPU path. Very happy to defer to that PR and to #44394.

If a follow-up from my side is still useful, I built a small multi-backend mel-spectrogram benchmark that computes the *same* log-mel several different ways and cross-checks their outputs. Here's what each one actually is:

[collapsed block '<b>The backends I compared</b>', 41 lines omitted]

Either way, thanks for the quick triage — glad this is already in motion. Let me know if you want more details about the scripts I used to test. 🙏

## [c4577198077] eustlb (COLLABORATOR) · issue_comment · 2026-05-29
Thanks a lot for raising this issue !! As stated above this will indeed be addressed in the larger refacto in #44394
This adds to the motivation to ship it, thanks a lot for being patient I'll do my best to have it merged asap

## [c4577249106] BBC-Esq (NONE) · issue_comment · 2026-05-29
Looking forward to benchmarking it when it's done!  Closing...

---
outcome: state=closed
