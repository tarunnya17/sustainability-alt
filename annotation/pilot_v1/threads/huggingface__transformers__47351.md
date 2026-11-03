# #47351 [pull_request] CPU can incur a slow path on non-contiguous magnitudes
https://github.com/huggingface/transformers/pull/47351  ·  opened by vbayanag on 2026-07-15

## [body] vbayanag (CONTRIBUTOR) · opening post · 2026-07-15
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47351)


# What does this PR do?
`stft[..., :-1]` produces a non-contiguous view. Asobserved on MI355X ROCm vLLM latest build image, the audio preprocessing function `mel_filters.T @ magnitudes` falls onto a strided-GEMM path that runs ~8x slower (~43 ms vs ~3 ms for a 30 s audio window), dominating Whisper feature extraction.

Forcing `magnitudes` contiguous costs ~0.18 ms but restores the fast matmul path, cutting per-request audio preprocessing substantially.

Fixes # (issue)

Make the CPU op run on contiguous memory layout for ~8x speedup

### Before/after (single 30 s window, ROCm vLLM image)
| stage | non-contiguous | contiguous |
| --- | --- | --- |
| `mel_filters.T @ magnitudes` | ~43 ms | ~3 ms |
| `.contiguous()` overhead | – | ~0.18 ms |

Benchmark Code
```
import torch
import time 
import numpy as np
from transformers import WhisperFeatureExtractor

[… 65 lines of code omitted]
```

## [c4713972680] vasqu (COLLABORATOR) · review COMMENTED · 2026-07-16
Happy to add the fix on 2 conditions
1. Shorten the comment --> `cpu can incurr a slow path on non-contiguous magnitutes`
2. Add a fast test that can show this or a benchmark with this PR so we can verify as well

## [c4996527975] vbayanag (CONTRIBUTOR) · issue_comment · reply to [c4713972680] · 2026-07-16
> Happy to add the fix on 2 conditions
> 
> 1. Shorten the comment --> `cpu can incurr a slow path on non-contiguous magnitutes`
> 2. Add a fast test that can show this or a benchmark with this PR so we can verify as well

Hi @vasqu, the posted benchmarks in the description are from before/after change. Kindly let me know if you need anymore information.

## [c5003247702] vasqu (COLLABORATOR) · issue_comment · reply to [c4996527975] · 2026-07-17
@vbayanag I meant like a small benchmark script :D the numbers are impressive but imo for anyone stumbling on this, it's always nice to have the exact code that produced the numbers along with it

## [c5008135389] vbayanag (CONTRIBUTOR) · issue_comment · reply to [c5003247702] · 2026-07-17
> @vbayanag I meant like a small benchmark script :D the numbers are impressive but imo for anyone stumbling on this, it's always nice to have the exact code that produced the numbers along with it

Hi @vasqu, I added a test case in the relevant test location. Please check.

Local testing shows 5x uplift. This number is different from what's in description, likely due to difference in the way metrics were extracted. The initial posted numbers were extracted from feature_extractor API call from the trace file of end-to-end inference testing of whisper model, whereas the uplift observed now is localised only to the matmul operation.

`[whisper mel matmul] non-contiguous=10.269 ms contiguous=1.999 ms speedup=5.14x`

System details:

1. Docker pull and launch `vllm/vllm-openai-rocm:latest` image
```
docker run -it  --group-add=video    --ipc=host    --cap-add=SYS_PTRACE    --security-opt seccomp=unconfined    --device /dev/kfd    --device /dev/dri -v /:/data --entrypoint ''  vllm/vllm-openai-rocm:latest /bin/bash
```
2. Install depenencies (optional, only required for other test cases to run smoothly)
```
apt-get update && apt-get install -y ffmpeg
pip install soundfile librosa torchcodec==0.12
```
3. Run the python unit test inside your clone dir
```
PYTHONPATH=src python3 -m pytest     tests/models/whisper/test_feature_extraction_whisper.py -k contiguous_magnitudes -sv
```

## [c3614992845] vasqu (COLLABORATOR) · review_comment on tests/models/whisper/test_feature_extraction_whisper.py:272 · 2026-07-20
Ok sorry for backtracking a bit but I'd rather have the benchmark within the PR description. A benchmark as test is probably brittle, wdyt?

My goal is just to have documentation that shows why it was done and for ppl to repro

## [c4735796967] vasqu (COLLABORATOR) · review COMMENTED · 2026-07-20 (no text)

## [c3616819723] vbayanag (CONTRIBUTOR) · review_comment on tests/models/whisper/test_feature_extraction_whisper.py:272 · reply to [c3614992845] · 2026-07-20
Hi @vasqu, I think that this small test case would be evidence of the change I am submitting. By having a test case that's part of the code, it stays persistent and can be removed upon future rebuttals.

If you still think benchmark is unnecessary, I can remove that part from the test case and leave the rest alone. I can post the benchmark script on description. Kindly let me know what you think

## [c4738056391] vbayanag (CONTRIBUTOR) · review COMMENTED · 2026-07-20 (no text)

## [c3622456891] vasqu (COLLABORATOR) · review_comment on tests/models/whisper/test_feature_extraction_whisper.py:272 · reply to [c3614992845] · 2026-07-21
> I can post the benchmark script on description.

Yea I think this is the better option tbh. It doesn't really make much sense as standalone test. We should refer to the benchmark script in some shape or form, e.g. a PR reference (that has it in the description) or a gh gist.

## [c4744969197] vasqu (COLLABORATOR) · review COMMENTED · 2026-07-21 (no text)

## [c3624215768] vbayanag (CONTRIBUTOR) · review_comment on tests/models/whisper/test_feature_extraction_whisper.py:272 · reply to [c3614992845] · 2026-07-21
Done.

1. I removed benchmark code from test case and added in the description.
2. The test case now only verifies the validity of the matrix when changed to contiguous.

## [c4747195483] vbayanag (CONTRIBUTOR) · review COMMENTED · 2026-07-21 (no text)

## [c3624338428] vasqu (COLLABORATOR) · review_comment on tests/models/whisper/test_feature_extraction_whisper.py:285 · 2026-07-21
Let's also add a reference to this PR in the description please

## [c4747340832] vasqu (COLLABORATOR) · review APPROVED · 2026-07-21
Thanks, one small nit and can you run make style for CI?

## [c5037274888] vbayanag (CONTRIBUTOR) · issue_comment · reply to [c4747340832] · 2026-07-21
> Thanks, one small nit and can you run make style for CI?

How to do that?

## [c5037502384] vasqu (COLLABORATOR) · issue_comment · 2026-07-21
You just have to run `make style` on the root of the folder

## [c5038629426] vbayanag (CONTRIBUTOR) · issue_comment · reply to [c5037502384] · 2026-07-21
> You just have to run `make style` on the root of the folder

Done

## [c4756397486] vasqu (COLLABORATOR) · review APPROVED · 2026-07-22 (no text)

## [c5048579155] vasqu (COLLABORATOR) · issue_comment · 2026-07-22
Merging it, thanks for your patience!

---
outcome: state=closed, merged=True
