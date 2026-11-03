# #45134 [pull_request] Optimize Parakeet feature extraction on CUDA
https://github.com/huggingface/transformers/pull/45134  ·  opened by milesial on 2026-03-31

## [body] milesial (CONTRIBUTOR) · opening post · 2026-03-31
# What does this PR do?

Add support for CUDA parakeet preprocessor, running STFT and mel spectrogram extraction on the GPU.
This refactor also speeds up the CPU implementation.

Tested on `nvidia/parakeet-ctc-0.6b`, B200, 300s audio:

Before this PR, CPU: 28ms
After this PR, CPU: 21ms
After this PR, GPU: 1.7ms

No impact on accuracy (VoxPopuli).


Context for this one is to accelerate vLLM for our multimodal nemotron model. Processing audio inputs has a bottleneck on this CPU feature extractor. This PR does a small refactor that gives some good speedup for CPU, and also enables the CUDA backend to be used to accelerate even further.

Some rough numbers, processing a 30min audio of 16 kHz:

    Before this PR, CPU: 770ms
    After this PR, CPU: 265ms (~3x)
    After this PR, CUDA: 20ms (~38x)

Several changes in this PR:

    Enabling CUDA backend
    Caching of the Hann window and mel filters
    Dynamic torch compile at strategic locations
    GPU-friendly padding
    More efficient and accurate complex number magnitude computation


- [x] I confirm that this is not a pure code agent PR.

## Before submitting
- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

## [c4162372292] Rocketknight1 (MEMBER) · issue_comment · 2026-03-31
cc @eustlb @ebezzam

## [c4162407953] eustlb (COLLABORATOR) · issue_comment · reply to [body] · 2026-03-31
Hey @milesial, interesting PR! This should be covered out of the box by #44394, so I'll wait it to land (coming days) beofre doing a full review. Context on what motivated you to do so and limitations you had with the current implem would be golden feedback for us 🙏

## [c4181744895] milesial (CONTRIBUTOR) · issue_comment · reply to [c4162407953] · 2026-04-03
Hi @eustlb , thanks for linking your PR!
Context for this one is to accelerate vLLM for our multimodal nemotron model. Processing audio inputs has a bottleneck on this CPU feature extractor. This PR does a small refactor that gives some good speedup for CPU, and also enables the CUDA backend to be used to accelerate even further.

Some rough numbers, processing a 30min audio of 16 kHz:
- Before this PR, CPU:  770ms
- After this PR, CPU: 265ms (~3x)
- After this PR, CUDA: 5.5ms (~140x)


Several changes in this PR:
- Enabling CUDA backend
- Caching of the Hann window and mel filters
- Dynamic torch compile at strategic locations
- GPU-friendly padding
- More efficient and accurate complex number magnitude computation

I see your PR is a major refactor of audio processors, does it address these points as well?
I tried your PR and these are the numbers I got:
- CPU: 690ms
- CUDA (after some patches, did not work OOTB): 16.5ms


Other question would be about timelines, I'm guessing your PR can take a while to get merged, while this smaller one can get merged faster in the meantime and unblock us, what do you think?

## [c4199940217] milesial (CONTRIBUTOR) · issue_comment · reply to [c4162407953] · 2026-04-07
Gentle ping @eustlb @ebezzam on the above comment

## [c4278892285] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-05-13 (no text)

---
outcome: state=open, merged=False
