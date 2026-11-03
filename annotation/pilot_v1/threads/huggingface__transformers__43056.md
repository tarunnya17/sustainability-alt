# #43056 [pull_request] Perf: enable pin_memory in DataLoader for CLM no_trainer example
https://github.com/huggingface/transformers/pull/43056  ·  opened by RinZ27 on 2025-12-27

## [body] RinZ27 (NONE) · opening post · 2025-12-27
This PR enables `pin_memory=True` in the `DataLoader` for the causal language modeling standalone example.

Pinned memory is a PyTorch best practice that improves host-to-device transfer efficiency.

## [c3710934121] ArthurZucker (COLLABORATOR) · issue_comment · 2026-01-05
thanks! tho we don't really maintain these anymore! @merveenoyan do you want to merge?

## [c3732319715] RinZ27 (NONE) · issue_comment · reply to [c3710934121] · 2026-01-10
Hi @ArthurZucker, thanks for the response. @merveenoyan, just checking in to see if you have any thoughts on merging this. It's a small change but would be beneficial for anyone still using these examples for reference. Thanks!

## [c3773577576] RinZ27 (NONE) · issue_comment · 2026-01-20
I've squashed the commits into a single clean one and ensured the `pin_memory` configuration properly checks for CUDA availability. The implementation is now streamlined and ready. @merveenoyan, would you mind taking a look for a final review when you have a moment? Thanks!

---
outcome: state=open, merged=False
