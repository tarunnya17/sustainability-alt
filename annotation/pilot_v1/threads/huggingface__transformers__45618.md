# #45618 [pull_request] Add MTP speculative decoding via MTPCandidateGenerator
https://github.com/huggingface/transformers/pull/45618  ·  opened by ArthurZucker on 2026-04-24

## [body] ArthurZucker (COLLABORATOR) · opening post · 2026-04-24
Adds `use_mtp=True` to `generate` for DeepSeek-V3 / GLM-4 MoE. MTP modules live in a new `generation.candidate_generators.MTPCandidateGenerator` (loaded via its `from_pretrained`) — the base model stays clean.

## [c4496572551] curnane-lab (NONE) · issue_comment · reply to [body] · 2026-05-20
Hi @ArthurZucker, really excited about this PR! MTP speculative decoding is a huge win for inference throughput.

I noticed the branch currently has merge conflicts — would it help if I (or someone from the community) rebased it against latest main? Also, for awareness, I'm working on MTP training support for Qwen3.5 in [#45638](https://github.com/huggingface/transformers/pull/45638), which could complement this nicely if we align on the config naming (num_nextn_predict_layers vs mtp_num_hidden_layers).

No pressure, just offering help if it speeds things up! 🙏

## [c4591620537] ArthurZucker (COLLABORATOR) · issue_comment · 2026-06-01
This was taken over in [46229](https://github.com/huggingface/transformers/pull/46229) i

---
outcome: state=closed, merged=False
