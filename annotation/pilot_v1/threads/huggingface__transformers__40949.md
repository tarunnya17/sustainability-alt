# #40949 [pull_request] Update expected values for some `test_speculative_generation`
https://github.com/huggingface/transformers/pull/40949  ·  opened by ydshieh on 2025-09-17

## [body] ydshieh (COLLABORATOR) · opening post · 2025-09-17
# What does this PR do?

These tests are failing after #40657.

As discussed offline [here](https://huggingface.slack.com/archives/C01NE71C4F7/p1758117410373159?thread_ts=1758031736.841469&cid=C01NE71C4F7), it's expected 

TL:DR

> The logits are updated due to changes in assistant temperature, as we move from implicit T=1 to explicit calibrated defaults.

Details

> turns out the PR inadvertently changed an implicit algorithmic bias: candidate_generator (the assistant model) was getting logits_processor while the decoding method (main model) was getting prepared_logits_processor. This meant that the assistant was running with T=1 while the main model was using lower temp.
We investigated and its good for speculation to have a hotter assistant model (so it was a good bug that we were not applying the lower temp to the assistant),
But it should be explicitly set and not a hidden argument forwarding consequence. So its correct as is for now, and in the future we will bring better defaults for assisted generation, after benchmarking a little more.

[image]

## [c3304176906] ydshieh (COLLABORATOR) · issue_comment · 2025-09-17
Merge directly as we all agree offline.

---
outcome: state=closed, merged=True
