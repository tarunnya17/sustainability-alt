# #39653 [pull_request] revert change to cu_seqlen_k and max_k when preparing from position_ids
https://github.com/huggingface/transformers/pull/39653  ·  opened by winglian on 2025-07-25

## [body] winglian (COLLABORATOR) · opening post · 2025-07-25
# What does this PR do?
follow up to PR #[39622](https://github.com/huggingface/transformers/pull/39622) as these new lines from #39474 causes this during training w multi-gpu

```
[rank2]:   File "/root/miniconda3/envs/py3.11/lib/python3.11/site-packages/flash_attn/flash_attn_interface.py", line 165, in _flash_attn_varlen_forward
[rank2]:     out, softmax_lse, S_dmask, rng_state = flash_attn_gpu.varlen_fwd(
[rank2]:                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank2]: RuntimeError: cu_seqlens_k must have shape (batch_size + 1)
[… 26 lines of code omitted]

## [c3054574172] ArthurZucker (COLLABORATOR) · review APPROVED · 2025-07-25
okay, as we discussed, generate now prepares the inputs for us so does not use this branch

---
outcome: state=closed, merged=True
