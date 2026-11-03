# #40870 [pull_request] Reduce vRAM usage during generation by allowing to transfer logits to CPU
https://github.com/huggingface/transformers/pull/40870  ·  opened by SamuelBarryCS on 2025-09-13

## [body] SamuelBarryCS (CONTRIBUTOR) · opening post · 2025-09-13
## What
- Fixes https://github.com/huggingface/transformers/issues/40794 by adding a parameter offload_logits_to_cpu to GenerationConfig which transfers logits and scores tensors to the CPUs after generation.
- Frees up memory during large runs, trading decreased vRAM usage for CPU/ GPU communication time, enabling potentially higher batch size or seq length during training.
- Adds test `tests.generation.test_utils.test_offload_logits_to_cpu` to test non regression

## How to review
- Read diff
- Check that new tests test_offload_logits_to_cpu is correct and that tests are still passing

## Testing performed
- All existing tests still passing fine
- tests.generation.test_utils.test_offload_logits_to_cpu passing fine as well:
```
(hf) samuel.barry@RNO:slurm-h100-reserved-rno-199-065:~/workspace/transformers(transfer-logits-to-cpu)$ python -m unittest tests.models.gpt2.test_modeling_gpt2.GPT2ModelTest.test_offload_logits_to_cpu -v
[2025-09-14 03:50:08,430] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
test_offload_logits_to_cpu (tests.models.gpt2.test_modeling_gpt2.GPT2ModelTest) ... ok

----------------------------------------------------------------------
[… 35 lines of code omitted]
```

## [c3289156256] SamuelBarryCS (CONTRIBUTOR) · issue_comment · 2025-09-14
cc @gante @SunMarc for a quick review when you get the time! :)

## [c2347103924] SamuelBarryCS (CONTRIBUTOR) · review_comment on memory_test.py:1 · 2025-09-14
(This script will of course be deleted before merging)

## [c3221710331] SamuelBarryCS (CONTRIBUTOR) · review COMMENTED · 2025-09-14 (no text)

## [c3290101803] YunruiZhang (NONE) · issue_comment · 2025-09-15
Thank you for the rapid response to the request! Since we’re at it, maybe it would be a good idea to do the same for output_attentions?

## [c3290353076] SamuelBarryCS (CONTRIBUTOR) · issue_comment · reply to [c3290101803] · 2025-09-15
> Thank you for the rapid response to the request! Since we’re at it, maybe it would be a good idea to do the same for output_attentions?

Very fair point @YunruiZhang 
Done here https://github.com/huggingface/transformers/pull/40870/commits/d1691e54c3e053e5af37af740f31a9c3165c0d95 :) 
I feel like the parameter `offload_logits_to_cpu` isn't well named anymore. Maybe I should rename to offload_output_to_cpu or something like this, wdyt ?

## [c3291396754] ArthurZucker (COLLABORATOR) · issue_comment · 2025-09-15
cc @gante

## [c3244335087] gante (CONTRIBUTOR) · review CHANGES_REQUESTED · 2025-09-19
Please see my full reply here: https://github.com/huggingface/transformers/issues/40794#issuecomment-3311719647

TL;DR the feature is desirable! But it will clash with an ongoing refactor, it will be much simpler if we add the feature after the refactor 💛

---
outcome: state=open, merged=False
