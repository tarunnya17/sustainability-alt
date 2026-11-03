# #42508 [pull_request] Fix Qwen-VL family with prompt tuning
https://github.com/huggingface/transformers/pull/42508  ·  opened by zucchini-nlp on 2025-12-01

## [body] zucchini-nlp (MEMBER) · opening post · 2025-12-01
# What does this PR do?

Fixes https://github.com/huggingface/transformers/pull/41173#issuecomment-3586424032.

TL;DR: PEFT pop position ids and cache position when doing prompt tuning, and expects that they will be reconstructed in forward using past cache length. However Qwen assumes that `cache_position == None` is always a prefill stage which will not be correct with PEFT

This PR changes the is-prefill check in qwen forward pass to be more explicit and rely on cache length. Checked with the below script

cc @BenjaminBossan . Btw, there are some more issues if we install "rc0" and run PEFT. I guess one of the breaking PR affected it, lmk if you need help on that as well

```python
import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from peft import get_peft_model, PromptTuningConfig, PromptTuningInit, TaskType

[… 25 lines of code omitted]

## [c3595835514] BenjaminBossan (MEMBER) · issue_comment · reply to [body] · 2025-12-01
Thanks @zucchini-nlp this indeed solves the issue. The generations with cache enabled are still bad compared to cache being disabled, but that was already true before #41173 so it must be something else. 

```
With cache:
Discuss the most important work by Mary Shelley. Mary Shelley,,, and, and, and and and and the novel Shelley W., and and and, and the, and Mary Shelley, and and Mary Shelley Shelley, and Mary Shelley,
```

```
Without cache:
Discuss the most important work by Mary Shelley. Mary Shelley was a British author who wrote "Frankenstein" in 1818. She was born in 1797 and died in 1851. She was a
```

> Btw, there are some more issues if we install "rc0" and run PEFT. I guess one of the breaking PR affected it, lmk if you need help on that as well

We have a backlog of a couple of things that are broken since then, but maybe not everything. What do you mean specifically?

## [c3595923789] zucchini-nlp (MEMBER) · issue_comment · reply to [c3595835514] · 2025-12-01
> The generations with cache enabled are still bad compared to cache being disabled,

Hmm, I ran the script with 4.57.3 and got a normal generation. Smth might have been broken after that in peft

> We have a backlog of a couple of things that are broken since then, but maybe not everything. What do you mean specifically?

Nothing specifically, just saw a few issues which I think are related to recent breaking changes I introduced in v5. If peft might requires fixes or changes on transformers side re these, lmk

## [c3596218388] zucchini-nlp (MEMBER) · issue_comment · 2025-12-01
run-slow: qwen2_5_omni, qwen2_5_vl, qwen2_vl, qwen3_omni_moe, qwen3_vl, qwen3_vl_moe

## [c3596943351] BenjaminBossan (MEMBER) · issue_comment · reply to [c3595923789] · 2025-12-01
> Hmm, I ran the script with 4.57.3 and got a normal generation. Smth might have been broken after that in peft

Hmm, how? Did you apply your patch from this PR to 4.57.3? Also, what PEFT version did you use? I tried a few PEFT versions back and each time, the generation was just gibberish.

> Nothing specifically, just saw a few issues which I think are related to recent breaking changes I introduced in v5. If peft might requires fixes or changes on transformers side re these, lmk

Thanks for the offer, I'll ping you if something comes up. The remaining issues I'm aware of are related to weight conversion or tied weights/gradient checkpointing.

## [c2577980081] molbap (COLLABORATOR) · review_comment on src/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1449 · 2025-12-01
seems ok, so aligned with qwen2- 2.5?

## [c3526170044] molbap (COLLABORATOR) · review COMMENTED · 2025-12-01 (no text)

## [c3526173365] molbap (COLLABORATOR) · review APPROVED · 2025-12-01
Thanks for fixing! Trusting you on the harmonisation between qwen family members.

## [c2580313674] zucchini-nlp (MEMBER) · review_comment on src/transformers/models/qwen3_vl/modeling_qwen3_vl.py:1449 · reply to [c2577980081] · 2025-12-02
yep, mostly copy from qwen2-vl. Modular is overwriting this class, so we're not copying automatically apparently

## [c3529050284] zucchini-nlp (MEMBER) · review COMMENTED · 2025-12-02 (no text)

## [c3601365884] zucchini-nlp (MEMBER) · issue_comment · reply to [c3596943351] · 2025-12-02
> Hmm, how? Did you apply your patch from this PR to 4.57.3? Also, what PEFT version did you use? I tried a few PEFT versions back and each time, the generation was just gibberish.

@BenjaminBossan oh you're right, my mind made sense of gibberish and assumed it was normal haha. The issue is again related to popping cache position because the fallback was initialize position ids of all zeros. I fixed it in my last commit to rely on cache length as it is equivalent to `cache_position[0]`

## [c3601370834] zucchini-nlp (MEMBER) · issue_comment · 2025-12-02
run-slow: qwen2_5_omni, qwen2_5_vl, qwen2_vl, qwen3_omni_moe, qwen3_vl, qwen3_vl_moe

## [c3601417277] BenjaminBossan (MEMBER) · issue_comment · reply to [c3601365884] · 2025-12-02
> The issue is again related to popping cache position because the fallback was initialize position ids of all zeros. I fixed it in my last commit to rely on cache length as it is equivalent to `cache_position[0]`

Oh really nice, now I get the same generations with and without cache, thank you so so much!

---
outcome: state=closed, merged=True
