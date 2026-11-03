# #39164 [pull_request] enable static cache on TP model
https://github.com/huggingface/transformers/pull/39164  ·  opened by jiqing-feng on 2025-07-02

## [body] jiqing-feng (CONTRIBUTOR) · opening post · 2025-07-02
The TP model usually needs torch.compile to get speed-up, but the static cache failed on running TP model. This PR will check the tp size and then allocate the correct cache shape.

Reproduce error:
`torchrun --standalone --nproc-per-node 2 tp_hf.py`
```python
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM


[… 24 lines of code omitted]
```

## [c3031060931] jiqing-feng (CONTRIBUTOR) · issue_comment · 2025-07-03
Hi @yao-matrix . Could you please review the PR? Thanks!

## [c2982033480] yao-matrix (CONTRIBUTOR) · review APPROVED · 2025-07-03 (no text)

## [c3031172851] jiqing-feng (CONTRIBUTOR) · issue_comment · 2025-07-03
Hi @Rocketknight1 . Would you please review this PR? Thanks!

## [c3034224445] jiqing-feng (CONTRIBUTOR) · issue_comment · 2025-07-04
Hi @SunMarc . This PR changes the StaticCache api, needs your review. Thanks!

## [c2987284460] SunMarc (MEMBER) · review COMMENTED · reply to [c3034224445] · 2025-07-04
Nice feature @jiqing-feng ! Can you also add a test for that ? Also, what is the speedup that we can get from tp -> tp + compile with the script you shared ?

## [c2185466898] gante (CONTRIBUTOR) · review_comment on src/transformers/cache_utils.py:1153 · 2025-07-04
assert -> raise ValueError

(we don't use assert in our codebase 🤗 )

## [c2987389549] gante (CONTRIBUTOR) · review COMMENTED · 2025-07-04
Nice 🔥 

Left a minor nit. I'm also curious to see the speedup numbers 👀

## [c2185482947] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/cache_utils.py:1153 · reply to [c2185466898] · 2025-07-04
more than that the way to go is to use lazy init. We never at tp specific code. #38495 

The main thing is making that lazy init compile compatible!

## [c2987420125] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-07-04
Eager to have that working!

## [c3043243610] jiqing-feng (CONTRIBUTOR) · issue_comment · reply to [c2987420125] · 2025-07-07
Hi @ArthurZucker . I didn't meet any compile issue. We usually only use compile on the model forward rather than the whole generation. The init cache process will not be traced in my script.

`TORCHINDUCTOR_CPP_WRAPPER=1 TORCHINDUCTOR_FREEZING=1 torchrun --standalone --nproc-per-node 2 tp_hf.py`
```python
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM


[… 12 lines of code omitted]
```

## [c3043326578] jiqing-feng (CONTRIBUTOR) · issue_comment · 2025-07-07
I've added the test, please run `CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/tensor_parallel/test_tensor_parallel.py`.

## [c3043329271] jiqing-feng (CONTRIBUTOR) · issue_comment · 2025-07-07
I can get the 30% speed-up when tp=2 and 60% speed-up when tp=4 on Intel Xeon 6th Gen CPU, and 20%+ speed-up on Intel XPU. 

```
| TP size | CPU  |  XPU  | A100  |
| 2       | 1.3x | 1.24x | 1.07x |
| 4       | 1.6x | 1.54x | 1.07x |
```

The A100 has almost no speed-up because the communication is slow via PCIe. I don't have other NV devices other than A100 PCIe. You can test it on an NVlink device like H100 to see the speed-up.

## [c2191375309] jiqing-feng (CONTRIBUTOR) · review_comment on src/transformers/cache_utils.py:1153 · reply to [c2185466898] · 2025-07-08
OK, I get your point. We don't need to use `config.num_key_value_heads` to init the kv-cache as long as we init the cache in the update function.

## [c2995824839] jiqing-feng (CONTRIBUTOR) · review COMMENTED · 2025-07-08 (no text)

## [c2191397557] jiqing-feng (CONTRIBUTOR) · review_comment on src/transformers/cache_utils.py:1153 · reply to [c2185466898] · 2025-07-08
However, it will make the compile more complicated. It will also trigger recompile and graph break because you use `torch.compiler.disable`. I haven't found a way to solve the recompile and break issue.

## [c2995856884] jiqing-feng (CONTRIBUTOR) · review COMMENTED · 2025-07-08 (no text)

## [c3051565587] ArthurZucker (COLLABORATOR) · issue_comment · 2025-07-09
Ah sorry, if you use static cache, generate will compile by default!

## [c3051567119] ArthurZucker (COLLABORATOR) · issue_comment · 2025-07-09
```diff
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM


[… 13 lines of code omitted]
```

## [c2194329226] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/cache_utils.py:1153 · reply to [c2185466898] · 2025-07-09
Okay, I am wondering if we can use https://docs.pytorch.org/docs/stable/generated/torch.compiler.allow_in_graph.html ?

## [c3000457568] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-07-09 (no text)

## [c3000473600] ArthurZucker (COLLABORATOR) · review APPROVED · 2025-07-09
Very nice! I am just wondering if we have a way to lazy init... 
https://docs.pytorch.org/docs/stable/generated/torch.compiler.allow_in_graph.html or https://github.com/pytorch/pytorch/issues/156377 

As it does not really seem possible, can you make sure the tp_size fix is propagated to other caches if possible?

## [c2194340457] jiqing-feng (CONTRIBUTOR) · review_comment on src/transformers/cache_utils.py:1153 · reply to [c2185466898] · 2025-07-09
I am not in favor of using any workaround that may make the compilation complicated. It's better to prepare the kv cache in init stage.

## [c3000473982] jiqing-feng (CONTRIBUTOR) · review COMMENTED · 2025-07-09 (no text)

## [c3051596133] jiqing-feng (CONTRIBUTOR) · issue_comment · reply to [c3051567119] · 2025-07-09
> ```diff
> -model.forward = torch.compile(model.forward)
> outputs = model.generate(**inputs, max_new_tokens=10, cache_implementation="static")
> ```
> 
> should already use compile!

It seems that only cuda will trigger the compile when using static cache, cpu still needs to compile manually.

## [c3051629054] jiqing-feng (CONTRIBUTOR) · issue_comment · reply to [c3000473600] · 2025-07-09
> Very nice! I am just wondering if we have a way to lazy init... https://docs.pytorch.org/docs/stable/generated/torch.compiler.allow_in_graph.html or [pytorch/pytorch#156377](https://github.com/pytorch/pytorch/issues/156377)
> 
> As it does not really seem possible, can you make sure the tp_size fix is propagated to other caches if possible?

Done.

## [c3054002107] ArthurZucker (COLLABORATOR) · issue_comment · 2025-07-09
Thanks! I am a bit sad that we are not able to hide the tensor parallel but hoping we can in the future!

---
outcome: state=closed, merged=True
