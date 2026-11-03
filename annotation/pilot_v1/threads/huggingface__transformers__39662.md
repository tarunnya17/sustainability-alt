# #39662 [pull_request] [CI] revert device in `test_export_static_cache`
https://github.com/huggingface/transformers/pull/39662  ·  opened by gante on 2025-07-25

## [body] gante (CONTRIBUTOR) · opening post · 2025-07-25
# What does this PR do?

#38976 makes our torch export integration compatible with models on GPU.

However, there is possibly something wrong with it: small models cause GPU OOM in our related tests. This PR reverts the device while we [I need help from the export folks] are exploring the issue, to prevent other regressions.

## [c3117821466] ydshieh (COLLABORATOR) · issue_comment · 2025-07-25
OK, I confirm it works when using `cpu`, at least for `qwen3` and `phi3`

But for those 2 models, the error doesn't seem to be GPU OOM, but

```bash
E       torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors: call_method index_copy_(*(FakeTensor(..., size=(1, 8, 26, 128), dtype=torch.bfloat16), 2, FakeTensor(..., device='cuda:0', size=(1,), dtype=torch.int64), FakeTensor(..., device='cuda:0', size=(1, 8, 1, 128), dtype=torch.bfloat16,
E                  grad_fn=<AddBack
```
(But maybe related)
[… 508 lines of code omitted]

## [c3055408430] ydshieh (COLLABORATOR) · review APPROVED · 2025-07-25
Thank you!

(if you think it helps, a comment above the line `device = "cpu"` would be nice)

---
outcome: state=closed, merged=True
