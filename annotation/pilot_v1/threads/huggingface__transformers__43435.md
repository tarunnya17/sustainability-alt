# #43435 [pull_request] XPU now supports MoE kernel(MegaBlocks) implementation
https://github.com/huggingface/transformers/pull/43435  ·  opened by YangKai0616 on 2026-01-23

## [body] YangKai0616 (CONTRIBUTOR) · opening post · 2026-01-23
# What does this PR do?





XPU now supports the **MoE kernel(MegaBlocks)** implementation, and includes XPU checks for GPT-OSS UTs compatibility.

## [c3789428854] YangKai0616 (CONTRIBUTOR) · issue_comment · 2026-01-23
run-slow: gpt_oss

## [c3789495223] YangKai0616 (CONTRIBUTOR) · issue_comment · 2026-01-23
Hi @ydshieh, please help review, thanks!

## [c3816390076] YangKai0616 (CONTRIBUTOR) · issue_comment · 2026-01-29
Hi @ydshieh , feel free to reach out if you have any questions!

## [c3821842117] sywangyi (CONTRIBUTOR) · issue_comment · 2026-01-30
@MekkCyber please help review

## [c2745332561] MekkCyber (CONTRIBUTOR) · review_comment on tests/models/gpt_oss/test_modeling_gpt_oss.py:141 · 2026-01-30
I guess it's not needed ?
```suggestion
```

## [c3727172771] MekkCyber (CONTRIBUTOR) · review COMMENTED · 2026-01-30
very niiice thank you ! Could you add to the pr description some benchmarks with and without kernels please ? That would be awesome to showcase why kernels are useful

## [c3830971192] YangKai0616 (CONTRIBUTOR) · issue_comment · 2026-02-01
1. Ran command `python3 run_benchmarks.py --model-id openai/gpt-oss-20b  --batch-size 2  --num-tokens-to-generate 256 --level 0 --sequence-length 256 --warmup 1 --iterations 1 --no-gpu-monitoring` on XPU. Test results are as follows:

**eager expert**:
```
Config: 1 warmup, 1 iterations, no GPU monitoring, batch size 2, sequence length 256, 256 generated tokens, eager attention, not compiled, kernelized, experts_eager, regular generate
 
  E2E Latency.........: avg=29.67s     std=0.00ns    min=29.67s     med=29.67s     max=29.67s     p95=29.67s
  Time to First Token.: avg=500.00ms   std=0.00ns    min=500.00ms   med=500.00ms   max=500.00ms   p95=500.00ms
  Inter-Token Latency.: avg=113.77ms   std=0.00ns    min=113.77ms   med=113.77ms   max=113.77ms   p95=113.77ms
[… 23 lines of code omitted]
```

## [c3738103005] MekkCyber (CONTRIBUTOR) · review APPROVED · 2026-02-02
Very nice! Thank you! cc @ArthurZucker if you can check and merge

## [c3738769777] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-02-02
Nice! Thanks

---
outcome: state=closed, merged=True
