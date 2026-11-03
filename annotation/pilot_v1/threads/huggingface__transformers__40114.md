# #40114 [pull_request] Fix torch.export compatibility for Mixtral MoE models 
https://github.com/huggingface/transformers/pull/40114  ·  opened by akacmazz on 2025-08-12

## [body] akacmazz (NONE) · opening post · 2025-08-12
- Replace data-dependent .nonzero() operation with static expert loop
- Resolves GuardOnDataDependentSymNode error during torch.export
- Maintains identical functionality while enabling export compatibility
- Fixes issue introduced in PR #32429
- Add tests for torch.export compatibility

# What does this PR do?

This PR fixes a torch.export compatibility issue #38518 with Mixtral MoE models that was introduced in PR #32429.

 Problem

The optimization in PR #32429 introduced a .nonzero() operation that creates data-dependent tensor shapes, causing torch.export to fail with:
torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: Could not extract specialized integer from data-dependent expression

Solution

Replace the dynamic expert selection loop:
 
expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
 for expert_idx in expert_hit:

With a static loop over all experts:
for expert_idx in range(self.num_experts):

Impact
  - ✅ Enables torch.export compatibility for Mixtral models
  - ✅ Maintains identical functionality (empty experts contribute 0 naturally)
  - ✅ Minimal performance impact (same computation, different loop structure)
  - ✅ Consistent with other MoE implementations (Jamba, DBRX)

  Testing

  - Verified torch.export works without errors
  - Confirmed functionality preservation with identical outputs
  - Tested with various input configurations

  Fixes torch.export compatibility issues reported for Mixtral-8x7B models.

## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [X] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [ ] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [X] Did you write any new necessary tests?


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag members/contributors who may be interested in your PR.

@Cyrilvallez
@ArthurZucker
 @gante

## [c3114407803] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-08-13
Thanks for the PR, but the reason we cannot do that is because it is really a lot lot less efficient! I would recommend to rather have 2 paths, 1 for training one for inference, inference can just not loop at all and repeat the inputs

## [c3182704117] akacmazz (NONE) · issue_comment · reply to [c3114407803] · 2025-08-13
> the reason we cannot do that is because it is really a lot lot less efficient! I would recommend to rather have 2 paths, 1 for training one for inference, inference can just not loop at all and repeat the inputs

Thx for the feedback, i will work on this

## [c3292957874] sshonTT (NONE) · issue_comment · reply to [c3182704117] · 2025-09-15
@akacmazz , @ArthurZucker Hi, could I know the blocker for this? I found this commit from #38518 , and I just want to know how things are going on.

## [c2351633747] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/mixtral/modeling_mixtral.py:160 · 2025-09-16
Hey! we should avoid this, it will really slow down "all" use cases. Is there no way to still export but have a different forward? Like using a kernel from the hub / using the bmm version? Or guarding manually or something?

## [c3228633107] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-09-16 (no text)

## [c4353763685] t8 (NONE) · issue_comment · reply to [c3228633107] · 2026-04-30
Hi @ArthurZucker @Cyrilvallez @akacmazz, dropping CPU benchmark data in case it helps unblock review.

I needed to export Mixtral via `torch.export` and ran the perf comparison between three forms of `MixtralSparseMoeBlock.forward`:

- **A — `.nonzero()`**: the form on `main` (and 4.53.3) that this PR is trying to fix.
- **B — static `range(num_experts)` loop**: the inference path proposed by this PR.
- **C — dense (no loop, repeat inputs through all experts, gate by routing weight)**: the form @ArthurZucker suggested in the first review (`inference can just not loop at all and repeat the inputs`).

## Setup

- `transformers==4.53.3` (the `.nonzero()` inference path matches `main` for the relevant code), `torch==2.7.0+cpu`
- Single `MixtralSparseMoeBlock` at random init, Mixtral 8x7B config: `hidden_size=4096`, `intermediate_size=14336`, `num_local_experts=8`, `num_experts_per_tok=2`
- Linux x86_64, 24-core Xeon Gold 6252, 24 PyTorch threads, fp32
- Median of 10 iters per scenario, 3 warmup iters
- Output equality verified: A vs B is bit-identical (max-abs-diff = 0.00e+00); A vs C is 3.05e-07 (different summation order)

## Results — single `MixtralSparseMoeBlock.forward`, fp32, ms

| scenario           | A `.nonzero()` | B static-loop (this PR) | C dense (no loop) | B/A    | C/A    |
|--------------------|----------------|-------------------------|-------------------|--------|--------|
| decode  b=1 s=1    |       31.5 ms  |                32.2 ms  |        123.9 ms   | 1.022x | 3.927x |
| decode  b=4 s=1    |       93.4 ms  |                93.4 ms  |        239.5 ms   | 1.000x | 2.564x |
| prefill b=1 s=128  |      302.2 ms  |               300.0 ms  |        487.7 ms   | 0.993x | 1.614x |
| prefill b=1 s=512  |      393.1 ms  |               394.7 ms  |       1593.1 ms   | 1.004x | 4.052x |
| prefill b=1 s=2048 |     1388.7 ms  |              1371.3 ms  |       3491.6 ms   | 0.987x | 2.514x |

I re-ran prefill seq=2048 separately with 5 warmups + 25 iters and again with A/B interleaved 10x each (to mitigate cache/thermal effects):

```
prefill seq=2048 confirmation (5 warmups + 25 iters):
  A nonzero:     median=1335.1 ms  IQR=[1333.7-1338.2]
  B static-loop: median=1334.8 ms  IQR=[1331.9-1336.1]   ratio=1.000x

[… 17 lines of code omitted]

---
outcome: state=open, merged=False
