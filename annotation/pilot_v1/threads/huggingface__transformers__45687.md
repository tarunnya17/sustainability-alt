# #45687 [pull_request] fix: Added Mps support in float fallback backends list 
https://github.com/huggingface/transformers/pull/45687  ·  opened by rigen1048 on 2026-04-28

## [body] rigen1048 (CONTRIBUTOR) · opening post · 2026-04-28
# What does this PR do?

Fixes a NotImplementedError: "histogram_mps" not implemented for 'Int' when running Mixture-of-Experts (MoE) models on Apple Silicon (MPS backend).
The error occurred in src/transformers/integrations/moe.py because torch.histc does not support integer dtypes on MPS. The original condition failed to account for the MPS backend. This PR improves the logic by checking specifically for CUDA instead, as float operations are more reliably supported across a wider range of hardware, including legacy devices.

Before:
- Use `float` on CPU
- Use `int` on all other backend (CUDA, MPS, XPU, TPU, etc)

After:
- Use `int` on CUDA (best performance)
- Use `float32` on all other backends (CPU, MPS, XPU, etc.)

Fixes #45685

## Code Agent Policy

The Transformers repo is currently being overwhelmed by a large number of PRs and issue comments written by
code agents. We are currently bottlenecked by our ability to review and respond to them. As a result,
**we ask that new users do not submit pure code agent PRs** at this time.
You may use code agents in drafting or to help you diagnose issues. We'd also ask autonomous "OpenClaw"-like agents
not to open any PRs or issues for the moment.
PRs that appear to be fully agent-written will probably be closed without review, and we may block users who do this
repeatedly or maliciously.
This is a rapidly-evolving situation that's causing significant shockwaves in the open-source community. As a result,
this policy is likely to be updated regularly in the near future. For more information, please read [CONTRIBUTING.md](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).

- [X] I confirm that this is not a pure code agent PR.
  - AI was used to understand broader application and write a manual smoke test script for regression checking & bring clarity to the PR

## Before submitting

- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [X] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request), Pull Request section?
- [X] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)?  
  Yes, discussed in [#45685](https://github.com/huggingface/transformers/issues/45685)
- [ ] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [ ] Did you write any new necessary tests?

## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

- @IlyasMoutawwakil (author of the current `grouped_mm_experts_forward` logic)
- @ArthurZucker
- @cyrilvallez

## [c4337814478] IlyasMoutawwakil (MEMBER) · issue_comment · 2026-04-28
i think it makes more sense if we continue using int as the default for any hardware we are not aware of. because it's better to have the best perf on as many hardware as possible and only use float if it breaks it.

## [c4337818838] IlyasMoutawwakil (MEMBER) · issue_comment · 2026-04-28
because int histograms are easier to compute than floating ones

## [c4337962316] rigen1048 (CONTRIBUTOR) · issue_comment · reply to [c4337818838] · 2026-04-28
@IlyasMoutawwakil I agree with you — using int by default makes sense for performance whenever it's reliably supported, since integer histograms are generally faster and cheaper to compute.
That said, a broader range of hardware (such as MPS on Apple Silicon, and potentially some other backends) currently doesn't support integer dtypes for torch.histc, which is what caused the NotImplementedError.
I also considered the "try int first, fallback to float" pattern. While it's more aggressive at giving good performance on unknown hardware, it has a couple of downsides:
- It can hide bugs or lead to inconsistent behavior across devices.
- The fallback path adds a small runtime cost and makes the code slightly more complex.

Because of that, I went with the explicit approach in this PR: int on CUDA (where we know it works well and gives the best perf) and float32 everywhere else for broad compatibility.
Happy to adjust if there's a cleaner way to handle this.

## [c4338197473] rigen1048 (CONTRIBUTOR) · issue_comment · 2026-04-28
Given that the large majority of users (especially for MoE models) run on CUDA, this approach gives them the best performance while ensuring reliable behavior on other backends like MPS. We can always expand the int64 support list later as more backends (ROCm, XPU, etc.) stabilize.

## [c4340498787] Vincent08199 (NONE) · issue_comment · 2026-04-29
It feels like this is a classic performance vs portability tradeoff:
- int histograms → better performance
- float histograms → broader hardware support (MPS, etc.)

One question I had: do we know how significant the performance difference is in practice for MoE workloads?

If the gap is large, it might be worth keeping int as the default and falling back only when needed. But if the difference is small, the current approach (float for non-CUDA) seems safer.

Also wondering whether a "try int → fallback to float" approach could be acceptable if the failure is rare and predictable.

## [c4340853714] rigen1048 (CONTRIBUTOR) · issue_comment · reply to [c4340498787] · 2026-04-29
> It feels like this is a classic performance vs portability tradeoff:
> 
>     * int histograms → better performance
> 
>     * float histograms → broader hardware support (MPS, etc.)
> 
> 
> One question I had: do we know how significant the performance difference is in practice for MoE workloads?
> 
> If the gap is large, it might be worth keeping int as the default and falling back only when needed. But if the difference is small, the current approach (float for non-CUDA) seems safer.
> 
> Also wondering whether a "try int → fallback to float" approach could be acceptable if the failure is rare and predictable.

Yes, that's correct. However, the main issue here is more about hardware reality than a classic performance vs portability tradeoff.

Integer histograms for torch.histc are still poorly supported outside of CUDA. They tend to fail on MPS, CPU, and several other backends (APU, TPU, etc.), and broader support may take a long time to mature.

My current approach uses int only on CUDA (where we get the best performance) and float32 everywhere else. This gives strong performance for the majority of MoE users while ensuring reliable behavior across hardware.

Regarding the "try int first, then fallback to float" idea — I actually considered it as well. While it could offer better performance on unknown hardware, I’m concerned it might introduce subtle issues such as problems with torch.compile, Python exception handling interfering with the memory cache, or inconsistent states on backends like Metal (MPS). That said, I might be overthinking it, so I’d really appreciate the maintainers’ conservative feedback on this.

## [c4364574832] IlyasMoutawwakil (MEMBER) · issue_comment · 2026-05-02
i wouldn't wanna penalize xpu and xla/tpu for example or any other custom torch backend that supports int histc
so i'm more inclined towards expanding the float fallback backends list rather than defaulting to float

## [c4365041179] rigen1048 (CONTRIBUTOR) · issue_comment · 2026-05-03
> i wouldn't wanna penalize xpu and xla/tpu for example or any other custom torch backend that supports int histc so i'm more inclined towards expanding the float fallback backends list rather than defaulting to float

Thank you for the response. I now see that this repo follows a performance-first, stability-second approach, which is a valid stance. I had initially gotten the priorities the other way around.
Upon reflection, your approach makes complete sense. Going with a float-compatible hardware list is smart because PyTorch and other serious AI backends are likely to expand their integer histogram support soon. This makes the current pattern nicely future-proof.
Sorry for dragging this out. I’ll keep my lessons in mind to improve my future PRs in the repo.

## [c4366850339] IlyasMoutawwakil (MEMBER) · issue_comment · 2026-05-03
basically we've had this experts impl for a couple months now, and no reports from xpu/hpu. so i'd say they probably already support int, otherwise they'd be the first to report it (we have a couple contributors from intel).

## [c4366923547] rigen1048 (CONTRIBUTOR) · issue_comment · reply to [c4366850339] · 2026-05-03
> basically we've had this experts impl for a couple months now, and no reports from xpu/hpu. so i'd say they probably already support int, otherwise they'd be the first to report it (we have a couple contributors from intel).

That's reassuring.

## [c4217313973] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-05-04
sounds good to me

## [c3180353590] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/integrations/moe.py:406 · 2026-05-04
```suggestion
    # torch.histc() does not support integer dtypes on CPU and MPS.
    histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
```

## [c4218592600] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-05-04 (no text)

## [c4218595368] IlyasMoutawwakil (MEMBER) · review APPROVED · 2026-05-04
LGTM! thanks for the fix

## [c4372192362] rigen1048 (CONTRIBUTOR) · issue_comment · 2026-05-04
I couldn't reproduce the failure using this command in my local device: uv run pytest tests/cli/test_download.py::test_cli_download_trust_remote. Perhaps local cache and automated test cache is handled differently?

## [c4374446863] rigen1048 (CONTRIBUTOR) · issue_comment · reply to [c4218595368] · 2026-05-04
Hi @IlyasMoutawwakil @ArthurZucker @Cyrilvallez, could you check the latest change? CI was failing due to a distributed race condition with pytest-xdist.
When running with -n 8, workers have isolated sys.path. If Worker A downloads a model and initializes the cache, Worker B skips initialization because the files already exist, but then fails the import because its sys.path doesn't have the cache directory.
I've updated get_class_in_module to ensure every worker is initialized. I managed to reproduce this with a script—let me know if you'd prefer this in a separate PR!

## [c4377471091] IlyasMoutawwakil (MEMBER) · issue_comment · reply to [c4374446863] · 2026-05-05
@rigen1048 thanks for investigating that's very helpful ! let's keep it in a separate PR tho since it would require reviewing from our ci/testing mantainers.

## [c3186869309] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/dynamic_module_utils.py:285 · 2026-05-05
```suggestion
```

## [c4226239432] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-05-05 (no text)

## [c3186869772] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/dynamic_module_utils.py:607 · 2026-05-05
```suggestion
```

## [c4226239918] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-05-05 (no text)

---
outcome: state=closed, merged=True
