# #47316 [pull_request] [Improvement] Re-structure sonic MoE kernel interface to use a Handle object
https://github.com/huggingface/transformers/pull/47316  ·  opened by remi-or on 2026-07-14

## [body] remi-or (COLLABORATOR) · opening post · 2026-07-14
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47316)

In #47126 and offline we talked about how we want to take care of loading kernels like `DeepGEMM`.

Notably, we can put everything related to loading and calling the kernel inside a singleton `Handle` object. This object would only keep track of a few state variable and provide a single entry point to access the kernel. This has the following advantages:
- we can keep track of the cached state without using `@lru_cache` which can be opaque and cause memory leaks
- failures are actually cached: `@lru_cache` does not cache exceptions, so the cache never worked in a machine / env that did not support `sonicmoe`
- we can reset the cached state of the handle whenever we want through a simple `reset` method
- we can check the availability of the kernel easily through a `sonicmoe_is_available` property, that accesses the same underlying cached kernel as the function call itself (quite convenient, as is shown in the reworked `_test_eager_matches_batched_and_grouped_inference`)

The downside of the Handle API are that transient failures are also cached (hence the `reset` method, but it is not called automatically) and it's beefier than `@lru_cache`. IMO this is an ok tradeoff, but I would love some feedback on this.

Since I was fixing a failure in `sonicmoe`, I thought this was a good occasion to try out the Handle API. The fail was an image failure, which is now guarded against.

I also added a test for sonicmoe + compile and it passes on B200. Also tested the sonicmoe regular tests on a qwen and they pass.

## [c3578080441] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/integrations/sonicmoe.py:104 · 2026-07-14
can you please also add the specific versions that worked for you of tvm ffi and cutedsl, i didn't expect it to start failing so fast with newer versions, when we tell users to install them they'll just do "pip install nvidia-cutlass-dsl" which would then result in version mismatch between the vendored quack and the installed cutedsl
cc @vasqu

## [c4693280252] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-07-14 (no text)

## [c3578103476] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/integrations/sonicmoe.py:81 · 2026-07-14
sorry but i am once again totally against syncs and tensor creation in the loader (also does this not break cuda graph capture ?) when a more direct alternative exist, in this case checking cutedsl exists and explicitly verifying its version. here the user sees import error no package named cutlass.

## [c4693306865] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-07-14 (no text)

## [c3578121597] IlyasMoutawwakil (MEMBER) · review_comment on src/transformers/integrations/sonicmoe.py:174 · 2026-07-14
the scope of the handle is a bit fuzzy, it doesn't handle the full loading (lets _load_sonicmoe_kernel do part of it) and here it implements something very implementation specific.. With FinegrainedFP8 for example which implements multiple implementations, it will be a very big handle. I think the implementation part should stay outside of it.

## [c4693326445] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-07-14 (no text)

## [c4693370294] IlyasMoutawwakil (MEMBER) · review CHANGES_REQUESTED · 2026-07-14
love the idea of the handle and standardizing the interaction with the kernel package (loading, checking for availability, completeness and usability), i just think it can be scoped better and also it should generalize to other kernel integrations. for example we can make it so that it has a class attr of required packages and min/max versions etc, a bit like we do in exporters' `validate_environment`. the list of required attributes as well can be programatically checked. ofc some custom checks will be necessary for deepgemm and its cuda toolkit for example.

## [c3579089622] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/sonicmoe.py:104 · reply to [c3578080441] · 2026-07-14
:100: we just had discussions around that for the past few days so let's guard it on our side properly

## [c4694403637] vasqu (COLLABORATOR) · review COMMENTED · 2026-07-14 (no text)

## [c3579205099] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/sonicmoe.py:174 · reply to [c3578121597] · 2026-07-14
Yea I think the goal here was to only handle the import / cache structure; in that case we really should be strict and dont mix functionality

## [c3579242202] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/sonicmoe.py:233 · 2026-07-14
Tbh I'm not a fan of singleton objects (but that's more of a personal opinion) - it feels like OOP for the sake of OOP 😅 

I'm still a bit confused about the actual benefits and why we may not adopt a generalized pattern similar to flash attention https://github.com/huggingface/transformers/blob/a6895655b289cc3fdd29afec36904e0b8545ef92/src/transformers/modeling_flash_attention_utils.py#L124-L272 (which is more complicated but I mean the lazy import x globals structure)

I think the biggest point is `is_sonicmoe_available` and the avoidance of the lru cache
- `is_sonicmoe_available` could be properly moved to import utils and we make proper versioning guards, e.g. is kernels available and the respective cute version. 
- Re lru cache, I feel like this is valid but we can restructure to an FA like structure as well (I think deepgemm does something similar)

## [c4694539897] vasqu (COLLABORATOR) · review COMMENTED · 2026-07-14
I feel like the pain points are valid but the direction is not ideal

I think we already have existing patterns we could apply to solve all these problems, notably 
- the lru cache (with the loading structure of deepgemm for example)
- the availability check can very well be moved to import utils (and it makes things a lot clearer, not relying on try catch)

## [c3584055310] remi-or (COLLABORATOR) · review_comment on src/transformers/integrations/sonicmoe.py:104 · reply to [c3578080441] · 2026-07-15
Here you go:
```
nvidia-cutlass-dsl                       4.5.2
apache-tvm-ffi                           0.1.9
```

## [c4700351877] remi-or (COLLABORATOR) · review COMMENTED · 2026-07-15 (no text)

## [c3584066216] remi-or (COLLABORATOR) · review_comment on src/transformers/integrations/sonicmoe.py:81 · reply to [c3578103476] · 2026-07-15
Yeah the `synchronize` should be guarded against, mb. As for tensor creation, sure I don't mind avoiding them, I will look into the version check.

## [c4700363091] remi-or (COLLABORATOR) · review COMMENTED · 2026-07-15 (no text)

## [c3584070237] remi-or (COLLABORATOR) · review_comment on src/transformers/integrations/sonicmoe.py:174 · reply to [c3578121597] · 2026-07-15
Makes a lot of sense, will change it up

## [c4700367772] remi-or (COLLABORATOR) · review COMMENTED · 2026-07-15 (no text)

## [c3584077075] remi-or (COLLABORATOR) · review_comment on src/transformers/integrations/sonicmoe.py:233 · reply to [c3579242202] · 2026-07-15
Haha my personal would be a singleton object over globals, but then again it comes back to the same results (pretty much) and FA already uses globals so might as well do the same. Will reformat into the same pattern and we can re-evaluate from there.

## [c4700375239] remi-or (COLLABORATOR) · review COMMENTED · 2026-07-15 (no text)

## [c4976304105] remi-or (COLLABORATOR) · issue_comment · 2026-07-15
Lots of feedback, thanks! This was more of a rough PoC so lots to take in, and I did not plan to spend too much time on this, it was more because tests were crashing on main that I took the opportunity to look into it. Will update when I have the bandwidth!

## [c4980966887] vasqu (COLLABORATOR) · issue_comment · 2026-07-15
No worries, we also nudged the kernels team so maybe we can simplify a bit more in the future. It's a bit awkward to handle these kernels :cry:

## [c5326711371] remi-or (COLLABORATOR) · issue_comment · 2026-08-18
Re-visiting this when the `kernels` package has landed more checks on their side

## [c5391899150] vasqu (COLLABORATOR) · issue_comment · reply to [c5326711371] · 2026-08-24
I think the kernels team plans to ship the deps as separate kernels so the dep hell will be more properly handled in a future kernels release @remi-or @IlyasMoutawwakil 

(then we only need to bump kernels)

---
outcome: state=closed, merged=False
