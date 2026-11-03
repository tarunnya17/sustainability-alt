# #45635 [pull_request] qa: speed up dtype regex weight load + reduce dtype tests to 3 random
https://github.com/huggingface/transformers/pull/45635  ·  opened by tarekziade on 2026-04-24

## [body] tarekziade (COLLABORATOR) · opening post · 2026-04-24
# What does this PR do?

1. make sure we reuse the complied regex across calls
2. reduce the test to 3 dtypes only (picked randomly across all supported) so we speed up the tests but don't lose coverage over time.

on my m5 that drops

```
tests/models/d_fine/test_modeling_d_fine.py::DFineModelTest::test_bc_torch_dtype
```

from 7.13s to 2.17s.

## [c3144701061] Cyrilvallez (MEMBER) · review_comment on src/transformers/core_model_loading.py:662 · 2026-04-27
From my tests, compiling a regex is extremely fast, of the order of `1e-7`/`1e-6` s. Are you sure it's actually a bottleneck?

## [c3144702734] Cyrilvallez (MEMBER) · review_comment on src/transformers/core_model_loading.py:969 · 2026-04-27
Probably not needed to have an outer function here

## [c3144710402] Cyrilvallez (MEMBER) · review_comment on src/transformers/core_model_loading.py:1096 · 2026-04-27
`get_submodule` should be extremely fast as well, see [here](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule) where they say that the compelxity depends on the nesting, which is always small

## [c3144712144] Cyrilvallez (MEMBER) · review_comment on src/transformers/core_model_loading.py:1367 · 2026-04-27
Not really nice IMO to define an inner function like this

## [c4177970913] Cyrilvallez (MEMBER) · review COMMENTED · reply to [body] · 2026-04-27
Hey @tarekziade! I'm not really convinced that the regex compiling and the `get_submodule` calls are actual bottlenecks!
This very very simple snippet:
```python
import re
import time

N = 1000

[… 13 lines of code omitted]
```
shows that the `compile` call is only of the order of 1e-6/1e-7.
And for `get_submodule`, the complexity is bounded by the depth of the graph, which is super small as well (usually maximum 10 I'd say)

## [c4324554708] tarekziade (COLLABORATOR) · issue_comment · reply to [c4177970913] · 2026-04-27
> Hey @tarekziade! I'm not really convinced that the regex compiling and the `get_submodule` calls are actual bottlenecks! This very very simple snippet:
> 
> ```python
> import re
> import time
> 
> N = 1000
> 
> a = ["qwe|ui"] * 15
> # Construct the regex we will use to rename keys from the sources to the targets
> branches = []
> for i, source_pattern in enumerate(a):
>     group_name = f"g{i}"
>     pattern = source_pattern.replace(".*.", r"\..*\.")
>     branches.append(f"(?P<{group_name}>{pattern})")
> 
> t0 = time.time()
> for _ in range(N):
>     compiled_sources = re.compile("|".join(branches))
> dt = time.time() - t0
> print(f"Took {dt/N:.2e} s")
> ```
> 
> shows that the `compile` call is only of the order of 1e-6/1e-7. And for `get_submodule`, the complexity is bounded by the depth of the graph, which is super small as well (usually maximum 10 I'd say)

Sorry I should have been clearer, `re.compile` itself is not that slow, it's the fact that it's rarely needed, so by lazy loading it we get a small speedup because of the numerous `WeightRenaming` instances we create when loading weights. That class is a hot path and its constructor should stay as lightweight as possible.

Using ~18,000 keys and 9 repeats,  median times:

  - WeightRenaming init only
      - old: 0.589529s
      - new: 0.069270s
      - gain: 0.520258s (88.2%), 8.51x faster
  - WeightRenaming init + first rename_source_key()
      - old: 0.605083s
      - new: 0.602909s
      - gain: 0.002174s (0.4%), effectively flat
  - top-level rename_source_key() prefix remove path
      - old: 0.009985s
      - new: 0.005094s
      - gain: 0.004892s (49.0%), 1.96x faster
  - top-level rename_source_key() prefix add path
      - old: 0.004415s
      - new: 0.004897s
      - delta: -0.000483s (-10.9%)
      - this is a tiny absolute regression, about 0.48 ms over all 18,424 keys

Now with the tests being slow right now, random 3x dtypes solves it. I guess it's deciding if optimizing  `WeightRenaming` worth the added complexity.

## [c4324612616] tarekziade (COLLABORATOR) · issue_comment · 2026-04-27
Using the D-FINE `test_bc_torch_dtype` run as the baseline, the lazy regex compile is much smaller than cutting the dtype matrix, but it is still measurable. 

- Full 7-dtype version of tests/test_modeling_common.py 7.13s
- 3-random-dtype version 2.17s
- Benefit from reducing dtypes 4.96s saved, about 69.6% faster

For the lazy regex change in src/transformers/core_model_loading.py, I benchmarked the exact number of weights loaded by the old 7-dtype D-FINE run:

- The test does 28 loads total: 14 x 650 weights and 14 x 666 weights
- That is 18,424 loaded weights total
- The isolated old-vs-new regex benchmark over 18,424 keys showed 0.520s saved from lazy regex compilation

So, at the whole-test level:
- Lazy regex benefit on the old 7-dtype run: about 0.52s
- 3-random-dtype benefit: 4.96s
- Lazy regex is about 10.5% of the dtype-reduction benefit

To recap, the lazy regex compile is a smaller but real secondary win. But in practice, when running in production, regex is quite a small win since we don't load but a single dtype at a time. 

That said I would still recommend doing it and add a comment in that class to say it's instanciated a lot at load time and its consrtuctor should stay as thin as possible

## [c4325115545] Cyrilvallez (MEMBER) · issue_comment · 2026-04-27
Alright I see, thanks for the added explanation!
Agreed that we can keep the lazy regex compile, since it's a very easy and sometimes noticeable improvement! Let's just add why we do it in the property - mostly the fact that every key will go through a fresh WeightRenaming if not matching any weight ops, for convenience (but those don't need to call `rename_source_key`, so the compiling is wasted)

## [c3152172218] tarekziade (COLLABORATOR) · review_comment on src/transformers/core_model_loading.py:969 · reply to [c3144702734] · 2026-04-28
it's used in two spots that's why I vectorized it here

## [c4186537762] tarekziade (COLLABORATOR) · review COMMENTED · 2026-04-28 (no text)

## [c3152175406] tarekziade (COLLABORATOR) · review_comment on src/transformers/core_model_loading.py:1367 · reply to [c3144712144] · 2026-04-28
agreed, I will extract it

## [c4186541261] tarekziade (COLLABORATOR) · review COMMENTED · 2026-04-28 (no text)

## [c4335266883] ydshieh (COLLABORATOR) · issue_comment · reply to [c4186541261] · 2026-04-28
Hi @tarekziade . Thanks for the work.

The origin of this optimization is for make some tests faster. The part of random 3x dtypes solves it works great.

For the changes in the core model loading file, so far the benefit is nit (in our tests, or if I load a single tiny model). However I am not saying it's not useful, but instead I am wondering:

  - if that change would show more benefit when we load a large (huge) model a single time?
  - and for small models (like in our CircleCI), the (absolute) amount of saving would become more visible if we load a model multiple times?
  - For the `deepcopy` part, is it really necessary?  Does `re.compile` already use some cache mechanism so the compile of the **same** pattern would not be slow (I am not sure and I might be wrong here).
   
Overall, I am fine as long as @Cyrilvallez is happy with the change. But it's nice if we can identify which part is really improving the loading, and limit the change to the smallest scope as possible.

## [c4335491945] tarekziade (COLLABORATOR) · issue_comment · reply to [c4335266883] · 2026-04-28
@ydshieh Thanks for the thoughtful questions, that’s very helpful.

You’re right that most of the measurable speedup comes from reducing the number of dtypes we iterate over.

The changes in the core loading path are more about a design concern that surfaced while doing this: the weight class used for conversion does some preparation work in its constructor that is rarely used, while it sits directly on the critical path (one instance per key).

So the goal there is mainly to keep that constructor as lightweight as possible e.g. avoid creating objects (like regexps) that we don’t actually use, and prevent this from growing over time.

In terms of impact:

- On large key counts (~18k), this shaves ~0.5s (what we see in tests today)
- For typical models (~500 keys), the gain is indeed negligible on a single load

I haven’t tested large models yet, but I’d expect the benefit to scale with the number of keys rather than model size itself

So I agree this is more of a “keep the hot path clean” improvement than a big performance win.

On `deepcopy`: it’s needed because `collected_tensors`, `layer_targets`, and `_was_used` accumulate state during loading, so we need a fresh instance per target. This is orthogonal to `re.compile`.

And you’re right about `re.compile` it’s cached in the stdlib itself, so repeated calls are not a concern here.

Happy to reduce the scope if we feel this is too much for the current benefit 👍

## [c4336179560] ydshieh (COLLABORATOR) · issue_comment · reply to [c4335491945] · 2026-04-28
OK, thanks for explaining. So `deepcopy` is needed (but not for the reason of `re.compile`).

 > Happy to reduce the scope if we feel this is too much for the current benefit 👍
 
 Since Cyril is convinced, so fine from my side after your above comment 👍

## [c4350248646] tarekziade (COLLABORATOR) · issue_comment · reply to [c4336179560] · 2026-04-30
@ydshieh or @Cyrilvallez anything else to tweak?

## [c3216243525] Cyrilvallez (MEMBER) · review_comment on src/transformers/core_model_loading.py:684 · 2026-05-11
We don't need this as we already have the `compiled_sources` property avoiding to run `re.compile` on `__init__`!

## [c3216255718] Cyrilvallez (MEMBER) · review_comment on src/transformers/core_model_loading.py:1095 · 2026-05-11
Would really much rather revert that as well, see previous comment - too hard to read for no gains IMO

## [c3216259562] Cyrilvallez (MEMBER) · review_comment on src/transformers/core_model_loading.py:1203 · 2026-05-11
Same here, let's revert

## [c4260631817] Cyrilvallez (MEMBER) · review COMMENTED · reply to [c4350248646] · 2026-05-11
Hey @tarekziade! As discussed before, let's revert everything except the lazy `re.compile` call! (I did not comment on every places where we need to revert for simplicity)

---
outcome: state=open, merged=False
