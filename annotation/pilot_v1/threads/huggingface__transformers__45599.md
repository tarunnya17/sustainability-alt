# #45599 [pull_request] qa: more lazy loading
https://github.com/huggingface/transformers/pull/45599  ·  opened by tarekziade on 2026-04-23

## [body] tarekziade (COLLABORATOR) · opening post · 2026-04-23
# What does this PR do?

Per https://github.com/huggingface/transformers/issues/44273 there are a few spots where we can do more lazy loading to speed up `import transformers` without complexifying the code too much

## [c4303203897] tarekziade (COLLABORATOR) · issue_comment · 2026-04-23
The public API of `transformers.utils` forces me to be very agressive on `hub.py` -- I am not sure if it's worth the added complexity.  

The current patch reduces to 270 imports / and 0.5 seconds on my M5 

@LysandreJik WDYT? 

I feel like doing this can be quite fragile/painful to maintain..

## [c4325424425] LysandreJik (MEMBER) · issue_comment · 2026-04-27
@albertvillanova can you confirm whether installing this branch results in dramatic speedups on your machine? not sure we want to guard numpy imports like this either

## [c4332324489] albertvillanova (MEMBER) · issue_comment · reply to [c4325424425] · 2026-04-28
Thanks for the ping, @LysandreJik. I'm checking it right now.

## [c4332365066] albertvillanova (MEMBER) · issue_comment · 2026-04-28
Yes, I confirm I see a gain in speed:
```bash
time python -c "import transformers"

real	0m2.325s
user	0m0.518s
sys	0m0.150s
[… 9 lines of code omitted]
```

## [c4332420289] albertvillanova (MEMBER) · issue_comment · 2026-04-28
On another machine:
- Now:
```bash
time python -c "import transformers"

real	0m1.330s
user	0m1.230s
sys	0m0.100s
```
- Before:
```bash
time python -c "import transformers"

real	0m1.819s
user	0m3.204s
sys	0m0.176s
```

## [c4333228070] tarekziade (COLLABORATOR) · issue_comment · reply to [c4332420289] · 2026-04-28
@albertvillanova thanks for the tests!

Could you share a bit more about the use case where startup time is critical here?
In some CLI / serving setups, transformers can be lazy-loaded, so it’d be great to understand how this fits your scenario.

## [c4333764584] albertvillanova (MEMBER) · issue_comment · 2026-04-28
Lazy loading in transformers would be an ideal solution, but I think it is currently broken, e.g. importing anything from `transformers.utils`.

Indeed, my use case is improving the startup latency of the `trl` CLI. It imports transformers and this was a significant contributor to the overall startup time. 

As a result, we had to introduce some workarounds in trl to avoid importing some transformers functions at CLI startup. See, e.g.: 
- https://github.com/huggingface/trl/pull/5129
- https://github.com/huggingface/trl/pull/5208
- https://github.com/huggingface/trl/pull/5210
- https://github.com/huggingface/trl/pull/5257

If lazy loading in transformers is fixed and becomes more reliable, we could eventually revert these changes and simplify the current setup.

## [c4335590053] tarekziade (COLLABORATOR) · issue_comment · 2026-04-28
Transformers is a large codebase, so importing some modules will inevitably pull in dependencies. Lazy loading can help, but it also adds complexity because we want to keep APIs backward compatible (like what's exposed in `transformers.utils`), so there’s a tradeoff between startup speed and maintainability. And in practice, once you really use Transformers, you usually end up needing things like NumPy or Torch anyway.

For the TRL CLI, I totally understand the goal. That’s exactly why I worked on making startup faster and added a regression test to keep it that way. But I think we’re reaching a point where pushing this further makes some modules harder to maintain, so I’d rather defer more of that laziness to downstream projects, similar to what you did.

That said, I wouldn’t say Transformers lazy loading is “broken”; that feels a bit strong to me. It may be that some imports still have more overhead than expected.

What would be an acceptable import-time overhead for your use case?

## [c4706287126] tarekziade (COLLABORATOR) · issue_comment · 2026-06-15
closing, rationale: https://github.com/huggingface/transformers/issues/44273#issuecomment-4706182929

---
outcome: state=closed, merged=False
