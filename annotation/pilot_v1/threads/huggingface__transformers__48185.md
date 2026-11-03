# #48185 [pull_request] Warn once when a hub-kernel function falls back to its reference PyTorch path
https://github.com/huggingface/transformers/pull/48185  ·  opened by qgallouedec on 2026-08-21

## [body] qgallouedec (MEMBER) · opening post · 2026-08-21
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=48185) [[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=48185)

`use_kernel_func_from_hub_with_fallback` resolves, in order, a hub kernel, the original package, and finally the pure-PyTorch reference. The third case is completely silent. For `chunk_gated_delta_rule` the reference is **>20× slower** than the kernel, so a user training Qwen3-Next / Qwen3.5 / Qwen3.6-MoE / OLMo-hybrid can lose half their throughput with nothing in the logs. Emit a `warning_once` on first call naming the distribution to install.

```
`chunk_gated_delta_rule` is falling back to its reference PyTorch implementation because
`flash-linear-attention` is not installed. This is correct but much slower; install
`flash-linear-attention` for the optimized kernel.
```
[… 35 lines of code omitted]

## [c3894561950] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/hub_kernels.py:866 · 2026-08-31
You definitely need to guard this for when this is traced / compiled (see CI failures) as the logger is untracable

## [c3894567739] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/hub_kernels.py:823 · 2026-08-31
Imo, might be a bit overkill. I'd honestly just like to avoid another constant and users should understand fla (or we update afterwards to be more precise)

## [c3894585777] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/hub_kernels.py:874 · 2026-08-31
What makes me hesitant is the fact that some users may intentionally want to use the torch fallback no?

## [c5066610276] vasqu (COLLABORATOR) · review COMMENTED · 2026-08-31
I'm unsure about this path because some users may use the torch path intentionally, no?

The PR description has 2 very important points tho
1. Qwen 3.5 Moe has a missing decorator, will submit a PR in a second -> #48436
2. The bwd for fla in kernels doesn't work? Would be really great to have a small repro because this is super important to fix imo. We want to move people to use hf kernels tbh and the bwd should work

## [c5478700578] vasqu (COLLABORATOR) · issue_comment · 2026-08-31
Imo, maybe a logger info could be used instead of a warning?

## [c5487570137] qgallouedec (MEMBER) · issue_comment · 2026-09-01
thks for the review and for #48436.

On the fla backward:
it's real and it's a packaging bug rather than an fla bug:
The vendored source in kernels-community has 10 relative imports at the wrong depth, the gated_delta_rule backward hits three of them. Filed it as huggingface/kernels-community#1128 with a repro, fix in huggingface/kernels-community#1129. With that applied the backward runs and the gradients match fla's own naive reference.

(Worth knowing it only fires when tilelang is installed, otherwise the backward falls through to triton and works. So the kernel is fine for most people today, and broken for exactly the ones who installed tilelang to go faster.)

On warning vs info, I'd keep the warning:
4.43x end to end on Qwen3.6-35B-A3B, >20x on the layer itself. That's not a mild difference.
So the tradeoff is being noisy for the few people who take the torch path on purpose, against staying silent for the many who land there without ever knowing.
And info isn't a middle ground: default verbosity is WARNING, so `info_once` is as silent as today. Side note, I've been looking at how coding agents read logs and they rarely even consider warnings, so I'll let you imagine what happens to a silent info.
If you disagree, let me know, I'm happy to use info as suggested though.

Also added `causal_conv1d` -> `causal-conv1d` to the map, it was missing and the message would have printed the import name.

## [c3900123221] qgallouedec (MEMBER) · review_comment on src/transformers/integrations/hub_kernels.py:866 · reply to [c3894561950] · 2026-09-01
Good catch, should be fixed by 31254aeb00

## [c5073285405] qgallouedec (MEMBER) · review COMMENTED · 2026-09-01 (no text)

## [c3900130870] qgallouedec (MEMBER) · review_comment on src/transformers/integrations/hub_kernels.py:874 · reply to [c3894585777] · 2026-09-01
answer in https://github.com/huggingface/transformers/pull/48185#issuecomment-5487570137

## [c5073293502] qgallouedec (MEMBER) · review COMMENTED · 2026-09-01 (no text)

## [c5493373727] vasqu (COLLABORATOR) · issue_comment · reply to [c5487570137] · 2026-09-01
> On the fla backward

Thanks checking! Very much appreciate it :heart: 

> (Worth knowing it only fires when tilelang is installed, otherwise the backward falls through to triton and works. So the kernel is fine for most people today, and broken for exactly the ones who installed tilelang to go faster.)

Honestly that is to be expected tho because kernels is build around triton atm; any tilelang side effect is unintentional atp

> On warning vs info, I'd keep the warning

Gotcha, I don't have the strongest opinion here so I trust you as a user more than me (who's too used to his own code). Lemme just give a sanity review but should be mergable :hugs:

## [c3903735243] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/hub_kernels.py:861 · 2026-09-01
```suggestion
            if not is_new_implementation and not is_torchdynamo_compiling():
```

## [c3903741085] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/hub_kernels.py:866 · 2026-09-01
If we keep an alias, I don't mind the static constant but let's move it close to where we keep the internal mapping as well

## [c3903747758] vasqu (COLLABORATOR) · review_comment on src/transformers/integrations/hub_kernels.py:865 · 2026-09-01
Let's keep the comment general, e.g. "Warn on torch fallback as the runtimes are significantly slower (formerly also called slow path), e.g. see #48185"

## [c5077565238] vasqu (COLLABORATOR) · review APPROVED · 2026-09-01
Some small comments but overall good 🤗 

Let's fix the other issues as well
- #48436 for qwen moe
- https://github.com/huggingface/kernels-community/pull/1129 checking in a second

## [c3920009790] qgallouedec (MEMBER) · review_comment on src/transformers/integrations/hub_kernels.py:866 · reply to [c3903741085] · 2026-09-03
Moved next to `_KERNELS_INTERNAL_PATH_MAPPINGS` as `_PACKAGE_TO_DISTRIBUTION`.

## [c5096659648] qgallouedec (MEMBER) · review COMMENTED · 2026-09-03 (no text)

## [c3920011400] qgallouedec (MEMBER) · review_comment on src/transformers/integrations/hub_kernels.py:865 · reply to [c3903747758] · 2026-09-03
Reworded

## [c5096661245] qgallouedec (MEMBER) · review COMMENTED · 2026-09-03 (no text)

---
outcome: state=closed, merged=True
