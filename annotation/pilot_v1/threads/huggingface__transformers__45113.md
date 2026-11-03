# #45113 [pull_request] Add GDS support for safetensors loading 
https://github.com/huggingface/transformers/pull/45113  ·  opened by cyyever on 2026-03-30

## [body] cyyever (CONTRIBUTOR) · opening post · 2026-03-30
# What does this PR do?
This PR adds GPU Direct Storage (GDS) support for safetensors model loading via `torch.cuda.gds.GdsFile`. GDS is disabled by default, `HF_ENABLE_GDS=1` env is used to enable it.
  ## Benchmark

  A100 PCIe 40GB, Samsung NVMe 3.5TB, GDS compat mode (no nvidia-fs) in `from_pretrained`:

  | Model | safe_open | GDS | Speedup |
  |-------|-----------|-----|---------|
  | Qwen/Qwen2.5-0.5B | 1.011s | 1.027s | 0.98x |
  | Qwen/Qwen2.5-7B | 3.133s | 1.725s | **1.82x** |


## Code Agent Policy

The Transformers repo is currently being overwhelmed by a large number of PRs and issue comments written by
code agents. We are currently bottlenecked by our ability to review and respond to them. As a result, 
**we ask that new users do not submit pure code agent PRs** at this time. 
You may use code agents in drafting or to help you diagnose issues. We'd also ask autonomous "OpenClaw"-like agents
not to open any PRs or issues for the moment.

PRs that appear to be fully agent-written will probably be closed without review, and we may block users who do this
repeatedly or maliciously. 

This is a rapidly-evolving situation that's causing significant shockwaves in the open-source community. As a result, 
this policy is likely to be updated regularly in the near future. For more information, please read [`CONTRIBUTING.md`](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).

- [x] I confirm that this is not a pure code agent PR.

## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [ ] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [ ] Did you write any new necessary tests?


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

## [c3016199600] McPatate (MEMBER) · review_comment on src/transformers/utils/gds_io.py:157 · 2026-03-31
Does this mean you pull the full tensor for each slice that is requested? What does your memory footprint on device look like once the model is loaded?

## [c4037725138] McPatate (MEMBER) · review COMMENTED · 2026-03-31
Hi thanks for the contribution! We're working on these kinds of topics over in the https://github.com/huggingface/safetensors repo directly, probably would be best to have GDS support directly in lib rather than transformers, cc @ArthurZucker.

Would be curious to see the following:
- larger model load
- distributed loading
- running iostat during load to measure throughput
- fio theoritical max throughput on your machine for reference
- warm vs cold cache test (`sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`)
- OS, I assume linux here?
- I assume each issued read is not sequential since there might be overlap between tensor ranges as you're pulling full tensor data for each slice, IIUC, wonder how much it impacts reads in the context of GDS though. Would be cool if you could test that but it'll require, I think, a more involved setup.

Not sure this will scale well with larger models, lmk if you want to test some more after my feedback. IMO not worth it as we're going to tackle this in the coming weeks in a "specialised manner" over in https://github.com/mfuntowicz/hmll, cc @mfuntowicz

## [c3016359850] cyyever (CONTRIBUTOR) · review_comment on src/transformers/utils/gds_io.py:157 · reply to [c3016199600] · 2026-03-31
Not sure your question. The purpose is to load tensors via GDS api, which works best with aligned file offsets.

## [c4037899691] cyyever (CONTRIBUTOR) · review COMMENTED · 2026-03-31 (no text)

## [c4163158521] cyyever (CONTRIBUTOR) · issue_comment · reply to [c4037725138] · 2026-03-31
@McPatate I saw a similar PR of safetensor repo but unfortunately it was denied. I prefer to apply GDS in transformers because it provides global overview of IO bottlenecks in large-scale LLM training/inference scenarios. 
This PR is also easier to review because PyTorch already provides GDS primitives, while we have to write rust code to encapsulate cuFile API in safe tensor.

For your concerns:
1. larger model load => provided.
2. distributed loading => we can load to different GPU cards on the same host via `mapping`
3. running iostat during load to measure throughput => I run benchmarks on a working server with other processes were training models, it was hard to isolate the workset.
4. fio theoritical max throughput on your machine for reference => it depends on the specification of tested SSDs, filesystem settings (we use EXT4 with noatime) and other kernel configs.
5. The reported figures are from warm cache test. Code cache tests have less sense on real servers where multiple users share a comment mount point to store downloaded LLMs.
6. GDS only supports Linux.
7. sequential IO access dominates the benchmark due to tensor disk layout, especially when loading checkpoints of LLMs.
Could you help verify my findings using the benchmark code?
```
#!/usr/bin/env python3
"""Benchmark GDS vs safe_open for from_pretrained loading.

Usage:
    HF_ENABLE_GDS=1 python benchmarks/benchmark_gds.py
[… 95 lines of code omitted]
```

## [c3016545294] McPatate (MEMBER) · review_comment on src/transformers/utils/gds_io.py:157 · reply to [c3016199600] · 2026-03-31
How do you guarantee file offsets are aligned here? safetensors files aren't written with that constraint in mind, you need to do some extra processing (we're thinking of supporting writing aligned offsets, but it's tricky wrt backwards compatibility).

What I'm asking, is that from your implementation, it seems you're loading the full tensor `self._name` on each call to `GdsSlice.__getitem__`. That is why I asked what the memory footprint (total used memory on device) looks like. If you can run nvidia-smi after loading the model, that'd be a good test to see if that happens.

## [c4038121046] McPatate (MEMBER) · review COMMENTED · 2026-03-31 (no text)

## [c4163429913] McPatate (MEMBER) · issue_comment · reply to [c4163158521] · 2026-03-31
> larger model load => provided.
> distributed loading => we can load to different GPU cards on the same host via mapping

For larger I meant, larger than qwen 7b! But I assume we're entering in distributed territory after that size, so consider these two points as the same. I would appreciate a benchmark to see how you're impl is performing in that scenario.

> I run benchmarks on a working server with other processes were training models, it was hard to isolate the workset.

No way to run your code on an isolated machine?

> fio theoritical max throughput on your machine for reference => it depends on the specification of tested SSDs, filesystem settings (we use EXT4 with noatime) and other kernel configs.

I was asking if you could run [`fio`](https://github.com/axboe/fio) to have a reference point of "theoritical disk throughput speed" vs "actual throughput of GDS loading".

> The reported figures are from warm cache test. Code cache tests have less sense on real servers where multiple users share a comment mount point to store downloaded LLMs.

There are scenarios where cold cache runs make sense (e.g. loading a model after a restart with files already present on disk).

> sequential IO access dominates the benchmark due to tensor disk layout, especially when loading checkpoints of LLMs.

I'm not convinced you are issuing reads sequentially, as for each slice you read the full tensor, again IIUC.

---
outcome: state=open, merged=False
