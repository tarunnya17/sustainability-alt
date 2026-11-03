# #42435 [pull_request] :rotating_light: [`FA4`] Initial support
https://github.com/huggingface/transformers/pull/42435  ·  opened by vasqu on 2025-11-26

## [body] vasqu (COLLABORATOR) · opening post · 2025-11-26
## :rotating_light: Breaking change
- FA2 is only supported from version `2.3.3` and on

This is due to the fact that this is older than 2+ years (we deprecate torch in 2 year cycles for example) as well as it giving fairly high maintenance burden.

## Related issues and PRs
Fixes #42405
Closes #42404 as it has a lot of unnecessary logic and tests alongside it

## Testing
Sanity Testing
```bash
RUN_SLOW=1 pytest tests/models/llama/test_modeling_llama.py -k flash
```
- [x] Passes all flash attention 4 tests

First quick numbers (hopper)
[… 15 lines of code omitted]
```

## [c2566439500] vasqu (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:1745 · 2025-11-26
should be not in

## [c3512690922] vasqu (COLLABORATOR) · review COMMENTED · 2025-11-26 (no text)

## [c3589997869] vasqu (COLLABORATOR) · issue_comment · 2025-11-28
FA4 support cc @stas00 if you wanna play around with this PR. It's pretty much ready, just not convinced by the numbers but I also don't have quick access to a blackwell GPU (at least today :D)

## [c3523189964] sfc-gh-sbekman (CONTRIBUTOR) · review COMMENTED · 2025-12-01
Thank you for working on this, Anton. Going to try it out.

To make it easier to try your PR please add to the OP how to install FA4, since it's non-trivial to find.
```
git clone https://github.com/Dao-AILab/flash-attention/
cd flash-attention
cd flash_attn/cute
uv build --wheel . -v --no-build-isolation --out-dir flash-attention/wheels
uv pip install flash-attention/wheels/flash_attn_cute*.whl --prerelease=allow
```

## [c3594675200] sfc-gh-sbekman (CONTRIBUTOR) · issue_comment · 2025-12-01
OK, gave it a test ride using your PR and the above comment's install of FA4 on B200.

I did a quick test with Llama-8b and the integration worked smoothly but the tflops performance is much worse than FA2 - 2-5x slower. Not sure if it's an issue with integration or the FA4 code or the pytorch version - most likely the upstream since the integration is just a wrapper

I tried pt-2.9.1-cu130 and pt-nightly-cu130 - same outcome

edit: `sdpa` in pt-nightly, which supposedly backported FA4, is about 3x faster than FA4 on its own using the same llama-8b - since they both should be using the same code, perhaps there is an issue with the integration?

## [c3598468299] vasqu (COLLABORATOR) · issue_comment · reply to [c3523189964] · 2025-12-01
Thanks for checking this out and all the pointers @sfc-gh-sbekman :heart: 

> To make it easier to try your PR please add to the OP how to install FA4, since it's non-trivial to find.

For sure, I'll add some docs for FA4 before release. Maybe also FA3 in a different PR.

> I did a quick test with Llama-8b and the integration worked smoothly but the tflops performance is much worse than FA2 - 2-5x slower. Not sure if it's an issue with integration or the FA4 code or the pytorch version - most likely the upstream since the integration is just a wrapper

Shoot, so it wasn't an GPU arch issue... This is weird

> `sdpa` in pt-nightly, which supposedly backported FA4, is about 3x faster than FA4 on its own using the same llama-8b - since they both should be using the same code, perhaps there is an issue with the integration?

Do you have a code snippet? There are so many edge cases with sdpa that it maybe is not even entering the FA backend path? Could be quickly checked by restricting the backend usage on SDPA with their context manager 
```python
with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
    pass  # do your thing here
```
I'm also unsure how FA4 is integrated in SDPA? Do we need to use a flag there? I remember that cudnn backend needed special treatment

## [c3599780455] stas00 (CONTRIBUTOR) · issue_comment · reply to [c3598468299] · 2025-12-02
> Shoot, so it wasn't an GPU arch issue... This is weird

Did you mean that you too have observed a similar slowdown?

> Do you have a code snippet? 

I was just using https://github.com/snowflakedb/ArcticTraining/ normal SFT training recipe where I tried different attention mechanisms. Just normal fwd/bwd/step - nothing special added.

> I'm also unsure how FA4 is integrated in SDPA? Do we need to use a flag there? I remember that cudnn backend needed special treatment

They copied/adapted the FA4 kernels see: https://github.com/huggingface/transformers/pull/42435 - you'd need pt nightly for that to work.

## [c3602736394] vasqu (COLLABORATOR) · issue_comment · reply to [c3599780455] · 2025-12-02
> Did you mean that you too have observed a similar slowdown?

I just did some quick numbers on inference, see the test I noted down in the PR description. I used an H100 there and as you can see it's slower (not on the same magnitude as in your samples - would say it's a mixture of model size / context size)

> I was just using https://github.com/snowflakedb/ArcticTraining/ normal SFT training recipe where I tried different attention mechanisms. Just normal fwd/bwd/step - nothing special added.

Gotcha, I will try to separate our implementation from the base fn of the FA library to see if our wrappers cause this or maybe some perf regression happened sometime else.

> They copied/adapted the FA4 kernels see: https://github.com/huggingface/transformers/pull/42435 - you'd need pt nightly for that to work.

Wrong link? My assumption / hunch was that maybe
1. Even nightly might not use FA4 per default and sticks to FA2 per default, i.e. might need some extra flags to enable that specific backend. But that's just my feeling, need to look into it.
2. Our implementations has some issues where attention masks are created even when it is not needed (full (causal) attention). If a mask is passed to SDPA, then the FA backend can never be entered per their restrictions. So I thought, maybe, we have this case (SDPA with xformers faster than FA4 - xformers is not so bad on short contexts <2k).

## [c3603295951] stas00 (CONTRIBUTOR) · issue_comment · 2025-12-02
My apologies, here is the correct link https://github.com/pytorch/pytorch/pull/167348

## [c3648377543] sfc-gh-sbekman (CONTRIBUTOR) · issue_comment · 2025-12-12
Some useful updates from talking to Tri:

- FA4 is supposed to replace FA2 and FA3 and would work with A/H/B archs
- FA4 varlen support is planned in a few weeks time

## [c3850433583] edixiong (NONE) · issue_comment · reply to [c3648377543] · 2026-02-05
Hi @sfc-gh-sbekman @vasqu Thanks for contributing! I think the varlen is supported. Do you mind testing FA4 again?

## [c3855778177] sfc-gh-sbekman (CONTRIBUTOR) · issue_comment · 2026-02-05
just tried with the `main` version of fa - `bwd` doesn't work:

```
[rank3]:   File "/home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/torch/utils/checkpoint.py", line 325, in backward
[rank3]:     torch.autograd.backward(outputs_with_grad, args_with_grad)
[rank3]:   File "/home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/torch/autograd/__init__.py", line 364, in backward
[rank3]:     _engine_run_backward(
[… 44 lines of code omitted]

## [c3954952028] sfc-gh-sbekman (CONTRIBUTOR) · issue_comment · 2026-02-24
Tried again with today's FA4 and a new error is reported on H200 w/ varlen.
```
[rank6]:   File "/home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/torch/autograd/function.py", line 311, in apply
[rank6]:     return user_fn(self, *args)
[rank6]:            ^^^^^^^^^^^^^^^^^^^^
[rank6]:   File "/home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/flash_attn/cute/interface.py", line 1395, in backward
[rank6]:     dq, dk, dv = _flash_attn_bwd(
[rank6]:                  ^^^^^^^^^^^^^^^^
[rank6]:   File "/home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/flash_attn/cute/interface.py", line 617, in _flash_attn_bwd
[rank6]:     assert not is_varlen, "varlen backward is not yet supported on sm90"
[rank6]:            ^^^^^^^^^^^^^
[rank6]: AssertionError: varlen backward is not yet supported on sm90
```

non-varlen works, but haven't measured the performance comparison

## [c3963774435] edixiong (NONE) · issue_comment · 2026-02-26
Thanks for checking!

## [c3995423295] edixiong (NONE) · issue_comment · 2026-03-04
Seems like FA4 will be published soon: https://github.com/Dao-AILab/flash-attention/pull/2282
I will ping back later when they release the package.

## [c3997187951] vasqu (COLLABORATOR) · issue_comment · 2026-03-04
Thanks for keeping us in the loop :saluting_face: will check it out then when we got the release

## [c4006560630] stas00 (CONTRIBUTOR) · issue_comment · 2026-03-05
Feels like the announcement was premature, I can't get it to even load:

```
$ uv pip install fa4
Using Python 3.12.12 environment at: /home/yak/miniconda3/envs/dev
Resolved 37 packages in 56ms
Prepared 1 package in 52ms
[… 16 lines of code omitted]

## [c4006735786] stas00 (CONTRIBUTOR) · issue_comment · 2026-03-05
OK, @mayank31398 helped me to find `cutlass.cute` - NVIDIA hid those packages from python - no clue why - instead of putting them into `site-packages/cutlass` it puts them under `site-packages/nvidia_cutlass_dsl/python_packages/cutlass` ?!?!?!

So I had to do:
```
export PYTHONPATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/
```
to load `cutlass.cute`

## [c4006739632] stas00 (CONTRIBUTOR) · issue_comment · reply to [c3997187951] · 2026-03-05
ok, so now the integration code fails to load fa4 so back to you @vasqu:

```
[rank0]:   File "/code/users/stas/github/transformers/src/transformers/modeling_utils.py", line 1255, in __init__
[rank0]:     self.config._attn_implementation_internal = self._check_and_adjust_attn_implementation(
[rank0]:                                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/code/users/stas/github/transformers/src/transformers/modeling_utils.py", line 1838, in _check_and_adjust_attn_implementation
[… 12 lines of code omitted]

## [c2891516371] stas00 (CONTRIBUTOR) · review_comment on src/transformers/modeling_utils.py:1614 · 2026-03-05
```suggestion
            if importlib.util.find_spec("flash_attn.cute") is None:
                raise ImportError(f"{preface} the package flash_attn (cute) seems to be not installed.")
```

## [c2891517720] stas00 (CONTRIBUTOR) · review_comment on src/transformers/modeling_utils.py:1623 · 2026-03-05
```suggestion
```

## [c3898715916] stas00 (CONTRIBUTOR) · review COMMENTED · 2026-03-05
these are needed to make it work with the current fa4 as of this morning

## [c4007245536] vasqu (COLLABORATOR) · issue_comment · 2026-03-05
Thanks a lot, will try to take a stab tomorrow :saluting_face:

## [c4008219660] stas00 (CONTRIBUTOR) · issue_comment · 2026-03-05
FWIW, I run a benchmark on a B300 comparing FA2 to FA4 and I see only a ~3% improvement end-to-end, which makes no sense - it should be much higher. 

Either something isn't right, or FA4 isn't quite there yet, I'm not sure.

This use case uses varlen.

edit: my colleague clocked an almost 2x speed up end-to-end with FA4 over FA2 on B200 - different workload. So that's a great sign. Perhaps his wasn't varlen.

## [c4013315104] vasqu (COLLABORATOR) · issue_comment · 2026-03-06
Ok, so I did not go as far as I wanted to because I had torch 2.10 issues in regards to cublass (for anyone interested, pin `nvidia-cublas-cu12==12.9.1.4` re [issue](https://github.com/pytorch/pytorch/issues/174949))

I got FA4 working with my commit 776a1af358ce1d187e927caee228c7a16f393d8c and checked the parity test where I fell into a rabbit hole:
- #41427 changed the loading logic and truly enabled properly loading different FA versions in the same process, i.e. the same FA was run twice
- Fixed loading already shows diffs within FA3 and FA2 

My initial numbers (still hopper, will check with blackwell later on if I can) show that it's around the same time as FA3. Iirc, this should be varlen as well.

Edit: another note `head dim == 256` does not work and results in a tvm-ffi error which is not so obvious

## [c2897282919] sfc-gh-sbekman (CONTRIBUTOR) · review_comment on src/transformers/modeling_utils.py:1619 · 2026-03-06
I don't think this is correct, as it'll find FA2 and succeed, no?

was there something wrong with the fixes I proposed yesterday? You ignored those and wrote a different version w/o saying why they didn't work for you

## [c3905372491] sfc-gh-sbekman (CONTRIBUTOR) · review COMMENTED · 2026-03-06 (no text)

## [c2897340553] vasqu (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:1619 · reply to [c2897282919] · 2026-03-06
We shouldn't have entered that path as the import shouldn't fail so I fixed the import utils first. This here only happens if we know that fa4 is requested so it doesn't really matter in relation to fa2. The or is to first check the base package and then the underlying fa4 cute

## [c3905438241] vasqu (COLLABORATOR) · review COMMENTED · 2026-03-06 (no text)

## [c2897451079] sfc-gh-sbekman (CONTRIBUTOR) · review_comment on src/transformers/modeling_utils.py:1619 · reply to [c2897282919] · 2026-03-06
I rewrote my comments - as they made no sense. All is good.

## [c3905563839] sfc-gh-sbekman (CONTRIBUTOR) · review COMMENTED · 2026-03-06 (no text)

## [c2897559573] vasqu (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:1619 · reply to [c2897282919] · 2026-03-06
I should have communicated the changes a bit more as well, no worries. Will check a bit more tomorrow probably

## [c3905691483] vasqu (COLLABORATOR) · review COMMENTED · 2026-03-06 (no text)

## [c4014865689] vasqu (COLLABORATOR) · issue_comment · reply to [c3905563839] · 2026-03-07
@sfc-gh-sbekman So it was pretty much ready already, I added some small details especially on the parity test. These deviation are somewhat expected and was already happening on FA3 (but went undetected for quite a while).

I added support for attention sinks so people may run GPT Oss which produces fairly sane output, similar to FA3 (vllm version).

I will revisit with some blackwell gpu to properly benchmark, hopper just really ain't the target device group (at least so far). Probably also adding some docs but thankfully the install is easy

## [c4015286615] jiosephlee (CONTRIBUTOR) · issue_comment · reply to [c4014865689] · 2026-03-07
> I added support for attention sinks so people may run GPT Oss which produces fairly sane output, similar to FA3 (vllm version).

Butting in with a naive question but does FA4 include bwd kernels? afaik, bwd kernels for sink attention are not supported in the main release of FA2/3

## [c4015316674] vasqu (COLLABORATOR) · issue_comment · 2026-03-07
Iirc, attention sinks are only in form of PRs available for fa2/3. On a quick glance, fa4 only support the fwd for the sinks, the bwd is ignored.

You could quickly check it by just creating random tensors, run some arbitrary loss and check the grads of your tensors. Can do so tomorrow or so if no one has checked until then

The more general stuff has a bwd version

## [c2905324380] Cyrilvallez (MEMBER) · review_comment on src/transformers/modeling_flash_attention_utils.py:654 · 2026-03-09
Why do we also check `is_tracing` here? I believe it would make it fail for previous FA version and no compile

## [c2905341368] Cyrilvallez (MEMBER) · review_comment on src/transformers/modeling_utils.py:1617 · 2026-03-09
Any way we can take the opportunity to make this slightly simpler instead of more or less duplicating 3 times for each fa?

## [c2905344011] Cyrilvallez (MEMBER) · review_comment on src/transformers/modeling_utils.py:1883 · 2026-03-09
SHould'nt we have a new path for fa4 here?

## [c3915297826] Cyrilvallez (MEMBER) · review COMMENTED · 2026-03-09
Nice! Just a few comments!

## [c2906975588] vasqu (COLLABORATOR) · review_comment on src/transformers/modeling_flash_attention_utils.py:654 · reply to [c2905324380] · 2026-03-09
This doesn't fail for any FA version. In non-compile environments, FA can handle this and we avoid device syncs with this. Only compile fails, because the fake ops and the underlying signature forces base int which dynamo cannot reason about if it is a scalar int tensor instead.

Initially, this change was specifically for FA4 which did not have max_seqlen_q/k but at some point it was added back with the same signature issue. This way we boost non-compile forward at least

## [c3917146587] vasqu (COLLABORATOR) · review COMMENTED · 2026-03-09 (no text)

## [c2907005424] vasqu (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:1617 · reply to [c2905341368] · 2026-03-09
Not really, I already refactored as much as I could into https://github.com/vasqu/transformers/blob/65912b2c02fe2f0653eb9710dc32b84aa209ae56/src/transformers/modeling_utils.py#L1635

There are subtle differences that cannot be ignored and these are the `_flash_attn_x_import_error` ones
- FA2 has very specific versioning stuff along kernels support for more devices like xpu
- FA3 has only support for ampere and hopper
- FA4 has only support for hopper and blackwell

## [c3917180459] vasqu (COLLABORATOR) · review COMMENTED · 2026-03-09 (no text)

## [c2907019690] vasqu (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:1883 · reply to [c2905344011] · 2026-03-09
There is no FA4 hf kernel yet. It needs support for cute and tvm ffi first. Iirc, cute is done but tvm still needs work

## [c3917196561] vasqu (COLLABORATOR) · review COMMENTED · 2026-03-09 (no text)

## [c2907021086] vasqu (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:1617 · reply to [c2905341368] · 2026-03-09
Let me check if I can simplify a bit more tho

## [c3917198255] vasqu (COLLABORATOR) · review COMMENTED · 2026-03-09 (no text)

## [c2907192300] sfc-gh-sbekman (CONTRIBUTOR) · review_comment on src/transformers/modeling_utils.py:1617 · reply to [c2905341368] · 2026-03-09
If I remember correctly the plan is to make fa4 work for any gpu arch, so there is only one package to maintain. But perhaps this has changed recently - or more likely the older archs are just not a priority.

e.g. FA4 on hopper still doesn't support varlen

## [c3917388242] sfc-gh-sbekman (CONTRIBUTOR) · review COMMENTED · 2026-03-09 (no text)

## [c2907200315] vasqu (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:1617 · reply to [c2905341368] · 2026-03-09
That's a good point, I need to write some docs for compatibility tbh. It may not be super obvious atm

## [c3917396294] vasqu (COLLABORATOR) · review COMMENTED · 2026-03-09 (no text)

## [c2907733332] vasqu (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:1617 · reply to [c2905341368] · 2026-03-09
Ok I refactored a bit in d07749f4c6c11d0d4726d5d6a9ca272e76424d34

It will force a lower bound on FA2 tho. Imo, it doesn't make sense to support these very old versions which atp are older than 2 years as it introduces additional checks. Wdyt about this refactored version

## [c3917991358] vasqu (COLLABORATOR) · review COMMENTED · 2026-03-09 (no text)

## [c2926186270] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:1678 · 2026-03-12
we could bring this outside this scope!

## [c2926187041] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/modeling_utils.py:1678 · reply to [c2926186270] · 2026-03-12
mayne in fa2 utils

## [c2926190901] ArthurZucker (COLLABORATOR) · review_comment on tests/generation/test_flash_attention_parity.py:87 · 2026-03-12
```suggestion
    @pytest.mark.all_flash_attn_tes
```
or 2 is the only not supported ahha

## [c3938397584] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-03-12
Ty!

## [c4143234736] imstevenpmwork (CONTRIBUTOR) · issue_comment · reply to [c3917991358] · 2026-03-27
Hey @vasqu  👋 Just a quick heads-up that the API changes in this PR (shipped in the 5.4.0 release) introduced a breaking change for us.

Over at [lerobot](https://github.com/huggingface/lerobot), we were relying on `is_flash_attn_greater_or_equal_2_10`, which looks like it got removed during the `import_utils.py` refactor. Just wanted to flag this to save some debugging time in case anyone else is suddenly staring at a [red CI](https://github.com/huggingface/lerobot/actions/runs/23629232685)!

Cheers!

Second thought:
Maybe we can add it into the Release Notes. Users should move from `is_flash_attn_greater_or_equal_2_10` to `is_flash_attn_greater_or_equal("2_10")`

## [c4143266313] ArthurZucker (COLLABORATOR) · issue_comment · reply to [c3917991358] · 2026-03-27
@vasqu let's add BC unless we had a deprecation cycle

## [c4143308703] vasqu (COLLABORATOR) · issue_comment · 2026-03-27
Check out #45061, mb

---
outcome: state=closed, merged=True
