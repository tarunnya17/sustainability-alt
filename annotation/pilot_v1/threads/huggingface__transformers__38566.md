# #38566 [pull_request] Fix `FalconMambaIntegrationTests`
https://github.com/huggingface/transformers/pull/38566  ·  opened by ydshieh on 2025-06-03

## [body] ydshieh (COLLABORATOR) · opening post · 2025-06-03
# What does this PR do?

All pass on A10. 

One fails on T4

> tests/models/falcon_mamba/test_modeling_falcon_mamba.py::FalconMambaIntegrationTests::test_generation_fp16

(# On T4, get `NotImplementedError: Cannot copy out of meta tensor; no data!`)

I decided to just skip it on T4

## [c2124441021] ydshieh (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:464 · 2025-06-03
Regarding bfloat16 --> float16, see

https://github.com/huggingface/transformers/pull/38384#pullrequestreview-2868942294

## [c2893367713] ydshieh (COLLABORATOR) · review COMMENTED · 2025-06-03 (no text)

## [c2124443971] ydshieh (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:553 · 2025-06-03
@gante I guess it's normal that, with `inputs_embeds`, we don't have the prompt included, right?

## [c2893371854] ydshieh (COLLABORATOR) · review COMMENTED · 2025-06-03 (no text)

## [c2125987388] vasqu (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:464 · reply to [c2124441021] · 2025-06-04
I don't think the reasoning matches here tho :D Mamba is a different architecture and doesn't have eager/sdpa. However, I know that the original author of mamba also suggested inference in fp16 (training in mixed bf16), so I'm still fine with this change.

## [c2125988650] vasqu (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:461 · 2025-06-04
Might be too little memory or some cpu offloading issues?

## [c2125996409] vasqu (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:553 · reply to [c2124443971] · 2025-06-04
Imo, that doesn't seem correct to me. It would be weird to expect different behaviour here since we generate from the same "prompt". That might be a regression somewhere.

## [c2895763043] vasqu (COLLABORATOR) · review COMMENTED · 2025-06-04
I would wait for Joao on the input embeds side because that seems really weird to me. Otherwise, mostly nits.

## [c2126062675] ydshieh (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:461 · reply to [c2125988650] · 2025-06-04
Yeah I guess too. But if I remove `device_map="auto"` and add `.to(torch_device)`

> model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16).to(torch_device)

the generation could run without OOM.

So it's a bit strange to me why `auto` would cause problem.

## [c2895877618] ydshieh (COLLABORATOR) · review COMMENTED · 2025-06-04 (no text)

## [c2126101165] ydshieh (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:464 · reply to [c2124441021] · 2025-06-04
Here the reason I change to fp16 is to

- try to keep the results on T4/A10 closer for all integration tests in this file.
- (and sometimes, looks a better outputs)

None of them are really required however. But fp16 is indeed better at numerical precision.


For example, `test_batched_generation` with bf16

on A10
```
[
"Hello today I'm going to show you how to make a 3D model of a house.\n",
'Hello my name is Younes and today I will be talking about the topic of “The importance of the internet in our life”.\n'
]
```
[… 24 lines of code omitted]
```

## [c2895938217] ydshieh (COLLABORATOR) · review COMMENTED · 2025-06-04 (no text)

## [c2126104966] ydshieh (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:553 · reply to [c2124443971] · 2025-06-04
I am not expert (that is why I ping @gante ), but I kind think it's normal. For the prompt part, we only pass embedding not the token ids. And for those part, we can't recover the token ids from the embedding. That is why it only gives the part that are `generated`.

## [c2895943849] ydshieh (COLLABORATOR) · review COMMENTED · 2025-06-04 (no text)

## [c2126154301] vasqu (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:461 · reply to [c2125988650] · 2025-06-04
Maybe someone to cc here? It's not a deal breaker here but might be interesting to investigate if someone wants to / has the time.

## [c2896019307] vasqu (COLLABORATOR) · review COMMENTED · 2025-06-04 (no text)

## [c2126161703] vasqu (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:553 · reply to [c2124443971] · 2025-06-04
It would be nice to know if the original commit worked on this test. Or at least if on input embeds the same issues persisted (no prompt).

Both behaviours are justified imo.

## [c2896030793] vasqu (COLLABORATOR) · review COMMENTED · 2025-06-04 (no text)

## [c2126163747] vasqu (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:464 · reply to [c2124441021] · 2025-06-04
Wow, thanks for the examples and context :D That's indeed way better.

## [c2896034220] vasqu (COLLABORATOR) · review COMMENTED · 2025-06-04 (no text)

## [c2126200704] ydshieh (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:553 · reply to [c2124443971] · 2025-06-04
It's failing when the test is written 😢 

(That's why I always it's important to use `run-slow` on PR )

## [c2896092865] ydshieh (COLLABORATOR) · review COMMENTED · 2025-06-04 (no text)

## [c2126203338] ydshieh (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:553 · reply to [c2124443971] · 2025-06-04
proof (on 2024/08/23 -- the test is added on 2024/08/12, but we moved the CI report channels)

https://huggingface.slack.com/archives/C06LR9PQA00/p1724428864646269?thread_ts=1724428831.756269&cid=C06LR9PQA00

## [c2896097123] ydshieh (COLLABORATOR) · review COMMENTED · 2025-06-04 (no text)

## [c2126214677] ydshieh (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:461 · reply to [c2125988650] · 2025-06-04
I will open an issue and ping

## [c2896114137] ydshieh (COLLABORATOR) · review COMMENTED · 2025-06-04 (no text)

## [c2126236892] vasqu (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:553 · reply to [c2124443971] · 2025-06-04
Sorry, maybe I'm seeing this not correctly but I don't see `test_batched_generation` among them? Only `test_generation_4bit`, `test_generation_bf16`, and `test_generation_torch_compile` are failing there.

## [c2896149000] vasqu (COLLABORATOR) · review COMMENTED · 2025-06-04 (no text)

## [c2126285427] ydshieh (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:553 · reply to [c2124443971] · 2025-06-04
You are right, my bad, move too fast to get the wrong results.

It's on 

https://huggingface.slack.com/archives/C06LR9PQA00/p1725987868413609?thread_ts=1725987848.555519&cid=C06LR9PQA00

I will check the commit the day before it to see what happened

## [c2896226664] ydshieh (COLLABORATOR) · review COMMENTED · 2025-06-04 (no text)

## [c2126478597] ydshieh (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:553 · reply to [c2124443971] · 2025-06-04
I confirmed that this test is failing when the test is added on 2025/06/19 - I checkout to the commit, run the test.

(around that time, there are several CIs triggered manually on different commits, so hard to check on slack channels)

## [c2896539732] ydshieh (COLLABORATOR) · review COMMENTED · 2025-06-04 (no text)

## [c2126744249] vasqu (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:553 · reply to [c2124443971] · 2025-06-04
Sry to be so picky but is it also having failures on (at least) no prefix being returned :eyes: If yes, I think this is fine to merge then.

## [c2896972389] vasqu (COLLABORATOR) · review COMMENTED · 2025-06-04 (no text)

## [c2126791653] ydshieh (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:553 · reply to [c2124443971] · 2025-06-04
Yes, same failing reason. Not picky, it's fine. I am also happy to wait joao's response. No urgent to merge at all :-)

## [c2897055035] ydshieh (COLLABORATOR) · review COMMENTED · 2025-06-04 (no text)

## [c2154712800] ydshieh (COLLABORATOR) · review_comment on tests/models/falcon_mamba/test_modeling_falcon_mamba.py:553 · reply to [c2124443971] · 2025-06-18
Gentilly ping @gante

## [c2939402490] ydshieh (COLLABORATOR) · review COMMENTED · 2025-06-18 (no text)

## [c2987778660] ydshieh (COLLABORATOR) · issue_comment · 2025-06-19
Although it is not urgent, but let's merge. If @gante has any comment that would lead to any further change(s), we could do it in a follow-up PR.

Keeping failing tests have some risk: if other merged commits cause troubles, it won't be detected and make further fixes more difficult to handle.

(I have been in these cases several times ... trust me the pain)

---
outcome: state=closed, merged=True
