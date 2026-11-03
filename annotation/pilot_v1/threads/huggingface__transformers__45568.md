# #45568 [pull_request] Gemma4: fix failed test cases
https://github.com/huggingface/transformers/pull/45568  ·  opened by kaixuanliu on 2026-04-22

## [body] kaixuanliu (CONTRIBUTOR) · opening post · 2026-04-22
# What does this PR do?

This PR did several things:
1. Skip some test cases that are not suitbale for gemma4 model
2. Fix bug when `attention_mask` is None(tests/models/gemma4/test_modeling_gemma4.py::Gemma4Audio2TextModelTest::test_eager_matches_fa2_generate)
3. fix some failed test cases related to `test_flash_attn_x_from_config`
4. Add XPU related Expectations

Fixes # (issue)

## Code Agent Policy


- [X] I confirm that this is not a pure code agent PR.



## Who can review?
@ydshieh pls help review

## [c3161002960] ydshieh (COLLABORATOR) · review_comment on src/transformers/models/gemma4/modeling_gemma4.py:1945 · 2026-04-29
@Cyrilvallez  any opinion.

From PR descriptioin

> Fix bug when attention_mask is None(tests/models/gemma4/test_modeling_gemma4.py::Gemma4Audio2TextModelTest::test_eager_matches_fa2_generate)

## [c4196965478] ydshieh (COLLABORATOR) · review COMMENTED · 2026-04-29 (no text)

## [c3161025205] ydshieh (COLLABORATOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:142 · 2026-04-29
OK for me. Just let one of @Cyrilvallez or @vasqu to also valid or comment

## [c4196991082] ydshieh (COLLABORATOR) · review COMMENTED · 2026-04-29 (no text)

## [c3161027346] ydshieh (COLLABORATOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:142 · reply to [c3161025205] · 2026-04-29
They are indeed failing on our CI too

## [c4196993388] ydshieh (COLLABORATOR) · review COMMENTED · 2026-04-29 (no text)

## [c3161044181] ydshieh (COLLABORATOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:478 · reply to [body] · 2026-04-29
@kaixuanliu I didn't see these 2 failing on our Flash Attn CI job.

Could you share more info / error logs ?

## [c4197013323] ydshieh (COLLABORATOR) · review COMMENTED · 2026-04-29 (no text)

## [c3161158292] vasqu (COLLABORATOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:478 · reply to [c3161044181] · 2026-04-29
Our flash attn ci doesn have FA3 - I think it's hard to install because you need to compile from source and it's much longer than FA2 build from source

Maybe we could add a separate FA4 CI - not sure how stable it is tho since it's still in beta

## [c4197145157] vasqu (COLLABORATOR) · review COMMENTED · 2026-04-29 (no text)

## [c3161161581] vasqu (COLLABORATOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:142 · reply to [c3161025205] · 2026-04-29
Hmm, iirc the fallback should be compile compatible cc @IlyasMoutawwakil

## [c4197148657] vasqu (COLLABORATOR) · review COMMENTED · 2026-04-29 (no text)

## [c3161170061] vasqu (COLLABORATOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:509 · 2026-04-29
Can we have something like https://github.com/huggingface/transformers/blob/727741f533b1f47120db47ef5934296d07214d7e/tests/models/dia/test_modeling_dia.py#L258-L271

But for FA? Imo it will always be quite a lot to skip these manually like that

## [c4197157736] vasqu (COLLABORATOR) · review COMMENTED · 2026-04-29 (no text)

## [c3161174465] vasqu (COLLABORATOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:142 · reply to [c3161025205] · 2026-04-29
Which torch version is CI now at btw?

## [c4197162526] vasqu (COLLABORATOR) · review COMMENTED · 2026-04-29 (no text)

## [c3161235406] IlyasMoutawwakil (MEMBER) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:142 · reply to [c3161025205] · 2026-04-29
it is torch compileable, just not any mode that uses cuda graphs (like max-autotune), where the torch.grouped_mm also fails on <sm90 (it also uses the fallback path)

## [c4197233765] IlyasMoutawwakil (MEMBER) · review COMMENTED · 2026-04-29 (no text)

## [c3161239845] kaixuanliu (CONTRIBUTOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:478 · reply to [c3161044181] · 2026-04-29
Well, for FA3 and FA4, on my env they are skipped as well. I can delete these two.

## [c4197238865] kaixuanliu (CONTRIBUTOR) · review COMMENTED · 2026-04-29 (no text)

## [c3161265004] vasqu (COLLABORATOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:478 · reply to [c3161044181] · 2026-04-29
Ah no, see my comment below https://github.com/huggingface/transformers/pull/45568#discussion_r3161170061

## [c4197268222] vasqu (COLLABORATOR) · review COMMENTED · 2026-04-29 (no text)

## [c3161756531] ydshieh (COLLABORATOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:478 · reply to [c3161044181] · 2026-04-29
No, I mean for 

test_flash_attn_2_from_config

our CI is `[PASSED]`. So I am not sure why we need this fix, at least for FA2.

Our CI runner don't have FA3 or FA4, so they are skipped. But the question may still valid: do we really this fix?

## [c4197860381] ydshieh (COLLABORATOR) · review COMMENTED · 2026-04-29 (no text)

## [c3161913691] kaixuanliu (CONTRIBUTOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:478 · reply to [c3161044181] · 2026-04-29
well, the code is added before this PR is merged: https://github.com/huggingface/transformers/pull/45454, it will crash for this case before [here](https://github.com/huggingface/transformers/pull/45454/changes#diff-663099e7a4d592c4a9bcad8f04f555f71d0475b8fa9e7ec399845d29ef5ba18dL2100) is removed. After this PR this case can pass. I will update the code.

## [c4198055237] kaixuanliu (CONTRIBUTOR) · review COMMENTED · 2026-04-29 (no text)

## [c3162105221] kaixuanliu (CONTRIBUTOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:509 · reply to [c3161170061] · 2026-04-29
Thx for the advice. Is it OK now?

## [c4198299303] kaixuanliu (CONTRIBUTOR) · review COMMENTED · 2026-04-29 (no text)

## [c3162152159] vasqu (COLLABORATOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:142 · reply to [c3161025205] · 2026-04-29
So it shouldn't fail iiuc?

## [c4198358088] vasqu (COLLABORATOR) · review COMMENTED · 2026-04-29 (no text)

## [c3162154088] vasqu (COLLABORATOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:509 · reply to [c3161170061] · 2026-04-29
Yep much better thanks

## [c4198360336] vasqu (COLLABORATOR) · review COMMENTED · 2026-04-29 (no text)

## [c3165197022] kaixuanliu (CONTRIBUTOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:142 · reply to [c3161025205] · 2026-04-30
In the skipped test case, they use mode="reduce-overhead", which use cuda graphs and will fail at [L238](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/moe.py#L238) with error: 
```
E                       torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
E                       GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u0, 0) (unhinted: Eq(u0, 0)).  (Size-like symbols: none)
E
E                       consider using data-dependent friendly APIs such as guard_or_false, guard_or_true and statically_known_true.
E                       Caused by: (oot/upstream/transformers/src/transformers/integrations/moe.py:238 in _grouped_mm_fallback_backward)
[… 10 lines of code omitted]
```

## [c4201921754] kaixuanliu (CONTRIBUTOR) · review COMMENTED · 2026-04-30 (no text)

## [c4351307643] Qodo-Free-For-OSS (NONE) · issue_comment · 2026-04-30
Hi, Integration tests add device-specific Expectations entries for XPU without a default fallback, so running these tests on an unsupported accelerator type (or an XPU generation not covered) can select an unintended expectation or raise if no expectation matches. This makes the tests more brittle to new device properties.

**Severity:** informational | **Category:** reliability

**How to fix:** Add default expectation fallback

**Agent prompt to fix** - you can give this to your LLM of choice:

> ### Issue description
> XPU expectations were added without a `(None, None)` default. This can make the test brittle when run on different XPU generations or unexpected device properties.
> 
> ### Fix Focus Areas
> - tests/models/gemma4/test_modeling_gemma4.py[534-542]
> - tests/models/gemma4/test_modeling_gemma4.py[575-593]
> - tests/models/gemma4/test_modeling_gemma4.py[621-629]
> - tests/models/gemma4/test_modeling_gemma4.py[675-681]
> - tests/models/gemma4/test_modeling_gemma4.py[745-755]
> 
> ### Recommended changes
> - Add a default expectation `(None, None): <existing cuda expectation>` if you want to preserve previous behavior for other devices.
> - Or, add additional XPU keys if multiple gens are expected to run these tests.
> - Ensure the intended device coverage is explicit to avoid accidental matching on future hardware.

---
*Found by [Qodo](https://qodo.ai) code review*

## [c3167963858] vasqu (COLLABORATOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:142 · reply to [c3161025205] · 2026-04-30
Can we adjust the comment then at least?

## [c4205162340] vasqu (COLLABORATOR) · review COMMENTED · 2026-04-30 (no text)

## [c3168504518] kaixuanliu (CONTRIBUTOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:142 · reply to [c3161025205] · 2026-04-30
Done.

## [c4205798982] kaixuanliu (CONTRIBUTOR) · review COMMENTED · 2026-04-30 (no text)

## [c3169056948] vasqu (COLLABORATOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:478 · reply to [c3161044181] · 2026-04-30
cc @zucchini-nlp for viz, don't think it's super important but would still be nice to fix at some point ig

## [c4206429286] vasqu (COLLABORATOR) · review COMMENTED · 2026-04-30 (no text)

## [c3172622127] zucchini-nlp (MEMBER) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:478 · reply to [c3161044181] · 2026-05-01
` Gemma4 requires mm_token_type_ids in train mode, so we test in eval mode` > was fixed already no?

If not, @kaixuanliu can you open an issue and ping me there. I might forget to come back when bugs are reported under PRs 😅

## [c4210578363] zucchini-nlp (MEMBER) · review COMMENTED · 2026-05-01 (no text)

## [c3176692797] kaixuanliu (CONTRIBUTOR) · review_comment on tests/models/gemma4/test_modeling_gemma4.py:478 · reply to [c3161044181] · 2026-05-02
It's already fixed. And I have removed this part.

## [c4215023573] kaixuanliu (CONTRIBUTOR) · review COMMENTED · 2026-05-02 (no text)

## [c4228806443] vasqu (COLLABORATOR) · review APPROVED · 2026-05-05
It kind of got lost, sorry let me run slow tests just for sanity checking and then merge

## [c4380103706] vasqu (COLLABORATOR) · issue_comment · 2026-05-05
run-slow: gemma4

---
outcome: state=closed, merged=True
