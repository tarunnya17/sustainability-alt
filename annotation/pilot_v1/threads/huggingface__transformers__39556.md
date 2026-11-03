# #39556 [pull_request] bad_words_ids no longer slow on mps
https://github.com/huggingface/transformers/pull/39556  ·  opened by dwarez on 2025-07-21

## [body] dwarez (CONTRIBUTOR) · opening post · 2025-07-21
# What does this PR do?
Using the `bad_words_ids` on mps is slowing down a lot the text generation, this PR tries to address that.

Fixes https://github.com/huggingface/transformers/issues/39512


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

 @ArthurZucker

## [c3099956490] xenova (CONTRIBUTOR) · issue_comment · 2025-07-21
Thanks for identifying the issue and opening a PR! This does speed up things quite a lot, but it's still around 6 times slower than without setting bad_words_ids 🤔 

Pretty much the same as just doing:
```py
            eos_token_id_list = eos_token_id.tolist() # convert to python list before
            bad_words_ids = list(
                filter(lambda bad_token_seq: all(bad_token_seq != [i] for i in eos_token_id_list), bad_words_ids)
            )
```

## [c3099983703] xenova (CONTRIBUTOR) · issue_comment · 2025-07-21
Though... I guess, since this is just done once (in the init function), 2 seconds vs 36 seconds is still much better!

## [c2221520853] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/logits_process.py:1318 · 2025-07-22
`SequenceBiasLogitsProcessor` is where I would look into slow downs!

## [c3041563604] ArthurZucker (COLLABORATOR) · review APPROVED · 2025-07-22
Already much better, forward pass could then be slow because of torch ops from `SequenceBiasLogitsProcessor` happy to merge as is, but if you have time nice to investigate in the forward see if we can squeeze more perfs!

## [c3101654293] dwarez (CONTRIBUTOR) · issue_comment · reply to [c3041563604] · 2025-07-22
Hi @ArthurZucker, the problem was in the `_prepare_bias_variables` method, in particular here https://github.com/huggingface/transformers/blob/3bc726b381592601cd9dd0fdcff5edcb02f3a85b/src/transformers/generation/logits_process.py#L1187-L1190

In the last commit I made, I modified it with a vectorized access and it's much faster, almost reaching the perfs when not using the `bad_words_ids` parameter.

## [c3046116362] ArthurZucker (COLLABORATOR) · review APPROVED · 2025-07-23
Nice let's go! 🤗

## [c3106342793] ArthurZucker (COLLABORATOR) · issue_comment · 2025-07-23
Could you check the failing test:
```python
=========================== short test summary info ============================
FAILED tests/generation/test_logits_process.py::LogitsProcessorTest::test_no_bad_words_dist_processor - AssertionError: Lists differ: [[True, True, False, True, True], [True, True, True, False, True]] != [[True, True, False, True, False], [True, True, True, False, False]]

First differing element 0:
[True, True, False, True, True]
[True, True, False, True, False]

- [[True, True, False, True, True], [True, True, True, False, True]]
?                            ^^^                              ^^^

+ [[True, True, False, True, False], [True, True, True, False, False]]
?                            ^^^^                              ^^^^
============= 1 failed, 43 passed, 4 warnings in 107.08s (0:01:47) =============
```

## [c3106459607] dwarez (CONTRIBUTOR) · issue_comment · reply to [c3099983703] · 2025-07-23
Oops, my bad, the previous fix had a small bug, which I fixed using @xenova's  suggested code 🤗
We lose 0.1s on average with respect to the previous version, but at least it's correct.

## [c3119601790] ArthurZucker (COLLABORATOR) · issue_comment · 2025-07-25
@bot /style

## [c3119736696] ArthurZucker (COLLABORATOR) · issue_comment · 2025-07-25
Thanks!

---
outcome: state=closed, merged=True
