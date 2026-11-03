# #40058 [pull_request] GGUF Qwen2VL
https://github.com/huggingface/transformers/pull/40058  ·  opened by RevanthGundala on 2025-08-09

## [body] RevanthGundala (NONE) · opening post · 2025-08-09
# What does this PR do?





Fixes #40049 

## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [X] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [ ] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [X] Did you write any new necessary tests?


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

## [c2267380749] Isotr0py (CONTRIBUTOR) · review_comment on tests/quantization/ggml/test_ggml.py:307 · 2025-08-11
```suggestion
    qwen2vl_model_id = "unsloth/Qwen2.5-VL-3B-Instruct-GGUF"
    original_qwen2vl_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
```
I think we can use a smaller model for test.

## [c2267381162] Isotr0py (CONTRIBUTOR) · review_comment on tests/quantization/ggml/test_ggml.py:342 · 2025-08-11
```suggestion
    q8_0_qwen2vl_model_id = "Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
```

## [c2267386432] Isotr0py (CONTRIBUTOR) · review_comment on tests/quantization/ggml/test_ggml.py:997 · 2025-08-11
```suggestion
    @unittest.skipUnless(is_gguf_available("0.16.0"), "test requires gguf version >= 0.16.0")
```

## [c3106762119] Isotr0py (CONTRIBUTOR) · review APPROVED · 2025-08-11
LGTM, just some nits.

---
outcome: state=open, merged=False
