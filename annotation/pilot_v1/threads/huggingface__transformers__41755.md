# #41755 [pull_request] Swap columns and rows of the grid layout in LFM2-VL
https://github.com/huggingface/transformers/pull/41755  ·  opened by ankke on 2025-10-21

## [body] ankke (CONTRIBUTOR) · opening post · 2025-10-21
Fixing swapped rows and columns that were resulting in incorrect positional special tokens. Fixed integration tests on the modeling file.

@zucchini-nlp

## [c3364596535] zucchini-nlp (MEMBER) · review APPROVED · 2025-10-22
Thanks! And for the batched test, I think we will have to fix it in separate PR. I will trigger slow CI and it might complain that the test is failing

## [c3431298830] zucchini-nlp (MEMBER) · issue_comment · 2025-10-22
run-slow: lfm2_vl

## [c3431565964] zucchini-nlp (MEMBER) · issue_comment · 2025-10-22
Interesting, in the runners the batched test gives better generation such as `"In this image, we see a cat and a dog lying on a pink blanket. They are both sleeping peacefully. They are"` . Could then prob be smth with the versions of installed packaged or hardware

---
outcome: state=closed, merged=True
