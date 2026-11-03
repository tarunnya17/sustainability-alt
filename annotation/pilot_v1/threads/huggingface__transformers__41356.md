# #41356 [pull_request] Add DEIMv2 model, image processor, and basic tests
https://github.com/huggingface/transformers/pull/41356  ·  opened by Vinayaktoor on 2025-10-05

## [body] Vinayaktoor (NONE) · opening post · 2025-10-05
# What does this PR do?

This PR introduces the DEIMv2 model along with its configuration, image processor, and unit tests. The changes include:

Model Implementation: Added Deimv2ForObjectDetection supporting object detection tasks.

Configuration: Added Deimv2Config for model hyperparameters.

Image Processing: Added Deimv2ImageProcessor to handle preprocessing and postprocessing of images.

Unit Tests: Added tests for configuration, image processing, and model forward pass (tests/models/deimv2/).

This PR builds upon and makes changes to PR by @kekwlboy12469, PR #41291.
Fixes #41211
Fixes missing or incomplete DEIMv2 model integration.

Fixes absence of unit tests for DEIMv2 model and image processor.

Fixes compatibility issues with Transformers pipelines for object detection.


## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
      Pull Request section?
- [x] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [x] Did you make sure to update the documentation with your changes? Here are the
      [documentation guidelines](https://github.com/huggingface/transformers/tree/main/docs), and
      [here are tips on formatting docstrings](https://github.com/huggingface/transformers/tree/main/docs#writing-source-documentation).
- [x] Did you write any new necessary tests?


## Who can review?

Anyone in the community is free to review the PR once the tests have passed. Feel free to tag
members/contributors who may be interested in your PR.

## [c3371598883] Rocketknight1 (MEMBER) · issue_comment · 2025-10-06
cc @nielsrogge @yonigozlan

## [c3870033507] peacefulotter (NONE) · issue_comment · 2026-02-09
Hi, 

Is there still a will to merge this PR at some point ? 

I think the implementation is unfinished yet; DEIMV2_PRESETS defines dinov2 as backbone instead of dinov3 and the published model.safetensors doesn't align with the proposed model anyway. 

Best

## [c3872953722] Rocketknight1 (MEMBER) · issue_comment · 2026-02-09
gentle ping @molbap @nielsrogge @mcpatate if anyone wants to take it!

---
outcome: state=open, merged=False
