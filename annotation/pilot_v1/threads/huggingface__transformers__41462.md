# #41462 [issue] Support encoder text classification for sequence to sequence models like BART and T5
https://github.com/huggingface/transformers/issues/41462  ·  opened by cbhyphen on 2025-10-09

## [body] cbhyphen (NONE) · opening post · 2025-10-09
### Feature request

Add support for encoder-only text classification for models like BART and T5.

### Motivation

The current sequence classifiers for BART and T5 include the full encoder-decoder.  While the decoder is used to produce the sentence representation for the classification head, the following paper shows that mean pooling of encoder final hidden states may be a better option (at least for T5):  https://arxiv.org/abs/2108.08877.  In addition to this, not using the decoder would reduce the size of the model and speedup fine-tuning as well as inference.

### Your contribution

PR Submission

## [c3387125082] Sheshank-singh (NONE) · issue_comment · reply to [body] · 2025-10-09
Hi @cbhyphen,

I’m interested in working on this feature request to add encoder-only text classification support for BART and T5. Could you please assign this issue to me? I’d love to contribute and submit a PR for Hacktoberfest.

Thank you!

## [c3388046032] chrisbw-premier (NONE) · issue_comment · 2025-10-10
Would love to see this!

## [c3391489121] developing-human (NONE) · issue_comment · 2025-10-10
I'd also find this useful 👍

## [c3391509299] cbhyphen (NONE) · issue_comment · reply to [c3387125082] · 2025-10-10
hey @Sheshank-singh thanks for your interest!  I have a [PR out here for T5](https://github.com/huggingface/transformers/pull/40898) but feel free to work on BART!

## [c3391687834] Sheshank-singh (NONE) · issue_comment · reply to [c3391509299] · 2025-10-10
Thank you for letting me know, @cbhyphen ! 
Sure, I’ll start working on the BART encoder-only text classification implementation.

## [c3393807558] lasl321 (NONE) · issue_comment · 2025-10-12
This is interesting. Have some ideas we might discuss.

## [c3398977111] EdgarJiJ (NONE) · issue_comment · 2025-10-13
I would like to see this

## [c3434075923] africker (NONE) · issue_comment · 2025-10-22
This would be useful to me, but I'd love to see more encoders for sequence classification.

---
outcome: state=open
