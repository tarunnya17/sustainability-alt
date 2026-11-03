# #48368 [pull_request] fix qwen4exp-fp8 ple embedding
https://github.com/huggingface/transformers/pull/48368  ·  opened by JJJYmmm on 2026-08-27

## [body] JJJYmmm (CONTRIBUTOR) · opening post · 2026-08-27
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=48368) [[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=48368)

# What does this PR do?

fixes https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/discussions/2

add a new class `FP8Embedding` (nn.Embedding with per-tensor weight scale), also add a private attr `_quantizable_embeddings` to determine the embedding which may be quantized


## Code Agent Policy

3.8max

## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [ x] Did you read the [contributor guideline](https://huggingface.co/docs/transformers/contributing) and the
      [Pull Request](https://huggingface.co/docs/transformers/pr_checks) checks?
- [ ] Was this discussed/approved via a Github issue or the [forum](https://discuss.huggingface.co/)? Please add a link
      to it if that's the case.
- [ ] Did you make sure to update the documentation with your changes according to the [guidelines](https://github.com/huggingface/transformers/tree/main/docs)?
- [x] Did you write any new necessary [tests](https://huggingface.co/docs/transformers/testing)?


## Who can review?

 @ArthurZucker @Cyrilvallez @vasqu @zucchini-nlp

## [c3880848362] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/models/qwen4_exp/modeling_qwen4_exp.py:1269 · 2026-08-28
the modeling should never have anything related to quantization. We can update the FP8 quantizer tho. 

Also for quantizing the embeddings, this should be classic with `modules_to_convert` in the quantization_config

## [c3880850016] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/quantizers/quantizer_finegrained_fp8.py:133 · 2026-08-28
it should come from the config not from the model

## [c3880851330] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/conversion_mapping.py:1810 · 2026-08-28
The other models do not require this. Might cause issues when saving

## [c5051394459] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-08-28
Ty! its a great addition, let's make sur it follows our standard API! Happy to do it myself

## [c3890629958] JJJYmmm (CONTRIBUTOR) · review_comment on src/transformers/conversion_mapping.py:1810 · reply to [c3880851330] · 2026-08-30
we need `$` when saving fp8 ckpt. without `$`, `ngram_embedding.weight_scale` also matches the target_patterns and would be saved as `ngram_embedding.shard_*.weight_scale`, which is not expected.

## [c5061980628] JJJYmmm (CONTRIBUTOR) · review COMMENTED · 2026-08-30 (no text)

## [c3890631693] JJJYmmm (CONTRIBUTOR) · review_comment on src/transformers/models/qwen4_exp/modeling_qwen4_exp.py:1269 · reply to [c3880848362] · 2026-08-30
got it, let's use explicit `modules_to_convert`🫡  it requires updating the hub ckpt's config, I'll ~~do it asap~~ done

## [c5061982124] JJJYmmm (CONTRIBUTOR) · review COMMENTED · 2026-08-30 (no text)

## [c3900437289] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/utils/quantization_config.py:1707 · 2026-09-01
We really needed this granularity, indeed! ty for that. 
cc @SunMarc it allows 0 changes to other quantization config!

## [c3900441741] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/utils/quantization_config.py:1728 · 2026-09-01
```suggestion
        # TODO: check overlap with not to convert
        self.modules_to_convert = modules_to_convert
```

## [c3900458656] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/conversion_mapping.py:1810 · reply to [c3880851330] · 2026-09-01
Yes, but: https://github.com/huggingface/transformers/blob/add-h4/src/transformers/quantizers/quantizer_finegrained_fp8.py#L266-L268 ! It should "automatically" happen here as the equivalent is done for linear case

## [c5073643556] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-09-01
GG great PR 😉 
One nit !

## [c3900814474] JJJYmmm (CONTRIBUTOR) · review_comment on src/transformers/conversion_mapping.py:1810 · reply to [c3880851330] · 2026-09-01
done! the quantizer now anchors target patterns ending in `.weight` [here](https://github.com/huggingface/transformers/commit/e6b7d9c1a3570c312378ebec485e773d1fd61213#diff-deafed3bd8ea5077fce3122a1c779d31d41d3686c4454de6e29b481995e84485R273). this should not affect other checkpoints, as weights and their corresponding scales are always handled separately.

## [c5074056521] JJJYmmm (CONTRIBUTOR) · review COMMENTED · 2026-09-01 (no text)

---
outcome: state=closed, merged=True
