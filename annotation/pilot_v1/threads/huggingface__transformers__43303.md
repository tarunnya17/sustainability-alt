# #43303 [pull_request] [DeepSpeed] add weight_mapping to _load_state_dict_into_zero3_model
https://github.com/huggingface/transformers/pull/43303  ·  opened by kashif on 2026-01-15

## [body] kashif (CONTRIBUTOR) · opening post · 2026-01-15
# What does this PR do?

This pull request introduces enhancements to the DeepSpeed integration for model loading, specifically improving how weight conversions (such as renaming and merging/splitting of weights) are handled when loading state dictionaries into models using DeepSpeed ZeRO-3. The changes add a new utility for applying weight conversion logic before loading, and update the model loading pipeline to use this utility when needed.

Key changes:

### Weight Conversion Utilities

* Added a new function `_apply_weight_conversions_to_state_dict` in `deepspeed.py` to handle weight renaming and merging/splitting operations on state dicts before loading them into a model. This function supports both simple renaming and more complex conversions using `WeightConverter` and `WeightRenaming` objects.

### DeepSpeed Model Loading Logic

* Updated `_load_state_dict_into_zero3_model` to accept an optional `weight_mapping` argument. If provided, it uses the new `_apply_weight_conversions_to_state_dict` function to preprocess the state dict before loading. It also stores the applied weight conversions on the model for later reference.
* Modified the call to `_load_state_dict_into_zero3_model` in `_load_pretrained_model` (in `modeling_utils.py`) to pass the `weight_mapping` argument, ensuring that weight conversions are applied during DeepSpeed ZeRO-3 model loading.

Fixes #43257


## Before submitting
- [ ] This PR fixes a typo or improves the docs (you can dismiss the other checks if that's the case).
- [ ] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request),
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

## [c2694210186] kashif (CONTRIBUTOR) · review_comment on src/transformers/integrations/deepspeed.py:365 · 2026-01-15
is the warning ok or should we have an error?

## [c3665387031] kashif (CONTRIBUTOR) · review COMMENTED · 2026-01-15 (no text)

## [c2699685184] stas00 (CONTRIBUTOR) · review_comment on src/transformers/integrations/deepspeed.py:365 · reply to [c2694210186] · 2026-01-16
assert for sure. how will it work if it's invalid?

## [c3672356885] stas00 (CONTRIBUTOR) · review COMMENTED · 2026-01-16 (no text)

## [c2699837785] kashif (CONTRIBUTOR) · review_comment on src/transformers/integrations/deepspeed.py:365 · reply to [c2694210186] · 2026-01-16
thanks changing

## [c3672557886] kashif (CONTRIBUTOR) · review COMMENTED · 2026-01-16 (no text)

## [c2720679220] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/integrations/deepspeed.py:400 · 2026-01-23
we need to delete the old value otherwise we are gonna explose memory consumption. We don't have to in the other path because we read only once

## [c3696895640] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-01-23
This is great thanks for the initiative! 
- will have device issues probably no? 
- for experts, if deepspeed places some on device x,y,z merging or etc could break this
- memory consumption issue!

## [c2720861030] kashif (CONTRIBUTOR) · review_comment on src/transformers/integrations/deepspeed.py:400 · reply to [c2720679220] · 2026-01-23
i see checking

## [c3697105475] kashif (CONTRIBUTOR) · review COMMENTED · 2026-01-23 (no text)

## [c2721101012] kashif (CONTRIBUTOR) · review_comment on src/transformers/integrations/deepspeed.py:400 · reply to [c2720679220] · 2026-01-23
i now free memory by clearing the source tensors

## [c3697366033] kashif (CONTRIBUTOR) · review COMMENTED · 2026-01-23 (no text)

## [c3701186042] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-01-24
Just merged #43261 which changes stuff a but I'll check this PR out to see if we can simplify a bit calls to re-use some stuff on it today

## [c3794411508] kashif (CONTRIBUTOR) · issue_comment · 2026-01-24
i'll use the new `LoadStateDictConfig`

## [c2724238894] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/integrations/deepspeed.py:321 · 2026-01-24
```suggestion
    model_state_dict = model.state_dict()
```

## [c2724239024] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/integrations/deepspeed.py:330 · 2026-01-24
```suggestion
            renamed_key, _ = rename_source_key(original_key, renamings, [], prefix, model_state_dict)
```

## [c2724239092] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/integrations/deepspeed.py:331 · 2026-01-24
```suggestion
            if renamed_key in model_state_dict:
```

## [c2724242051] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/integrations/deepspeed.py:381 · 2026-01-24
IDK exactly how well python handles this, but `mapping.add_tensor(renamed_key, original_key, source_pattern, tensor)` adds the reference to the tensor. convert creates new and then we can indeed clear the tensor that was in the `sorted_state_dict`. We should propably iterate by key and pop them when needed?

## [c2724242504] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/integrations/deepspeed.py:349 · 2026-01-24
```suggestion
            original_key, renamings, converters, prefix, model_state_dict
```

## [c2724242576] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/integrations/deepspeed.py:356 · 2026-01-24
```suggestion
        if renamed_key in model_state_dict:
```

## [c2724242688] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/integrations/deepspeed.py:392 · 2026-01-24
```suggestion
        if renamed_key not in new_state_dict and renamed_key in model_state_dict:
```

## [c3701788226] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-01-24
LGTM

## [c2724244751] kashif (CONTRIBUTOR) · review_comment on src/transformers/integrations/deepspeed.py:381 · reply to [c2724242051] · 2026-01-24
checking

## [c3701795816] kashif (CONTRIBUTOR) · review COMMENTED · 2026-01-24 (no text)

---
outcome: state=closed, merged=True
