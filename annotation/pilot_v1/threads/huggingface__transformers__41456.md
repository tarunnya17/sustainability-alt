# #41456 [pull_request] Add vision contribution guide
https://github.com/huggingface/transformers/pull/41456  ·  opened by molbap on 2025-10-08

## [body] molbap (COLLABORATOR) · opening post · 2025-10-08
# What does this PR do?

This updates the vision contribution guide, to help external contributors merge their work faster and reduce the load on maintainers with a checklist. It is a very minimal checklist that will need to be completed with specific model implementation guides following the new API.

## [c2416039081] zucchini-nlp (MEMBER) · review_comment on CONTRIBUTING.md:119 · 2025-10-09
maybe  "or any multimodal model that processes `images/videos`" or simply vision objects?

## [c2416041440] zucchini-nlp (MEMBER) · review_comment on CONTRIBUTING.md:128 · 2025-10-09
"it's better if configuration and processing is in it"

## [c2416049180] zucchini-nlp (MEMBER) · review_comment on CONTRIBUTING.md:157 · 2025-10-09
can we nudge to add tests for all components which are not 100%-copies? In other words, if model has its own processing, then we also need processor tests etc. I've seen several times ppl adding only model tests and then we find issues in processing...

## [c2416052003] zucchini-nlp (MEMBER) · review_comment on CONTRIBUTING.md:161 · 2025-10-09
to clarify maybe: checkpoints should fit in our runners so if the model size is huge, they need to load in 4-bit or in half precision etc. Also using small image sizes helps

## [c3317875193] zucchini-nlp (MEMBER) · review COMMENTED · 2025-10-09
Nice, my top pain points when reviewing model PR are below. Feel free to re-use them if you think it's valid to be added in the doc 😄 

- Do not add code path with if/else if the released weights have only one path supported. Code path used for experiments are not needed and bloat up in the end
- Try to use @autodocstring and @check_model_inputs 
- The model structure has to follow standards of other VLMs with correct class names. Some users add a vision backbone with its original name, i.e. RiceModel used in llava-onevision-1.5. IMO we can either add a separate vision model and call it RiceModel, or we add it in llava model-file and call it LlavaOnevision1_5VisionModel
- Users add many assertions, some are basic shape checks and not needed, while some are valid but without actionable error message. Let's nudge to add only valid checks with good error messages
- Users call model inputs as `x, y, z`, we can nudge for better naming conventions `pixel_values, hidden_states`. Especially in audio backbones, apparently copy from source code
- Always go over the PR again , look for commented code or left-over TODO/FIXME comments
- Do not use _checkpoint_conversion_mapping, convert checkpoints and release in the hub if needed
- The vision language models should have `get_image_features()`  and `get_placeholder_mask()` methods
- When copying tests from similar models, delete all skipped tests and check if that applies for your model as well. Blind test skipping is not good

## [c2426137397] yonigozlan (CONTRIBUTOR) · review_comment on CONTRIBUTING.md:196 · 2025-10-13
I would add `auto_docstring`, `can_return_tuple`, `check_model_inputs` and `_can_record_outputs`. I think the more precise we can be the better! I see a lot of confusion on the use of the decorators, and how it's not necessary to add the output_attention, output_hidden_states etc.

## [c2426141762] yonigozlan (CONTRIBUTOR) · review_comment on CONTRIBUTING.md:214 · 2025-10-13
🤗

## [c3331333469] yonigozlan (CONTRIBUTOR) · review APPROVED · 2025-10-13
One small comment, but very eager to have this merge in general, this should simplify first reviews quite a bit 🤗

## [c2438898611] ariG23498 (CONTRIBUTOR) · review_comment on CONTRIBUTING.md:127 · 2025-10-17
```suggestion
- Use [`transformers-cli add-new-model-like`](https://github.com/huggingface/transformers/blob/main/src/transformers/cli/add_new_model_like.py) to generate a modular skeleton and get started
```

Could not find docs for this, so referring the users to the script would help them get a better overview of the command.

## [c2438904687] ariG23498 (CONTRIBUTOR) · review_comment on CONTRIBUTING.md:153 · 2025-10-17
```suggestion
- Examples: [`convert_llava_onevision_weights_to_hf.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_onevision/convert_llava_onevision_weights_to_hf.py), [`convert_idefics2_weights_to_hf.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/convert_idefics2_weights_to_hf.py)
```

## [c3348952289] ariG23498 (CONTRIBUTOR) · review COMMENTED · 2025-10-17
Adding links to scripts might make it more useful for the readers.

## [c2443958910] ariG23498 (CONTRIBUTOR) · review_comment on CONTRIBUTING.md:119 · reply to [c3317875193] · 2025-10-20
```suggestion
If you're contributing a **vision-language model** (or any multimodal model that processes images/videos), please follow this checklist. Maintainers will use this to review your PR, and completing these steps will significantly increase the likelihood of your PR being merged quickly.
```

Adding @zucchini-nlp's suggestion

## [c2443962095] ariG23498 (CONTRIBUTOR) · review_comment on CONTRIBUTING.md:157 · 2025-10-20
```suggestion
At minimum, add an `IntegrationTest` class that tests end-to-end generation (processing and modelling) with **exact** output matching:
```

## [c2443963297] ariG23498 (CONTRIBUTOR) · review_comment on CONTRIBUTING.md:161 · 2025-10-20
```suggestion
- Tests should use real checkpoints (load in 4-bit or half precision if the checkpoint is too big) and inputs
```

## [c2443967248] ariG23498 (CONTRIBUTOR) · review_comment on CONTRIBUTING.md:196 · 2025-10-20
```suggestion
- Use the decorators like (`auto_docstring`, `can_return_tuple`, `check_model_inputs` and `_can_record_outputs`) in the correct manner.
- Don't reinvent the wheel
```

## [c3355197519] ariG23498 (CONTRIBUTOR) · review COMMENTED · reply to [c3331333469] · 2025-10-20
I have added the review from @zucchini-nlp and @yonigozlan as suggestions for the ease of @molbap.

Hope this helps!

---
outcome: state=closed, merged=True
