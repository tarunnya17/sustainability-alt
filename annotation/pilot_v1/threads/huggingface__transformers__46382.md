# #46382 [issue] Propagation of output_hidden_states to sub_configs
https://github.com/huggingface/transformers/issues/46382  ·  opened by julien-gadonneix on 2026-06-03

## [body] julien-gadonneix (NONE) · opening post · 2026-06-03
### Feature request

Hi !

I was wondering if it was possible to automatically propagate the `output_hidden_states` attributes to `sub_configs`. For some pre-trained models such as the multimodal ones, this could be convenient to avoid manually mentioning the configs of each modality.

Thanks :)

### Motivation

I use the Qwen3-VL models and my pipeline is common for many different models but for this model, adding the `output_hidden_states` in the config does not give the hidden activations as the output of the model.

### Your contribution

I created this PR: https://github.com/huggingface/transformers/pull/46348

## [c4612287246] julien-gadonneix (NONE) · issue_comment · 2026-06-03
Tell me if you think it's a bad idea !

## [c4621194331] zucchini-nlp (MEMBER) · issue_comment · 2026-06-04
Personally, I think it is not a bad idea but rather slightly breaking if we start accumulating intermediate outputs for all subconfig, that might incur a tiny memory overhead

I see what you mean tho, atm setting the flag `True` on the highest level model has zero effect unless you set it on text config. For vision config, it is mostly unused because we don't return vision hidden states afair. Therefore, I am not really sure if we want to set `True` for all sub-configs tbh

You can also pass it as a `kwarg` and that usually is used by all backbones in that case, that should do imo

## [c4621800539] julien-gadonneix (NONE) · issue_comment · 2026-06-04
Indeed, I noticed using a `kwarg` worked and this was my work-around, thanks !
But it surprised me at first that setting `output_hidden_states=True` in the `config` did not work.

For the vision models indeed it is unused for now and changing this config has no impact. But it could come in the future for some models where these representations are particularly interesting ?

## [c4622274556] zucchini-nlp (MEMBER) · issue_comment · 2026-06-04
yeah, I think we need a proper and intuitive way to get intermediate outputs in multimodal models, since setting `output_hidden_states=True` at hight level config also has zero effect iirc. That is quite a niche use-case so we didn't get much feedback on that

Let's keep this issue and I will come back with suggestions or a PR in the next few weeks. If we touch it, we'd want smth long-term working and minimally BC breaking

[Unless you have an idea, please share if so😄 ]

---
outcome: state=open
