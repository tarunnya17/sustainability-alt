# #43778 [pull_request] Mamba-1/-2 init weights in mixer class
https://github.com/huggingface/transformers/pull/43778  ·  opened by kevinli573 on 2026-02-05

## [body] kevinli573 (CONTRIBUTOR) · opening post · 2026-02-05
# What does this PR do?

Initializes dt_bias in the mixer class for both Mamba-1 and Mamba-2 (also initialized in PreTrainedModel class for meta device test).

Addresses [#43717](https://github.com/huggingface/transformers/issues/43717)

## [c3856148096] kevinli573 (CONTRIBUTOR) · issue_comment · 2026-02-05
> **[For maintainers]** Suggested jobs to run (before merge)
> 
> run-slow: mamba, mamba2

The following tests are failing, but they are unrelated to the changes proposed in this PR and also fail on the main branch.

```
FAILED tests/models/mamba2/test_modeling_mamba2.py::Mamba2ModelTest::test_mamba2_slow_vs_fast_forward - TypeError: 'NoneType' object is not callable
FAILED tests/models/mamba2/test_modeling_mamba2.py::Mamba2ModelTest::test_mamba2_slow_vs_fast_forward_grouped - TypeError: 'NoneType' object is not callable
FAILED tests/models/mamba2/test_modeling_mamba2.py::Mamba2ModelTest::test_resize_tokens_embeddings - AssertionError: Tensor-likes are not close!
```

```
FAILED tests/models/mamba/test_modeling_mamba.py::MambaModelTest::test_resize_tokens_embeddings - AssertionError: Tensor-likes are not close!
```

## [c3762946552] ArthurZucker (COLLABORATOR) · review COMMENTED · 2026-02-06
LGTM, @Cyrilvallez I am not 100% this won't prevent `_init_weight` from overwriting it :
```python

def copy_(tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        with torch.no_grad():
            return tensor.copy_(other)
    return tensor
```
does not set it to `True`

## [c3861161905] kevinli573 (CONTRIBUTOR) · issue_comment · reply to [c3762946552] · 2026-02-06
> LGTM, @Cyrilvallez I am not 100% this won't prevent `_init_weight` from overwriting it :
> 
> ```python
> def copy_(tensor: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
>     if not getattr(tensor, "_is_hf_initialized", False):
>         with torch.no_grad():
>             return tensor.copy_(other)
>     return tensor
> ```
> 
> does not set it to `True`

I might be misinterpreting, so please don't hesitate to correct me if so.

I had initially removed the redundant init_weights of A_log and dt_bias in the PreTrainedModel class, but that caused the a test for meta device initialization to fail. So currently, if you instantiate a Mamba2PreTrainedModel, the dt_bias and A_log will be init twice.

## [c2787143883] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/mamba/modeling_mamba.py:196 · reply to [c3762946552] · 2026-02-10
Indeed, as @ArthurZucker said, this will be overwritten in most case. We start by initializing the models on meta device, so all weights defined in `__init__` do not get instantiated (only their shape etc).
Then, if the weight is missing, or if we want a fully randomly instantiated model, the proper way is to it in `_init_weights` currently, which will be called later on.
So here this should go inside this block of the `_init_weights`:

```python
def _init_weights(self, module):
        """Initialize the weights."""
        std = self.config.initializer_range
        if isinstance(module, MambaMixer):
```

## [c3778129016] Cyrilvallez (MEMBER) · review COMMENTED · 2026-02-10 (no text)

## [c2787163608] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/mamba/modeling_mamba.py:196 · reply to [c2787143883] · 2026-02-10
Oh but I see it's already the same there!

## [c3778148669] Cyrilvallez (MEMBER) · review COMMENTED · 2026-02-10 (no text)

## [c2787175431] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/mamba/modeling_mamba.py:196 · reply to [c2787143883] · 2026-02-10
Then if you want to avoid (potentially costly) double init of the same weights in case of random initialization, you should declare all nn.Parameter and buffers as `torch.empty` tensor inside `__init__` (the same way it's done in nn.Linear for example)

## [c3778161158] Cyrilvallez (MEMBER) · review COMMENTED · 2026-02-10 (no text)

## [c2788829522] kevinli573 (CONTRIBUTOR) · review_comment on src/transformers/models/mamba/modeling_mamba.py:196 · reply to [c2787143883] · 2026-02-10
Hmm I see, so what you are recommended is just declare the weights as torch.empty within the mixer and have a specialized init_weights function for the MambaMixer class?

## [c3779940763] kevinli573 (CONTRIBUTOR) · review COMMENTED · 2026-02-10 (no text)

## [c2792488560] Cyrilvallez (MEMBER) · review_comment on src/transformers/models/mamba/modeling_mamba.py:196 · reply to [c2787143883] · 2026-02-11
Yes - well rn it's unfortunately not on the MambaMixer itself that it needs to live, but on the `MambaPreTrainedModel` class. We are thinking about making those module-level though, and will do in the future

## [c3783961712] Cyrilvallez (MEMBER) · review COMMENTED · 2026-02-11 (no text)

## [c3956792464] kevinli573 (CONTRIBUTOR) · issue_comment · reply to [c3783961712] · 2026-02-25
@ArthurZucker @Cyrilvallez sorry for the late follow-up.

I've updated the implementation so the select params are declared as empty with a local mixer-based init helper that runs only when the mixer is instaniated on a non-meta device. To prevent double inits, I had to add a custom `initialize_mixer_weights` argument (which might still cause some issues if people decide to copy over both the mixer and block).

This does seem a bit unwieldy, but correct initialization is really important for users who instantiate/copy the mixer directly. The same issue was also present in the [flash-linear-attention](https://github.com/fla-org/flash-linear-attention/pull/739) repo, where it had a noticable impact on pretrained performance.

If you all have other suggestions on how to handle the initialization, I'd be more than happy to adjust and implement it!

## [c3983448192] ArthurZucker (COLLABORATOR) · issue_comment · reply to [c3956792464] · 2026-03-02
@kevinli573 took the liberty to fix CIs, we'll merge this as is for the release!

## [c3983523924] ArthurZucker (COLLABORATOR) · issue_comment · 2026-03-02
Thanks again for the PR and the reports! 🤗

## [c3988225023] kevinli573 (CONTRIBUTOR) · issue_comment · 2026-03-03
Fantastic, thank you both!

---
outcome: state=closed, merged=True
