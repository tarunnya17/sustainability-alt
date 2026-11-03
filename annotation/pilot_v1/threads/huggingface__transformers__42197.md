# #42197 [issue] Attempt to access socket despite HF_HUB_OFFLINE = 1 if cache warmed outside current process
https://github.com/huggingface/transformers/issues/42197  ·  opened by fr1ll on 2025-11-13

## [body] fr1ll (NONE) · opening post · 2025-11-13
### System Info

- `transformers` version: 4.57.1
- Platform: Linux-6.6.84.1-microsoft-standard-WSL2-x86_64-with-glibc2.35
- Python version: 3.13.0
- Huggingface_hub version: 0.36.0
- Safetensors version: 0.6.2
- Accelerate version: not installed
- Accelerate config: not found
- DeepSpeed version: not installed
- PyTorch version (accelerator?): 2.9.1+cpu (NA)
- Tensorflow version (GPU?): not installed (NA)
- Flax version (CPU?/GPU?/TPU?): not installed (NA)
- Jax version: not installed
- JaxLib version: not installed
- Using distributed or parallel set-up in script?: No

### Who can help?

@ydshieh I have created a reproducible example of the issue I mentioned in https://github.com/huggingface/transformers/issues/41311#issuecomment-3508674325.

### Information

- [ ] The official example scripts
- [x] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [x] My own task or dataset (give details below)

### Reproduction

Reproducible example: https://github.com/fr1ll/HF_HUB_OFFLINE

Warming the cache in a subprocess, then disabling sockets, then loading the same model should work.
However, it fails with an attempt to access a socket and then "Can't load" errors.

The script named `subprocess-warm_then_offline-load.py` reproduces this error.

Interestingly, warming the cache in process, then disabling sockets, then loading the same model works.
This is reproduced in `inprocess-warm_then_offline-load.py` in the repo above.

### Expected behavior

When a model has already been loaded into the cache ("warm cache"), if `HF_OFFLINE_MODE` = `"1"`, a Transformers pipeline should be able to load the model without accessing any network sockets.

## [c3532993720] ydshieh (COLLABORATOR) · issue_comment · reply to [body] · 2025-11-14
Thank you @fr1ll 

It's indeed valid and I can reproduce. Even with `inprocess-warm_then_offline-load.py`, if I comment out the line that remove the cache dir. Run it twice, I will get failure.

Quite mysterious, but maybe related to the slow/fast image process issues.

I will need to dive into this more. But let me call for contribution/help from the community first.

## [c3559187695] Aaraviitkgp (CONTRIBUTOR) · issue_comment · reply to [c3532993720] · 2025-11-20
@ydshieh I have made a pr on this issue.

## [c3559560155] fr1ll (NONE) · issue_comment · 2025-11-20
https://github.com/huggingface/transformers/pull/42308

## [c3567019533] fr1ll (NONE) · issue_comment · 2025-11-22
Closed with explanation in #42308 by @Wauplin:
> HF_HUB_OFFLINE environment variable is evaluated once at import time. If you really want to update it at runtime, you need to patch huggingface_hub.constant.HF_HUB_OFFLINE

tl;dr: When possible, set the environment variable before importing transformers.

Note: Possible that `local_files_only` argument can be used to disable hub access at runtime for some actions. #42269 indicates this may not work for `transformers.pipeline`; #42318 offers a fix.

## [c3567020322] fr1ll (NONE) · issue_comment · reply to [c3532993720] · 2025-11-22
@ydshieh sorry if I closed this with the wrong `reason`- probably it should be something besides `completed`. I don't have access to reopen and amend the reason.

## [c3569762840] Wauplin (CONTRIBUTOR) · issue_comment · 2025-11-24
closing as "completed" is fine here, considering https://github.com/huggingface/transformers/issues/42197#issuecomment-3567019533 contains the solution (_set the environment variable before importing transformers_)

---
outcome: state=closed
