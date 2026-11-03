# #44246 [issue] import transformers takes long sometimes
https://github.com/huggingface/transformers/issues/44246  ·  opened by audricschiltknecht on 2026-02-23

## [body] audricschiltknecht (NONE) · opening post · 2026-02-23
### System Info

- `transformers` version: 5.2.0
- Platform: macOS-15.7.3-arm64-arm-64bit-Mach-O
- Python version: 3.13.11
- Huggingface_hub version: 1.4.1
- Safetensors version: 0.7.0
- Accelerate version: not installed
- Accelerate config: not found
- DeepSpeed version: not installed
- PyTorch version (accelerator?): not installed (NA)
- Using distributed or parallel set-up in script?: Unknown

### Who can help?

_No response_

### Information

- [ ] The official example scripts
- [x] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [x] My own task or dataset (give details below)

### Reproduction

Use the following script:
```python
from datetime import datetime
print("Before importing files...")
start = datetime.now()
import transformers
delta = datetime.now() - start
print(f"Import done: {delta.seconds}.{delta.microseconds} seconds")
```

You get:

```
Before importing files...
PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Import done: 10.512374 seconds
```

### Expected behavior

Relatively small import time:

```
Before importing files...
PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Import done: 0.836140 seconds
```

## [c3947585920] audricschiltknecht (NONE) · issue_comment · 2026-02-23
This is really similar to #16863, except I reproduce the issue in a brand new venv with only transformers installed.

## [c3947595766] audricschiltknecht (NONE) · issue_comment · 2026-02-23
Here is an extract of a cprofile's profile:

```
>>> import pstats
>>> p = pstats.Stats("transformers-5.2.0.profile")
>>> p.strip_dirs().sort_stats('cumtime').print_stats(.1)
Mon Feb 23 17:00:31 2026    transformers-5.2.0.profile
[… 38 lines of code omitted]

seems a lot of time is spent in the `define_import_structure` and `create_import_structure_from_path` methods.

## [c3951656464] Rocketknight1 (MEMBER) · issue_comment · reply to [c3947595766] · 2026-02-24
Hi @audricschiltknecht, it's not as slow on my machine, so it's trickier for my to benchmark. However, I made some optimizations which should help. Can you try installing from the PR branch with `pip install git+https://github.com/huggingface/transformers.git@faster_create_import_structure_from_path` and let me know if it fixes things?

## [c3953069336] audricschiltknecht (NONE) · issue_comment · reply to [c3951656464] · 2026-02-24
Thanks for the patch @Rocketknight1 
Unfortunately, it does not seem to have improved by a lot:
```
(test-transformers) ~/test-transformers (master)> uv pip install git+https://github.com/huggingface/transformers.git@faster_create_import_structure_from_path
    Updated https://github.com/huggingface/transformers.git (375a659370db5dbf3bf8440fa150e2c8962ccc38)
Resolved 28 packages in 6m 04s
      Built transformers @ git+https://github.com/huggingface/transformers.git@375a659370db5dbf3bf8440fa150e2c8962ccc38
Prepared 1 package in 28.02s
Uninstalled 1 package in 580ms
Installed 1 package in 32ms
 - transformers==5.2.0
 + transformers==5.3.0.dev0 (from git+https://github.com/huggingface/transformers.git@375a659370db5dbf3bf8440fa150e2c8962ccc38)
(test-transformers) ~/test-transformers (master)> python3 -m cProfile -o transformers-faster_create_import_structure_from_path.profile reproduce.py
Before importing files...
PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Import done: 9.308399 seconds
```

## [c3953220179] Rocketknight1 (MEMBER) · issue_comment · reply to [c3953069336] · 2026-02-24
Hi @audricschiltknecht, can you give me a bit more detail about the hardware? Also, is transformers installed on a non-SSD hard disc, or a network drive? This section of the codebase does a lot of file scanning, but it usually completes quickly in my testing.

## [c3954073797] audricschiltknecht (NONE) · issue_comment · 2026-02-24
It's a MacBook Pro M4, running Sequoia. `transformers` is installed on the built-in SSD.
However, it is a corporate-provided laptop, so I'm wondering if there are any monitoring softwares that could cause the issue.
I'm looking into the tracing/instrumentation capabilities of MacOS to see if it's possible to get more data, I'll keep you posted.

## [c3959356730] Rocketknight1 (MEMBER) · issue_comment · reply to [c3954073797] · 2026-02-25
@audricschiltknecht one other thing that would be very helpful would be if you could run a `git bisect` with `pip install -e` to identify the commit where this slowdown first occurs. This would give us a lot of info about the cause!

## [c3975162071] audricschiltknecht (NONE) · issue_comment · reply to [c3959356730] · 2026-02-27
@Rocketknight1 so I'm not 100% sure due to the fact that it takes a while to reproduce the issue, but the bisect points 54a123f068c57abe8bc27a507d05d5674f5862bf as the commit with the first observable slowdown.
Given the commit had a pretty heavy refactor of the `_init__.py` file, it still not a completely frivolous possibility.

Before that, I'm seeing import time of max 2-3sec, and after that it jumps to 7-11sec.

## [c3984949618] LysandreJik (MEMBER) · issue_comment · reply to [c3975162071] · 2026-03-02
Hey @audricschiltknecht, thanks a lot for your work on this issue.

A few clarification questions:
- Are you using ipython in your example?
- If yes, do you observe the same slowdowns within non-ipython runtimes?
- If yes, can you let me know if running `%config IPCompleter.use_jedi=False` before anything else changes anything?
- If yes, can you also let me know if uninstalling `jedi` solves this?

Thanks for your help!

## [c3985308380] audricschiltknecht (NONE) · issue_comment · 2026-03-02
I'm using cpython directly:
> Python 3.14.2 (main, Jan 13 2026, 19:27:04) [Clang 21.1.4 ]

---
outcome: state=closed
