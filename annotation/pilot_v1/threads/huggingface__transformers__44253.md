# #44253 [pull_request] Speed create_import_structure up with os.scandir()
https://github.com/huggingface/transformers/pull/44253  ·  opened by Rocketknight1 on 2026-02-24

## [body] Rocketknight1 (MEMBER) · opening post · 2026-02-24
`create_import_structure_from_path` does some redundant `os` calls, so I'm experimenting with changes to see if we can speed up loading a lot.

Related to #44246

## [c3985179134] Rocketknight1 (MEMBER) · issue_comment · 2026-03-02
cc @lysandrejik while you're working on this! I don't think it addresses the contributor's issue but it might be some free speedup regardless.

The tl;dr is that the existing code does two wasteful things. Firstly, it doesn't skip `convert_*.py` files in the module dir, which are never relevant. Those files are excluded in release versions but are still present if people install from `main`.

Secondly, the old approach of `os.listdir(module_path)` followed by `os.path.isdir()` requires multiple system calls per model, but it's much faster to use [os.scandir](https://docs.python.org/3/library/os.html#os.scandir) because the OS actually returns the information we need in the directory listing call, so we avoid thousands of system stat calls this way. The speedup may not be huge, but there's no downside, so it's probably worth merging!

## [c3986117069] Rocketknight1 (MEMBER) · issue_comment · 2026-03-02
(Failing test is unrelated and goes on my list of flaky tests to fix)

## [c3919938683] LysandreJik (MEMBER) · review APPROVED · reply to [c3986117069] · 2026-03-10
Thank you @Rocketknight1 !

---
outcome: state=closed, merged=True
