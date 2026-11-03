# #45338 [pull_request] docs: document known limitations of _can_set_attn/experts_implementation source inspection
https://github.com/huggingface/transformers/pull/45338  ·  opened by RudrenduPaul on 2026-04-09

## [body] RudrenduPaul (CONTRIBUTOR) · opening post · 2026-04-09
Closes #45162

## What this PR does

Expands the docstrings of `_can_set_attn_implementation` and `_can_set_experts_implementation` in `modeling_utils.py` to explicitly document the known limitations of their source-inspection heuristic.

**Coordination:** I posted on the issue before opening this PR. The maintainer comment (#45162 (comment)) confirmed the approach is "definitely hacky" and that the team is "open to ideas" — this PR addresses the immediate documentation gap as a first step.

## Limitations now documented

Both methods now document four known edge cases:

1. **False positives from comments/strings** — the searched patterns (`eager_attention_forward`, `ALL_ATTENTION_FUNCTIONS.get_interface(`, `@use_experts_implementation`) may appear inside docstrings, comments, or string constants rather than actual code, causing a spurious `True`.
2. **Compiled modules** (`.so` / `.pyd`) — no readable source file. Already handled safely by the `__file__` guard (returns `False`), but this was not documented.
3. **Non-standard import environments** (PyInstaller, `zipimport`, custom loaders) — `__file__` may be absent or non-filesystem. Also caught by the guard, but undocumented.
4. **Interactive environments** (Jupyter, REPL) — no `__file__`. Caught by the guard.

## No behaviour change

This is a documentation-only PR. No logic was modified.

## Testing

No tests changed. `make style` was run to verify formatting.

> **Note:** Claude Code was used to assist in drafting the expanded docstrings. All changes were reviewed by the submitter.

## [c4214710165] Rocketknight1 (MEMBER) · issue_comment · 2026-04-09
Hi, I'm sorry, we probably don't want this PR! The new docstring is very long, and most of these risks are obvious once you see that it's string-searching inside the file

---
outcome: state=closed, merged=False
