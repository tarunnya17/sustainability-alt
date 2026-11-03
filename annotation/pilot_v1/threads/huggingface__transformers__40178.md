# #40178 [pull_request] [`kernels`] If flash attention2 is not installed / fails to import (cc on our cluster) default to kernels
https://github.com/huggingface/transformers/pull/40178  ·  opened by ArthurZucker on 2025-08-14

## [body] ArthurZucker (COLLABORATOR) · opening post · 2025-08-14
# Improves handling of FlashAttention2 + add community kernel fallback 

* Updated `_check_and_adjust_attn_implementation` to set the attention implementation to `kernels-community/flash-attn` when FlashAttention2 is requested but not available, ensuring seamless fallback and proper kernel registration.

**Testing Improvements:**

* Modified `require_flash_attn` in `testing_utils.py` to allow tests to run if either FlashAttention2 or the community kernel is available, broadening test coverage and reliability.

## [c3191087784] ArthurZucker (COLLABORATOR) · issue_comment · 2025-08-15
run-slow: flash_attention_2

## [c3206282762] ArthurZucker (COLLABORATOR) · issue_comment · 2025-08-20
run-slow: flash_attention_2

## [c3206302652] ArthurZucker (COLLABORATOR) · issue_comment · 2025-08-20
run-slow: flash_attention_2

## [c2288150283] Cyrilvallez (MEMBER) · review_comment on src/transformers/modeling_utils.py:2493 · 2025-08-20
I don't think we need this part!

## [c2288192731] Cyrilvallez (MEMBER) · review_comment on src/transformers/modeling_utils.py:2736 · 2025-08-20
Only change needed should be in this function
```suggestion
        applicable_attn_implementation = attn_implementation
        # If FA not installed, do not fail but use kernels
        if attn_implementation == "flash_attention_2" and self._supports_flash_attn and not is_flash_attn_2_available() and is_kernels_available():
            applicable_attn_implementation = "kernels-community/flash-attn"
        if re.match(r"^[^/:]+/[^/:]+(?:@[^/:]+)?(?::[^/:]+)?$", applicable_attn_implementation):
[… 34 lines of code omitted]
```

## [c3136578769] Cyrilvallez (MEMBER) · review COMMENTED · 2025-08-20
I think the following is a bit more appropriate, although this code part is starting to be quite convoluted

## [c3233532949] ArthurZucker (COLLABORATOR) · issue_comment · 2025-08-28
run-slow: flash_attention_2

## [c3233574411] ArthurZucker (COLLABORATOR) · issue_comment · 2025-08-28
run-slow: flash_attention_2

## [c3233700616] Cyrilvallez (MEMBER) · issue_comment · 2025-08-28
Confirmed on tests locally that it works! Merging

---
outcome: state=closed, merged=True
