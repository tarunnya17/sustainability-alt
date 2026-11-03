# #45627 [pull_request] Processing Utils: honor pre-built sub-processor kwargs in from_pretrained
https://github.com/huggingface/transformers/pull/45627  ·  opened by javierdejesusda on 2026-04-24

## [body] javierdejesusda (CONTRIBUTOR) · opening post · 2026-04-24
## What does this PR do?

Addresses @ArthurZucker's follow-up on #44987. When a caller passes a pre-built sub-processor to `AutoProcessor.from_pretrained` — e.g. `tokenizer=tok` or `bpe_tokenizer=tok` — the instance is now used directly instead of being silently forwarded into the sub-loader calls.

Exact attribute names always take precedence (`bpe_tokenizer=`). For processors with a single sub-processor of a given modality, the canonical modality name (`tokenizer=`) is also accepted as an alias — this matches the reproducer in the issue comment, where the remote `UniversalActionProcessor` takes `bpe_tokenizer` but the caller passes `tokenizer=`.

The OP's zero-kwarg reproducer (`AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)`) is a separate hub repo layout issue and is not addressed here.

## Before submitting

- [x] Did you read the [contributor guideline](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md#create-a-pull-request)?
- [x] Did you write any new necessary tests?

## Who can review?

cc @ArthurZucker @itazap

## [c3137000589] zucchini-nlp (MEMBER) · review_comment on src/transformers/processing_utils.py:1452 · 2026-04-24
not really sure about this. The error from GH issue is due to old remote code, and we don't yet support the Pi0-FAST natively in transformers. also cc @yonigozlan ig you might have seen similar issue when refactoring processor loading

We're planning native support though, and waiting for lerobot team to test and convert the configs correctly

## [c4169676141] zucchini-nlp (MEMBER) · review COMMENTED · 2026-04-24 (no text)

## [c3137076073] javierdejesusda (CONTRIBUTOR) · review_comment on src/transformers/processing_utils.py:1452 · reply to [c3137000589] · 2026-04-24
Thanks for taking a look, @zucchini-nlp!

Quick scope note: this PR isn't targeting the OP's zero-kwarg traceback (that's the hub layout / old remote code path you mentioned, which I agree is out of scope here and will be obsoleted by native support). It's targeting @ArthurZucker's follow-up comment on the issue:

> `p = AutoProcessor.from_pretrained("physical-intelligence/fast", tokenizer=tokenizer, trust_remote_code=True, use_fast=False)`
>
> *"this does not work and it should!"*

The underlying behavior is general to `ProcessorMixin`: when a caller supplies a pre-built sub-processor via kwargs (whether `tokenizer=` or the exact attribute name like `bpe_tokenizer=`), the instance is silently dropped and the loader tries to reload from disk anyway. Any processor with a non-primary tokenizer attribute runs into this, so native Pi0-FAST support wouldn't fix it on its own, it'd just mean one fewer processor hitting it.

That said, happy to defer fully. If you and @yonigozlan / @ArthurZucker feel this should wait (or be folded into the native support work, or handled differently), I'm glad to close or rescope, just let me know.

## [c4169761066] javierdejesusda (CONTRIBUTOR) · review COMMENTED · 2026-04-24 (no text)

## [c3147562059] zucchini-nlp (MEMBER) · review_comment on src/transformers/processing_utils.py:1452 · reply to [c3137000589] · 2026-04-27
yeah, totally get it. Personally, I think we can deliberately not support it as remote code and not-v5 compatible unless Arthur/Yoni have a different opinion

## [c4181205578] zucchini-nlp (MEMBER) · review COMMENTED · 2026-04-27 (no text)

## [c4763632375] javierdejesusda (CONTRIBUTOR) · issue_comment · 2026-06-21
@ArthurZucker gentle ping, would appreciate a review when you have time.

## [c5916041219] javierdejesusda (CONTRIBUTOR) · issue_comment · reply to [c4181205578] · 2026-09-30
Closing this since #44987 is closed and, as @zucchini-nlp noted, the motivating case came from old remote code that transformers does not intend to support. Thanks for the review.

---
outcome: state=closed, merged=False
