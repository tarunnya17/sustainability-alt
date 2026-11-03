# #45183 [issue] [Bug] XOR logic for `input_ids`/`inputs_embeds` validation produces wrong or misleading error messages across multiple models
https://github.com/huggingface/transformers/issues/45183  ·  opened by XanxusCrypto on 2026-04-02

## [body] XanxusCrypto (CONTRIBUTOR) · opening post · 2026-04-02
### System Info

## Summary

Multiple models across the library use an XOR (`^`) condition to validate 
`input_ids` and `inputs_embeds` inputs. This pattern has two distinct issues 
depending on the model:

1. **Severe (11 models)**: XOR is paired with the error message 
   `"You cannot specify both ..."`, which is factually wrong when *neither* 
   input is provided — the opposite situation.
2. **Minor (most other models, e.g. BART)**: XOR is paired with the generic 
   message `"You must specify exactly one of ..."`, which is functionally 
   correct but conflates two opposite error cases into one vague message.

---

## Background and History

### Before PR #43590

`BartEncoder` used a correct explicit 4-branch check:
```python
if input_ids is not None and inputs_embeds is not None:
    raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
elif input_ids is not None:
    ...  # use input_ids
elif inputs_embeds is not None:
    ...  # use inputs_embeds
else:
    raise ValueError("You have to specify either input_ids or inputs_embeds")
```

`BartDecoder` already had the **severe bug** (XOR + wrong message):
```python
# Pre-#43590 BartDecoder — already broken
if (input_ids is None) ^ (inputs_embeds is not None):
    raise ValueError(
        "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
    )
```

### After PR #43590

The refactor unified both encoder and decoder to:
```python
if (input_ids is None) ^ (inputs_embeds is not None):
    raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
```

This fixed the wrong message in `BartDecoder`, but the XOR logic itself 
still conflates two opposite error cases. Meanwhile, **11 other older models 
were not updated** and still carry the original severe bug.

---

## Why XOR Is Problematic Here

The truth table for `(input_ids is None) ^ (inputs_embeds is not None)`:

| `input_ids` | `inputs_embeds` | XOR result | Correct action |
|---|---|---|---|
| provided | provided | `True`  | ✅ raise error: *"cannot specify both"* |
| provided | None     | `False` | ✅ use `input_ids` |
| None     | provided | `False` | ✅ use `inputs_embeds` |
| None     | None     | `True`  | ✅ raise error: *"must specify one"* |

XOR fires for **both** the "too many" and "too few" cases, but these require 
**different, opposite error messages**. Using a single message for both makes 
it impossible for users to know whether they passed too many or too few inputs.

---

## Issue 1 (Severe): XOR + Wrong Message — 11 Models Not Updated by #43590

These models still use XOR paired with `"You cannot specify both ..."`, 
which is **factually incorrect** when neither input is provided:

| Model | File |
|---|---|
| BigBird-Pegasus | `modeling_bigbird_pegasus.py` |
| BioGPT | `modeling_biogpt.py` |
| Blenderbot | `modeling_blenderbot.py` |
| Blenderbot Small | `modeling_blenderbot_small.py` |
| M2M-100 | `modeling_m2m_100.py` |
| Marian | `modeling_marian.py` |
| mBART | `modeling_mbart.py` |
| Pegasus | `modeling_pegasus.py` |
| Pegasus-X | `modeling_pegasus_x.py` |
| Speech-to-Text | `modeling_speech_to_text.py` |
| Whisper | `modeling_whisper.py` |

## Suggested Fix

Replace the XOR with two explicit conditions, which is unambiguous and 
consistent with the original `BartEncoder` logic before #43590:

```python
if input_ids is not None and inputs_embeds is not None:
    raise ValueError(
        "You cannot specify both input_ids and inputs_embeds at the same time"
    )
if input_ids is None and inputs_embeds is None:
    raise ValueError(
        "You have to specify either input_ids or inputs_embeds"
    )
# No elif needed — the embed_tokens call below handles the remaining cases cleanly
if inputs_embeds is None:
    inputs_embeds = self.embed_tokens(input_ids)
```

This preserves the structural simplification introduced in #43590 while 
restoring precise, actionable error messages for each failure case.

### Who can help?

_No response_

### Information

- [x] The official example scripts
- [ ] My own modified scripts

### Tasks

- [x] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [ ] My own task or dataset (give details below)

### Reproduction

**Reproduction** (example with Blenderbot):
```python
from transformers import BlenderbotModel, BlenderbotConfig

config = BlenderbotConfig()
model = BlenderbotModel(config)

model.model.decoder(input_ids=None, inputs_embeds=None)

```

---

### Expected behavior

# Pass neither input — should say "must specify one", but says "cannot specify both"
# ValueError: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time

## Issue 2 (Minor): XOR + Vague Message — BART and Most Other Models
Models updated by #43590 (including BART) now use:
```python
if (input_ids is None) ^ (inputs_embeds is not None):
    raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
```

This message is technically acceptable for the "neither provided" case, but 
is misleading for the "both provided" case, where the user would expect to 
be told they passed *too many* inputs, not that they need to pick one.

---
outcome: state=closed
