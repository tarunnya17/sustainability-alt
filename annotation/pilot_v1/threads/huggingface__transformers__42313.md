# #42313 [issue] [v5, Mistral models broken] Embeddings are automatically tied
https://github.com/huggingface/transformers/issues/42313  ·  opened by patrickvonplaten on 2025-11-20

## [body] patrickvonplaten (CONTRIBUTOR) · opening post · 2025-11-20
### System Info

...

### Who can help?

```py
import torch
from transformers import Mistral3ForConditionalGeneration, AutoTokenizer

model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
[… 40 lines of code omitted]

is broken an current main

### Information

- [ ] The official example scripts
- [ ] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [ ] My own task or dataset (give details below)

### Reproduction

execute the code

### Expected behavior

To give non gibberish

**Note**: Only happens on v5

## [c3560475901] patrickvonplaten (CONTRIBUTOR) · issue_comment · 2025-11-20
Can be fixed by setting `tie_word_embeddings=False` for the `config.json`

## [c3562860967] molbap (COLLABORATOR) · issue_comment · 2025-11-21
Acknowledged - current bug is on `main` branch (4.57 dev), to fix before v5 release.

## [c3563111771] ArthurZucker (COLLABORATOR) · issue_comment · 2025-11-21
Yeah, we need to change the default to `False` in individual config. It was always `True` but never enforced. Sorry about that

## [c3681619644] molbap (COLLABORATOR) · issue_comment · 2025-12-22
This should be solved thanks to https://github.com/huggingface/transformers/pull/42507. Tested it, just FYI this also is needed to make it work https://github.com/huggingface/transformers/pull/42997 (there's a bug in the automapping). 

On our radar in any case for the full v5.0 release!

## [c3685942588] molbap (COLLABORATOR) · issue_comment · 2025-12-23
Edit: now this is fixed. From repro code posted, now the output yields a warning relative to config correctness, and generates correctly. 

```bash
Fetching 10 files: 100%|███████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 9993.58it/s]
Download complete: : 0.00B [00:00, ?B/s]                                                      | 0/10 [00:00<?, ?it/s]
Loading weights: 100%|█| 585/585 [00:05<00:00, 113.46it/s, Materializing param=model.vision_tower.transformer.layers.
The tied weights mapping and config for this model specifies to tie model.language_model.embed_tokens.weight to lm_head.weight, but both are present in the checkpoints, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning
The following generation flags are not valid and may be ignored: ['temperature']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
<s>[INST]Hello[/INST]Hello! 😊 How can I assist you today?</s>

```

---
outcome: state=closed
