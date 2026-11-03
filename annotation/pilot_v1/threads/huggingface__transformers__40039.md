# #40039 [pull_request] New DynamicSlidingWindowLayer & associated Cache
https://github.com/huggingface/transformers/pull/40039  ·  opened by Cyrilvallez on 2025-08-08

## [body] Cyrilvallez (MEMBER) · opening post · 2025-08-08
# What does this PR do?

As per the title. To avoid wasting memory for models with sliding window. As I don't want to reintroduce static hybrid caches by default to avoid all the pitfalls of automatic compilation, but don't want to waste that memory, this is definitely the way to go.

The only change that is needed is to pass the `config` to `DynamicCache`, to be able to parse `sliding_window`/`layer_types`. If we don't, then the behavior is exactly the same as before.

See the following figures for an illustration:
- top: Mistral 7B, all layers are sliding, so the cache stops growing after reaching the window size of 4096 
- bottom: Gemma 2 9B, 1 out of 2 layers are sliding, so the Cache grows "sublinearly" after reaching the window size of 4096
[image]
[image]


Bonus:
- Gpt OSS 20B: 1 out of 2 layers is sliding with window_size=128, so we basically fully divide memory requirements by 2 throughout the whole range (except super small input sizes < 128 of course) (it has lower absolute cache size because of only 24 layers and head_dim=64)
[image]


Adding the benchmark script for posterity:

```python
from transformers import AutoModelForCausalLM, DynamicCache
import torch
from tqdm import tqdm

[… 38 lines of code omitted]

## [c3177131462] Cyrilvallez (MEMBER) · issue_comment · 2025-08-11
All good now, slow tests on mistral, gemma2 and t5gemma are all similar to main (only a slight fa2 issue that surfaced on a slow test for mistral, but it's unrelated and solved by https://github.com/huggingface/transformers/pull/40002)

## [c2269543094] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/cache_utils.py:999 · 2025-08-12
well you can put this in the docstring above!

## [c2269547205] ArthurZucker (COLLABORATOR) · review_comment on tests/models/mistral/test_modeling_mistral.py:379 · 2025-08-12
nice test!

## [c3110007788] ArthurZucker (COLLABORATOR) · review APPROVED · 2025-08-12
Perfect! This is long due especially for mistral models, but will also be effective for gpt oss (hybrid sliding I think, sliding window is 128 so would generate the graph for this one as well!)

---
outcome: state=closed, merged=True
