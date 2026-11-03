# #43521 [pull_request] [CB] Minor perf improvements and ty compatibility
https://github.com/huggingface/transformers/pull/43521  ·  opened by remi-or on 2026-01-27

## [body] remi-or (COLLABORATOR) · opening post · 2026-01-27
# Summary
This PR introduces minor changes to the continuous batching feature:

#### Performance
- optimize the `get_seqlens_k` function to avoid looping over all cache managers
- logits indexing is now done inside the cuda graph, as there was no apparent reason it should be done outside
- fuse the `get_read/write_indices` functions, which should also make the code more readable
#### Typing
- add a `ProtoPretrainedModel` class for typing purposes
- make the code `ty` compliant, using as little `# type:ignore` as possible
- add the `set_continuous_batching_context` to the `LogitsProcessorsList` class to stay coherent with typing 
#### Usability
- rename a variable in `copy_cache`
- we now add a temporary token to the request that get scheduled, which is replaced when a new real token is produced. This is not very usefull as of now but will be when we add CPU / GPU overlapping scheduling, which is soon™

## Performances
| Arguments                                          | Throughput on main | Throughput with PR | Delta    |
|----------------------------------------------------|--------------------|--------------------|----------|
| --samples 10                                       | 530.45             | 531.02             | +0.57    |
| --samples 20 --num-blocks 16/20                    | 100.14             | 126.91             | N/A*     |
| --samples 50                                       | 1477.42            | 1478.66            | +1.24    |
| --samples 100                                      | 2266.81            | 2243.32            | -23.49   |
| --samples 100 --attn flash_attention_2             | 2076.81            | 2050.43            | -26.38   |
| --samples 100 --attn sdpa                          | 840.72             | 831.78             | -8.94    |
| --samples 500                                      | 5281.98            | 5287.99            | +6.01    |
| --samples 500 --add-prefix --compile               | 6743.21            | 6732.60            | -10.61   |
| --samples 50 --num-return-sequences 8 --do-sample  | 5362.52            | 5359.59            | -2.93    |
| --samples 100 --num-return-sequences 4 --do-sample | 4674.91            | 4697.05            | +22.14   |

## Tests
Tests pass, including the tests added for this PR.

## Sanity check
Looks good.

## [c2735897294] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/continuous_api.py:287 · 2026-01-28
you no longer use these no? you could type them as type dict if you still wanted to use them

## [c2735905073] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/continuous_api.py:1377 · 2026-01-28
not sure its really necessary

## [c3715828980] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-01-28
thanks

## [c2735948489] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/continuous_api.py:287 · reply to [c2735897294] · 2026-01-28
Right, removing it.

## [c3715888136] remi-or (COLLABORATOR) · review COMMENTED · 2026-01-28 (no text)

## [c2735954178] remi-or (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/continuous_api.py:1377 · reply to [c2735905073] · 2026-01-28
Actually it is kind of helpful if you call CB through `subcommand.run` because trying to capture `stderr` means you also capture the TQDM bars, which is a pain. Open to any other alternatives!

## [c3715894764] remi-or (COLLABORATOR) · review COMMENTED · 2026-01-28 (no text)

---
outcome: state=closed, merged=True
