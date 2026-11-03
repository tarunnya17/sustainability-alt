# #45530 [pull_request] [CB] Changes for long generation
https://github.com/huggingface/transformers/pull/45530  ·  opened by remi-or on 2026-04-20

## [body] remi-or (COLLABORATOR) · opening post · 2026-04-20
## Summary                                                     
                                                                                                                                                                                                                                          
This PR fixes some issues related to memory to make long generation (16K+) easier.                                                                                                      
                                                                                                                                                                                                                                          
  - Fix KV dedup for decode batches (scheduler.py): Decode-only batches don't consume the read_indices cache budget, so don't reject them on that basis. Also gate the decode fast path on max_blocks_per_request > 0 instead of unconditionally enabling it.        
  - Fix memory estimation (requests.py): Use torch.cuda.mem_get_info instead of device_properties().total_memory, which ignored CUDA context/driver overhead (~0.5 GiB) and caused overcommit/OOM.                                        
  - Raise max_memory_percent default 0.8 → 0.9 (configuration_utils.py): Now safe with the corrected memory accounting above.                                                                                                             
  - Write-only fast path (cache.py, input_outputs.py, scheduler.py): When a batch has no past-cache reads (pure prefills), skip the index_select read-back, avoid allocating/transferring read_index, and return the input KV states directly. Also adjusts the CUDA-graph key to depend on the block-table path rather than max_kv_read > 0.
  - Two-peak memory model (cache.py): Replace the single peak_activation_per_token with two activation peaks — LM head (hidden + logits, N-independent) and attention (hidden + Q + new K/V + cache K/V reads, grows with N). Solve the memory polynomial for each peak independently and take the most restrictive (num_blocks, max_batch_tokens). Bumps _upper_bound_max_batch_tokens 256 → 1024.

# Performances

Pretty good, lot of workloads benefit from 80% to 90% raise in cache space

| Arguments                                                       | Main (tok/s) | Current (tok/s) | Diff (%) |
|-----------------------------------------------------------------|--------------|-----------------|----------|
| --samples 10                                                    | 869.0        | 890.37          | +2.5%    |
| --samples 20 --num-blocks 20                                    | 517.95       | 520.16          | +0.4%    |
| --samples 50                                                    | 3629.88      | 3638.6          | +0.2%    |
| --samples 100                                                   | 5375.41      | 5522.83         | +2.7%    |
| --samples 100 --attn flash_attention_2                          | 3666.82      | 3743.47         | +2.1%    |
| --samples 100 --attn sdpa                                       | 1030.21      | 1053.57         | +2.3%    |
| --samples 500 --no-use-async                                    | 6621.78      | 8020.64         | +21.1%   |
| --samples 500 --use-async                                       | 7963.66      | 9332.71         | +17.2%   |
| --samples 32 --max-new-tokens 2048 --use-async                  | 2033.87      | 2064.29         | +1.5%    |
| --samples 32 --max-new-tokens 2048 --use-async --block-table 32 | 2716.64      | 2734.81         | +0.7%    |
| --samples 500 --add-prefix --compile                            | 7649.48      | 8882.62         | +16.1%   |
| --samples 50 --num-return-sequences 8 --do-sample               | 869.94       | 980.13          | +12.7%   |
| --samples 100 --num-return-sequences 4 --do-sample              | 1708.88      | 1925.25         | +12.7%   |

# Tests

- [x] `make style` and `make_typing` pass
- [x] RUN_SLOW=1 pytest tests/generation/test_continuous_batching.py
- [x] RUN_SLOW=1 pytest tests/cli/test_serve.py
- [x] RUN_SLOW=1 pytest tests/generation/test_paged_attention.py

## [c3129186458] ArthurZucker (COLLABORATOR) · review_comment on src/transformers/generation/continuous_batching/requests.py:51 · 2026-04-23
nice!

## [c4160622476] ArthurZucker (COLLABORATOR) · review APPROVED · 2026-04-23
some of the names are a tad bit unfamiliar to me, but LGTM!

---
outcome: state=closed, merged=True
