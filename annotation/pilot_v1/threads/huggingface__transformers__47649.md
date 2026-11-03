# #47649 [issue] `modeling_qwen3_vl.py` performed D2H makes cpu wait gpu
https://github.com/huggingface/transformers/issues/47649  ·  opened by NewUserHa on 2026-07-30

## [body] NewUserHa (NONE) · opening post · 2026-07-30
https://github.com/huggingface/transformers/blob/516a6b40b7d5ae7147306661378ea0ebe4ee238e/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L207-L274
This code
```py
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)
            ]
```
creates `D2H pageable->host`, it costs more than `250ms` and makes gpu is waited by cpu that already ahead.

https://github.com/huggingface/transformers/blob/516a6b40b7d5ae7147306661378ea0ebe4ee238e/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L924-L1015
The 
`current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size` `position_ids[:, batch_idx] = llm_positions.to(position_ids.device)` `mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)` also having this issue.

Can this be improved?

### Expected behavior

No D2H or H2D during forward, to make model run faster

## [c5135344531] chavalasantosh (NONE) · issue_comment · reply to [body] · 2026-07-30
Thanks @NewUserHa for identifying this and providing the profiler observation. I’d like to investigate and work on a production-safe optimization.

I see three potentially separate synchronization problems:

1. `lengths.tolist()` is executed inside every vision-attention layer for non-Flash Attention backends.
2. `input_token_type.tolist()` followed by Python `itertools.groupby()` moves modality metadata into Python during RoPE-index construction.
3. CUDA scalar tensors are accumulated in Python containers and later reconstructed with `torch.tensor(...)`, which can introduce scalar extraction and a new host-to-device copy.

I would treat these separately so the final change remains measurable, reviewable, and correctness-preserving.

### Performance contract

The optimized path should:

* avoid repeated device-to-host synchronization inside vision transformer layers;
* avoid rebuilding GPU tensors through Python scalar lists;
* preserve exact sequence segmentation and multimodal position semantics;
* preserve outputs for eager, SDPA, and Flash Attention;
* remain compatible with CPU execution, tracing, and `torch.compile`;
* avoid replacing synchronization with a large block-diagonal attention mask that causes a material memory regression.

### Phase 1: establish a reliable baseline

I would add a focused benchmark using a tiny Qwen3-VL configuration and synthetic image/video grids.

The benchmark matrix would include:

* eager, SDPA, and Flash Attention where available;
* one image, multiple images, one video, and mixed image/video inputs;
* different numbers and sizes of packed vision sequences;
* warm-up followed by measured iterations;
* CPU and CUDA execution;
* eager execution and `torch.compile` where supported.

I would capture:

* CPU self time;
* CUDA kernel time;
* D2H/H2D memcpy events;
* host synchronization events;
* graph breaks;
* peak allocated GPU memory;
* end-to-end vision-prefill latency.

Timing would use CUDA events or explicit synchronization outside the measured region, rather than allowing an implicit synchronization to distort the result.

### Phase 2: remove repeated synchronization from vision attention

The largest immediate issue appears to be:

```python
lengths = cu_seqlens[1:] - cu_seqlens[:-1]
lengths.tolist()
```

because it is executed for every vision-attention block even though the packed sequence partition is constant for the complete vision forward.

My preferred minimal design would be:

1. Derive immutable segment metadata once in `Qwen3VisionModel.forward()`.
2. Pass the same segment metadata through each vision block and attention layer.
3. Avoid recomputing or transferring it inside every layer.
4. Preserve the existing `cu_seqlens` tensor for Flash Attention.
5. Use precomputed host split sizes only for backends whose public PyTorch API requires Python integer split sizes.

Where possible, the host split sizes should be derived from the original CPU-side grid metadata before tensors are moved to the accelerator.

If callers supply only CUDA `cu_seqlens`, a compatibility fallback may require one explicit conversion before the block loop. That would still remove an `O(number_of_layers)` synchronization pattern, but I would document and benchmark that fallback rather than claiming it is fully device-native.

I would also evaluate a fully device-native non-Flash path, but only if it does not create worse trade-offs. For example, replacing segmented attention with a dense block-diagonal mask could eliminate Python splitting while increasing memory and compute quadratically. I would not adopt that as the default without benchmark evidence.

### Phase 3: tensorize multimodal RoPE construction

For `get_rope_index()`, I would remove Python conversions from GPU metadata where practical.

The proposed direction is:

* detect modality-run boundaries with tensor operations rather than `input_token_type.tolist()` and `itertools.groupby()`;
* keep `current_pos` as a device scalar tensor;
* replace Python `max(grid_thw[1], grid_thw[2])` with `torch.maximum(...)`;
* preserve image/video grid ordering deterministically;
* replace `torch.tensor(mrope_position_deltas, device=...)` with a device-preserving `torch.stack(...)`;
* avoid redundant device conversions.

I would verify the reported `.to(position_ids.device)` operation through profiling before changing it. If `llm_positions` and `position_ids` are already on the same device, that call should be a no-op rather than an actual H2D transfer. The optimization should target measured transfers, not merely remove calls that look suspicious.

The most delicate part is preserving the exact mapping between modality runs and entries consumed from `image_grid_thw` and `video_grid_thw`. I would keep explicit assertions around:

* number of detected image/video runs;
* number of available grid entries;
* generated position count matching the unmasked token count;
* output dtype, device, and shape.

### Regression tests

I would add correctness tests using a tiny randomly initialized Qwen3-VL model so no large checkpoint is required.

Coverage would include:

* text-only input;
* one image;
* multiple images;
* one video;
* interleaved text, image, and video runs;
* left- and right-padded batches;
* multiple packed vision sequences with unequal lengths;
* eager, SDPA, and Flash Attention where available;
* CPU and CUDA paths;
* compiled and uncompiled execution where supported.

For every case, the test would compare the optimized implementation against the current reference implementation for:

* position IDs;
* MRoPE deltas;
* attention segmentation;
* hidden states;
* final logits;
* gradients when training mode is supported.

The test should use exact equality for integer metadata and appropriate numerical tolerances for floating-point model outputs.

### Performance regression protection

I would avoid adding a fragile wall-clock assertion to normal CI.

Instead, I would add:

1. A deterministic benchmark script that reports before/after profiler data.
2. A CUDA-only targeted test, where reliable, verifying that the optimized hot path does not contain the known scalar-extraction or `.tolist()` synchronization operations.
3. A benchmark result in the PR description showing:

   * number of D2H/H2D events;
   * CPU and CUDA time;
   * peak memory;
   * speedup across several segment counts.

### Production safeguards

The implementation should:

* preserve the existing public API;
* avoid caching request-specific tensors on the module instance;
* remain safe when the same model serves concurrent requests;
* avoid global mutable state;
* validate malformed `cu_seqlens` early;
* handle zero-length or single-segment edge cases;
* avoid changing Flash Attention behavior;
* avoid increasing asymptotic memory usage for eager/SDPA;
* maintain device and dtype propagation.

### Acceptance criteria

I would consider the issue resolved when:

* `lengths.tolist()` is no longer executed once per vision layer;
* modality grouping does not require full CUDA-tensor conversion to a Python list;
* MRoPE deltas are constructed without a host round trip;
* outputs match the existing implementation;
* eager, SDPA, and Flash Attention tests pass;
* no material GPU-memory regression is introduced;
* profiler evidence shows the reported synchronization removed or reduced to one unavoidable setup operation;
* `torch.compile` behavior is unchanged or improved;
* CPU execution remains supported;
* changes are made in `modular_qwen3_vl.py` and the generated modeling file is updated through the repository tooling.

I can start by providing a profiler trace and a small patch that hoists the sequence-partition conversion outside the repeated layer loop. After that is validated, I can follow with the more involved device-native RoPE tensorization in the same PR if maintainers prefer, or keep it as a separate focused change.

Please let me know if I may take ownership of this optimization.

## [c5142736806] Mahnoor-Zaffar (NONE) · issue_comment · 2026-07-31
I've identified that the current tensor splitting implementation causes unnecessary data transfers between CPU and GPU, which slows down the model. I'm planning to refactor the tensor operations to eliminate these transfers and improve performance during the forward pass. Could you please assign this issue to me?

## [c5144794943] Rocketknight1 (MEMBER) · issue_comment · 2026-07-31
We don't assign issues, but if anyone wants to test it out and make a PR, feel free! I suspect a lot of this overhead will be removed when users compile the model though, so I'd like to see numbers there too before we merge anything!

## [c5146357603] Mahnoor-Zaffar (NONE) · issue_comment · 2026-07-31
This seems to be the simplest fix in my mind. Any concerns?

| # | Fix | Approach |
|---|-----|----------|
| 1 | `get_vision_cu_seqlens` in `vision_utils.py` | Force CPU computation. `cu_seqlens` is tiny (<100 elements), so this is trivial. Flash attention path does a cheap H2D (.to(device)) instead of the current D2H (.tolist()). |
| 2 | `get_vision_position_ids` method in `qwen2_vl` | Replace `.item()` with `int()` on CPU tensor, or accept pre-computed int args |
| 3 | `get_rope_index` in `qwen2_vl` | Replace `.tolist()` groupby with tensor `diff`/`nonzero`; accumulate deltas in pre-allocated tensor |
| 4 | Regenerate | `make fix-repo` to propagate |

This avoids O(N²) memory entirely, as it moves tiny tensor to CPU

## [c5146620722] NewUserHa (NONE) · issue_comment · 2026-07-31
The issue of no d2h/h2d is it creates `cudasynchronize`, it breaks `cuda stream` overlap makes cpu stop to wait gpu before send next kernels; and breaks `cuda graphs` in which it's disallowed; and breaks `torch.compile` into pieces of small graphs that downgrades speed.

The `cu_seqlens` and relates data that needs could have been staying on cpu instead of sent to gpu, these data is small and to provides info to cpu only about how to operate the tensor. The `cu_seqlens[1:] - cu_seqlens[:-1]`.`tolist()` were here is becase tensor.split() only support python list as the argument. 
I think it probably is the place at upper level where sent these arguments to the functions wrong

## [c5470196761] Marchematics (NONE) · issue_comment · 2026-08-30
Opened #48418 to compute vision attention split sizes once per forward and reuse them across layers.

## [c5972772660] NewUserHa (NONE) · issue_comment · 2026-10-03
still relevant

---
outcome: state=closed
