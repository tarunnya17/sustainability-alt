# #45931 [pull_request] Add TurboQuant KV cache backend
https://github.com/huggingface/transformers/pull/45931  ·  opened by Yeuvoir on 2026-05-13

## [body] Yeuvoir (NONE) · opening post · 2026-05-13
# What does this PR do?

This PR adds a Transformers-native `TurboQuantCache` backend for quantized KV cache generation.

TurboQuant is an online vector quantization method for KV-cache compression. It normalizes KV vectors, applies a data-oblivious random rotation, quantizes rotated coordinates with Lloyd-Max scalar codebooks, and optionally applies a 1-bit QJL residual correction for inner-product preservation. The TurboQuant paper reports near-optimal distortion rates and strong KV-cache results, including quality neutrality around 3.5 bits per channel and marginal degradation around 2.5 bits per channel.

KV cache memory is one of the main memory constraints for long-context generation. A cache-level TurboQuant backend lets users reduce KV-cache memory while staying inside the standard `generate()` / `Cache` API.

vLLM has also recently added TurboQuant KV-cache quantization support through `--kv-cache-dtype turboquant_*` variants. This PR brings the same class of KV-cache compression to Transformers, exposed through the existing quantized cache generation path:

```python
cache_implementation="quantized"
cache_config={"backend": "turboquant"}
```

This PR adds:

- `TurboQuantQuantizedLayer`
- `TurboQuantCache`
- `backend="turboquant"` support in `QuantizedCache`
- generation integration for `cache_config={"backend": "turboquant"}`
- PyTorch dummy exports
- cache utility tests and a tiny end-to-end generation smoke test

References:

- TurboQuant paper: https://huggingface.co/papers/2504.19874
- vLLM TurboQuant docs: https://docs.vllm.ai/en/stable/api/vllm/model_executor/layers/quantization/turboquant/
- vLLM TurboQuant blog: https://vllm.ai/blog/turboquant

# Validation

Unit and style checks:

```bash
.venv/bin/python -m pytest tests/utils/test_cache_utils.py::CacheTest -q
# 5 passed

.venv/bin/python -m ruff check src/transformers/__init__.py src/transformers/cache_utils.py src/transformers/generation/utils.py src/transformers/utils/dummy_pt_objects.py tests/utils/test_cache_utils.py
# All checks passed

.venv/bin/python -m ruff format --check src/transformers/__init__.py src/transformers/cache_utils.py src/transformers/generation/utils.py src/transformers/utils/dummy_pt_objects.py tests/utils/test_cache_utils.py
# 5 files already formatted

.venv/bin/python utils/check_dummies.py
git diff --check
```

## End-to-end Transformers generation

End-to-end generation was validated with locally cached model weights on an NVIDIA H100 GPU.

Long-context test on `Qwen/Qwen3-8B`:

| cache | input tokens | generated tokens | final cache | allocation delta | peak delta | time |
|---|---:|---:|---:|---:|---:|---:|
| `DynamicCache` | 4096 | 32 | 580.36 MiB | 612.39 MiB | 1026.06 MiB | 2.87s |
| `TurboQuantCache`, nbits=3 | 4096 | 32 | 123.61 MiB | 123.64 MiB | 537.31 MiB | 4.32s |
| `TurboQuantCache`, nbits=2 | 4096 | 32 | 87.61 MiB | 87.64 MiB | 501.31 MiB | 3.25s |

Compared with `DynamicCache`:

- `TurboQuantCache`, nbits=3: 78.70% final cache reduction, 47.63% peak allocation reduction
- `TurboQuantCache`, nbits=2: 84.90% final cache reduction, 51.14% peak allocation reduction

Additional end-to-end smoke tests with `TurboQuantCache`, nbits=3:

| model | input tokens | generated tokens | DynamicCache final cache | TurboQuantCache final cache | cache reduction | Dynamic peak delta | TurboQuant peak delta | peak reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen3-0.6B` | 1024 | 16 | 113.64 MiB | 27.45 MiB | 75.84% | 170.52 MiB | 104.08 MiB | 38.96% |
| `Qwen/Qwen2.5-0.5B-Instruct` | 1024 | 16 | 12.18 MiB | 3.46 MiB | 71.59% | 48.77 MiB | 39.05 MiB | 19.93% |
| `Qwen/Qwen3-1.7B` | 1024 | 16 | 113.64 MiB | 27.45 MiB | 75.84% | 164.52 MiB | 110.08 MiB | 33.09% |
| `unsloth/Llama-3.2-3B-Instruct` | 1024 | 16 | 113.64 MiB | 27.45 MiB | 75.84% | 184.52 MiB | 118.08 MiB | 36.01% |

Also verified the default generation path on `Qwen/Qwen3-8B`:

```python
cache_implementation="quantized"
cache_config={"backend": "turboquant"}
```

with 512 input tokens and 8 generated tokens. The returned cache type was `TurboQuantCache`.

## kvpress validation

I also validated the TurboQuant behavior in the local `kvpress` evaluation framework, using the existing TurboQuant press path to compare against an uncompressed baseline and reproduce the expected TurboQuant paper trend.

Setup:

- model: `unsloth/Llama-3.2-3B-Instruct`
- benchmark: LongBench `hotpotqa`
- samples: 40
- compression setting: `fraction=0.100`, `key_channel_cr=0.25`
- compared methods:
  - no compression
  - TurboQuant b8
  - TurboQuant b4
  - TurboQuant b3

Results:

| model | task | method | score | compression ratio |
|---|---|---:|---:|---:|
| `unsloth/Llama-3.2-3B-Instruct` | `hotpotqa` | no press | 60.83 | 0.00 |
| `unsloth/Llama-3.2-3B-Instruct` | `hotpotqa` | TurboQuant b8 | 63.89 | 0.50 |
| `unsloth/Llama-3.2-3B-Instruct` | `hotpotqa` | TurboQuant b4 | 58.06 | 0.75 |
| `unsloth/Llama-3.2-3B-Instruct` | `hotpotqa` | TurboQuant b3 | 47.86 | 0.81 |

These local kvpress results reproduce the expected TurboQuant paper behavior: higher-bit TurboQuant preserves LongBench quality close to the uncompressed baseline while substantially reducing KV cache storage, and lower-bit settings trade more quality for higher compression.

## [c4459884643] Yeuvoir (NONE) · issue_comment · 2026-05-15
@ArthurZucker
Hi maintainers,
This PR adds a TurboQuant KV cache backend following the existing cache interface in Transformers.

I’ve tested the implementation locally.

Would appreciate a review when you have time. Thanks!

## [c4460707207] Rocketknight1 (MEMBER) · issue_comment · 2026-05-15
Sorry, this just looks like code agent output! We'd really prefer if people opened an issue to check if there was interest in a feature before dropping 500 lines of Claude on us and asking us to review it

---
outcome: state=closed, merged=False
