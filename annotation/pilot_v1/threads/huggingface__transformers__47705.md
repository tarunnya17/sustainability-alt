# #47705 [issue] Gemma-4-E4B: the PLE table holds ~5.6 GB of VRAM that can live on CPU bit-identically - is there a supported device_map path?
https://github.com/huggingface/transformers/issues/47705  ·  opened by MyceliaLabsBV on 2026-08-02

## [body] MyceliaLabsBV (NONE) · opening post · 2026-08-02
### Summary

On `google/gemma-4-E4B-it`, the per-layer-embedding table (`embed_tokens_per_layer`) accounts for the large majority of resident VRAM. Moving that single module to CPU after load cuts residency by more than half and is **bit-identical** — only the gathered rows travel to the GPU, so no arithmetic changes, only where the lookup runs.

This matters more than a usual memory tweak because Gemma-4-E4B is positioned as an on-device model, and the difference lands squarely on whether it fits a 16 GB card at useful context lengths.

### Measured

Single RTX A5000 16 GB (laptop), bnb-4bit NF4 with the PLE-sensitive modules kept in bf16:

| | PLE on GPU | PLE on CPU |
|---|---|---|
| resident after load | **10.11 GB** | **4.48 GB** |
| peak, 8.5k-token prefill | **OOM at 15.5 GB** | **11.84 GB** |
| generation throughput | baseline | slightly **faster** |

The throughput result was not what we expected — we assumed a penalty and measured a small gain, which we attribute to reduced allocator pressure rather than to anything clever.

**Correctness:** verified by greedy parity at 2287 tokens — identical output token-for-token. The lookup is an `index_select`; running it on CPU and moving the result up changes only the device of the gather. Transfer volume is roughly 20 MB per 1k tokens.

### The workaround we're using

```python
def offload_ple_table(model) -> None:
    """Move the PLE embedding table to CPU; gather on CPU, ship rows to GPU."""
    for name, mod in model.named_modules():
        if name.endswith("embed_tokens_per_layer"):
            mod.to("cpu")

            def _cpu_gather(input_ids, *a, _f=mod.forward, **kw):
                return _f(input_ids.to("cpu"), *a, **kw).to("cuda:0", non_blocking=True)

            mod.forward = _cpu_gather   # instance attr shadows Module.forward
            torch.cuda.empty_cache()
            return
```

Matched by name **suffix** rather than full path, so a PEFT wrapper doesn't hide it.

### The question

We first tried the supported route — passing a `device_map` that pins `embed_tokens_per_layer` to `"cpu"` — and it did not hold: the module appeared to be back on CUDA by the time init finished, and we OOMed. We worked around it post-load rather than chasing it, so we have **not** re-verified this carefully and are **not** reporting it as a bug.

So, as a question:

1. Is there a supported way to keep `embed_tokens_per_layer` on CPU through `from_pretrained` / `device_map`, and were we simply holding it wrong?
2. If not — would PLE-aware placement be welcome as a feature? For a model family whose whole point is on-device deployment, "the biggest tensor can sit in host RAM for free" seems like it should be a documented option rather than folklore.

Happy to open a PR if you can point at the right layer to do it in, and happy to run measurements on other E-sizes if useful.

### Environment

```
transformers  5.8.1
accelerate    1.13.0
torch         2.6.0+cu124
GPU           NVIDIA RTX A5000 Laptop, 16 GB
model         google/gemma-4-E4B-it
```

## [c5164686634] zucchini-nlp (MEMBER) · issue_comment · 2026-08-03
hey, you can definitely define a fine-grained device map and offload the PLE layer on cpu. In that case the device map should be a dict with entries per each module. See example below which offloads the 31st layer on CPU

```
device_map = {"model.layers.1": 0, "model.layers.14": 1, "model.layers.31": "cpu", "lm_head": "disk"}
```

As for making it the default way to load a model, I am not sure we want to do that. It's really interesting you got better speed with offloaded cpu, which is one of the reasons we might not want to offload by default 🤔

---
outcome: state=closed
