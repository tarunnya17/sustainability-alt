# #47225 [issue] orthogonal_ weight init crashes with `geqrf_cuda not implemented for BFloat16` under v5 bf16 on-device init (RWKV and any QR-based init)
https://github.com/huggingface/transformers/issues/47225  ·  opened by janbernloehr on 2026-07-09

## [body] janbernloehr (CONTRIBUTOR) · opening post · 2026-07-09
### System Info

`transformers env` could not be run in the CI artifact context; closest captured environment:
- transformers 5.x (v5 on-device init path; `torch_dtype is deprecated` warning present)
- torch 2.14.0a0 (nightly)
- Python 3.12
- CUDA 13.x
- GPU: NVIDIA H100 (cc 9.0); also reproduced on B200 and GB200 — independent of GPU architecture
- Platform: Linux x86_64 (and arm64 on GB200)

### Who can help?

Core modeling / weight-initialization maintainers (new transformers/initialization.py wrappers and modeling_utils.initialize_weights/smart_apply) and RWKV model maintainers.

### Information
- [x] The official example scripts
- [ ] My own modified scripts

### Tasks
- [x] An officially supported task (model construction via from_config)
- [ ] My own task or dataset

### Reproduction

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("RWKV/rwkv-4-169m-pile")

# transformers v5 materializes weights in the requested compute dtype on-device
# before running _init_weights, so orthogonal_ (QR) runs on a bf16 CUDA tensor.
with torch.device("cuda"):
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
```

### Expected behavior

Constructing a model with dtype=torch.bfloat16 on CUDA should succeed. Under transformers 4.x, orthogonal_ init ran in fp32 so torch.linalg.qr never hit the unsupported geqrf_cuda bf16 path. Since v5 now materializes weights in the compute dtype on-device before _init_weights, orthogonal/QR-based initialization should remain dtype-safe.

### Actual behavior

Model construction aborts immediately (before any forward/training step — at ~0.17 GB GPU memory) with an uncaught `NotImplementedError`. The failure is deterministic: it happens on every run and reproduces identically on H100, B200, and GB200, and for both eager and torch.compile paths (the crash is in weight init, before compilation). Full traceback:

```
File ".../transformers/models/rwkv/modeling_rwkv.py", line 502, in __init__
    self.post_init()
File ".../transformers/modeling_utils.py", line 1448, in post_init
    self.init_weights()
[… 13 lines of code omitted]

Note: this is distinct from the CPU-only low-precision NaN init discussed in issue #46002 and the closed PR #46014 — that fallback guarded on `tensor.device.type == "cpu"` and does not address the CUDA `geqrf_cuda` failure shown here.

### Proposed fix

In transformers/initialization.py::orthogonal_, when the target tensor is a low-precision floating dtype (bf16/fp16), perform the orthogonal QR in float32 and copy the result back (device-agnostic — must also cover CUDA, not just CPU):

```python
def orthogonal_(tensor, gain=1, generator=None):
    if not getattr(tensor, "_is_hf_initialized", False):
        if tensor.is_floating_point() and tensor.dtype not in (torch.float32, torch.float64):
            tmp = torch.empty_like(tensor, dtype=torch.float32)
            TORCH_INIT_FUNCTIONS["orthogonal_"](tmp, gain=gain, generator=generator)
            with torch.no_grad():
                tensor.copy_(tmp)
            return tensor
        return TORCH_INIT_FUNCTIONS["orthogonal_"](tensor, gain=gain, generator=generator)
    return tensor
```

Validated experimentally: with the patch applied, the RWKV model above constructs successfully in bf16 on H100 and trains for 100 steps with a smoothly decreasing loss, with no change to other models' behavior.

_This issue was drafted with assistance from the `opus` AI model._

---
outcome: state=closed
