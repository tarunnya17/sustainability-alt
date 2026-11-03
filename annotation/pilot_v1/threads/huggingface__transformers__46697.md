# #46697 [issue] Swin2SR image processor over-pads images when dimensions are already divisible by size_divisor
https://github.com/huggingface/transformers/issues/46697  ·  opened by arnavkewalram on 2026-06-16

## [body] arnavkewalram (CONTRIBUTOR) · opening post · 2026-06-16
## System Info

```
- `transformers` version: 4.57.6
- Platform: macOS-26.5.1-arm64-arm-64bit
- Python version: 3.9.6
- PyTorch version: 2.8.0
- CUDA available: False
```

## Who can help?

@yonigozlan @molbap (vision models)

## Information

- [ ] The official example scripts
- [x] My own modified scripts

## Tasks

- [ ] An officially supported task in the `examples` folder
- [x] My own task or dataset (see details below)

## Reproduction

```python
from transformers import Swin2SRImageProcessor
from PIL import Image
import numpy as np

processor = Swin2SRImageProcessor()

# 1024 is divisible by 8 (the default size_divisor), so no padding should occur
img = Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8))
out = processor(img, return_tensors="pt")
print(out.pixel_values.shape)  # torch.Size([1, 3, 1032, 1032]) — should be [1, 3, 1024, 1024]
```

The root cause is in `image_processing_swin2sr.py` (lines 75–76) and `image_processing_pil_swin2sr.py` (lines 76–77):

```python
# current (wrong) — always pads by a full size_divisor even when already aligned
pad_height = (height // size_divisor + 1) * size_divisor - height
pad_width  = (width  // size_divisor + 1) * size_divisor - width
```

The `+ 1` forces an extra `size_divisor` of padding even when the dimension is already divisible. The model itself (`modeling_swin2sr.py`, line 747) uses the correct formula:

```python
modulo_pad_height = (window_size - height % window_size) % window_size
```

The test file uses the same wrong formula for expected values, so the tests pass silently.

## Expected behavior

For an image whose dimensions are already divisible by `size_divisor`, the processor should apply zero padding. For a 1024×1024 image with the default `size_divisor=8`, `out.pixel_values.shape` should be `torch.Size([1, 3, 1024, 1024])`, not `torch.Size([1, 3, 1032, 1032])`.

## [c4732073883] nightcityblade (NONE) · issue_comment · 2026-06-17
Hi, I'd like to work on this. I'll submit a focused PR with the padding fix and regression tests.

---
outcome: state=closed
