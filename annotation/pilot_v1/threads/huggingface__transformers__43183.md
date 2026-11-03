# #43183 [issue] Regression in Qwen2_5_VLProcessor video kwargs routing (4.56.2 -> 4.57.1+): videos_kwargs{min_pixels,max_pixels} ignored
https://github.com/huggingface/transformers/issues/43183  ·  opened by sidezrw on 2026-01-08

## [body] sidezrw (NONE) · opening post · 2026-01-08
### System Info

Transformers versions: 4.56.2 (works), 4.57.1 (regression), 4.57.3 (still regressed)
numpy==2.2.6
pillow==12.1.0
torch==2.8.0+cu128
torchvision==0.23.0

CUDA 12.8
Python 3.10.12

CPU: Intel(R) Xeon(R) Platinum 8480C
GPU: NVIDIA H100 80GB HBM3

Model / processor: nvidia/cosmos-reason1-7b via AutoProcessor -> Qwen2_5_VLProcessor (video input)
Input: 1 video as np.uint8 array with shape T=10, H=1080, W=1920, C=3
Constraints used in repro: min_pixels=100352, max_pixels=151200

### Who can help?

@zucchini-nlp 

### Information

- [ ] The official example scripts
- [x] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [x] My own task or dataset (give details below)

### Reproduction

1) Enter python environment with specific transformers version
2) Run the repro script:
```bash
python test_qwen2_5_vl_kwargs_routing.py --model nvidia/cosmos-reason1-7b
```
2) Observe the printed `video_grid_thw` for these cases:
  A) `videos_kwargs={"min_pixels": min_pixels, "max_pixels": max_pixels}`
  B) `videos_kwargs={"size": {"shortest_edge": min_pixels, "longest_edge": max_pixels}}`
  C) `top-level size={"shortest_edge": min_pixels, "longest_edge": max_pixels}`

[test_qwen2_5_vl_kwargs_routing.py](https://github.com/user-attachments/files/24510556/test_qwen2_5_vl_kwargs_routing.py)

### Expected behavior

`videos_kwargs={"min_pixels":..., "max_pixels":...}` should continue to constrain video resizing (or raise a clear error / emit a deprecation warning if the API changed), because it works in transformers==4.56.2.

Concretely, for the same input and pixel budget, case A should produce a downsized grid similar to `video_grid_thw=[[5, 20, 36]]` (small patch/token count), not the large grid.

### What actually happens (summary of the issue)
In 4.56.2, case A downsizes as expected, but cases B/C do not.

In 4.57.1+, this flips: case A no longer downsizes (appears ignored), while cases B/C downsize.
This silent routing change causes a large increase in `video_grid_thw` (e.g., `[[5,20,36]]` -> `[[5,78,138]]`), which is ~15x more patches/visual tokens and leads to major performance regressions in downstream serving (e.g., vLLM throughput drop).

### API naming/semantics confusion
For Qwen2‑VL video resizing, the code path uses `size["shortest_edge"]` and `size["longest_edge"]` as pixel-budget bounds passed into `smart_resize(min_pixels=..., max_pixels=...)` (i.e., total pixel count constraints), not literal edge lengths in pixels. This makes it non-obvious that `size={"shortest_edge": min_pixels, "longest_edge": max_pixels}` is effectively the same as `{min_pixels, max_pixels}`, and it’s easy for users to interpret these keys as geometric edge lengths instead of area bounds. Is this intended?

## [c3728589956] zucchini-nlp (MEMBER) · issue_comment · reply to [body] · 2026-01-09
Hey @sidezrw ,

IIUC you want to be able to pass `min/max_pixels` as kwargs and be sure that they take effect at processing. Since that was supported, we can add a similar workaround to [this code](https://github.com/huggingface/transformers/blob/e5a860837f8a4f577b93e35677584906860bb6eb/src/transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py#L87-L108) in video processor. It will use either `min/max pixels` or `size`, whichever is present in kwargs

We can patch this in `4.57.4` along with other PRs. Do you want to submit a PR?

---
outcome: state=closed
