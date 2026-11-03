# #41543 [issue] Add VideoInpaintPipeline for temporally-consistent diffusion-based video inpainting
https://github.com/huggingface/transformers/issues/41543  ·  opened by Aki-07 on 2025-10-13

## [body] Aki-07 (CONTRIBUTOR) · opening post · 2025-10-13
### Feature request

Introduce a new pipeline to extend existing image inpainting capabilities (StableDiffusionInpaintPipeline) to videos. The goal is to provide a native, GPU-optimized API within Diffusers that performs temporally coherent video inpainting instead of independent per-frame processing.

### Motivation

Current video inpainting approaches in the community simply loop over frames and call the image inpainting pipeline repeatedly.
This leads to:
 
- Temporal flicker and inconsistent textures between frames.
- Poor GPU utilization and high memory overhead.
- Lack of tools to maintain motion coherence or reuse diffusion latents across time.

A built-in VideoInpaintPipeline would make it possible to remove objects, restore scenes, or creatively edit videos using diffusion models while keeping motion and lighting consistent across frames.

### Your contribution

I plan to:

- Implement VideoInpaintPipeline as a subclass of DiffusionPipeline, leveraging StableDiffusionInpaintPipeline under the hood.
- Add temporal consistency mechanisms, such as latent reuse between frames and optional optical-flow–guided warping (RAFT / GMFlow).
- Optimize performance through batched FP16 inference, scheduler noise reuse, and optional torch.compile acceleration.

Provide a clean user API compatible with existing pipelines:
```
from diffusers import VideoInpaintPipeline

pipe = VideoInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    use_optical_flow=True,
[… 11 lines of code omitted]
```
- Contribute documentation and tests demonstrating temporal coherence, performance benchmarks, and example notebooks for real-world use.

## [c3397743389] Rocketknight1 (MEMBER) · issue_comment · 2025-10-13
Could be interesting! Not sure who to ping - cc @molbap @yonigozlan @NielsRogge @sayakpaul. I'm happy to review the pipeline code, but I'd like someone more familiar with video diffusers to tell me if it's worth having a pipeline for this or if it's very specific/niche

## [c3397812669] yonigozlan (CONTRIBUTOR) · issue_comment · reply to [body] · 2025-10-13
Hey @Aki-07! Please open an issue on diffusers ;)

## [c3397887480] Aki-07 (CONTRIBUTOR) · issue_comment · reply to [c3397812669] · 2025-10-13
Sure @yonigozlan, updated :)
https://github.com/huggingface/diffusers/issues/12477

## [c3414086005] ariG23498 (CONTRIBUTOR) · issue_comment · reply to [c3397887480] · 2025-10-17
Hey @Aki-07 would it be okay if you closed this issue as it is already tracked in diffusers?

## [c3414092482] Aki-07 (CONTRIBUTOR) · issue_comment · 2025-10-17
Sure, Linking the diffusers issue here
[Add VideoInpaintPipeline for temporally-consistent diffusion-based video inpainting](https://github.com/huggingface/diffusers/issues/12477)

---
outcome: state=closed
