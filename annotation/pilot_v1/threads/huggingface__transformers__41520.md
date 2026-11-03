# #41520 [issue] Add GPU-Accelerated Video Decoding for VideoMAE, TimeSformer, and Similar Models
https://github.com/huggingface/transformers/issues/41520  ·  opened by Aki-07 on 2025-10-11

## [body] Aki-07 (CONTRIBUTOR) · opening post · 2025-10-11
### Feature request

I’d like to propose adding a lightweight, GPU-friendly video decoding utility to Transformers.
Right now, models like VideoMAE and TimeSformer depend on PIL or OpenCV for frame extraction, which keeps preprocessing on the CPU and slows down training and inference. A unified VideoReader module that wraps libraries such as PyAV or Decord could decode and batch frames directly as CUDA tensors, eliminating a major bottleneck in current video workflows and making it easier for users to stream large clips efficiently into Transformer-based video and multimodal models.

### Motivation

Most video pipelines in Transformers still rely on CPU-bound decoding through PIL or OpenCV.
For large-batch or long-clip training, this quickly becomes the main bottleneck — GPU utilization stays low while the CPU decodes frames one by one.

Libraries like PyAV and Decord already provide fast, batched video loading that can hand off tensors directly to the GPU.
Integrating this capability natively would make training and inference with models such as VideoMAE, TimeSformer, or InternVideo far more efficient.

### Your contribution

Love working in this space, if given an green sign
Will start working on it

## [c3393249391] zucchini-nlp (MEMBER) · issue_comment · reply to [body] · 2025-10-11
hey @Aki-07 👋🏻 

We have a video processor class which handles video decoding with `torchcodec` (or `torchvision` but it is deprecated). The decoding in `torchcodec` can be done on GPU and in our experiments it was faster than decord/pyav for most use-cases. Unfortunately not all models have their own video processor, and still use an `image_processor` class by treating videos as a set of PIL images

If you want to contribute, feel free to add a video processor for the models of your interest following [Llava-next-video](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_next_video/video_processing_llava_next_video.py). The processing code has to be rewritten if the model has non-standard treatment in `self._preprocess()` method. See Qwen-VL video processors for that

And lmk if you decide to work on any of the models. I suggest to take one model for start, merge it to `main` and then take another model if you wish to continue 🤗

## [c3393448147] Aki-07 (CONTRIBUTOR) · issue_comment · reply to [c3393249391] · 2025-10-11
Thank you so much @zucchini-nlp  🙏

I'd love to start contributing! Quick question: which model would you recommend I start with? 

I see VideoMAE, TimeSformer, Vivit, and LLaVA-Onevision don't have video processors yet. Since you mentioned starting with one model and learning from LLaVA-NeXT-Video, which one would be:
1. Most useful for the community
2. Good complexity for a first contribution

Happy to take whatever you think is best! I'm ready to start today.

Thanks again for the opportunity

## [c3394067144] zucchini-nlp (MEMBER) · issue_comment · 2025-10-12
LLaVA-Onevision already has a video processor tbh so I think you can work on Vivit or VideoMAE as a first contribution. Please tag me under the PR if you need a review or have questions

## [c3394077134] Aki-07 (CONTRIBUTOR) · issue_comment · reply to [c3394067144] · 2025-10-12
thanks for the quick reply @zucchini-nlp, will raise a PR for VideoMAE in sometime

---
outcome: state=closed
