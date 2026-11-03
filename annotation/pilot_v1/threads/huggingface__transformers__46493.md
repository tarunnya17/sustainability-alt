# #46493 [issue] difference in performance of sam3 video model compared to original sam3 implementation
https://github.com/huggingface/transformers/issues/46493  ·  opened by iariav on 2026-06-08

## [body] iariav (NONE) · opening post · 2026-06-08
Hi,
i ran the original sam3 video model (from meta) and the transformers sam3 video model on the same video with the same text prompt. 
I noticed that the original sam3 model detections were more accurate, and even that in the transformers' version some frames didn't have detections at all (whereas the original sam3 did correctly detect the object). 
i attached an example frame.

[image]
[image]

i also noticed that the transformers model takes less GPU memory, and runs faster, so maybe it is initialized with different hyper-parameters compared to the original?

bottom line, i expected the results to be identical. any reason for such a difference? i use the same code for video loading\processing in both cases.

help would be greatly appriciated,.

## [c4648048130] zucchini-nlp (MEMBER) · issue_comment · 2026-06-08
@molbap @guarin

## [c4648969731] guarin (MEMBER) · issue_comment · 2026-06-08
Hi, thanks for reporting the issue! Do you mind sharing a reproducible example including the visualisation code? ? Ideally with an image where the detection fails or is wrong.

## [c4650437159] guarin (MEMBER) · issue_comment · 2026-06-08
Couldn't reproduce the issue on my side. Prompt was "vehicle". Note that the bounding box format after post processing is `(x1, y1, x2, y2)` in absolute pixel coordinates.

[image]
[image]

## [c4651721824] iariav (NONE) · issue_comment · reply to [c4650437159] · 2026-06-08
@guarin 
Thank you for your prompt response and check!
Perhaps it wasn't totally clear from my post - i am referring to the video model. i am assuming you checked the attached images using the image model?
I attach both scripts - using the original sam3 and the transformers sam3. I had to do minor modifications to the visualization function due to differences in the models' outputs (it's all there). I would be very happy to discover that the problem is with my code and not with the model itself :)
unfortunately, i cannot share the video itself. if it's relevant, the video has ~100 frames and the attached images are from frame ~65.
thanks again

[sam3_video_predictor_example_transformers.py](https://github.com/user-attachments/files/28720386/sam3_video_predictor_example_transformers.py)
[sam3_video_predictor_example.py](https://github.com/user-attachments/files/28720387/sam3_video_predictor_example.py)

## [c4657000278] iariav (NONE) · issue_comment · 2026-06-09
i think maybe i figured it out - 
just noticed that i was receiving this warning - 
"[transformers] kernels library is not installed. NMS post-processing, hole filling, and sprinkle removal will be skipped. Install it with `pip install kernels` for better mask quality."

i installed the kernels library as suggested and it does look better now. i still get a couple of frames with a "floating" small mask blob that causes the bbox to be inflated, but 99% of the frames are fine. any way to improve this further?

## [c4657415387] guarin (MEMBER) · issue_comment · 2026-06-09
Good to hear that the detections improved. I tested it with the video model by converting the frames into videos. Does the floating small mask problem also happen with the original model or is this transformers only? If it happens for transformers only could you share some examples? Helps a lot with debugging :)

## [c4659217677] iariav (NONE) · issue_comment · 2026-06-09
I didn't notice it in the original model. all the bbox were "tight" arounf the object (vehicle).

I attach one of the frames in which i notice the problem (the clean frame, the result of the original model, and the result of the transformers model). I plotted the masks as well this time, and you can notice in the transformers image the small blob in the leftmost part of the image that causes the bbox to inflate.

[image]
[image]
[image]

p.s
pardon if the images are not EXACTLY the same size, i had to crop these from the original image to remove parts of the frame that i couldn't upload. The models were both ran on the full res images.

## [c4659880109] guarin (MEMBER) · issue_comment · 2026-06-09
Hmm I couldn't reproduce on CPU or CUDA (H100, cuda 12.8) with your code. The transformers model always outputs a tight bounding box around the car even without the `cv_utils` kernel. Do you see a warning when running the code on your side? Something like this:
```
[transformers] Failed to load cv_utils kernel (your torch/cuda setup may not be supported): Cannot install kernel from repo kernels-community/cv-utils (revision: main). NMS post-processing, hole filling, and sprinkle removal will be skipped.
```

As you mentioned before, some post-processing steps require the `cv_utils` kernel. Even if you have installed `kernels` it might not be enough to actually load the kernel. If the kernel is not installed, the mask cleanup step [here](https://github.com/huggingface/transformers/blob/5047f086332cb3c05bf46ad4ca93ec2f73946164/src/transformers/models/sam3_video/modeling_sam3_video.py#L1973-L1978) is skipped which seems to be the case in your example.

I'll check if we can add fallbacks in case the kernel is missing or couldn't be loaded.

## [c4667279611] iariav (NONE) · issue_comment · 2026-06-10
Hi, just ran the code in debug - cv_utils_kernel does get loaded OK and the the run enters the "remove_sprinkles" and "fill_holes" parts.. 
i don't get any warning like the one you mentioned

## [c4668354284] guarin (MEMBER) · issue_comment · 2026-06-10
Ok if the remove sprinkles suppression runs as expected then it is probably rather due to the model outputs. Some things you could try:

1) Remove the `model.to(bfloat16)`conversion in the transformers script. The default in the original code is float32. If you would like to run with bfloat16 I suggest you use autocast instead:
```
accelerator = Accelerator(mixed_precision="bf16")
...
with accelerator.autocast():
     for model_outputs in model.propagate_in_video_iterator(...):
        ...
```
This should already improve model outputs

2) The `propagation_direction` is set to `both` in the original script. This setting doesn't exist in the transformers version. What you have to do instead is run it once with `reverse=False` (the default, corresponds to `propagation_direction="forward"`) and once with `reverse=True`:
```
# forward
for model_outputs in model.propagate_in_video_iterator(
                inference_session=inference_session,
                max_frame_num_to_track=len(pil_image_list),
                reverse=False,
[… 10 lines of code omitted]
```

And finally, which hardware, torch, and cuda versions are you running on? As mentioned above I did get good predictions from the provided images so it might also be a hardware or library version issue.

## [c4669836483] iariav (NONE) · issue_comment · 2026-06-10
thank a lot. it tried your suggestions:
1. if i run with the 
```
accelerator = Accelerator(mixed_precision="bf16")
...
with accelerator.autocast():
     for model_outputs in model.propagate_in_video_iterator(...):
        ...
```
i still get inflated bbox in some frames. 
2. when i run without autocast completely (and after removing the model.to(bfloat16), meaning i run on full float32) all the bbox are tight and i don't get any unexpected mask sprinkles. 
 
so apparently, it happens due to some sort of numerical instability

bf16:
[image]

float32:
[image]

---
outcome: state=closed
