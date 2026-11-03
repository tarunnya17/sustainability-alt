# #43278 [issue] Embedding layer dtype changed from BF16 in training to FP32 in evaluate.
https://github.com/huggingface/transformers/issues/43278  ·  opened by TriLoo on 2026-01-14

## [body] TriLoo (NONE) · opening post · 2026-01-14
### System Info

- `transformers` version: 4.57.1
- Platform: Linux-5.15.0-1071-azure-x86_64-with-glibc2.35
- Python version: 3.12.12
- Huggingface_hub version: 0.36.0
- Safetensors version: 0.7.0
- Accelerate version: 1.12.0
- Accelerate config:    not found
- DeepSpeed version: 0.18.4
- PyTorch version (accelerator?): 2.7.1+cu126 (CUDA)
- Tensorflow version (GPU?): not installed (NA)
- Flax version (CPU?/GPU?/TPU?): not installed (NA)
- Jax version: not installed
- JaxLib version: not installed
- Using distributed or parallel set-up in script?: <fill in>
- Using GPU in script?: <fill in>
- GPU type: NVIDIA H100 80GB HBM3

### Who can help?

I’m currently training a text translation task using Qwen3-0.6B on 8 × 8 (64) H100 GPUs. I’m using Seq2SeqTrainer with --fsdp "hybrid_shard_zero2 auto_wrap" for training.
However, during training I noticed something odd: the embedding layer’s dtype looks correct—bf16, as I set it. But during evaluation, the embedding output dtype becomes fp32. Then after the computation goes through the Transformer layers, the dtype returns to the expected bf16. What could be causing this?
There are also two other abnormal behaviors:

After evaluation finishes (evaluation takes around 4 hours, which seems unusually slow), the training loss spikes sharply, which also doesn’t look normal.
Regarding evaluation time: evaluation is fast on a single node with 8 GPUs, but when I scale to 8 nodes, evaluation time increases to over 4 hours.

Do you have any suggestions on what might be going on and how to troubleshoot or improve this?

[image]

### Information

- [x] The official example scripts
- [ ] My own modified scripts

### Tasks

- [x] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [ ] My own task or dataset (give details below)

### Reproduction

To reproduce the issue, you may try training Qwen3-0.6B with the translation example provided in the examples/translation directory.

### Expected behavior

The training loss should be smooth, and the evaluation time shouldn’t get significantly slower.

## [c3749361461] Rocketknight1 (MEMBER) · issue_comment · 2026-01-14
cc @SunMarc maybe? Not sure if this is a bug or just a data/script issue

## [c3759594730] karthik-0306 (NONE) · issue_comment · 2026-01-16
While experimenting with a fix/test for this issue, I ran into a separate CI failure in generation tests.

GlmImageModel.get_rope_index() can raise an IndexError during prompt-lookup (assisted) decoding when attention_mask and input_ids have different sequence lengths. This appears to be a latent model-specific issue and seems unrelated to the bf16/fp32 train vs eval behavior discussed here.

If this turns out to be something maintainers want addressed, I’d be happy to look into a small, focused fix.

## [c3760709748] SunMarc (MEMBER) · issue_comment · reply to [body] · 2026-01-16
Hey @TriLoo, can you share the exact command you used so that we can easily reproduce ? Also, are you using mixed precision or not ?

## [c5669860360] Roy3838 (NONE) · issue_comment · 2026-09-14
Hey! That loss spike looks weird! If you want to monitor the training run and get a notification when the eval is finished, I built a free/FOSS tool for that: https://github.com/Roy3838/Observer 

It's a tiny local agent that watches your terminal and pings or calls you the moment the eval actually finishes/spikes, so you're not watching logs scroll for hours per attempt. 
Runs locally, nothing leaves your machine. Happy to answer anything :D

---
outcome: state=closed
