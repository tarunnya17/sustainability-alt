# #42947 [issue] Gradient Checkpointing Ineffective with PEFT LoRA Despite Proper Configuration
https://github.com/huggingface/transformers/issues/42947  ·  opened by yurkoff-mv on 2025-12-18

## [body] yurkoff-mv (NONE) · opening post · 2025-12-18
### System Info

- `transformers` version: 4.57.1
- Platform: Linux-5.15.0-161-generic-x86_64-with-glibc2.35
- Python version: 3.12.12
- Huggingface_hub version: 0.36.0
- Safetensors version: 0.7.0
- Accelerate version: 1.11.0
- Accelerate config: 	not found
- DeepSpeed version: not installed
- PyTorch version (accelerator?): 2.9.0+cu126 (CUDA)
- Tensorflow version (GPU?): not installed (NA)
- Flax version (CPU?/GPU?/TPU?): not installed (NA)
- Jax version: not installed
- JaxLib version: not installed
- Using distributed or parallel set-up in script?: <fill in>
- Using GPU in script?: <fill in>
- GPU type: NVIDIA L40

### Who can help?

@ArthurZucker @Cyrilvallez @SunMarc

### Information

- [x] The official example scripts
- [ ] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [ ] My own task or dataset (give details below)

### Reproduction

**Description:**
Gradient checkpointing fails to reduce memory usage when using PEFT LoRA adapters, even with correct configuration (use_cache=False, proper call order). The model reports is_gradient_checkpointing=True but shows identical memory consumption to non-checkpointed training.

**Detailed Analysis:**
1. The issue occurs specifically with PEFT LoRA fine-tuning
2. `use_cache=False` is set in model configuration
3. The model reports `is_gradient_checkpointing=True` but memory usage doesn't change
4. Without LoRA, gradient checkpointing works as expected
5. Tested with Llama 3 8B, but likely affects other models

**Minimal Reproducible Example:**
```python
import torch
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

[… 104 lines of code omitted]
```

**Results:**
(`model_name = "meta-llama/Meta-Llama-3-8B-Instruct"`)
```
print_header('Test 1: WITHOUT GRADIENT CHECKPOINTING')
test_memory_usage(
    model_name=model_name,
    use_gc=False,
    use_lora=True,
[… 21 lines of code omitted]
```

```
print_header('Test 2: WITH GRADIENT CHECKPOINTING')
test_memory_usage(
    model_name=model_name,
    use_gc=True,
[… 23 lines of code omitted]

### Expected behavior

1. Significant memory reduction with gradient checkpointing (40-50% typical)
2. Lower peak memory during backward pass

## [c3675232659] SunMarc (MEMBER) · issue_comment · 2025-12-19
cc @BenjaminBossan

## [c3675606216] BenjaminBossan (MEMBER) · issue_comment · reply to [body] · 2025-12-19
Hey @yurkoff-mv for PEFT usage, we need to enable gradients on the input embeddings (`model.enable_input_require_grads()`). This can be done via `model.prepare_model_for_gradient_checkpointing(model)` or `prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)` for quantized models. Yes, ideally, we should detect this automatically and make the call unnecessary, we'll check how best to do that.

Note that `model.gradient_checkpointing_enable()` should be called after `get_peft_model`. In my local testing with [this benchmark](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA) and LoRA with rank 32, this addition would reduce memory usage from 19 GB peak reserved memory to 12 GB, while speed would drop from 7200 to 5300 tokens per second. So this is exactly what we'd expect.

In your example, adding this actually makes no difference. I'm not sure why, there is a lot happening there, maybe it's not measuring the right thing. Could you please test my suggestion on your original problem and report back if you see the same effect as I did?

## [c3681235336] yurkoff-mv (NONE) · issue_comment · 2025-12-22
Hi!
Thanks for your reply.
I've made the suggested changes to the code:
```
    if use_lora:
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()
```
I got the following results:
```
╔══════════════════════════════════════════════════════════════════════╗
║                Test 1: WITHOUT GRADIENT CHECKPOINTING                ║
╚══════════════════════════════════════════════════════════════════════╝
trainable params: 335,544,320 || all params: 8,365,805,568 || trainable%: 4.0109
║    After loading model: 16.24 GB                                     ║
[… 13 lines of code omitted]
```

```
╔══════════════════════════════════════════════════════════════════════╗
║                 Test 2: WITH GRADIENT CHECKPOINTING                  ║
╚══════════════════════════════════════════════════════════════════════╝
Loading checkpoint shards: 100%
[… 18 lines of code omitted]

The results are quite strange.
I started running these experiments when I started fine-tuning this model (Meta-Llama-3-8B-Instruct) on a context of 32,768 tokens. Back then, using FSDP and a mini-batch size of 1, I got a GPU load of 44 GB with [Activation Checkpointing enabled](https://github.com/pytorch/pytorch/issues/159359#issuecomment-3664894659).

[image]

I found this very strange, and I tried to theoretically calculate the memory required to train such a model. I'm attaching a file with my conclusions:

[GPU RAM Estimation for seq length 1024.md](https://github.com/user-attachments/files/24288669/GPU.RAM.Estimation.for.seq.length.1024.md)

[GPU RAM Estimation for seq length 32768.md](https://github.com/user-attachments/files/24288716/GPU.RAM.Estimation.for.seq.length.32768.md)

## [c3686616385] BenjaminBossan (MEMBER) · issue_comment · reply to [c3681235336] · 2025-12-23
Thanks for running the tests. What you find is strange and different from what I get. Here are my results:

```
╔══════════════════════════════════════════════════════════════════════╗
║   GPU Memory Usage Test with grad checkpointing=False, LoRA=False    ║
╚══════════════════════════════════════════════════════════════════════╝
║    After loading model: 5.98 GB                                      ║
[… 52 lines of code omitted]

I'm using PEFT 0.18.0, transformers 4.57.3, and torch 2.8.0. For completeness, here is my script, which is basically the same as yours but with `meta-llama/Llama-3.2-3B` and some argparse on top:

[collapsed block 'click to open', 135 lines omitted]

However, as mentioned, I'm not sure if this script really captures real world usage, I would suggest using a small training script to measure.

> I started running these experiments when I started fine-tuning this model (Meta-Llama-3-8B-Instruct) on a context of 32,768 tokens

Did you try this again with `model.enable_input_require_grads()` as suggested? Did it make any difference?

> I found this very strange, and I tried to theoretically calculate the memory required to train such a model.

This is a good idea and at a glance, I don't see anything wrong there, assuming a batch size of 1. However, it is very easy to miss activations that are being stored and FSDP adds another layer of complexity on top of everything. If I want to check the activations, what I would typically do is to add something like:

```python

storage = []

def pack(x):
[… 11 lines of code omitted]

When I run [this memory estimation script](https://github.com/huggingface/peft/blob/261366de2e40cde64b702d6b9c527081ad850549/scripts/train_memory.py) like so:

```sh
python scripts/train_memory.py "meta-llama/Llama-3.1-8B-Instruct" --max_seq_length 1024 --batch_size 1 --dtype bfloat16 --monitor_tensors --path_config /tmp/peft/lora/adapter_config.json
```

where the LoRA config sets rank 32 with `target_modules="all-linear"`, I get:

```
Total size of tensors:          14706MB
Total size of activations:       3714MB
Total size of parameters:       10992MB
```

The memory required for activations is thus much greater than from your calculation, but I can't say where the difference comes from.

## [c3687345982] yurkoff-mv (NONE) · issue_comment · 2025-12-23
Hi, Benjamin!
My measurements from the previous post were exactly with the model.enable_input_require_grads() option

When I run your script (for `lora_rank=128` and `lora_rank=32`), I get OOM () error on the L40:

For **seq_length=32768**:
```
python ./tests/resource_estimation.py --model_name ./models/Llama-3.2-3B-Instruct --use_gc --use_lora --seq_len 32768

╔══════════════════════════════════════════════════════════════════════╗
║    GPU Memory Usage Test with grad checkpointing=True, LoRA=True     ║
╚══════════════════════════════════════════════════════════════════════╝
[… 29 lines of code omitted]
```

My environment:
```
accelerate                         1.11.0
numpy                              2.3.3
peft                               0.18.0
tokenizers                         0.22.1
torch                              2.9.0+cu126
[… 10 lines of code omitted]
```

This is very strange.

For **seq_length=1024** and `lora_rank=128` :
```
python ./tests/resource_estimation.py --model_name ./models/Llama-3.2-3B-Instruct --use_gc --use_lora --seq_len 1024 

╔══════════════════════════════════════════════════════════════════════╗
║    GPU Memory Usage Test with grad checkpointing=True, LoRA=True     ║
╚══════════════════════════════════════════════════════════════════════╝
[… 17 lines of code omitted]
```

## [c3687619649] yurkoff-mv (NONE) · issue_comment · 2025-12-23
I tried running [your script to evaluate memory during training](https://github.com/huggingface/peft/blob/261366de2e40cde64b702d6b9c527081ad850549/scripts/train_memory.py). Unfortunately, the sequence length in the dataset doesn't exceed 256 tokens. So I decided to slightly modify the script and generate the input synthetically using randomness, as in my previous example. I also added memory profiling.

At a context length of `32,768 tokens`, I get an OOM error.

At a context length of `1,024 tokens`, I get the following results:
```
python tests/train_memory.py "./models/Llama-3.2-3B-Instruct" --max_seq_length 1024 --batch_size 1 --dtype bfloat16 --max_steps 10 --path_config tests/adapter_config.jso
n
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.47it/s]
trainable params: 2,293,760 || all params: 3,215,043,584 || trainable%: 0.0713
step   0 seq_length 1024 loss 10.955673 time 1.20s
[… 13 lines of code omitted]
```

At a context length of `4,096 tokens`, I get the following results:
```
python tests/train_memory.py "./models/Llama-3.2-3B-Instruct" --max_seq_length 4096 --batch_size 1 --dtype bfloat16 --max_steps 10 --path_config tests/adapter_config.jso
n
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.52it/s]
trainable params: 2,293,760 || all params: 3,215,043,584 || trainable%: 0.0713
step   0 seq_length 4096 loss 10.936045 time 3.01s
[… 13 lines of code omitted]
```
At the peak of `nvidia-smi` I observed **32 GB** of used memory.

I am attaching files for reproducing the experiment.
[adapter_config.json](https://github.com/user-attachments/files/24316870/adapter_config.json)
[train_memory.py](https://github.com/user-attachments/files/24316869/train_memory.py)

I also attach the memory profiling file. It can be analyzed using https://docs.pytorch.org/memory_viz
[memory_snapshot_seq_length_1024.zip](https://github.com/user-attachments/files/24316883/memory_snapshot_seq_length_1024.zip)

I also made a screenshot for convenience.
[image]

## [c3687639583] yurkoff-mv (NONE) · issue_comment · 2025-12-23
I decided to add a screenshot of memory profiling for a sequence length of 4096 tokens.

[image]

[image]

## [c3690264343] yurkoff-mv (NONE) · issue_comment · 2025-12-24
I've slightly modified your script for evaluating memory during training. This time, I added the Gradient Checkpointing option. I ran some measurements, but the memory consumption didn't change with this option enabled.
[train_memory.py](https://github.com/user-attachments/files/24332081/train_memory.py)

Memory profile **without Gradient Checkpointing**:
[image]
[memory_snapshot_4096_liger.zip](https://github.com/user-attachments/files/24332098/memory_snapshot_4096_liger.zip)

Memory profile **with Gradient Checkpointing**:
[image]
[memory_snapshot_4096_liger_gc.zip](https://github.com/user-attachments/files/24332103/memory_snapshot_4096_liger_gc.zip)

## [c3710533232] BenjaminBossan (MEMBER) · issue_comment · reply to [c3690264343] · 2026-01-05
Thanks for running these further experiments. Note that with my script, to check the size of activations, you have to pass `--monitor_tensors`.

> I've slightly modified your script for evaluating memory during training

It seems like you attached the wrong file, it is identical to my script.

> Could you please test my suggestion on your original problem and report back if you see the same effect as I did?

Did you actually try this yet? IME gradient checkpointing is difficult to get right with toy examples, I would suggest testing it in a realistic setting.

## [c3755098323] yurkoff-mv (NONE) · issue_comment · reply to [c3710533232] · 2026-01-15
Hi, @BenjaminBossan!
Sorry for the delay in replying.  I'm sorry, I actually attached the wrong file. I am attaching the correct one.
[train_memory.py](https://github.com/user-attachments/files/24644148/train_memory.py)

### Results
**With GC**
```
python tests/train_memory.py "./models/Llama-3.2-3B-Instruct" --max_seq_length 1024 --batch_size 1 --dtype bfloat16 --max_steps 10 --path_config tests/adapter_config.json --monitor_tensors --use_liger --use_gc
```
```
trainable params: 2,293,760 || all params: 3,215,043,584 || trainable%: 0.0713
Gradient Checkpointing enabled
step   0 seq_length 1024 loss 10.955275 time 22.27s
cuda memory avg: 9929MB
cuda memory max: 10892MB
total time: 25.81s
file size: 8.8MB
Dtype counts: [(torch.bfloat16, 506), (torch.float32, 483), (torch.int64, 57)]
Total size of tensors:          12109MB
Total size of activations:       5458MB
Total size of parameters:        6651MB
```
[image]

**Without GC**
```
python tests/train_memory.py "./models/Llama-3.2-3B-Instruct" --max_seq_length 1024 --batch_size 1 --dtype bfloat16 --max_steps 10 --path_config tests/adapter_config.json --monitor_tensors --use_liger
```
```
trainable params: 2,293,760 || all params: 3,215,043,584 || trainable%: 0.0713
step   0 seq_length 1024 loss 10.955275 time 1.63s
cuda memory avg: 9905MB
cuda memory max: 10874MB
total time: 4.97s
file size: 8.8MB
Dtype counts: [(torch.bfloat16, 502), (torch.float32, 477), (torch.int64, 57)]
Total size of tensors:          12055MB
Total size of activations:       5446MB
Total size of parameters:        6609MB
```

[image]

## [c3755294128] yurkoff-mv (NONE) · issue_comment · 2026-01-15
I forgot to write that now when using the **Liger Kernel** you need to add the `skip_logits` parameter:
```
batch = {
    "input_ids": input_ids,
    "labels": labels,
}

if use_liger:
    batch["skip_logits"] = True
```

[train_memory.py](https://github.com/user-attachments/files/24646482/train_memory.py)

## [c3755448462] BenjaminBossan (MEMBER) · issue_comment · 2026-01-15
Thanks for providing the updated script, I can indeed replicate your findings. However, I still find that in a proper training run, enabling GC has the expected effect. For instance, when trying this training example in PEFT:

https://github.com/huggingface/peft/tree/622c2821cb0d7897bee53aad7914d42b5fecbf61/method_comparison/MetaMathQA

I can reduce max reserved memory from 19 GB to 13 GB with GC. To replicate this, check out the PEFT repo, navigate to `peft/method_comparison/MetaMathQA` and then run:

`python run.py -v experiments/lora/llama-3.2-3B-rank32/`

You can edit the `max_steps` in `default_training_params.json` to a lower number (say, 500) to get results more quickly. Once the training finishes, a log file with the memory profile will be created -- check the print output, it'll say something like `Saved log to: ...`, you can grep `memory` against this file.

Now edit the `run.py` by inserting

```python
    model.prepare_model_for_gradient_checkpointing(model)
    model.gradient_checkpointing_enable()
```

[here](https://github.com/huggingface/peft/blob/622c2821cb0d7897bee53aad7914d42b5fecbf61/method_comparison/MetaMathQA/run.py#L166). When you re-run the same experiment, you should observe much lower memory usage.

I think a toy example will not capture the effect of GC in realistic settings. We could try to figure out how this training example differs from the toy example to pinpoint the deciding factor, but I think this won't teach us much for real use cases. This is also why I asked repeatedly if you have tried adding those lines in your original training example and checked if they worked.

## [c3758409155] yurkoff-mv (NONE) · issue_comment · 2026-01-16
My original training example uses `Trainer` from the `transformers` library. I encountered high memory consumption, so I created this issue. I'm not sure how `run.py` differs significantly from `train_memory.py`, since conceptually the lines
```
model.prepare_model_for_gradient_checkpointing(model)
model.gradient_checkpointing_enable()
```
should work the same in both.

---
outcome: state=closed
