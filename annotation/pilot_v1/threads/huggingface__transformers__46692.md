# #46692 [issue] Out of memory after some epochs while training RT-DETR v2 model
https://github.com/huggingface/transformers/issues/46692  ·  opened by Rayndell on 2026-06-16

## [body] Rayndell (NONE) · opening post · 2026-06-16
### System Info

Unfortunately transformers env fails with the following error: NameError: name 'CompletionCreateParamsStreaming' is not defined. Please find manuel configuration instead:
System: Linux 6.1.0-40-amd64 (Docker)
Python version: 3.12.13
transformers==5.12.0

### Who can help?

_No response_

### Information

- [ ] The official example scripts
- [x] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [x] My own task or dataset (give details below)

### Reproduction

I am trying to fine tune a RT DETR v2 detection model on my custom dataset. I am following the notebook provided here: https://github.com/qubvel/transformers-notebooks/blob/main/notebooks/RT_DETR_v2_finetune_on_a_custom_dataset.ipynb, but I am facing some issues regarding a lack of memory after several epochs:

{'loss': '39.8', 'grad_norm': '150.6', 'learning_rate': '0.0009995', 'epoch': '0.06573'}                                                                              
{'loss': '46.91', 'grad_norm': '1.044e+04', 'learning_rate': '0.0009982', 'epoch': '0.1315'}                                                                          
{'loss': '49.9', 'grad_norm': '9.306e+04', 'learning_rate': '0.0009968', 'epoch': '0.1972'}                                                                           
{'loss': '48.61', 'grad_norm': '1022', 'learning_rate': '0.0009955', 'epoch': '0.2629'}                                                                               
{'loss': '47.9', 'grad_norm': '1.658e+05', 'learning_rate': '0.0009942', 'epoch': '0.3286'}                                                                           
{'loss': '47.41', 'grad_norm': '1.303e+05', 'learning_rate': '0.0009929', 'epoch': '0.3944'}                                                                          
{'loss': '47.88', 'grad_norm': '2.339e+04', 'learning_rate': '0.0009916', 'epoch': '0.4601'}                                                                          
{'loss': '47.76', 'grad_norm': '1.749e+05', 'learning_rate': '0.0009903', 'epoch': '0.5258'}                                                                          
{'loss': '49.36', 'grad_norm': '5.75e+05', 'learning_rate': '0.000989', 'epoch': '0.5916'}                                                                            
{'loss': '52.91', 'grad_norm': '4.086e+05', 'learning_rate': '0.0009876', 'epoch': '0.6573'}                                                                          
{'loss': '53.59', 'grad_norm': '3.539e+04', 'learning_rate': '0.0009863', 'epoch': '0.723'}                                                                           
{'loss': '49.96', 'grad_norm': '1.536e+04', 'learning_rate': '0.000985', 'epoch': '0.7887'}                                                                           
{'loss': '55.76', 'grad_norm': '221.7', 'learning_rate': '0.0009837', 'epoch': '0.8545'}                                                                              
{'loss': '52.76', 'grad_norm': '4660', 'learning_rate': '0.0009824', 'epoch': '0.9202'}                                                                               
{'loss': '39.24', 'grad_norm': '1.069e+04', 'learning_rate': '0.0009811', 'epoch': '0.9859'}                                                                          
  2%|██▍                                                                                                                     | 7607/380350 [25:11<23:35:13,  4.39it/s]ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py", line 1285, in _try_get_data
    data = self._data_queue.get(timeout=timeout)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/queue.py", line 180, in get
    self.not_empty.wait(remaining)
  File "/usr/lib/python3.12/threading.py", line 359, in wait
    gotit = waiter.acquire(True, timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/data/_utils/signal_handling.py", line 73, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 365) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspace/detection/test_rt_detr.py", line 213, in <module>
    trainer.train()
  File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 1433, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 1515, in _inner_training_loop
    self._run_epoch(
  File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 1816, in _run_epoch
    self._maybe_log_save_evaluate(
  File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 2095, in _maybe_log_save_evaluate
    metrics = self._evaluate(trial, ignore_keys_for_eval)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 3016, in _evaluate
    metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 2603, in evaluate
    output = self.evaluation_loop(
             ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 2718, in evaluation_loop
    for step, inputs in enumerate(dataloader):
                        ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/accelerate/data_loader.py", line 585, in __iter__
    current_batch = next(dataloader_iter)
                    ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py", line 734, in __next__
    data = self._next_data()
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py", line 1492, in _next_data
    idx, data = self._get_data()
                ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py", line 1444, in _get_data
    success, data = self._try_get_data()
                    ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py", line 1298, in _try_get_data
    raise RuntimeError(
RuntimeError: DataLoader worker (pid(s) 365) exited unexpectedly
  2%|▏         | 7607/380350 [25:15<20:37:25,  5.02it/s]

Here is a code snippet to reproduce:

```python
from dataset import get_dataloader
import albumentations as A
from PIL import Image, ImageDraw
import numpy as np
[… 351 lines of code omitted]

I have also tried with batch_size=2 and the crash occurs after 2 or 3 epochs. I am suspecting a memory leak since the training does well on a smaller dataset.

### Expected behavior

The training is expected to run without crashing due to lack of memory.

## [c4749461079] Rayndell (NONE) · issue_comment · 2026-06-19
Thank you for the in-depth explaantion, it was indeed a shared memory related issue. However I have other troubles now during training, it seems that there is a lack of CUDA memory at some time during the training:

Traceback (most recent call last): | 44/268 [00:24<02:03, 1.81it/s]
File "/workspace/detection/test_rt_detr.py", line 213, in
trainer.train()
File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 1433, in train
return inner_training_loop(
^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 1515, in _inner_training_loop
self._run_epoch(
File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 1816, in _run_epoch
self._maybe_log_save_evaluate(
File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 2095, in _maybe_log_save_evaluate
metrics = self._evaluate(trial, ignore_keys_for_eval)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 3016, in _evaluate
metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 2603, in evaluate
output = self.evaluation_loop(
^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 2728, in evaluation_loop
losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 2982, in prediction_step
loss, outputs = self.compute_loss(
^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/trainer.py", line 1987, in compute_loss
outputs = model(**inputs)
^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
return self._call_impl(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
return forward_call(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py", line 907, in wrapper
output = func(self, *args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py", line 1867, in forward
outputs = self.model(
^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
return self._call_impl(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
return forward_call(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py", line 907, in wrapper
output = func(self, *args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py", line 1514, in forward
features = self.backbone(pixel_values, pixel_mask)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
return self._call_impl(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
return forward_call(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py", line 810, in forward
features = self.model(pixel_values).feature_maps
^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
return self._call_impl(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
return forward_call(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/utils/generic.py", line 907, in wrapper
output = func(self, *args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/backbone_utils.py", line 172, in wrapper
output = forward_function(self, *args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/models/rt_detr/modeling_rt_detr_resnet.py", line 400, in forward
embedding_output = self.embedder(pixel_values)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
return self._call_impl(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
return forward_call(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/models/rt_detr/modeling_rt_detr_resnet.py", line 112, in forward
embedding = self.embedder(pixel_values)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
return self._call_impl(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
return forward_call(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/container.py", line 244, in forward
input = module(input)
^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
return self._call_impl(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
return forward_call(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/models/rt_detr/modeling_rt_detr_resnet.py", line 66, in forward
hidden_states = self.normalization(hidden_states)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
return self._call_impl(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
return forward_call(*args, **kwargs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/usr/local/lib/python3.12/dist-packages/transformers/models/rt_detr_v2/modeling_rt_detr_v2.py", line 761, in forward
return x * scale + bias
~~~~~~~~~~^~~~~~
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.11 GiB. GPU 0 has a total capacity of 31.47 GiB of which 481.06 MiB is free. Including non-PyTorch memory, this process has 30.98 GiB memory in use. Of the allocated memory 25.30 GiB is allocated by PyTorch, and 5.31 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
2%|▏ | 2535/126750 [22:59<18:46:50, 1.84it/s]

This happened after 5 epochs (not right away), I tried to decrease the batch size but it still happens. Is there some memory that is not properly deallocated?

## [c4756900269] lcheng321 (CONTRIBUTOR) · issue_comment · 2026-06-20
This looks like a GPU memory fragmentation issue rather than a bug in transformers. The OOM is from memory fragmentation caused by the large input resolution (1216x1920). Try PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True and adding torch.cuda.empty_cache() in your compute_metrics function.

## [c4767577021] Rayndell (NONE) · issue_comment · 2026-06-22
Hello, thank you for your answer. However the solution you propose changed nothing, I still have the same memory error.

## [c4768554878] Rayndell (NONE) · issue_comment · 2026-06-22
I also tried the fix proposed by @gmrnlg1971 (detaching tensors) but it did not work either.

## [c4774044507] jashwanth-reddy-g (NONE) · issue_comment · reply to [c4768554878] · 2026-06-22
Hi @Rayndell, sorry to hear the detach fix did not resolve the OOM for you — high-resolution inputs like 1216×1920 put serious pressure on fragmented GPU memory, so a single change often is not enough. Here are a few targeted strategies that should help:

**1. Reduce `max_num_pages` in the dataset config**
If your dataset config exposes a `max_num_pages` parameter, lowering it directly reduces the maximum sequence length fed to the model at once, which is usually the quickest win for large-resolution inputs.

**2. Enable gradient checkpointing**
This trades compute for memory by recomputing activations during the backward pass rather than storing them:
```python
model.gradient_checkpointing_enable()
```
This can cut activation memory by 30–60% with only a modest throughput penalty.

**3. Free the fragment cache inside the training loop**
CUDA memory fragmentation accumulates quickly at high resolutions. Adding a periodic `empty_cache()` call helps:
```python
for step, batch in enumerate(dataloader):
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if step % 50 == 0:
        torch.cuda.empty_cache()
```

**4. Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`**
This tells PyTorch's caching allocator to avoid holding onto very large contiguous blocks, which is a primary driver of fragmentation at high resolutions. Set it before launching your training script:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```
or in Python:
```python
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
```

I would recommend trying these in combination — particularly gradient checkpointing + the allocator config — as they address complementary causes of OOM. Feel free to share the full traceback if the issue persists and I'll take a closer look! 🙂

## [c4779411006] Rayndell (NONE) · issue_comment · 2026-06-23
If this could help, it seems that the crash occur at the eval, since setting eval_strategy="no" prevents from crashes. Could this be a bug in the MAPEvaluator given in the notebook? I tried to move all I could to the cpu (by detaching the evaluation and ground truth tensors), but none of it works. I will try what you suggest but it does not seem to be a fragmented memory related problem.

## [c4784992237] Prachi-kushwaha (NONE) · issue_comment · 2026-06-24
do compute_metrics = None
if crash disappear that means culprit is most likely MAPEvaluator  or some memory accumulation happening during metric computation
if crash still happen culprit is most likely  evaluation pipeline such as : -  trainer evaluation loop, prediction accumulation, Rt-detr output

## [c4789137367] Rayndell (NONE) · issue_comment · 2026-06-24
Yes, the crash disappears when I disable the evaluation, but I have currently no clue on what to change to prevent this.

I already tried setting eval_accumulation_steps=1 and per_device_eval_batch_size=1, without success.

## [c4789312460] Rayndell (NONE) · issue_comment · 2026-06-24
Actually the program seems to crash during the first evaluation process:

[image]

## [c4792304825] Prachi-kushwaha (NONE) · issue_comment · 2026-06-24
I'm not sure but this might be another possibility

If it crashes on first evaluation then 
evaluation dataset is too large so this can explode memory immediately during the first evaluation.

you can use small validation subset with this if crash disappearss then that would suggest evaluation memory usage scales with dataset size

if not then then I would start suspecting a bug in the  evaluation pipeline itself or metric code.

## [c4903680412] Rayndell (NONE) · issue_comment · 2026-07-07
Hello,

I have finally solved this problem. It was in fact during the evaluation, where the processed stored all the outputs on the GPU which resulted in a lack of GPU memory. As a workaround, I retained only the outputs with a score > threshold:

```python
def preprocess_logits_for_metrics(outputs, labels):
    probs = logits.softmax(-1)
    scores, labels_idx = probs.max(-1)  # (B, N)

[… 26 lines of code omitted]

I also added these parameters in the TrainingArguments:

```python
remove_unused_columns=False,
eval_do_concat_batches=False,
eval_accumulation_steps=1,
```

However I am still unable to obtain proper results on my dataset: the loss stays big, with big gradients, and it seems that the model isn't learning anything. I tried different strategies: freezing the backbone, reducing the learning rate, trying different optimizers and schedulers, ... Can you guys help me on this? My dataset has large images, which I reduce to 1920x1216 pixels (I cannot reduce it further since some objects are very small). One challenging issue is that on top of that some objects are well represented and are very big and elongated (see example below). I also tried a version of the dataset where these objects are not present but the problem still remain.

[image]

Here is my current configuration:

```python
checkpoint = "PekingU/rtdetr_v2_r50vd"

train_augmentation_and_transform = A.Compose(
    [
[… 32 lines of code omitted]

Am I doing something wrong here? I also carefully double-checked my data and what is being fed to the network during training, everything is fine.

---
outcome: state=closed
