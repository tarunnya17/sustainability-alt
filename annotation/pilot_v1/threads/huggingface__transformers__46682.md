# #46682 [issue] [Flex Attention] `Tensor` q_offset will cause wrong mask shape
https://github.com/huggingface/transformers/issues/46682  ·  opened by hoshibara on 2026-06-16

## [body] hoshibara (NONE) · opening post · 2026-06-16
### System Info

```
torch                    2.14.0.dev20260613+cu126
torchao                  0.18.0.dev20260615+cu126
transformers             5.12.1
triton                   3.7.1+git5d6048aa
```

### Who can help?

@ArthurZucker @CyrilVallez

### Information

- [ ] The official example scripts
- [x] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [x] My own task or dataset (give details below)

### Reproduction

### Description

`q_offset` passed into `flex_attention_mask` is a `torch.Tensor` rather than a plain `int` which causing introducing an extra dimension. Causing the mask to have more than 4 dimensions and triggering the `ValueError`. 
WA: Converting `q_offset` to a Python `int` via `.item()` before passing it into `flex_attention_mask` avoids this.

```
def flex_attention_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    **kwargs,
) -> BlockMask:
```

### Reproducer
```
python -u run_generation.py -m Qwen/Qwen3-4B --input-tokens 1024 --max-new-tokens 1024 --num-iter 8 --num-warmup 4 --batch-size 1 --device cuda --inductor --dtype bfloat16 --attn-type flex_attention --disable-skip-guard-eval --disable-cpp-wrapper --num-beams 1  --use-static-cache --use-hf-code False --woq --woq-type rtn --quant-dtype uint4 --group-size 128
```
Another simpler reproducer
```
python run_generation.py \
  -m Qwen/Qwen3-4B \
  --input-tokens 1024 --max-new-tokens 1024 \
  --device cuda --dtype bfloat16 \
  --attn-type flex_attention \
  --use-static-cache \
  --inductor \
  --num-beams 1
```
### Script

[run_generation.py](https://github.com/user-attachments/files/28983711/run_generation.py)
[prompt.json](https://github.com/user-attachments/files/28983710/prompt.json)

### Error log
```
2026-06-16 03:22:50,627 INFO root Namespace(model_id='Qwen/Qwen3-4B', sub_model_name='', device='cuda', dtype='bfloat16', input_tokens='1024', max_new_tokens=1024, prompt=None, batch_size=1, num_iter=8, num_warmup=4, num_profile=1, num_beams=1, greedy=False, use_hf_code=False, use_static_cache=True, amp=False, inductor=True, profile=False, unitrace=False, accuracy_only=False, acc_tasks='gsm8k', acc_iter=-1, print_memory=False, token_latency=False, output_csv_path='output.csv', quant_mode='woq', woq=True, group_size=128, ZPFLOAT=False, calibration_samples=10, model_save_path=None, load_quantize_model=False, woq_type='rtn', quant_dtype='uint4', use_hqq=False, granularity='per_tensor', attn_type='flex_attention', disable_cpp_wrapper=True, disable_skip_guard_eval=True)
W0616 03:22:51.452000 1853176 site-packages/torch/utils/_pytree.py:630] <enum 'KernelPreference'> is an Enum subclass and is now natively supported by torch.compile as an opaque value type. Calling register_constant() on Enum subclasses is deprecated and will be an error in a future release.
2026-06-16 03:22:51,494 INFO root Using cuda device for WoQ RTN mode, Using TorchAoConfig: TorchAoConfig(quant_method=<QuantizationMethod.TORCHAO: 'torchao'>, quant_type=Int4WeightOnlyConfig(group_size=128, set_inductor_config=True, int4_packing_format='tile_packed_to_4d', int4_choose_qparams_algorithm=<Int4ChooseQParamsAlgorithm.TINYGEMM: 'tinygemm'>, int4_tile_packed_ntile=8, version=2), modules_to_not_convert=None, include_input_output_embeddings=False, untie_embedding_weights=False)
W0616 03:22:51.689000 1853176 site-packages/torch/utils/_pytree.py:630] <enum 'ScaleCalculationMode'> is an Enum subclass and is now natively supported by torch.compile as an opaque value type. Calling register_constant() on Enum subclasses is deprecated and will be an error in a future release.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 398/398 [00:02<00:00, 183.31it/s]
[… 79 lines of code omitted]
```

### Expected behavior

Fixing this bug

## [c4715683387] Tobi-Adesoye (NONE) · issue_comment · reply to [body] · 2026-06-16
Hey @hoshibara—ran into a parallel layout validation block when dealing with custom memory offsets on dynamic sequence profiles. The mask shape mutation usually happens because when q_offset slices the tensor under varying batch timelines, PyTorch produces non-contiguous memory strides. If your downstream attention mask logic assumes regular contiguous blocks, the hidden graph stride mismatches, throwing off the memory offset shape calculation.

We isolated this layout-stride translation failure in renorm-native. It catches non-contiguous strides via a __torch_dispatch__ layout hook and runs an inline zero-copy pointer compaction step right before the attention execution pipeline handles the dynamic offset masks. It locks memory ceilings without forcing manual contiguous clones. You can check our memory-stride validation scripts under the /benchmark directory to see if dropping the layout-aware layer into your runtime fixes your mask shapes:
 [github.com/Tobi-Adesoye/renorm-native](https://github.com/Tobi-Adesoye/renorm-native)

## [c4716308515] Tobi-Adesoye (NONE) · issue_comment · 2026-06-16
The startup failure is an infrastructure permission mismatch in the GitHub Actions orchestration layer, not a runtime code bug. 

The sub-workflow `.github/workflows/get-pr-number.yml` requires explicit read clearance to fetch the metadata block, but the caller pipeline is passing a restricted `contents: none` token policy. 

To unblock the docs compiler, we need to explicitly declare the `contents: read` permission scope at the job or workflow root inside `.github/workflows/pr_build_doc_with_comment.yml`.

### The Resolution:
Update line 15 of the parent workflow block to declare the necessary baseline access:

```yaml
permissions:
  contents: read
  pull-requests: read

## [c4752312055] Rocketknight1 (MEMBER) · issue_comment · 2026-06-19
cc @arthurzucker because this looks like an issue with the packed tensors for attention

## [c4778478967] Rocketknight1 (MEMBER) · issue_comment · 2026-06-23
I think #46802 resolves this!

---
outcome: state=closed
