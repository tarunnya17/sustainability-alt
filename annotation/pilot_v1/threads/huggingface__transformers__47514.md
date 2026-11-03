# #47514 [issue] Transformers 5.13 CPU memory growth loading Qwen3-235B-A22B with DeepSpeed ZeRO-3; 4.51.0 remains bounded
https://github.com/huggingface/transformers/issues/47514  ·  opened by narankhuut on 2026-07-24

## [body] narankhuut (NONE) · opening post · 2026-07-24
### System Info

**System information**
- `transformers` version: 5.13.0
- Platform: Linux-5.14.0-284.11.1.el9_2.x86_64-x86_64-with-glibc2.34
- Python version: 3.12.3
- Huggingface_hub version: 1.22.0
- Safetensors version: 0.8.0
- Accelerate version: 1.14.0
- DeepSpeed version: 0.19.2
- PyTorch version: 2.8.0+cu128 (CUDA)
- Using distributed or parallel set-up in script?: Distributed
- Using GPU in script?: Yes
- GPU type: NVIDIA A100-SXM4-80GB

**Summary**
Loading Qwen/Qwen3-235B-A22B with Transformers 5.13.0 under DeepSpeed ZeRO-3 causes host CPU memory to continuously increase until approximately 1 TB, after which the job is killed by OOM.

Downgrading only Transformers to 4.51.0 fixes the problem completely.

The same DeepSpeed configuration successfully loads gpt-oss-120b-bf16 with Transformers 5.13.0, suggesting this regression is specific to the Qwen3 and GLM MoE loading path.

**Setup**
Model: Qwen/Qwen3-235B-A22B
Transformers 5.13.0: CPU RSS grows until >1 TB and OOM
Transformers 4.51.0: model loads successfully on 16×A100
DeepSpeed: 0.19.2
ZeRO stage: 3
CPU offload: disabled
dtype: BF16
Global ZeRO partition ratio: verified as 1/world_size
Control model: GPT-OSS-120B BF16 loads successfully under Transformers 5.13

### Who can help?

@Cyrilvallez

### Information

- [x] The official example scripts
- [ ] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [x] My own task or dataset (give details below)

### Reproduction

1. Install Transformers 5.13.0 or pip install -r [requirements.txt](https://github.com/user-attachments/files/30333258/requirements.txt):
`pip install transformers==5.13.0`

2. Download train script: [https://github.com/huggingface/peft/tree/main/examples/sft](url)
3. Launch:
`deepspeed --hostfile hostfile --master_addr MASTER_IP --master_port MASTER_PORT train.py \
--seed 100 \
--model_name_or_path "Qwen/Qwen3-235B-A22B" \
--dataset_name "smangrul/ultrachat-10k-chatml" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_length 2048 \
--num_train_epochs 1 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--eval_strategy "epoch" \
--save_strategy "epoch" \
--push_to_hub \
--hub_private_repo True \
--hub_strategy "every_save" \
--bf16 True \
--packing True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_steps 0 \
--max_grad_norm 1.0 \
--output_dir "mistral-sft-lora-deepspeed" \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 4 \
--gradient_checkpointing True \
--use_reentrant False \
--dataset_text_field "content" \
--use_flash_attn True \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "all-linear" \
--use_4bit_quantization False \
--deepspeed ds_zero3.json`
4. Deepspeed config:
`
{
  "bf16": {
    "enabled": true
  },
  "train_micro_batch_size_per_gpu": 1,
  "train_batch_size": 16,
  "gradient_accumulation_steps": 1,
  "gradient_clipping": "auto",
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_gather_16bit_weights_on_model_save": false
  },
  "steps_per_print": 100
}
`

### Expected behavior

**_CPU memory should remain bounded while checkpoint shards are loaded into ZeRO-3-partitioned parameters, as happens with Transformers 4.51.0._**

**Observed behavior**

Using Transformers 5.13.0:
- DeepSpeed initializes correctly.
- Global world size is correct.
- ZeRO-3 parameter partitioning is correct.
- GPU memory allocation completes (~35 GB reserved/GPU).
- CPU memory then continuously increases.
- Eventually host RAM reaches ~1 TB and the process is killed.

Using Transformers 4.51.0:
- CPU memory remains bounded.
- Model loads successfully.

## [c5074342564] eslam-ahmed43 (NONE) · issue_comment · 2026-07-24
Hi, I would like to work on this issue if it is still available. I have been working with AI and ML projects and I think I can contribute here. I will look into it and open a PR when ready.

## [c5187753791] narankhuut (NONE) · issue_comment · 2026-08-05
After reviewing the changes in [Transformers 5.x change](https://huggingface.co/blog/moe-transformers), I found that it requires dynamic weight loading via [weightconverter](https://huggingface.co/blog/moe-transformers#dynamic-weight-loading-with-weightconverter). 

Transformers 5.x added on-the-fly MoE expert fusion (**core_model_loading.py**, **MergeModulelist** + **Concatenate** ops via **WeightConverter**) that doesn't exist in 4.51.0 at all. It fuses per-layer **experts.{i}.{gate,up,down}_proj** into stacked **gate_up_proj/down_proj** params, which requires all experts of a layer  to be resident simultaneously to fuse correctly. 
Since a layer's experts can span multiple shards with boundaries falling mid-layer (confirmed against my checkpoint's **model.safetensors.index.json**), a naive per-shard streaming loop like 4.51.0's would try to fuse an incomplete expert set and either crash or silently produce the wrong shape. 
So the full-merge-before-partition approach in 5.x looks like it was chosen to make fusion correctness simple, at the cost of losing the CPU memory bound entirely for Deepspeed ZeRO loading of large models. 

I think a possible fix would be to stream by fusion unit — i.e. the group of shards needed to fuse one layer's experts — rather than by individual shard or by the whole checkpoint. **_load_state_dict_into_zero3_model** and the weight-conversion path already accept partial state dicts, so this could be done without touching the fusion logic itself, just changing the granularity at which shards are loaded and freed.

Before I go further down that path, though — is anyone from the Transformers team already working on a fix for this internally? Wanted to check in case there's ongoing work I should coordinate with rather than duplicate.

## [c5204228241] Rocketknight1 (MEMBER) · issue_comment · 2026-08-06
I don't think anyone is working on this, but a fix here sounds like it might be very complex!

---
outcome: state=closed
