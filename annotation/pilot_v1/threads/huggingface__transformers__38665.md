# #38665 [issue] Exception while inference Qwen2VL and Qwen2VL, assert module.weight.shape[1] == 1
https://github.com/huggingface/transformers/issues/38665  ·  opened by iglaweb on 2025-06-07

## [body] iglaweb (NONE) · opening post · 2025-06-07
### System Info

    transformers version: 4.52.3
    Platform: Linux-5.10.0-1029-oem-x86_64-with-glibc2.31
    GPU device: Quadro RTX 8000
    Python version: 3.10
    Huggingface_hub version: 0.32.2
    Safetensors version: 0.5.3
    Accelerate version: 0.34.2
    PyTorch version (GPU?): 2.5.0+cu124
    Using distributed or parallel set-up in script?: No

### Who can help?

@zucchini-nlp 
@qubvel 
@ArthurZucker 

### Information

- [ ] The official example scripts
- [x] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [x] My own task or dataset (give details below)

### Reproduction

I followed these tutorials:
https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/fine_tuning_vlm_trl.ipynb
https://github.com/QwenLM/Qwen2.5-VL/blob/d2240f11656bfe404b9ba56db4e51cd09f522ff1/qwen-vl-finetune/qwenvl/train/train_qwen.py

Steps to reproduce the issue:

1. Fine-tune Qwen2VL or Qwen2.5VL (e.g. "Qwen/Qwen2.5-VL-3B-Instruct") model on custom dataset (Qlora and LoRA enabled, use cuda)
2. Run the inference on a video (use cuda).

Full log and exception:

```
- This IS expected if you are initializing Qwen2_5_VLForConditionalGeneration from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing Qwen2_5_VLForConditionalGeneration from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of Qwen2_5_VLForConditionalGeneration were not initialized from the model checkpoint at /home/user/Desktop/demo/tmp/weights_2025-05-30_13.06.42.192256_qwen_qwen2.5-vl-3b-instruct_b2_e1_vf16_fps1.0/model_qwen2vl_video-lora and are newly initialized: ['model.language_model.layers.0.self_attn.k_proj.bias', 'model.language_model.layers.0.self_attn.k_proj.weight', 'model.language_model.layers.0.self_attn.o_proj.weight', 'model.language_model.layers.0.self_attn.q_proj.bias', 'model.language_model.layers.0.self_attn.q_proj.weight', 'model.language_model.layers.0.self_attn.v_proj.bias', 'model.language_model.layers.0.self_attn.v_proj.weight', 'model.language_model.layers.1.self_attn.k_proj.bias', 'model.language_model.layers.1.self_attn.k_proj.weight', 'model.language_model.layers.1.self_attn.o_proj.weight', 'model.language_model.layers.1.self_attn.q_proj.bias', 'model.language_model.layers.1.self_attn.q_proj.weight', 'model.language_model.layers.1.self_attn.v_proj.bias', 'model.language_model.layers.1.self_attn.v_proj.weight', 'model.language_model.layers.10.self_attn.k_proj.bias', 'model.language_model.layers.10.self_attn.k_proj.weight', 'model.language_model.layers.10.self_attn.o_proj.weight', 'model.language_model.layers.10.self_attn.q_proj.bias', 'model.language_model.layers.10.self_attn.q_proj.weight', 'model.language_model.layers.10.self_attn.v_proj.bias', 'model.language_model.layers.10.self_attn.v_proj.weight', 'model.language_model.layers.11.self_attn.k_proj.bias', 'model.language_model.layers.11.self_attn.k_proj.weight', 'model.language_model.layers.11.self_attn.o_proj.weight', 'model.language_model.layers.11.self_attn.q_proj.bias', 'model.language_model.layers.11.self_attn.q_proj.weight', 'model.language_model.layers.11.self_attn.v_proj.bias', 'model.language_model.layers.11.self_attn.v_proj.weight', 'model.language_model.layers.12.self_attn.k_proj.bias', 'model.language_model.layers.12.self_attn.k_proj.weight', 'model.language_model.layers.12.self_attn.o_proj.weight', 'model.language_model.layers.12.self_attn.q_proj.bias', 'model.language_model.layers.12.self_attn.q_proj.weight', 'model.language_model.layers.12.self_attn.v_proj.bias', 'model.language_model.layers.12.self_attn.v_proj.weight', 'model.language_model.layers.13.self_attn.k_proj.bias', 'model.language_model.layers.13.self_attn.k_proj.weight', 'model.language_model.layers.13.self_attn.o_proj.weight', 'model.language_model.layers.13.self_attn.q_proj.bias', 'model.language_model.layers.13.self_attn.q_proj.weight', 'model.language_model.layers.13.self_attn.v_proj.bias', 'model.language_model.layers.13.self_attn.v_proj.weight', 'model.language_model.layers.14.self_attn.k_proj.bias', 'model.language_model.layers.14.self_attn.k_proj.weight', 'model.language_model.layers.14.self_attn.o_proj.weight', 'model.language_model.layers.14.self_attn.q_proj.bias', 'model.language_model.layers.14.self_attn.q_proj.weight', 'model.language_model.layers.14.self_attn.v_proj.bias', 'model.language_model.layers.14.self_attn.v_proj.weight', 'model.language_model.layers.15.self_attn.k_proj.bias', 'model.language_model.layers.15.self_attn.k_proj.weight', 'model.language_model.layers.15.self_attn.o_proj.weight', 'model.language_model.layers.15.self_attn.q_proj.bias', 'model.language_model.layers.15.self_attn.q_proj.weight', 'model.language_model.layers.15.self_attn.v_proj.bias', 'model.language_model.layers.15.self_attn.v_proj.weight', 'model.language_model.layers.16.self_attn.k_proj.bias', 'model.language_model.layers.16.self_attn.k_proj.weight', 'model.language_model.layers.16.self_attn.o_proj.weight', 'model.language_model.layers.16.self_attn.q_proj.bias', 'model.language_model.layers.16.self_attn.q_proj.weight', 'model.language_model.layers.16.self_attn.v_proj.bias', 'model.language_model.layers.16.self_attn.v_proj.weight', 'model.language_model.layers.17.self_attn.k_proj.bias', 'model.language_model.layers.17.self_attn.k_proj.weight', 'model.language_model.layers.17.self_attn.o_proj.weight', 'model.language_model.layers.17.self_attn.q_proj.bias', 'model.language_model.layers.17.self_attn.q_proj.weight', 'model.language_model.layers.17.self_attn.v_proj.bias', 'model.language_model.layers.17.self_attn.v_proj.weight', 'model.language_model.layers.18.self_attn.k_proj.bias', 'model.language_model.layers.18.self_attn.k_proj.weight', 'model.language_model.layers.18.self_attn.o_proj.weight', 'model.language_model.layers.18.self_attn.q_proj.bias', 'model.language_model.layers.18.self_attn.q_proj.weight', 'model.language_model.layers.18.self_attn.v_proj.bias', 'model.language_model.layers.18.self_attn.v_proj.weight', 'model.language_model.layers.19.self_attn.k_proj.bias', 'model.language_model.layers.19.self_attn.k_proj.weight', 'model.language_model.layers.19.self_attn.o_proj.weight', 'model.language_model.layers.19.self_attn.q_proj.bias', 'model.language_model.layers.19.self_attn.q_proj.weight', 'model.language_model.layers.19.self_attn.v_proj.bias', 'model.language_model.layers.19.self_attn.v_proj.weight', 'model.language_model.layers.2.self_attn.k_proj.bias', 'model.language_model.layers.2.self_attn.k_proj.weight', 'model.language_model.layers.2.self_attn.o_proj.weight', 'model.language_model.layers.2.self_attn.q_proj.bias', 'model.language_model.layers.2.self_attn.q_proj.weight', 'model.language_model.layers.2.self_attn.v_proj.bias', 'model.language_model.layers.2.self_attn.v_proj.weight', 'model.language_model.layers.20.self_attn.k_proj.bias', 'model.language_model.layers.20.self_attn.k_proj.weight', 'model.language_model.layers.20.self_attn.o_proj.weight', 'model.language_model.layers.20.self_attn.q_proj.bias', 'model.language_model.layers.20.self_attn.q_proj.weight', 'model.language_model.layers.20.self_attn.v_proj.bias', 'model.language_model.layers.20.self_attn.v_proj.weight', 'model.language_model.layers.21.self_attn.k_proj.bias', 'model.language_model.layers.21.self_attn.k_proj.weight', 'model.language_model.layers.21.self_attn.o_proj.weight', 'model.language_model.layers.21.self_attn.q_proj.bias', 'model.language_model.layers.21.self_attn.q_proj.weight', 'model.language_model.layers.21.self_attn.v_proj.bias', 'model.language_model.layers.21.self_attn.v_proj.weight', 'model.language_model.layers.22.self_attn.k_proj.bias', 'model.language_model.layers.22.self_attn.k_proj.weight', 'model.language_model.layers.22.self_attn.o_proj.weight', 'model.language_model.layers.22.self_attn.q_proj.bias', 'model.language_model.layers.22.self_attn.q_proj.weight', 'model.language_model.layers.22.self_attn.v_proj.bias', 'model.language_model.layers.22.self_attn.v_proj.weight', 'model.language_model.layers.23.self_attn.k_proj.bias', 'model.language_model.layers.23.self_attn.k_proj.weight', 'model.language_model.layers.23.self_attn.o_proj.weight', 'model.language_model.layers.23.self_attn.q_proj.bias', 'model.language_model.layers.23.self_attn.q_proj.weight', 'model.language_model.layers.23.self_attn.v_proj.bias', 'model.language_model.layers.23.self_attn.v_proj.weight', 'model.language_model.layers.24.self_attn.k_proj.bias', 'model.language_model.layers.24.self_attn.k_proj.weight', 'model.language_model.layers.24.self_attn.o_proj.weight', 'model.language_model.layers.24.self_attn.q_proj.bias', 'model.language_model.layers.24.self_attn.q_proj.weight', 'model.language_model.layers.24.self_attn.v_proj.bias', 'model.language_model.layers.24.self_attn.v_proj.weight', 'model.language_model.layers.25.self_attn.k_proj.bias', 'model.language_model.layers.25.self_attn.k_proj.weight', 'model.language_model.layers.25.self_attn.o_proj.weight', 'model.language_model.layers.25.self_attn.q_proj.bias', 'model.language_model.layers.25.self_attn.q_proj.weight', 'model.language_model.layers.25.self_attn.v_proj.bias', 'model.language_model.layers.25.self_attn.v_proj.weight', 'model.language_model.layers.26.self_attn.k_proj.bias', 'model.language_model.layers.26.self_attn.k_proj.weight', 'model.language_model.layers.26.self_attn.o_proj.weight', 'model.language_model.layers.26.self_attn.q_proj.bias', 'model.language_model.layers.26.self_attn.q_proj.weight', 'model.language_model.layers.26.self_attn.v_proj.bias', 'model.language_model.layers.26.self_attn.v_proj.weight', 'model.language_model.layers.27.self_attn.k_proj.bias', 'model.language_model.layers.27.self_attn.k_proj.weight', 'model.language_model.layers.27.self_attn.o_proj.weight', 'model.language_model.layers.27.self_attn.q_proj.bias', 'model.language_model.layers.27.self_attn.q_proj.weight', 'model.language_model.layers.27.self_attn.v_proj.bias', 'model.language_model.layers.27.self_attn.v_proj.weight', 'model.language_model.layers.28.self_attn.k_proj.bias', 'model.language_model.layers.28.self_attn.k_proj.weight', 'model.language_model.layers.28.self_attn.o_proj.weight', 'model.language_model.layers.28.self_attn.q_proj.bias', 'model.language_model.layers.28.self_attn.q_proj.weight', 'model.language_model.layers.28.self_attn.v_proj.bias', 'model.language_model.layers.28.self_attn.v_proj.weight', 'model.language_model.layers.29.self_attn.k_proj.bias', 'model.language_model.layers.29.self_attn.k_proj.weight', 'model.language_model.layers.29.self_attn.o_proj.weight', 'model.language_model.layers.29.self_attn.q_proj.bias', 'model.language_model.layers.29.self_attn.q_proj.weight', 'model.language_model.layers.29.self_attn.v_proj.bias', 'model.language_model.layers.29.self_attn.v_proj.weight', 'model.language_model.layers.3.self_attn.k_proj.bias', 'model.language_model.layers.3.self_attn.k_proj.weight', 'model.language_model.layers.3.self_attn.o_proj.weight', 'model.language_model.layers.3.self_attn.q_proj.bias', 'model.language_model.layers.3.self_attn.q_proj.weight', 'model.language_model.layers.3.self_attn.v_proj.bias', 'model.language_model.layers.3.self_attn.v_proj.weight', 'model.language_model.layers.30.self_attn.k_proj.bias', 'model.language_model.layers.30.self_attn.k_proj.weight', 'model.language_model.layers.30.self_attn.o_proj.weight', 'model.language_model.layers.30.self_attn.q_proj.bias', 'model.language_model.layers.30.self_attn.q_proj.weight', 'model.language_model.layers.30.self_attn.v_proj.bias', 'model.language_model.layers.30.self_attn.v_proj.weight', 'model.language_model.layers.31.self_attn.k_proj.bias', 'model.language_model.layers.31.self_attn.k_proj.weight', 'model.language_model.layers.31.self_attn.o_proj.weight', 'model.language_model.layers.31.self_attn.q_proj.bias', 'model.language_model.layers.31.self_attn.q_proj.weight', 'model.language_model.layers.31.self_attn.v_proj.bias', 'model.language_model.layers.31.self_attn.v_proj.weight', 'model.language_model.layers.32.self_attn.k_proj.bias', 'model.language_model.layers.32.self_attn.k_proj.weight', 'model.language_model.layers.32.self_attn.o_proj.weight', 'model.language_model.layers.32.self_attn.q_proj.bias', 'model.language_model.layers.32.self_attn.q_proj.weight', 'model.language_model.layers.32.self_attn.v_proj.bias', 'model.language_model.layers.32.self_attn.v_proj.weight', 'model.language_model.layers.33.self_attn.k_proj.bias', 'model.language_model.layers.33.self_attn.k_proj.weight', 'model.language_model.layers.33.self_attn.o_proj.weight', 'model.language_model.layers.33.self_attn.q_proj.bias', 'model.language_model.layers.33.self_attn.q_proj.weight', 'model.language_model.layers.33.self_attn.v_proj.bias', 'model.language_model.layers.33.self_attn.v_proj.weight', 'model.language_model.layers.34.self_attn.k_proj.bias', 'model.language_model.layers.34.self_attn.k_proj.weight', 'model.language_model.layers.34.self_attn.o_proj.weight', 'model.language_model.layers.34.self_attn.q_proj.bias', 'model.language_model.layers.34.self_attn.q_proj.weight', 'model.language_model.layers.34.self_attn.v_proj.bias', 'model.language_model.layers.34.self_attn.v_proj.weight', 'model.language_model.layers.35.self_attn.k_proj.bias', 'model.language_model.layers.35.self_attn.k_proj.weight', 'model.language_model.layers.35.self_attn.o_proj.weight', 'model.language_model.layers.35.self_attn.q_proj.bias', 'model.language_model.layers.35.self_attn.q_proj.weight', 'model.language_model.layers.35.self_attn.v_proj.bias', 'model.language_model.layers.35.self_attn.v_proj.weight', 'model.language_model.layers.4.self_attn.k_proj.bias', 'model.language_model.layers.4.self_attn.k_proj.weight', 'model.language_model.layers.4.self_attn.o_proj.weight', 'model.language_model.layers.4.self_attn.q_proj.bias', 'model.language_model.layers.4.self_attn.q_proj.weight', 'model.language_model.layers.4.self_attn.v_proj.bias', 'model.language_model.layers.4.self_attn.v_proj.weight', 'model.language_model.layers.5.self_attn.k_proj.bias', 'model.language_model.layers.5.self_attn.k_proj.weight', 'model.language_model.layers.5.self_attn.o_proj.weight', 'model.language_model.layers.5.self_attn.q_proj.bias', 'model.language_model.layers.5.self_attn.q_proj.weight', 'model.language_model.layers.5.self_attn.v_proj.bias', 'model.language_model.layers.5.self_attn.v_proj.weight', 'model.language_model.layers.6.self_attn.k_proj.bias', 'model.language_model.layers.6.self_attn.k_proj.weight', 'model.language_model.layers.6.self_attn.o_proj.weight', 'model.language_model.layers.6.self_attn.q_proj.bias', 'model.language_model.layers.6.self_attn.q_proj.weight', 'model.language_model.layers.6.self_attn.v_proj.bias', 'model.language_model.layers.6.self_attn.v_proj.weight', 'model.language_model.layers.7.self_attn.k_proj.bias', 'model.language_model.layers.7.self_attn.k_proj.weight', 'model.language_model.layers.7.self_attn.o_proj.weight', 'model.language_model.layers.7.self_attn.q_proj.bias', 'model.language_model.layers.7.self_attn.q_proj.weight', 'model.language_model.layers.7.self_attn.v_proj.bias', 'model.language_model.layers.7.self_attn.v_proj.weight', 'model.language_model.layers.8.self_attn.k_proj.bias', 'model.language_model.layers.8.self_attn.k_proj.weight', 'model.language_model.layers.8.self_attn.o_proj.weight', 'model.language_model.layers.8.self_attn.q_proj.bias', 'model.language_model.layers.8.self_attn.q_proj.weight', 'model.language_model.layers.8.self_attn.v_proj.bias', 'model.language_model.layers.8.self_attn.v_proj.weight', 'model.language_model.layers.9.self_attn.k_proj.bias', 'model.language_model.layers.9.self_attn.k_proj.weight', 'model.language_model.layers.9.self_attn.o_proj.weight', 'model.language_model.layers.9.self_attn.q_proj.bias', 'model.language_model.layers.9.self_attn.q_proj.weight', 'model.language_model.layers.9.self_attn.v_proj.bias', 'model.language_model.layers.9.self_attn.v_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
[… 64 lines of code omitted]

### Expected behavior

I expect the inference to complete without errors.

## [c2954962673] zucchini-nlp (MEMBER) · issue_comment · reply to [body] · 2025-06-09
@iglaweb we recently fixed one bug when saving Qwen-VL models and it is in the latest patch release. Can you try to update the transformers version?

Prob you'll need to re-save the checkpoint, I will help with that if you can share it on the hub

## [c2970777471] msciancalepore98 (NONE) · issue_comment · reply to [c2954962673] · 2025-06-13
hey @zucchini-nlp  I have trained qwen2.5-VL-3B model 1month ago and right now I have this error when attempting to load the adapters..

```
Unrecognized video processor in /content/lora_model_qwen2.5-VL-3B/lora_model_qwen2.5-VL-3B. Should have a `video_processor_type` key in its video_preprocessor_config.json of config.json, or one of the following `model_type` keys in its config.json: instructblip, instructblipvideo, internvl, llava_next_video, llava_onevision, qwen2_5_omni, qwen2_5_vl, qwen2_vl, smolvlm, video_llava
```

can you please guide me in how I should tweak the adapter config? I mean.. I hope I dont have to re-train..

## [c2972512986] msciancalepore98 (NONE) · issue_comment · reply to [c2954962673] · 2025-06-14
I've just seen that transformers has added VideoProcessors as first class citizens processors.. can it be related @zucchini-nlp @danielhanchen  ?

## [c2975296895] zucchini-nlp (MEMBER) · issue_comment · reply to [c2972512986] · 2025-06-16
@msciancalepore98 can you share your tuned weights and a minimal repro please?

## [c2975331102] msciancalepore98 (NONE) · issue_comment · 2025-06-16
Just using the official unsloth notebook with no changes. I've executed the cell down below that loads the model from a path.

```
from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = True, # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model) # Enable for inference!
```

This is the zip with the configs, I cannot share the weights unfortunately.
[lora_qwen2.5VL-3B-configs.zip](https://github.com/user-attachments/files/20751582/lora_qwen2.5VL-3B-configs.zip)

## [c2975428298] zucchini-nlp (MEMBER) · issue_comment · 2025-06-16
Hm I think I found it, can you check if this (https://github.com/huggingface/transformers/pull/38840) resolves your issue?

I see that the shared zip file has no config, and thus we weren't able to infer if model type. Now we check for image processor's type as well

## [c2975483351] msciancalepore98 (NONE) · issue_comment · reply to [c2975428298] · 2025-06-16
@zucchini-nlp I've tried to install it in the notebook at the top, but when I execute the model loading cell I have:

```
ImportError: cannot import name 'HybridCache' from 'transformers.models.gemma3.modeling_gemma3' (/usr/local/lib/python3.11/dist-packages/transformers/models/gemma3/modeling_gemma3.py)

```

installed with: !pip install -U "git+https://github.com/zucchini-nlp/transformers.git@video_processor"

## [c2975497971] zucchini-nlp (MEMBER) · issue_comment · 2025-06-16
Are you importing HybridCache anywhere or is that inside unsloth? Prob the version of unsloth isn't compatible with transformers from `main` branch. We just recently removed HybricCache from gemma and it should be imported only from `transformers.cache_utils`

In that case, even if the issue is fixed on `main` branch you might have to add a few hack to make current version compatible with unsloth or wait for for unsloth team bump transformers' next releases

## [c2975523053] msciancalepore98 (NONE) · issue_comment · 2025-06-16
nop not using that anywhere in the notebook. I'd say it's on the unsloth side, but idk.

I've created an issue on unsloth as well to track it.

thanks for your time!

## [c2981737866] iglaweb (NONE) · issue_comment · reply to [c2954962673] · 2025-06-17
> [@iglaweb](https://github.com/iglaweb) we recently fixed one bug when saving Qwen-VL models and it is in the latest patch release. Can you try to update the transformers version?
> 
> Prob you'll need to re-save the checkpoint, I will help with that if you can share it on the hub

@zucchini-nlp Thank you for reply. I updated the transformers library to **4.52.4**. Then, I fine-tuned "Qwen/Qwen2.5-VL-3B-Instruct" on a small portion of my dataset. Hm, it still does not work. I tested the model in two scenarios below.

Scenario 1 (train free). I implemented a script for loading the pre-trained model, saving with `model.save_pretrained` and loading the model again from a saved folder (`Qwen2_5_VLForConditionalGeneration.from_pretrained`), it works without any errors.

Scenario 2 (fine-tune). This is the issue I want to fix. I train a model using `SFTTrainer` with `LoraConfig` and `BitsAndBytesConfig` as follows:
```python
from trl import SFTConfig, SFTTrainer, get_kbit_device_map
from transformers import Qwen2VLProcessor, Qwen2_5_VLProcessor, \
    AutoModelForVision2Seq
from transformers import AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig
[… 41 lines of code omitted]
```
And, then I'm getting the same error while doing an inference on a fine-tuned model. The model is loaded through `Qwen2_5_VLForConditionalGeneration.from_pretrained`.
Full log with an error is here https://gist.github.com/iglaweb/87d8f29a02f0d2032566206711b8a9bb

## [c2982872407] msciancalepore98 (NONE) · issue_comment · 2025-06-18
In my case, downgrading to Transformers: 4.51.3 fixes this. (which was the transformer version I used along with unsloth ~1 month ago when training)

## [c2983011918] zucchini-nlp (MEMBER) · issue_comment · 2025-06-18
Hm I will take a look and train on dummy small dataset, maybe we need to patch something in the trainer 🙃

## [c2983992030] zucchini-nlp (MEMBER) · issue_comment · reply to [c2981737866] · 2025-06-18
@iglaweb the below sciprt works with the latest transformers installed from `main` if we let the trained save the checkpoint. But I am not sure if you are saving the model manually at the end with `model.save_pretrained()`

With `model.save_pretrained()` and loading back I get a state dict mismatch error (expected since you saved manually the model and happens with any other model), but no error about the shape. Can you provide a minimal reproducer with dummy dataset? 

```
from trl import SFTConfig, SFTTrainer, get_kbit_device_map
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
[… 66 lines of code omitted]

## [c2988778634] iglaweb (NONE) · issue_comment · reply to [c2983992030] · 2025-06-19
@zucchini-nlp Thanks a lot for the snippet. Once I changed `model.save_pretrained()` to `trainer.save_model` together with `args.output_dir` for the training, I was able to make an inference successfully on a fine-tuned model. But, then, the model output for any video used in the prompt, is the following: 

> I'm sorry, but I cannot answer your question as there is no video attached to this prompt. Please provide me with the necessary information so that I can assist you better. 

So, this is the answer which I always get from my fine-tuned model. If I use just the pre-trained model with `AutoModelForImageTextToText.from_pretrained`, the answer is fine, no warning about missing video. The videos do exist on the disk.
I used the following code for a video inference:

```python
def run_model_single_inference(model, processor, video_path, llm_prompt, num_frames=8, max_token_length=100):
    conversation = [
        {
            "role": "user",
[… 28 lines of code omitted]

The following code is being used for training data preparation:
```python
def prepare_dataset(example, llm_text_prompt, max_pixels, fps=1.0) -> dict[str, list[dict[str, Any]]]:
    video_path = example["short_clip_path"]
    answer = example["caption"]
    system_message = "You are an expert in video analysis."
    messages = [
[… 11 lines of code omitted]
```

> With model.save_pretrained() and loading back I get a state dict mismatch error (expected since you saved manually the model and happens with any other model), but no error about the shape.

Hm, why should I expect to get a state dict mismatch error? Could you explain the purpose of each method and when should I use each of them? Which one is best to use if I want to load the fine-tuned weights?
`model.save_pretrained()`
`trainer.save_model()`
`trainer.model.save_pretrained`
`training_args.output_dir`
`model.save_pretrained()` works without any errors for the Llava-Next-Video.

I fine-tuned a model with `peft_config=peft_config,` but, then the following code does not work. Is it normal?
```python
adapter_path = 'temp_qwen/adapter_config.json'
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
if os.path.exists(adapter_path):
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = PeftModel.from_pretrained(
      model,
     'temp_qwen',
      device_map="auto",
      trust_remote_code=True).eval().cuda()
```

## [c2990019974] zucchini-nlp (MEMBER) · issue_comment · reply to [c2988778634] · 2025-06-20
Hmm, I don't really know how SFT uses your input messages, probably it is expecting `video` inputs as a separate key in the dataset. I suggest to look at how the example scripts handle inputs [here](https://github.com/huggingface/trl/blob/main/examples/scripts/sft_video_llm.py). In any case, seems like the training went wrong

> Hm, why should I expect to get a state dict mismatch error? 

Using PEFT changes the model's state dict and when you just save the model, it will be saved with extra `base_model.` prefix in the state dict. Using `trainer` to save will remove the extra keys and hooks added on top of the model, and save it correctly for you. In general, when training you don't have to save the model yourself and the trainer saves the outputs every `K` steps as indicated in training args. For all other cases `model.save_pretrained()` works perfectly fine

## [c2998528329] iglaweb (NONE) · issue_comment · reply to [c2990019974] · 2025-06-24
@zucchini-nlp Thanks a lot for quick response. As you mentioned, the problem was in the message template. I used `"path": video_path,` instead of `"video": video_path`. Right now, the training and inference works. But...

There is one more issue. Once I fine-tuned the model and saved it with `trainer.save_model('./saved_model')`, I'm not able t load a correct processor through `processor = AutoProcessor.from_pretrained('./saved_model')`. For some reason, it is **Qwen2TokenizerFast** instead of **Qwen2_5_VLProcessor** (got an error that `images` and `videos` args not found). Then, I have to use the default name explicitly `processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')`. 
So, why does it happen? Should I use `processor.save_pretrained` in this case? 
Is it safe to use the default huggingface processor id after training? Does the Trainer update the tokenizer?

## [c2998998232] zucchini-nlp (MEMBER) · issue_comment · 2025-06-24
Tbh, I don't really know if the trainers can save the processor currently. I see that in Trainer you got `processing_class=processor.tokenizer` so it means the trainer has no chance to save the processor without access to it

My question is, if we pass the processor as a processing class, does that fail during training (I believe it shouldn't)? And yeah, using the official processor will also work since we didn't change when tuning :)

## [c3005640440] iglaweb (NONE) · issue_comment · reply to [c2998998232] · 2025-06-25
@zucchini-nlp 
> My question is, if we pass the processor as a processing class, does that fail during training (I believe it shouldn't)? And yeah, using the official processor will also work since we didn't change when tuning :)

Thank you. Training is not affected now, it runs successfully.

I just noticed a new error raising while doing an inference on a fine-tuned model (from "Qwen/Qwen2.5-VL-3B-Instruct"). It fails on a same input every time, but prompts are very similar (the video input is slightly different). 

```
C:\actions-runner\_work\pytorch\pytorch\pytorch\aten\src\ATen\native\cuda\TensorCompare.cu:110: block: [0,0,0], thread: [0,0,0] Assertion `input[0] != 0` failed.
 21%|██        | 23/111 [02:28<09:29,  6.48s/it]
Traceback (most recent call last):
  File "hf_qwen_demo_video.py", line 420, in <module>
[… 15 lines of code omitted]

Any suggestions how to avoid it? Let me know what additional logs may help here.

UPD.  Adding `do_sample=False` to `model.generate` eliminates an exception.

## [c3018200849] zucchini-nlp (MEMBER) · issue_comment · 2025-06-30
Might be some over/underflowing issue if the final logits have `NaN`. Can you try to change the `torch_dtype` and the `attn_implementation` when loading the model?

I remember Qwen had some issues with specific combinations of those, for example in https://github.com/huggingface/transformers/issues/33294

## [c3029835307] iglaweb (NONE) · issue_comment · reply to [c3018200849] · 2025-07-03
@zucchini-nlp I switched to `attn_implementation='flash_attention_2'` and then I got the following error:
`AttributeError: 'Qwen2_5_VLVisionAttention' object has no attribute 'is_causal'`
But it looks like it is a known issue in 4.53.0 https://github.com/huggingface/transformers/issues/39095
Can't use FA2 with 4.52.4 due to "RuntimeError: FlashAttention only supports Ampere GPUs or newer.". I have RTX 8000

I did tests with v. 4.52.4 and 4.53.0. Qwen2.5-VL model was trained with **LoRA only**:
`torch_dtype=torch.float16, attn_implementation='sdpa` produces "!!!" output for all cases
`torch_dtype=torch.bfloat16, attn_implementation='sdpa` leads to CUDA out of memory
`torch_dtype=torch.float32, attn_implementation='sdpa` produces "!!!" output for all cases

`attn_implementation='eager'` leads to CUDA out of memory

`do_sample=False`, no issue
`do_sample=True` raises exception
```
RuntimeError: CUDA error: device-side assert triggered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

Model trained **with LoRA and QLoRA** does not lead to `RuntimeError: CUDA error: device-side assert triggered...` and produces a valid output / answer. Checked with v. 4.52.4. 
It also works with `do_sample=False` as well as `do_sample=True`. Tested with `sdpa` only.

---
outcome: state=closed
