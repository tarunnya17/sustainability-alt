# #45229 [issue] Gemma4 31B-IT Multi-GPU inference CUDA OOM
https://github.com/huggingface/transformers/issues/45229  ·  opened by vaibhavBh-0 on 2026-04-03

## [body] vaibhavBh-0 (NONE) · opening post · 2026-04-03
### System Info

### Description
I updated my transformers module to 5.5.0 from 4.53.0 to try `google/gemma-4-31B-it` model. I was using `meta-llama/Llama-3.3-70B-Instruct` for the same set of prompts. The Llama model is able to process the prompt without any problems despite occupying more VRAM than Gemma4. Gemma4 on the other hand crashes giving CUDA OOM error in an isocompute multi-GPU setting of 4 A100s (80GB each)

[logs.txt](https://github.com/user-attachments/files/26471745/logs.txt)

### Environment
- transformers 5.5.0
- PyTorch 2.5.1
- Python 3.12.9
- Linux
- 4 A100s (80GB each)

### Who can help?

_No response_

### Information

- [x] The official example scripts
- [x] My own modified scripts

### Tasks

- [ ] An officially supported task in the `examples` folder (such as GLUE/SQuAD, ...)
- [x] My own task or dataset (give details below)

### Reproduction

### Steps to reproduce:
1. Load the model with `device_map='auto'` for `text-generation` pipeline.
2. Prepare a long (less than 128K tokens; gemma 4 32B IT supports 256K tokens) `prompt` following the chat template format.
3. Create some prompt_dict `prompt_dict = {'dummy section': prompt}`

In my use case, I am focusing on summarization of a private dataset which I cannot share.

### Code Snippets

```
model_id = MODEL_PATH_DICT['gemma-4-31B-it']
model_pipe = pipeline('text-generation', model=model_id, dtype=torch.bfloat16, device_map='auto') # 'auto' -> 'balanced' -> 'sequential'

def do_inference_gemma_4(model_pipe: TextGenerationPipeline, prompts_dict : dict):
[… 29 lines of code omitted]

### Crash log
[logs.txt](https://github.com/user-attachments/files/26471751/logs.txt)

### Expected behavior

### Expected Behavior
`google/gemma-4-31B-it` should process the prompt and generate the response without throwing any errors.

### Observed Behavior
`google/gemma-4-31B-it` loads using `auto` device_map, however, while generating VRAM is occupied for the first GPU leading to `OutOfMemoryError` .
`meta-llama/Llama-3.3-70B-Instruct` evenly distributes the kv cache across multiple GPUs and does not focus on the first GPU.

## [c4186306292] vaibhavBh-0 (NONE) · issue_comment · 2026-04-04
I also recreated this issue by installing a completely new environment from scratch.
### Environment
- transformers 5.5.0
- PyTorch 2.10.0
- Python 3.14.3
- Linux
- 4 A100s (80GB each)

## [c4241138572] vaibhavBh-0 (NONE) · issue_comment · 2026-04-14
I have upgraded to transformers 5.5.4 and I tried to compare the multi-GPU inference to ollama.
Ollama can easily handle 180K tokens of context, transformers still crashes due to OOM.

```
import torch
from transformers import pipeline, AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM, set_seed
import accelerate
from ollama import chat, ChatResponse
[… 84 lines of code omitted]

[oom_crash.txt](https://github.com/user-attachments/files/26695663/oom_crash.txt)

## [c4241245838] vaibhavBh-0 (NONE) · issue_comment · 2026-04-14
Hi @Cyrilvallez 

I am facing issues while generating (anything with >20K tokens) using `gemma-4-31B-it` on transformers (v.5.5.0 - v.5.5.4), but on ollama (v.0.20.5) `gemma4:31b-it-bf16` works without any model compression techniques like key value cache compression or model quantization for the entirety of 256K token context window.

Ollama uses flash attention by default whereas the crash indicates transformers was using SDPA. As per [Issue 45201: Support per-layer FlashAttention: FA2 for sliding layers, SDPA for global layers](https://github.com/huggingface/transformers/issues/45201) difference in memory consumption of SDPA and FA is not that great that it should cause a crash due to OOM error.

## [c4246565604] Aravind-11 (CONTRIBUTOR) · issue_comment · 2026-04-14
Hi, can i see the exact output? is it cuda oom or system oom?

## [c4246576192] vaibhavBh-0 (NONE) · issue_comment · 2026-04-14
It's cuda OOM. You can look at oom_crash.txt

## [c4256127921] Aravind-11 (CONTRIBUTOR) · issue_comment · reply to [c4246576192] · 2026-04-15
> It's cuda OOM. You can look at oom_crash.txt

Have you tried increasing memory to see if it crashes?? 

Also, it will be much easier to see the logs directly instead of having to download it. Thanks

## [c4256146185] Aravind-11 (CONTRIBUTOR) · issue_comment · reply to [c4256127921] · 2026-04-15
> > It's cuda OOM. You can look at oom_crash.txt
> 
> Have you tried increasing memory to see if it crashes?? 
> 
> Also, it will be much easier to see the logs directly instead of having to download it. Thanks

Also, where is the distributed call
For gpu sharding ?

## [c4256624815] vaibhavBh-0 (NONE) · issue_comment · reply to [c4256146185] · 2026-04-16
> > It's cuda OOM. You can look at oom_crash.txt
> 
> Have you tried increasing memory to see if it crashes??
> 
> Also, it will be much easier to see the logs directly instead of having to download it. Thanks

I tried to paste the log text here earlier, it didn't work, because it was too long. I will try that again when the opportunity arises. Or if you want me to post it here, I can paste it in a separate comment.

I am not sure what do you mean by increasing the memory? I have 320GB VRAM available (can't increase the GPU allocation). Both ollama and HF code are given all of it to use during the inference process. `ollama_inference` empties the GPUs as soon as the inference is done.

> > > It's cuda OOM. You can look at oom_crash.txt
> > 
> > 
> > Have you tried increasing memory to see if it crashes??
> > Also, it will be much easier to see the logs directly instead of having to download it. Thanks
> 
> Also, where is the distributed call For gpu sharding ?

`model = AutoModelForCausalLM.from_pretrained(hf_model_id, dtype='auto', device_map='auto')`

The `'auto'` takes care of the automatic GPU sharding, I have tried manual GPU sharding too, which distributes the model's layers (nearly uniformly - `gemma_4_interleaved_map`), but both gave the same result. 

GPU 0 (for `'auto'`) or GPU with index `gpu_offset` (for manual) is where the CUDA OOM occurs during the generation. The other GPUs only occupy the model layers but none of the tensors or KV cache - there is no growth in memory consumption during generation unlike other LLMs.

```
def gemma_4_interleaved_map(model_id: str, gpu_offset: int = 0, n_gpus: int = 4) -> dict[str, int]:
    config = AutoConfig.from_pretrained(model_id)
    layer_types = config.text_config.layer_types
    gpu_offset %= n_gpus
[… 32 lines of code omitted]

## [c4256633277] Aravind-11 (CONTRIBUTOR) · issue_comment · reply to [c4256624815] · 2026-04-16
Memory of each gpu.
On Wed, Apr 15, 2026 at 5:50 PM Vaibhav Bhargava ***@***.***>
wrote:

> *vaibhavBh-0* left a comment (huggingface/transformers#45229)
> <https://github.com/huggingface/transformers/issues/45229#issuecomment-4256624815>
>
> It's cuda OOM. You can look at oom_crash.txt
>
> Have you tried increasing memory to see if it crashes??
>
> Also, it will be much easier to see the logs directly instead of having to
> download it. Thanks
>
> I tried to paste the log text here earlier, it didn't work, because it was
> too long. I will try that again when the opportunity arises. Or if you want
> me to post it here, I can paste it in a separate comment.
>
> I am not sure what do you mean by increasing the memory? I have 320GB VRAM
> available. Both ollama and HF code are given all of it to use during the
> inference process. ollama_inference empties the GPUs as soon as the
> inference is done.
>
> It's cuda OOM. You can look at oom_crash.txt
>
> Have you tried increasing memory to see if it crashes??
> Also, it will be much easier to see the logs directly instead of having to
> download it. Thanks
>
> Also, where is the distributed call For gpu sharding ?
>
> model = AutoModelForCausalLM.from_pretrained(hf_model_id, dtype='auto',
> device_map='auto')
>
> The 'auto' takes care of the automatic GPU sharding, I have tried manual
> GPU sharding too, which distributes the model's layers (nearly uniformly -
> gemma_4_interleaved_map), but both gave the same result.
>
> GPU 0 (for 'auto') or GPU with index gpu_offset (for manual) is where the
> CUDA OOM occurs during the generation. The other GPUs only occupy the model
> layers but none of the tensors or KV cache - there is no growth in memory
> consumption during generation unlike other LLMs.
>
> def gemma_4_interleaved_map(model_id: str, gpu_offset: int = 0, n_gpus: int = 4) -> dict[str, int]:
>     config = AutoConfig.from_pretrained(model_id)
>     layer_types = config.text_config.layer_types
>     gpu_offset %= n_gpus
>     device_list = [(idx + gpu_offset) % n_gpus for idx in range(n_gpus)]
>
>     device_map = {'model.langauge_model.embed_tokens': device_list[0],
>                   'model.language_model.embed_tokens.weight': device_list[0]}
>
>     device_idx = 0
>
>     for block_idx, layer_type in enumerate(layer_types):
>         device_id = device_list[device_idx]
>         device_map[f'model.language_model.layers.{block_idx}'] = device_id
>
>         if layer_type == 'full_attention':
>             # Keep sliding window and full attention block on the same device. Shift by 1
>             # after full attention block.
>             device_idx += 1
>             device_idx %= n_gpus
>
>     device_id = device_list[device_idx]
>     device_map['model.language_model.norm'] = device_id
>     device_map['model.language_model.rotary_emb'] = device_id
>
>     # Gemma models share the embedding and unembedding layers. Tied parameters.
>     device_map['lm_head'] = device_map['model.langauge_model.embed_tokens']
>
>     # Non-textual encoders
>     device_map['model.vision_tower'] = device_id
>     device_map['model.embed_vision'] = device_id
>
>
>     return device_map
>
> model = AutoModelForCausalLM.from_pretrained(hf_model_id, dtype='auto', device_map=gemma_4_interleaved_map(hf_model_id))
>
> —
> Reply to this email directly, view it on GitHub
> <https://github.com/huggingface/transformers/issues/45229#issuecomment-4256624815>,
> or unsubscribe
> <https://github.com/notifications/unsubscribe-auth/AKDCEOXTCO5OQ3QHDZ5QEXL4WAU6TAVCNFSM6AAAAACXMEPT3SVHI2DSMVQWIX3LMV43OSLTON2WKQ3PNVWWK3TUHM2DENJWGYZDIOBRGU>
> .
> You are receiving this because you commented.Message ID:
> ***@***.***>
>

## [c4256647918] vaibhavBh-0 (NONE) · issue_comment · reply to [c4256633277] · 2026-04-16
> Memory of each gpu.
> […](#)

I can't this is all I have access to.

## [c4262519397] Aravind-11 (CONTRIBUTOR) · issue_comment · 2026-04-16
so the kv cache is only populated on GPU 0 and not distributing it based on the logs you pasted. @yonigozlan could you please help?

## [c4283294516] Aravind-11 (CONTRIBUTOR) · issue_comment · 2026-04-20
Hi @vasqu , is this an issue that you think would need fixes ? thank you.

## [c4284138598] vaibhavBh-0 (NONE) · issue_comment · 2026-04-20
```
causal_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
[… 12 lines of code omitted]
```

I did some some debugging. Shifting the masks creation to just the CPU prevents OOM for larger token inputs in [masking_utils.py](https://github.com/huggingface/transformers/blob/ef97a7525dc6e4665db4ba5e1ec40d79e43cc74e/src/transformers/masking_utils.py#L1192). If I do not pass `'cpu'` and just pass `device`, the underlying code creates two masks in it's computation which causes the first OOM on CUDA.

The second OOM CUDA crash happens while using the SDPA attention interface `sdpa_attention_forward`

What caught my eye was Gemma 3 (all it's variants) technical report mentions "[We use a Grouped-Query Attention (GQA) (Ainslie et al., 2023) with post-norm and pre-norm with RMSNorm (Zhang and Sennrich, 2019)](https://arxiv.org/pdf/2503.19786)". I was not sure if Gemma 4 employed GQA since it doesn't has a proper technical report. Gemma 4 models did not use

I decided to see if Gemma 3 models on HF use GQA during it's attention call. So I debug the script mentioned in my [ previous comment](https://github.com/huggingface/transformers/issues/45229#issuecomment-4241138572) with some modifications `hf_model_id = 'gemma-3-27B-it'` and a smaller context size of 123813 tokens (128K token limit).

In [sdpa_attention.py](https://github.com/huggingface/transformers/blob/ef97a7525dc6e4665db4ba5e1ec40d79e43cc74e/src/transformers/integrations/sdpa_attention.py#L58) `sdpa_kwargs` was never set with the key `'enable_gqa'` to `True` because `use_gqa_in_sdpa` returns `False` for Gemma 3 and Gemma 4 models.

As per the following code snippet (from [sdpa_attention.py](https://github.com/huggingface/transformers/blob/ef97a7525dc6e4665db4ba5e1ec40d79e43cc74e/src/transformers/integrations/sdpa_attention.py#L10)) GQA should be available to CUDA devices but because the attention_mask is already provided to `sdpa_attention_forward`, the code disables GQA for Gemma 3 and 4. This apparently contradicts what Google did with Gemma 3 and (I assume) Gemma 4. 

```
_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()
_is_torch_npu_available = is_torch_npu_available()
[… 24 lines of code omitted]

Could anyone explain me why GQA should be disabled in HF while GQA is present in the original implementation? LLama 3.3 70B Instruct on the other hand yields  `True` for `use_gqa_in_sdpa` as the causal mask passed to it is `None`

## [c4284263223] Aravind-11 (CONTRIBUTOR) · issue_comment · reply to [c4284138598] · 2026-04-20
> ```
> causal_mask = mask_interface(
>         batch_size=batch_size,
>         q_length=q_length,
>         kv_length=kv_length,
>         q_offset=q_offset,
>         kv_offset=kv_offset,
>         mask_function=mask_factory_function,
>         attention_mask=attention_mask,
>         allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
>         local_size=sliding_window,  # Additional kwarg for sdpa
>         dtype=dtype,  # Additional kwarg for eager
>         config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
>         use_vmap=use_vmap,  # Short-circuit to non-vmap expansions for the mask
>         device='cpu',
>     )
> # device = inputs_embeds.device
> causal_mask = causal_mask.to(device)
> ```
> 
> I did some some debugging. Shifting the masks creation to just the CPU prevents OOM for larger token inputs in [masking_utils.py](https://github.com/huggingface/transformers/blob/ef97a7525dc6e4665db4ba5e1ec40d79e43cc74e/src/transformers/masking_utils.py#L1192). If I do not pass `'cpu'` and just pass `device`, the underlying code creates two masks in it's computation which causes the first OOM on CUDA.
> 
> The second OOM CUDA crash happens while using the SDPA attention interface `sdpa_attention_forward`
> 
> What caught my eye was Gemma 3 (all it's variants) technical report mentions "[We use a Grouped-Query Attention (GQA) (Ainslie et al., 2023) with post-norm and pre-norm with RMSNorm (Zhang and Sennrich, 2019)](https://arxiv.org/pdf/2503.19786)". I was not sure if Gemma 4 employed GQA since it doesn't has a proper technical report. Gemma 4 models did not use
> 
> I decided to see if Gemma 3 models on HF use GQA during it's attention call. So I debug the script mentioned in my [ previous comment](https://github.com/huggingface/transformers/issues/45229#issuecomment-4241138572) with some modifications `hf_model_id = 'gemma-3-27B-it'` and a smaller context size of 123813 tokens (128K token limit).
> 
> In [sdpa_attention.py](https://github.com/huggingface/transformers/blob/ef97a7525dc6e4665db4ba5e1ec40d79e43cc74e/src/transformers/integrations/sdpa_attention.py#L58) `sdpa_kwargs` was never set with the key `'enable_gqa'` to `True` because `use_gqa_in_sdpa` returns `False` for Gemma 3 and Gemma 4 models.
> 
> As per the following code snippet (from [sdpa_attention.py](https://github.com/huggingface/transformers/blob/ef97a7525dc6e4665db4ba5e1ec40d79e43cc74e/src/transformers/integrations/sdpa_attention.py#L10)) GQA should be available to CUDA devices but because the attention_mask is already provided to `sdpa_attention_forward`, the code disables GQA for Gemma 3 and 4. This apparently contradicts what Google did with Gemma 3 and (I assume) Gemma 4.
> 
> ```
> _is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
> _is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)
> _is_torch_xpu_available = is_torch_xpu_available()
> _is_torch_npu_available = is_torch_npu_available()
> 
> 
> def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
>     """
>     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
>     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
>     """
>     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
>     if n_rep == 1:
>         return hidden_states
>     hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
>     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
> 
> 
> def use_gqa_in_sdpa(attention_mask: torch.Tensor | None, key: torch.Tensor) -> bool:
>     # GQA can only be used under the following conditions
>     # 1.cuda or Ascend NPU
>     #   - torch version >= 2.5
>     #   - attention_mask is None (otherwise it will fall back to the math kernel)
>     # 2.xpu
>     #   - torch version >= 2.8
>     if _is_torch_xpu_available:
>         return _is_torch_greater_or_equal_than_2_8
>     return _is_torch_greater_or_equal_than_2_5 and attention_mask is None
> ```
> 
> Could anyone explain me why GQA should be disabled in HF while GQA is present in the original implementation? LLama 3.3 70B Instruct on the other hand yields `True` for `use_gqa_in_sdpa` as the causal mask passed to it is `None`

we can still pass gqa = true for sdpa in the torch function right ? without creating the mask manually, just pass is_causal as true

## [c4284690536] vaibhavBh-0 (NONE) · issue_comment · 2026-04-20
So, I tried that by setting the full_attention and sliding_attention mask to None in [modular_gemma4.py](https://github.com/huggingface/transformers/blob/ef97a7525dc6e4665db4ba5e1ec40d79e43cc74e/src/transformers/models/gemma4/modular_gemma4.py#L1713). Since the mask passed is None, all the other bool flags are set appropriately and GQA on SDPA is enabled.

I am able to progress from the initial layer to the first `full_attention` layer where I face CUDA OOM error. Gemma 3 27B crashes after the first `sliding_attention` layer. Both the models had 123821 tokens to process in their context with `'auto'` sharding of the models on 4 A100s (320GB).

gemma-4-31B-it Logs
```
Number of tokens in prompt : 123821
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1188/1188 [00:08<00:00, 143.53it/s]
Attention mask is None : False
Decoder Block Layer Type: sliding_attention Layer Idx 0
Decoder Block Layer Type: sliding_attention Layer Idx 1
[… 115 lines of code omitted]
```
 I think the next step would be to try flash attention. But, that is currently is not supported for Gemma 4 based on [[Gemma 4] Support per-layer FlashAttention: FA2 for sliding layers, SDPA for global layers](https://github.com/huggingface/transformers/issues/45201)

## [c4284719446] Aravind-11 (CONTRIBUTOR) · issue_comment · 2026-04-20
does flash work with sliding attention as well?? PYTORCH_ALLOC_CONF=expandable_segments:True did you try this ?

## [c4284775532] vaibhavBh-0 (NONE) · issue_comment · reply to [c4284719446] · 2026-04-20
> does flash work with sliding attention as well?? 

It apparently works on Ollama. But, [[Gemma 4] Support per-layer FlashAttention: FA2 for sliding layers, SDPA for global layers](https://github.com/huggingface/transformers/issues/45201) says that FA can possibly work for `'sliding_attention'` but not for `'global_attention'` where the crash seems to happen.

> PYTORCH_ALLOC_CONF=expandable_segments:True did you try this ?

That would not work, though I did try adding this environment variable while running the script. The forward pass is trying to allocate 57.12 GB on top of already total capacity of 79.25 GiB of which 20.59 GiB is free. Only 1.83 GiB is reserved by PyTorch but unallocated.

Adding the environment variable reduced the memory reserved to 124.66 MiB. That's clearly not enough for ~ 30+GB. And this is just 5 / 60 layers. Assuming `'auto'` distributes the layers uniformly, that's just 5 / 15 layers per device.

## [c4284960939] vaibhavBh-0 (NONE) · issue_comment · 2026-04-20
So, I got the model to _work_ for both the types of attention, `'sliding_attention'` and `'global_attention'`.

I initialized the attention implementation as follows
`model : Gemma4ForConditionalGeneration = AutoModelForCausalLM.from_pretrained(hf_model_id, dtype='auto', device_map='auto', attn_implementation='paged|sdpa')` 

However, I have to set sliding_attention mask to None. (full_attention is already None)

The output of the model is gibberish (on HF)
`{'test': {'role': 'assistant', 'content': '//額s---<eos>'}}`

The output of the model here is consistent with the task the model was given (Ollama)

`{'test': ChatResponse(model='gemma4:31b-it-bf16', created_at='2026-04-20T23:38:14.993661107Z', done=False, done_reason=None, total_duration=None, load_duration=None, prompt_eval_count=None, prompt_eval_duration=None, eval_count=None, eval_duration=None, message=Message(role='assistant', content='needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle', thinking=None, images=None, tool_name=None, tool_calls=None), logprobs=None)}`

Both the libraries used the same sampling parameters 

```
# Ollama
 options = {
        'temperature': 1.0, 
        'num_ctx': ctx_size,
[… 9 lines of code omitted]

## [c4285004200] Aravind-11 (CONTRIBUTOR) · issue_comment · reply to [c4284960939] · 2026-04-20
> So, I got the model to _work_ for both the types of attention, `'sliding_attention'` and `'global_attention'`.
> 
> I initialized the attention implementation as follows `model : Gemma4ForConditionalGeneration = AutoModelForCausalLM.from_pretrained(hf_model_id, dtype='auto', device_map='auto', attn_implementation='paged|sdpa')`
> 
> However, I have to set sliding_attention mask to None. (full_attention is already None)
> 
> The output of the model is gibberish (on HF) `{'test': {'role': 'assistant', 'content': '//額s---<eos>'}}`
> 
> The output of the model here is consistent with the task the model was given (Ollama)
> 
> `{'test': ChatResponse(model='gemma4:31b-it-bf16', created_at='2026-04-20T23:38:14.993661107Z', done=False, done_reason=None, total_duration=None, load_duration=None, prompt_eval_count=None, prompt_eval_duration=None, eval_count=None, eval_duration=None, message=Message(role='assistant', content='needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle needle', thinking=None, images=None, tool_name=None, tool_calls=None), logprobs=None)}`
> 
> Both the libraries used the same sampling parameters
> 
> ```
> # Ollama
>  options = {
>         'temperature': 1.0, 
>         'num_ctx': ctx_size,
>         'num_predict': 256 * 10 if not thinking else 5120,
>         'top_p': 0.95,
>         'top_k': 64,
>         'repeat_penalty': 1.5, # default is 1.1
>         'seed': 42
> }
> 
> # HF
> output = model.generate(**inputs, max_new_tokens=256 * 10, temperature=1.0, top_p=0.95, top_k=64, repetition_penalty=1.5, cache_implementation='offloaded')
> ```

wdym? did sliding_window execute or no ?

## [c4285076211] vaibhavBh-0 (NONE) · issue_comment · reply to [c4285004200] · 2026-04-21
> wdym? did sliding_window execute or no ?

Your earlier suggestion was to disable the attention masks. `'full_attention'` is always None (I did not change anything here).
'`sliding_attention`' is something I set to None manually. That made it work until the first 5 transformer layers. After that it would crash on the  `'full_attention'` layer.

In my previous comment I switched from `sdpa` to `paged|sdpa` that did not lead to any CUDA OOM error, but it gave me gibberish output which is bad.

## [c4302835129] Cyrilvallez (MEMBER) · issue_comment · 2026-04-23
Please provide a clear repro of the issue if you're a human.
It otherwise looks like you're 2 bots arguing with each other and not making sense

---
outcome: state=closed
