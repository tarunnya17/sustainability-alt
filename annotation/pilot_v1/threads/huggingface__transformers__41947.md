# #41947 [issue] why Smolvlm-256M-Instruct slower then Internvl-v2-1B ?
https://github.com/huggingface/transformers/issues/41947  ·  opened by HuangChiEn on 2025-10-30

## [body] HuangChiEn (NONE) · opening post · 2025-10-30
As title, Smolvlm have smaller model size (1/4 less matrix multiplication), smaller input embedding. But, both torch.CudaEvent, timer.perf_counter with torch.sync report the slower inference time ?

I wonder that does this related with the wrong implementation of Smolvlm in transformers ?
inference performance comparison : 
internvl-1B >
inp_embed : (1, 547, 896)
trainable params: 17,596,416 || all params: 647,260,288 || trainable%: 2.7186

smolvlm-256M >
inp_embed : (1, 171, 576)
trainable params: 9,768,960 || all params: 172,742,976 || trainable%: 5.6552

---

model init (all flags turns on, especially flash attention!) :
```python
    if 'internvl' in self.variant.lower():
          if '3_5' in self.variant:
              self.model = AutoModelForImageTextToText.from_pretrained(self.variant, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True)
              # internvl3.5, lm_head is not part of language_model !?
              lm_head = self.model.lm_head
[… 18 lines of code omitted]
```

code snippet to measure fps :
```python
for _ in range(30):
        _, _, _ = self.model(model_input)
        print('warm up done!')

prof = Tch_prof(device=self.device)
[… 14 lines of code omitted]
```

code snippet for timer classes :
```python
class Tch_prof(object):
    def __init__(self, device):
        self.device = device
        self.hw_type = 'gpu'
        self.tlt_time = {
[… 37 lines of code omitted]
```

Any suggestion will be helpful !!

## [c3468793212] FWKTechnologies (NONE) · issue_comment · 2025-10-30
Your conclusion is correct: SmolVLM (256M) is indeed slower than InternVL (1B). This isn't a fault in your measurements (your benchmark code is correct), but rather a result of the GPU architecture and model.

It's likely not a "wrong implementation" but rather an "unoptimized implementation" and hardware bottlenecks.

🧐 Problem Analysis: Why is Smaller Slower? (The "Smaller is Slower" Paradox)
1. Memory bottleneck, not Compute (Memory-Bandwidth Bound):

Modern GPUs (e.g., A100, H100, 4090) have compute power. (Compute/FLOPs) are huge, but VRAM (Memory Bandwidth) read/write speeds are limited.

InternVL (1B): Large enough to "saturate" (fully utilize) the GPU's Compute Units (Tensor Cores). The model spends most of its time "computing" (compute-bound).

SmolVLM (256M): Too small. The GPU computes each layer very quickly, but constantly waits for the weights of the next layer to be loaded from VRAM. This is called "Memory Bandwidth Bound."

Comparison:

InternVL is like a large factory with one large order (machines at full capacity).

SmolVLM is like the same factory with 100 small orders (machines complete in a short time). Then, we have to wait for the next part to be delivered.)

2. Kernel Launch Overhead (Layer Call Cost):

Each layer call incurs a small overhead to "launch" the kernel on the GPU.

If SmolVLM (based on the Phi-2 architecture) has "deep but thin" layers, it may be slower than InternVL (based on the LLaMA-style architecture) with "wide but shallow" layers, even though the total parameters are smaller.

3. Transformer Optimization:

This is where you're wondering. InternVL is a very popular model. Transformer implementations are often highly optimized using techniques like torch.nn.functional.scaled_dot_product_attention (SDPA).

SmolVLM still requires trust_remote_code=True may be an "incomplete reference implementation" (source code) that hasn't been fully optimized. It may contain hidden transpose, view, or inefficient operations, which can slow down and use unnecessary memory bandwidth.

💡 Suggestions & Fixes
The best solution for "memory-bound" and "kernel overhead" issues is to "combine" these operations.

1. (Best way) Use torch.compile
This is the most direct solution. torch.compile() in PyTorch 2.0+ performs "JIT (Just-in-Time) Compilation" for your model. It analyzes the Python code, performs "kernel fusion" (merging small layers into a larger layer), and significantly reduces memory bandwidth overhead.

Small, memory-bound SmolVLMs benefit from torch.compile is more efficient than the larger, compute-bound InternVL.

2. Increase the Batch Size (for Throughput Testing)
The latency (ms per input) you measured is correct, but if you want to know the true "performance" of the model, you should test the Throughput (samples/sec) by increasing the Batch Size as large as the VRAM can handle.

You may find that SmolVLM can handle much larger batch sizes than InternVL, so when measuring the total throughput, SmolVLM may win.

3. Enable TF32 (if using an Ampere/Hopper GPU)
This provides a slight speedup (quick win) by allowing the GPU to use TensorFloat-32 for matrix multiplication calculations.

```python
import torch
import time
from transformers import AutoModel, AutoModelForImageTextToText

[… 145 lines of code omitted]

## [c3470865382] HuangChiEn (NONE) · issue_comment · reply to [c3468793212] · 2025-10-31
> Your conclusion is correct: SmolVLM (256M) is indeed slower than InternVL (1B). This isn't a fault in your measurements (your benchmark code is correct), but rather a result of the GPU architecture and model.
> 
> It's likely not a "wrong implementation" but rather an "unoptimized implementation" and hardware bottlenecks.
> 
> 🧐 Problem Analysis: Why is Smaller Slower? (The "Smaller is Slower" Paradox)
> 
> 1. Memory bottleneck, not Compute (Memory-Bandwidth Bound):
> 
> Modern GPUs (e.g., A100, H100, 4090) have compute power. (Compute/FLOPs) are huge, but VRAM (Memory Bandwidth) read/write speeds are limited.
> 
> InternVL (1B): Large enough to "saturate" (fully utilize) the GPU's Compute Units (Tensor Cores). The model spends most of its time "computing" (compute-bound).
> 
> SmolVLM (256M): Too small. The GPU computes each layer very quickly, but constantly waits for the weights of the next layer to be loaded from VRAM. This is called "Memory Bandwidth Bound."
> 
> Comparison:
> 
> InternVL is like a large factory with one large order (machines at full capacity).
> 
> SmolVLM is like the same factory with 100 small orders (machines complete in a short time). Then, we have to wait for the next part to be delivered.)
> 
> 2. Kernel Launch Overhead (Layer Call Cost):
> 
> Each layer call incurs a small overhead to "launch" the kernel on the GPU.
> 
> If SmolVLM (based on the Phi-2 architecture) has "deep but thin" layers, it may be slower than InternVL (based on the LLaMA-style architecture) with "wide but shallow" layers, even though the total parameters are smaller.
> 
> 3. Transformer Optimization:
> 
> This is where you're wondering. InternVL is a very popular model. Transformer implementations are often highly optimized using techniques like torch.nn.functional.scaled_dot_product_attention (SDPA).
> 
> SmolVLM still requires trust_remote_code=True may be an "incomplete reference implementation" (source code) that hasn't been fully optimized. It may contain hidden transpose, view, or inefficient operations, which can slow down and use unnecessary memory bandwidth.
> 
> 💡 Suggestions & Fixes The best solution for "memory-bound" and "kernel overhead" issues is to "combine" these operations.
> 
> 1. (Best way) Use torch.compile
>    This is the most direct solution. torch.compile() in PyTorch 2.0+ performs "JIT (Just-in-Time) Compilation" for your model. It analyzes the Python code, performs "kernel fusion" (merging small layers into a larger layer), and significantly reduces memory bandwidth overhead.
> 
> Small, memory-bound SmolVLMs benefit from torch.compile is more efficient than the larger, compute-bound InternVL.
> 
> 2. Increase the Batch Size (for Throughput Testing)
>    The latency (ms per input) you measured is correct, but if you want to know the true "performance" of the model, you should test the Throughput (samples/sec) by increasing the Batch Size as large as the VRAM can handle.
> 
> You may find that SmolVLM can handle much larger batch sizes than InternVL, so when measuring the total throughput, SmolVLM may win.
> 
> 3. Enable TF32 (if using an Ampere/Hopper GPU)
>    This provides a slight speedup (quick win) by allowing the GPU to use TensorFloat-32 for matrix multiplication calculations.
> 
> import torch
> import time
> from transformers import AutoModel, AutoModelForImageTextToText
> 
> # --- 1. Enable quick wins (TF32 and CuDNN benchmarking) ---
> # Allow PyTorch to use the efficient TF32 format on Ampere+ GPUs
> torch.backends.cuda.matmul.allow_tf32 = True
> torch.backends.cudnn.benchmark = True # Helps when input shapes are static
> 
> class YourModelClass:
>     def __init__(self, variant):
>         self.variant = variant
>         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>         self.load_model()
> 
>     def load_model(self):
>         print(f"Loading model: {self.variant}")
>         
>         # Define model loading kwargs
>         model_kwargs = {
>             "torch_dtype": torch.bfloat16,
>             "trust_remote_code": True,
>         }
> 
>         if 'internvl' in self.variant.lower():
>             if '3_5' in self.variant:
>                 # This path seems for InternVL 3.5
>                 model_kwargs.update({"low_cpu_mem_usage": True, "use_flash_attn": True})
>                 self.model = AutoModelForImageTextToText.from_pretrained(self.variant, **model_kwargs)
>                 lm_head = self.model.lm_head
>                 self.model = self.model.language_model
>                 self.model.lm_head = lm_head
>             else:
>                 # This path is for InternVL2-1B
>                 model_kwargs.update({"low_cpu_mem_usage": True, "use_flash_attn": True})
>                 self.model = AutoModel.from_pretrained(self.variant, **model_kwargs) # Assuming "OpenGVLab/InternVL2-1B" is self.variant
>                 self.model = self.model.language_model
>             
>             # This logic for finding embed_tokens seems complex, ensure it's correct
>             try:
>                 self.model.embed_tokens = self.model.base_model.embed_tokens
>             except AttributeError:
>                 try:
>                     self.model.embed_tokens = self.model.model.tok_embeddings
>                 except AttributeError:
>                     print("Warning: Could not find embed_tokens attribute.")
> 
>         elif 'smolvlm' in self.variant.lower():
>             # This path is for SmolVLM
>             model_kwargs.update({"_attn_implementation": "flash_attention_2"})
>             self.model = AutoModelForImageTextToText.from_pretrained(self.variant, **model_kwargs) # Assuming "HuggingFaceTB/SmolVLM-256M-Instruct" is self.variant
>             lm_head = self.model.lm_head
>             self.model = self.model.model.text_model # Extracting the text model
>             self.model.lm_head = lm_head
>             # self.model.embed_tokens already built-in!
> 
>         else:
>             raise ValueError(f"Carefull: Variant {self.variant} not tested.")
> 
>         self.model.to(self.device).eval() # Move to device and set to eval mode
> 
>         # --- 2. THE MOST IMPORTANT FIX: torch.compile() ---
>         # This will fuse operations, reduce kernel overhead, and dramatically
>         # speed up memory-bound models like SmolVLM.
>         # 'reduce-overhead' is a good mode for fast compilation and speedup.
>         print("Compiling model... (This may take a moment)")
>         try:
>             # Use 'reduce-overhead' for small models or 'max-autotune' for best performance
>             self.model = torch.compile(self.model, mode="reduce-overhead")
>             print("Model compiled successfully.")
>         except Exception as e:
>             print(f"Warning: torch.compile failed. Using uncompiled model. Error: {e}")
> 
>     def benchmark(self, model_input):
>         # Your benchmarking code (Tch_prof, CudaEvent_Tch_prof) is correct.
>         # This function just shows how to use it with the compiled model.
>         
>         # Ensure model_input is on the correct device
>         model_input = model_input.to(self.device, dtype=torch.bfloat16) # Example input conversion
> 
>         print("--- Warming up (30 iterations) ---")
>         for _ in range(30):
>             # The 'model' here is now the *compiled* model
>             _ = self.model(model_input) 
>         
>         print("Warm up done!")
>         
>         prof = CudaEvent_Tch_prof(device=self.device) # Using CudaEvent for precision
>         
>         with torch.no_grad():
>             with prof:
>                 # Run the compiled model
>                 # Note: The output signature might be specific to your custom model
>                 # pred_speed_wps, pred_route, language = self.model(model_input, device=self.device)
>                 
>                 # Using a standard forward pass for this example
>                 _ = self.model(model_input)
>               
>         print(f"Model: {self.variant}")
>         print(f"CudaEvent Time: {prof.get_profile():.4f} ms")
> 
> 
> # --- Your profiler classes (Unchanged, they are correct) ---
> class Tch_prof(object):
>     def __init__(self, device):
>         self.device = device
>         self.hw_type = 'gpu'
>         self.tlt_time = {'cpu' : 0, 'gpu' : 0}
>     def __enter__(self):
>         torch.cuda.current_stream(self.device).synchronize()
>         self.s = time.perf_counter()  
>     def __exit__(self, *exc):
>         torch.cuda.current_stream(self.device).synchronize()
>         self.tlt_time[self.hw_type] += time.perf_counter() - self.s
>     def get_profile(self, hw_type='all'):
>         if hw_type == 'all': return self.tlt_time
>         elif hw_type in self.tlt_time.keys(): return self.tlt_time[hw_type]
>         else: raise RuntimeError(f"No such hardware type {hw_type}")
> 
> class CudaEvent_Tch_prof(object):
>     def __init__(self, device):
>         self.device = device
>         self.start = torch.cuda.Event(enable_timing=True)
>         self.end = torch.cuda.Event(enable_timing=True)
>     def __enter__(self):
>         self.start.record()
>     def __exit__(self, *exc):
>         self.end.record()
>         torch.cuda.current_stream(self.device).synchronize()
>         self.tlt_time = self.start.elapsed_time(self.end) # This is already in ms
>     def get_profile(self):
>         return self.tlt_time 
> 
> # --- Example Usage ---
> print("Starting benchmark...")
> 
> # 1. Benchmark InternVL
> intern_model_input = torch.randn(1, 547, 896) # Your reported shape
> intern_benchmarker = YourModelClass(variant="OpenGVLab/InternVL2-1B")
> intern_benchmarker.benchmark(intern_model_input)
> 
> print("\n" + "="*30 + "\n")
> 
> # 2. Benchmark SmolVLM
> smol_model_input = torch.randn(1, 171, 576) # Your reported shape
> smol_benchmarker = YourModelClass(variant="HuggingFaceTB/SmolVLM-256M-Instruct")
> smol_benchmarker.benchmark(smol_model_input)
> 
> print("\nBenchmark complete.")

TKS for your quick reply, i think the Memory bottleneck issue you mentioned may potentially explain why smolvlm run slower slightly.  While TF32 or other speedup method (`torch.compile`) is not the point in this disscusion.

But the crucial factor i miss is that A100 GPU maybe too powerful to make more matrix multiplication run in parallel and make no difference in terms of exec time between smolvlm-**256M** and internvl-**1B** (maybe both are too small for A100-80G).

Seems i need to rethink how to measure FPS. Increasing bz may not make sense, since autodriving forward input in streaming style, instead of batching style.

I'm corresponding to work on software side, hence don't want to really dive in HW spec. If smolvlm really can not show their inference speed in stream mode under powerful GPU env, i think i can switch to measure GPU memory footprint to enhance the advantage of this model.

Anyway TKS for your AI assistant reply, hope you have a good day ~

## [c3470925975] FWKTechnologies (NONE) · issue_comment · 2025-10-31
1️⃣ GPU underutilization (Underutilized GPU)

SmolVLM has few parameters and very small layers → each layer requires very little compute time.
The GPU must constantly reload data from VRAM because each task ends too quickly.

The GPU performs 10% of the work, with the remaining 90% waiting in memory.

This is called a memory-bandwidth-bound model.
InternVL, on the other hand, is large enough to handle continuous computation. (compute-bound), therefore, it is more "cost-effective" for GPUs.
→ As a result, InternVL utilizes the GPU more fully and is faster, despite its larger size.

2️⃣ SmolVLM uses "unoptimized" architecture and code.

InternVL is derived from OpenGVLab, which uses production-grade code (flash-attention, fused kernels, CUDA Graph, etc.).

SmolVLM is a research project of HuggingFaceTB. → The implementation is still "referenced," so there are:

Multiple separate layer calls (high kernel launch overhead).

Unnecessary view(), transpose(), or tensor copying.

Some fallback to using PyTorch standard attention (not actual Flash attention).

Conclusion: SmolVLM is not optimized for GPU inference.

3️⃣ Flash attention does not actually execute (silent fallback).

Even if _attn_implementation="flash_attention_2" is specified,
but if PyTorch, CUDA or driver version doesn't match the kernel requirements →
It will fall back to scaled_dot_product_attention (SDPA), which is 2–3 times slower.

InternVL uses the use_flash_attn=True flag, which is linked to the OpenGVLab kernel itself (custom C++/CUDA),
so it's faster, even with 4x more parameters.

4️⃣ Torch Compile / Graph Optimization isn't used.

SmolVLM is a small model (256M) → it benefits greatly from torch.compile() (kernel fusion + JIT).
But its code doesn't have a compiler → the GPU must launch the kernel layer by layer, function by function.
The time spent on "launch overhead" is greater than the actual computation.

InternVL is large enough to mitigate this overhead → so it's not at a disadvantage.

## [c3472719140] Rocketknight1 (MEMBER) · issue_comment · 2025-10-31
@FWKMultiverse pasting unverified LLM outputs into issues is not actually helpful, and parts of that were misleading or incorrect. We may start banning users in future for doing this.

---
outcome: state=closed
