# #40328 [pull_request] [rfc] Prototype to make torch.compile work with DynamicCache
https://github.com/huggingface/transformers/pull/40328  ·  opened by anijain2305 on 2025-08-20

## [body] anijain2305 (CONTRIBUTOR) · opening post · 2025-08-20
# What does this PR do?

## Background on StaticCache vs DynamicCache
A static KV cache must be preallocated for the model’s maximum context window (e.g., 32k tokens), even if you only generate a few tokens. That leads to large memory pressure and wastage. A dynamic cache grows with the sequence actually in use: with P​ prefill tokens and D decode steps, memory scales with P + D instead of max_seq_length. For example, if you prefill 2,000 tokens and generate 50, the dynamic cache holds ~2,050 tokens’ worth of K/V per layer, versus 32k for a static allocation, significantly reducing memory pressure.

Lets do a quick experiment for Qwen3

[image]


In all cases, the output generated text was exactly the same.

As expected, StaticCache has a higher memory footprint. But the generation time is also higher. This is because the main tensors involved in [SDPA](https://github.com/huggingface/transformers/blob/242bb2cafccec9f90479f5f688bca9d240b1031f/src/transformers/integrations/sdpa_attention.py#L83) operation - (key, value, attention_mask) have max seq length shape. Even though attention_mask fires up only a portion of the actual compute, there is still more work to be done here compared to Dynamic KV cache. (Side chat - Driss and Boyuan told me that we might be able to use better eager code or flex decoding to make this better, but thats not the focus of this doc.)



## torch.compile and Dynamic Cache
transformers introduced StaticCache just for torch.compile. For Qwen3, torch.compile latency is 20 seconds, achieving 1.2x speedup and 25 seconds of compile time ([tlparse](https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/.tmpifaWBm/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000)). 

But as pointed out earlier, DynamicCache is a better default setting for eager PyTorch. So, we will look at the changes required to make torch.compile work with DynamicCache. WIth this PR,  I see torch.compile improve **generation latency to 3.6 (while eager was 8.88 seconds), leading to 2.4x speedup** and a cold start latency of 36 seconds ([tlparse](https://manifold.edge.x2p.facebook.net/v0/read/tree/logs/.tmpq4CUzG/index.html?bucketName=tlparse_reports&apiKey=tlparse_reports-key&withPayload=1&timeoutMsec=10000)) and warm start of 11 seconds.


For this RFC
* We will use fullgraph=True and do full model compilation (no regional compilation yet).
* We will compile only the decode step to keep the discussion simple.

There are two main changes
**Avoid recompilations (symbolic seq length)** : Tell torch.compile to treat sequence length as dynamic for all KV-cache–related tensors - key, value, and attention_mask, using torch._dynamo.mark_dynamic. To ensure a single shared symbolic size is used everywhere (since mark_dynamic alone can yield distinct symbols), wrap the model and insert explicit size-equality checks (e.g., via torch._check) that tie the symbols together. This nudges the compiler to unify on one seq-len symbol, improving inductor codegen and in future compile time.

**CUDA graphs**: Disable cudagraphs for DynamicCache. Because the sequence length changes step-to-step, cudagraphs would constantly re-record, which undercuts their benefit and can hurt latency. (They remain useful when shapes are fixed, e.g., with a static cache.)


### Update 

The PR now extends torch.compile to `DynamicSlidingWindowLayer`. There is one extra recompilation on the integer value of `cumulative_length` that I hope can be prevented by converting the integer into a scalar tensor.

Speedup - Pytorch eager has 6.66 seconds, compile takes 5.10 seconds for Gemma2. Although the compile time is high (currently due to 2 compilations).

## [c3209160907] anijain2305 (CONTRIBUTOR) · issue_comment · 2025-08-21
cc @ArthurZucker @gante

## [c2290300907] Cyrilvallez (MEMBER) · review_comment on src/transformers/cache_utils.py:978 · 2025-08-21
We need to be careful here, since https://github.com/huggingface/transformers/pull/40039 this is not necsarily the case on all layers (we need to do it separately on full and sliding layers)

## [c2290313414] Cyrilvallez (MEMBER) · review_comment on src/transformers/cache_utils.py:968 · 2025-08-21
Let's have a nice docstring here explaining that the `._check` calls here are used to tell the compiler in advance that they will all be the same symbolic size, which will allow it to optimize nicely in combination with `torch._dynamo.mark_dynamic` on this symbolic size.

Because `._check` is not documented as all, so we tried using it a few times already, but it's never clear how it works and what it allows 👌

## [c2290318260] Cyrilvallez (MEMBER) · review_comment on src/transformers/generation/utils.py:2111 · 2025-08-21
Agreed, we don't want to start compiling everything!

## [c2290325210] Cyrilvallez (MEMBER) · review_comment on src/transformers/generation/utils.py:3517 · 2025-08-21
When compiling, we create the final masks in advance between the forwards, because compilation of the mask creation is extremely tricky (borderline impossible in all scenarios), and it's not a bottleneck anyway (you can think of it as some kind of graph break, but we just do it outside)

## [c3139621190] Cyrilvallez (MEMBER) · review COMMENTED · 2025-08-21
Nice! Super happy to see you jump on this! Indeed, we waste A LOT of tokens when using StaticCache, so would be amazing if we can compile Dynamic with nice optimizations 🚀
Finally starting to understand the usage of `torch._check` a bit more!

## [c2290723526] gante (CONTRIBUTOR) · review_comment on src/transformers/generation/utils.py:2111 · reply to [c2290318260] · 2025-08-21
+1, compilation by default places additional requirements on the user's setup, which occasionally results in issues

## [c3140194727] gante (CONTRIBUTOR) · review COMMENTED · 2025-08-21
Very cool! 🔥

## [c3140656924] ArthurZucker (COLLABORATOR) · review COMMENTED · 2025-08-21
🚀

## [c3278869359] ArthurZucker (COLLABORATOR) · issue_comment · reply to [c3209160907] · 2025-09-11
@anijain2305 we are happy to help you / push this to merge it, tell us if you need anything!

## [c3285825235] Cyrilvallez (MEMBER) · issue_comment · reply to [c3209160907] · 2025-09-12
Yes sorry everyone, I told @anijain2305 I would take over and wanted to do it this week, but did not find the time 🥲 Will do next week!

## [c2352646676] Cyrilvallez (MEMBER) · review_comment on src/transformers/generation/utils.py:2789 · 2025-09-16
Do we need to mark them dynamic at each iteration, or could we do it only once at the beginning?
I assume we need it every iteration, but so can we move it in `model_wrapper` in `get_compiled_call` instead? Should it be outside the function that is actually being compiled? (i.e. any reason you defined it here specifically?) If possible, having all this logic in the wrapper would be simpler to understand!

## [c2352649605] Cyrilvallez (MEMBER) · review_comment on src/transformers/generation/utils.py:2797 · 2025-09-16
Any way to activate this when calling `torch.compile` on the forward instead of a context manager by any chance?

## [c2352665005] Cyrilvallez (MEMBER) · review_comment on src/transformers/cache_utils.py:1044 · 2025-09-16
Just setting them all to the same value in this way is enough right? Or does dynamo relies on the cross-objects dependencies somehow?

## [c3230198503] Cyrilvallez (MEMBER) · review COMMENTED · reply to [c3209160907] · 2025-09-16
Alright @anijain2305, just a few questions to better understand the compiler needs/structure before continuing and going for the full integration! 🤗

## [c3563505464] ArthurZucker (COLLABORATOR) · issue_comment · reply to [c3209160907] · 2025-11-21
@anijain2305 we are happy to push this further!

---
outcome: state=closed, merged=False
