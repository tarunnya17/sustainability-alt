# #47042 [pull_request] [`Gemma4`] Update 1 integration test
https://github.com/huggingface/transformers/pull/47042  ·  opened by vasqu on 2026-07-03

## [body] vasqu (COLLABORATOR) · opening post · 2026-07-03
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47042)

As per title, this comes from #46960 and the simple is fairly simple: SDPA selects its backend on certain criteria and in this case we switched the backend from math to the memory-efficient one which always introduces slighty different outputs due to kernel inaccuracies

cc @ydshieh

## [c4877303115] vasqu (COLLABORATOR) · issue_comment · 2026-07-03
run-slow: gemma4

## [c4626556819] ydshieh (COLLABORATOR) · review APPROVED · 2026-07-03 (no text)

## [c4877538819] Butterfingrz (CONTRIBUTOR) · issue_comment · reply to [c4877303115] · 2026-07-03
@vasqu Hi! I was wondering if you might have any more details about the root cause of the numerical accuracy issue in the SDPA-EA kernel? I’ve encountered a similar issue during training as well, and would really appreciate any insights. I’d also be happy to help investigate if needed.

## [c4877557966] vasqu (COLLABORATOR) · issue_comment · 2026-07-03
Tbh it's hard to tell. The bigger your model sizes the easier it is to slightly shift in exact numerics. This is like comparing FA to a simple math backend. Even small norms can have big influences on the final output and the more layers the more these different numerics accumulate

Does this mean that the EA backend is not working for you during training? Or that the results are different, e.g. different rate to converge?

## [c4877588734] Butterfingrz (CONTRIBUTOR) · issue_comment · 2026-07-03
The SDPA backend can train normally, but I noticed that the loss is slightly different from what I get when using FlexAttention

## [c4877627241] Butterfingrz (CONTRIBUTOR) · issue_comment · 2026-07-03
This might be due to the SDPA and Flex backends dispatching different kernels for the sliding-window layers in the Gemma4 model.

## [c4877642802] vasqu (COLLABORATOR) · issue_comment · reply to [c4877588734] · 2026-07-03
> The SDPA backend can train normally, but I noticed that the loss is slightly different from what I get when using FlexAttention

Well there will be other backends involved as flex attention uses other kernels as well. Not sure how it handles different head dims especially! You can try also eager and I would expect different values again

## [c4877655358] vasqu (COLLABORATOR) · issue_comment · 2026-07-03
This is the issue with normalization mostly: The attentions themself dont drift too much I would expect but when you combine attention + rms norm it will slightly drift more - repeat that x layers and you have bigger drifts at the end. It still has similar distributions I would expect

## [c4877796704] Butterfingrz (CONTRIBUTOR) · issue_comment · 2026-07-03
When I ran operator-level tests for attention with head_dim=512 under the SDPA-EA and Flex backends on an H200, I didn’t observe any numerical accuracy discrepancy.

## [c4877838839] Butterfingrz (CONTRIBUTOR) · issue_comment · 2026-07-03
Hi! Here are the kernels that are actually dispatched by the Flex and SDPA backends for the sliding-window and full-attention layers in the Gemma4 model.</p>
Backend | Layer type | head_dim | mask object | Dispatched aten/HOP op | Actual CUDA kernel
-- | -- | -- | -- | -- | --
sdpa | sliding | 256 | 4D bool Tensor | aten::_scaled_dot_product_cudnn_attention | cudnn_generated_fort_native_sdpa_sm90_flash_{fprop,bprop}_wgmma_f16
sdpa | full (no padding) | 512 | None (takes the is_causal fast path) | MATH — no fused op | decomposes into aten::bmm ×6 + aten::_softmax (materializes S×S)
sdpa | full (with padding) | 512 | 4D bool Tensor | aten::_scaled_dot_product_efficient_attention | cutlass mem-efficient: fmha_cutlassF/B_bf16_aligned_*_sm80
sdpa | sliding (with padding) | 256 | 4D bool Tensor | aten::_scaled_dot_product_cudnn_attention | same cuDNN flash as above
flex | sliding / full | 256 / 512 | BlockMask | torch.ops.higher_order.flex_attention (compiled) | inductor triton_tem_fused_flex_attention*

## [c4877854988] Butterfingrz (CONTRIBUTOR) · issue_comment · reply to [c4877655358] · 2026-07-03
> This is the issue with normalization mostly: The attentions themself dont drift too much I would expect but when you combine attention + rms norm it will slightly drift more - repeat that x layers and you have bigger drifts at the end. It still has similar distributions I would expect

Thanks a lot! I’ll take a closer look into this.

## [c4877860965] vasqu (COLLABORATOR) · issue_comment · 2026-07-03
Woah this is super interesting https://github.com/huggingface/transformers/pull/47042#issuecomment-4877838839

Could cudnn be the cause for more drifts?

## [c4877895937] vasqu (COLLABORATOR) · issue_comment · reply to [c4877854988] · 2026-07-03
I will let this be merged, @Butterfingrz would you like to open an issue on this. This is essentially a deeper topic tbh and reveals backend subtlety tbh

## [c4877917106] Butterfingrz (CONTRIBUTOR) · issue_comment · 2026-07-03
Sure, I can open an issue and follow up on this. Thanks a lot for your reply!

---
outcome: state=closed, merged=True
