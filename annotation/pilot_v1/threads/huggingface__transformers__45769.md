# #45769 [pull_request] feat: add bf16_loss training argument for VRAM-efficient QLoRA by keeping loss in BF16 during training to save ~1.4 GB VRAM
https://github.com/huggingface/transformers/pull/45769  ·  opened by butterwecksolutions on 2026-05-03

## [body] butterwecksolutions (NONE) · opening post · 2026-05-03
## Problem                                                                                                                                                                                                                                
                                                                                                                                                                                                                                            
  During QLoRA training with `--bf16`, logits are upcast from BF16 to                                                                                                                                                                       
  FP32 during loss computation, allocating 600 MB–1.4 GB per logit                                                                                                                                                                          
  tensor at model forward time. This is disproportionately expensive                                                                                                                                                                        
  when the model weights are already quantized to 4-bit precision.                                                                                                                                                                          
                                                                                                                                                                                                                                            
  ## Fix                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                            
  New `--bf16_loss` flag (default: `False`, opt-in). When set:                                                                                                                                                                              
  - Requires `--bf16` (validated at startup)                                                                                                                                                                                                
  - Wraps `model.forward()` in `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`
  - Logits stay in BF16 → ~600 MB–1.4 GB saved per forward pass                                                                                                                                                                             
                                                                                                                                                                                                                                            
  ## Trade-off                                                                                                                                                                                                                              
                                                                                                                                                                                                                                            
  Negligible precision loss for most workloads. QLoRA already                                                                                                                                                                               
  quantizes weights to 4-bit — BF16 logits preserve more signal                                                                                                                                                                             
  than the quantization boundaries discard.                       
                                                                                                                                                                                                                                            
  ## Verification                                                                                                                                                                                                                           
                                                                                                                                                                                                                                            
  Tested on Qwen3-VL-9B QLoRA training (RTX 3090 24 GB):                                                                                                                                                                                    
                                                                                                                                                                                                                                            
  | Metric | fp32 loss | bf16 loss |                                                                                                                                                                                                        
  |--------|----------|-----------|                               
  | Logit tensor dtype | FP32 (4 bytes/elt) | BF16 (2 bytes/elt) |                                                                                                                                                                          
  | Peak VRAM | 22.5 GB (before other optimizations) | ~21 GB (estimated) |                                                                                                                                                                 
  | Loss convergence | identical | identical |                                                                                                                                                                                              
  | Gradients | baseline | indistinguishable |                                                                                                                                                                                              
                                                                                                                                                                                                                                            
  Combined with [bitsandbytes#1935](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1935) and [TRL#5694](https://github.com/huggingface/trl/pull/5694), peak VRAM drops                                                                                                                                                                             
  from ~22.5 GB to ~12.5 GB in full QLoRA training.

## [c4379708831] Rocketknight1 (MEMBER) · issue_comment · 2026-05-05
cc @sunmarc @benjaminbossan

## [c4379998864] BenjaminBossan (MEMBER) · issue_comment · reply to [body] · 2026-05-05
Thanks for the PR @butterwecksolutions. Do you have a small script that can be used to measure this difference?

## [c4383232601] butterwecksolutions (NONE) · issue_comment · reply to [c4379998864] · 2026-05-05
> Thanks for the PR @butterwecksolutions. Do you have a small script that can be used to measure this difference?

OK, this request is a result of a debug marathon of my training pipeline. i currently build a reproducer script with all my fixes and update this within the next few days. Please wait for my results until further investigation.

## [c4387625020] BenjaminBossan (MEMBER) · issue_comment · 2026-05-06
Thanks for working on the reproducer. Something very simple, even with dummy data, would be absolutely fine.

## [c4394288579] butterwecksolutions (NONE) · issue_comment · 2026-05-07
## Systematic Isolation Complete — No Measurable Effect at Current Scale                                                                                                                                                                  
                                                                                                                                                                                                                                            
  Same 14-test reproducer (7 patches × 2 modes, fresh GPU per test)                                                                                                                                                                         
  used to isolate the VRAM leak root cause.                                                                                                                                                                                                 
                                                                                                                                                                                                                                            
  **bf16_loss training arg (#45769): 0.0 GB effect.**                                                                                                                                                                                       
                                                                                                                                                                                                                                            
  | Mode | Baseline | P2 (#45769) | Δ |                                                                                                                                                                                                     
  |------|----------|-------------|---|                           
  | VL (seq_len=512) | 20.3 GB | 20.3 GB | **0.0 GB** |                                                                                                                                                                                     
  | TEXT (seq_len=4096) | 8.5 GB | 8.5 GB | **0.0 GB** |                                                                                                                                                                                    
                                                                                                                                                                                                                                            
  The concept is sound — keeping loss in BF16 avoids an fp32 upcast that                                                                                                                                                                    
  saves VRAM in principle. But at this model scale (9B, QLoRA, RTX 3090),                                                                                                                                                                   
  the measurable impact is 0.0 GB — the loss tensor is not the bottleneck.                                                                                                                                                                  
                                                                                                                                                                                                                                            
  The dominant VRAM leak (7.0 GB) traces entirely to async CUDA streams in                                                                                                                                                                  
  OffloadActivations (TRL #5700). That fix alone drops VL training from                                                                                                                                                                     
  20.3→13.3 GB. All other candidate fixes show zero additional effect.                                                                                                                                                                      
                                                                                                                                                                                                                                            
  This PR isn't wrong — it just doesn't move the needle for the specific                                                                                                                                                                    
  leak we were chasing. The feature flag itself may still be useful for                                                                                                                                                                     
  other configurations.                                                                                                                                                                                                                     
                                                                                                                                                                                                                                            
  **Full reproducer, HTML report, raw training logs:**                                                                                                                                                                                      
  - https://github.com/butterwecksolutions/Testrepo/tree/main/vram-fix-reproducer                                                                                                                                                           
  - https://butterwecksolutions.github.io/Testrepo/vram-fix-reproducer/vram_test_results.html                                                                                                                                               
                                                                                                                                                                                                                                            
  Closing this PR. Thanks for the review.

## [c4396122589] BenjaminBossan (MEMBER) · issue_comment · reply to [c4394288579] · 2026-05-07
@butterwecksolutions Thanks for double-checking. If you find a situation where this leads to a significant reduction in memory usage, feel free to re-open the PR.

---
outcome: state=closed, merged=False
