# #44407 [pull_request] docs: add energy efficiency considerations to bitsandbytes quantization guide
https://github.com/huggingface/transformers/pull/44407  ·  opened by hongping-zh on 2026-03-03

## [body] hongping-zh (NONE) · opening post · 2026-03-03
## Summary

Adds an "Energy Efficiency Considerations" section to the bitsandbytes quantization documentation, providing practical guidance on the energy implications of different quantization configurations.

## Motivation

This addresses the suggestion from @SunMarc in bitsandbytes-foundation/bitsandbytes#1882 to update transformers documentation with energy efficiency insights based on systematic benchmarking.

## Changes

Added a new section covering:
- **INT8 mixed-precision trade-offs**: Explains the 17-33% energy overhead of default `llm_int8_threshold=6.0` as a justified accuracy trade-off
- **Threshold configuration guidance**: Documents why `threshold=0.0` is not recommended (only 3% energy savings vs 25% perplexity degradation)
- **Model size considerations**: NF4 efficiency crossover point at ~5B parameters
- **Batch size impact**: 84-96% energy reduction from batch_size=1 to batch_size=8-64

## Testing

- Documentation follows existing markdown formatting
- Content is concise and actionable for users
- Links verified

## References

- Related bitsandbytes PR: https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1882
- Benchmark data: https://github.com/hongping-zh/ecocompute-ai
- Interactive dashboard: https://hongping-zh.github.io/ecocompute-dynamic-eval/

## [c3988649322] hongping-zh (NONE) · issue_comment · 2026-03-03
Hi @SunMarc, 

Thank you for the suggestion in bitsandbytes-foundation/bitsandbytes#1882 to update the transformers documentation. I've added a concise "Energy Efficiency Considerations" section based on systematic benchmarking across multiple GPU architectures.

I would greatly appreciate your feedback on this documentation addition. Additionally, I have more comprehensive research on this topic that I'm preparing for academic publication. If you're interested, I'd be happy to discuss potential collaboration opportunities or seek your guidance on the work.

Looking forward to your thoughts!

## [c3882645545] SunMarc (MEMBER) · review APPROVED · 2026-03-03
Thanks !

## [c4016022752] hongping-zh (NONE) · issue_comment · 2026-03-07
Thank you for reviewing! Please let me know if any changes are needed for the CI checks, happy to address them.

## [c4020915492] hongping-zh (NONE) · issue_comment · reply to [c3882645545] · 2026-03-09
Hi @SunMarc, thank you for the approval! Just checking — is there anything else needed on my end before this can be merged? Happy to fix any CI issues if needed. Thanks!

## [c4125988544] hongping-zh (NONE) · issue_comment · reply to [c3882645545] · 2026-03-25
[Quantization_Energy_Crossover_Analysis.pdf](https://github.com/user-attachments/files/26242217/Quantization_Energy_Crossover_Analysis.pdf)
@SunMarc 

Hi Marc! Hope you're doing well. 

Following up on the energy efficiency guidance that was merged from this PR, I've expanded the study significantly and discovered something I think you'll find interesting: a **quantization-energy crossover effect**.

Key finding: NF4 quantization actually *increases* energy consumption for models below ~3.3B parameters (up to +29%) due to dequantization overhead, while providing expected savings for larger models. This holds across 5 GPU generations including RTX-5090.

I've uploaded the crossover analysis figure showing this effect. Would you be open to reviewing it and potentially providing a brief endorsement for the paper? The findings extend the guidance already in the Optimum docs with hardware-specific thresholds.

Happy to share the full details via email if you're interested. Thanks again for your support on the original PR!

---
outcome: state=open, merged=False
