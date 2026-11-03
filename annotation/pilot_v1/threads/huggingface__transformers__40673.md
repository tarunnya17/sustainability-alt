# #40673 [pull_request] [Quantization] Allow loading of transform configs
https://github.com/huggingface/transformers/pull/40673  ·  opened by kylesayrs on 2025-09-03

## [body] kylesayrs (CONTRIBUTOR) · opening post · 2025-09-03
## Purpose ##
* Support loading models with online transforms applied via Compressed Tensors (LLM Compressor)

## Prerequisites ##
* https://github.com/neuralmagic/compressed-tensors/pull/445

## Changes ##
* Require a minimum compressed tensors version of `0.11.0` (to support transform features)
* Load transform configs (if available), and apply them to the model before weight loading
* (misc) Refactor compressed tensors tests to check for perplexity, rather than exact output matches
* (misc) Remove `update_dtype` in order to reduce complexity and give users more control/predictability of model data types

## Testing ##
* Regression tested using `CompressedTensorsTest`, added an online quip-style transformed model for testing
  * Perplexity results match expectations

## [c3253403665] Rocketknight1 (MEMBER) · issue_comment · 2025-09-04
cc @mekkcyber

## [c3254436359] kylesayrs (CONTRIBUTOR) · issue_comment · 2025-09-04
Putting in draft for now, need to do some more testing

## [c2334270635] brian-dellabetta (NONE) · review_comment on src/transformers/quantizers/quantizer_compressed_tensors.py:165 · 2025-09-09
do we have to call this `dequantize` to match huggingface API? If not, `decompress` would be more accurate since it might involve something beyond quantization?

## [c3202540770] brian-dellabetta (NONE) · review COMMENTED · 2025-09-09 (no text)

## [c2415165008] kylesayrs (CONTRIBUTOR) · review_comment on src/transformers/quantizers/quantizer_compressed_tensors.py:165 · reply to [c2334270635] · 2025-10-08
Yes, `dequantize` is the name of the HF api. Falls in line with the "quantization_config" really  being an "optimization/compression" config

## [c3316698249] kylesayrs (CONTRIBUTOR) · review COMMENTED · 2025-10-08 (no text)

## [c3658073341] kylesayrs (CONTRIBUTOR) · issue_comment · 2025-12-15
https://github.com/huggingface/transformers/pull/42887

---
outcome: state=closed, merged=False
