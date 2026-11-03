# #40005 [pull_request] [fix] Pass video inputs to plm
https://github.com/huggingface/transformers/pull/40005  ·  opened by 4g on 2025-08-07

## [body] 4g (NONE) · opening post · 2025-08-07
1. video_inputs were not being passed to plm, resulting in same results for all videos.
2. This was breaking the official example. more @ https://github.com/huggingface/transformers/issues/40004
3. tested locally with different videos

# Pass video inputs to plm

Fixes # [40004](https://github.com/huggingface/transformers/issues/40004) 

 
## Who can review?
@zucchini-nlp

---
outcome: state=closed, merged=False
