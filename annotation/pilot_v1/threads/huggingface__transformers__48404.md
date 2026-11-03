# #48404 [pull_request] Add Shensi and Shensi-VL models
https://github.com/huggingface/transformers/pull/48404  ·  opened by zongzhilou on 2026-08-29

## [body] zongzhilou (NONE) · opening post · 2026-08-29
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=48404) [[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=48404)

## What does this PR do?

Adds two new models sharing a "DeepRecur" design: **Shensi** (text,
`ShensiConfig` / `ShensiForCausalLM`, checkpoint `louzongzhi/Shensi-Nano`) built on
the DeepSeek-V4 backbone, and **Shensi-VL** (multimodal, `ShensiVlConfig` /
`ShensiVlForConditionalGeneration`, checkpoint `louzongzhi/Shensi-VL-Nano`) pairing
the Shensi language tower with a Kimi K3 vision tower (MoonshotAI/Kimi-K3).

### Shensi (text)

Three changes vs. the vanilla V4 stack:

1. **GDAR-XHC** — residual streams reworked following xHC
   ([arXiv:2607.14530](https://arxiv.org/abs/2607.14530)): the residual runs as
   `hc_mult=16` parallel streams with per-stream pre/route/post gating, only the
   active streams are updated per step (no Sinkhorn constraint). The Gated Delta
   Attention Residual replaces the remaining mHC machinery: each stream does
   cross-layer attention *internally* — a delta rule keeps a per-block slot memory
   with decay/erase/write gates, normalized-key retrieval, and softmax routing over
   the slots of previous blocks — instead of stacking an extra attention module on
   top of the hyper-connection.

2. **Block Router & Block UniPool** — router sharing across a block of layers
   ([arXiv:2603.18297](https://arxiv.org/abs/2603.18297), Path-MoE) combined with a
   block-level unified expert pool ([arXiv:2605.06665](https://arxiv.org/abs/2605.06665),
   UniPool): the block's write layer owns the top-k router and the expert pool, the
   read layers of the same block share them — one router and one pool per block
   instead of per layer (parameters tied in the state dict). Shared experts are
   removed following [arXiv:2605.11689](https://arxiv.org/abs/2605.11689)
   (Slicing and Dicing: shared experts barely move the needle).

3. **Block aux_loss & erc_loss** — kept at block level from an information-theoretic
   angle: the standard load-balancing `aux_loss` minimizes the mutual information
   between inputs and routed experts (not strictly required once the path constraint
   removes the bias, but kept as a regularizer), while `erc_loss`
   ([arXiv:2512.23447](https://arxiv.org/abs/2512.23447)) maximizes the mutual
   information between outputs and routed experts. Together they give routing
   stability and expert specialization, stacking on the specialization already
   enforced by the path constraint for deep cooperation.

### Shensi-VL (multimodal)

Shensi language tower + Kimi K3 vision tower, interleaved into **DeepRecur** blocks
that rely on GDAR:

- cross blocks exchange evidence/guidance between the towers — language queries
  retrieve vision evidence, language guidance is written back to vision;
- the loop block runs a continuous latent reasoning loop (Coconut-style): the
  language stream is refined with damped write-backs, and the loop stops on a
  data-driven criterion — hidden-stream delta, reasoning-state delta and orbit
  recurrence are combined, no hard iteration cap.

### Notes

- Docs: `docs/source/en/model_doc/shensi.md`, `shensi_vl.md`; tests:
  `tests/models/shensi`, `tests/models/shensi_vl`.
- The VL composite owns the tower layers; blocks only reference slices of them, so
  state dicts stay alias-free and save/load round-trips exactly.

## Who can review?

- text models: @ArthurZucker @Cyrilvallez @vasqu
- multimodal models: @zucchini-nlp

---
outcome: state=closed, merged=False
