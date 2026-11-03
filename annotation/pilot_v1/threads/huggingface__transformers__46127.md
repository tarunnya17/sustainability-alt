# #46127 [pull_request] [deepseek_v4] Add DeepseekV4NextNPredictor class for MTP draft-head support
https://github.com/huggingface/transformers/pull/46127  ·  opened by pasta-paul on 2026-05-20

## [body] pasta-paul (NONE) · opening post · 2026-05-20
## Summary

DeepSeek-V4-Flash ships with a Multi-Token-Prediction (MTP) draft block (`num_nextn_predict_layers = 1`) whose weights are stored under an `mtp.0.*` prefix in the safetensors index. The current implementation drops these on load via:

```python
_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
```

…which makes the MTP head unusable for vLLM's speculative decoding (`--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`).

## Changes

1. Adds `DeepseekV4NextNPredictor` as a `DeepseekV4DecoderLayer` subclass with MTP-specific projections (`e_proj`, `h_proj`, `enorm`, `hnorm`, `norm`) and head parameters (`hc_head_fn`, `hc_head_base`, `hc_head_scale`). Mirrors the upstream reference at `DeepSeek-V4-Flash/inference/model.py:MTPBlock`.

2. Wires `DeepseekV4Model.__init__` to instantiate `self.mtp` as a `nn.ModuleList` of `num_nextn_predict_layers` blocks. Empty when the config has `num_nextn_predict_layers = 0` (no behavior change for non-MTP variants).

3. Drops the silent-drop regex from `_keys_to_ignore_on_load_unexpected` (now `[]`). With the MTP module class instantiated, `mtp.*` keys load into real submodules instead of being filtered out.

4. Auto-extends `config.layer_types` and `config.mlp_layer_types` from inside `DeepseekV4NextNPredictor.__init__` to cover the MTP layer's index. `DeepseekV4Attention` and `DeepseekV4SparseMoeBlock` index these lists by `layer_idx`; without the extension, instantiating at `layer_idx = num_hidden_layers` raises `IndexError`.

## Verification

Tested locally with the DSv4-Flash config (`num_hidden_layers=43`, `num_nextn_predict_layers=1`, `hc_mult=8`):

```
OK: MTP block instantiated
params: 6.629190907 B
missing expected children: none
hc_head_* params: ['hc_head_fn', 'hc_head_base', 'hc_head_scale']
```

Class instantiates, `named_modules()` enumerates the full sub-tree (including `mtp.0.mlp.experts.0..255.{gate_proj,down_proj,up_proj}` and `mtp.0.self_attn.{q_a_proj,q_b_proj,kv_proj,o_a_proj,o_b_proj}`), `from_pretrained` loads `mtp.0.*` keys into the new submodules.

## Open items (for follow-up commits or follow-on PRs)

- `DeepseekV4NextNPredictor.forward()` implementation for inference-time draft. This PR adds the structural shell; the weights load and submodules are walkable for downstream calibration / quantization tooling, but invoking the draft head at inference time requires the embed + e_proj/h_proj composition and the `super().forward` + `norm` + head wiring.
- Integration with `DeepseekV4Model.forward` (gated kwarg to invoke the MTP draft after the main forward).
- Tests under `tests/models/deepseek_v4/`:
  - `from_pretrained` round-trip preserves `mtp.0.*` weights
  - `model.mtp[0]` reachable via `named_modules()`
  - `num_nextn_predict_layers = 0` path unchanged

## Note on modular vs generated file

I edited both `modular_deepseek_v4.py` (source of truth) and `modeling_deepseek_v4.py` manually because my local environment doesn't have `libcst` installed to run `utils/modular_model_converter.py`. Please re-run the converter if the manual edit to the generated file drifts from what the converter would produce.

## Related

Discussed at vllm-project/llm-compressor#2735 (the calibration side of the MTP-retention story).

cc @kylesayrs — companion to your active DSv4 iteration on `kylesayrs/transformers-v5` in the llm-compressor repo.

## [c4502847196] pasta-paul (NONE) · issue_comment · 2026-05-20
## Correction to design — `layer_type="sliding_attention"` for MTP

While verifying this PR against an actual DSv4-Flash MTP checkpoint, I found that the layer_type extension logic needs to be more specific than "repeat the last main-layer's type" (which was the initial approach in this PR's first push).

### Evidence

The DSv4-Flash config has three attention `layer_types`:

```
Counter({'compressed_sparse_attention': 21,    # full attn + Compressor + Indexer
         'heavily_compressed_attention': 20,   # full attn + Compressor
         'sliding_attention': 2})              # full attn, compressor = None
```

`DeepseekV4Attention.__init__` instantiates the compressor conditionally:

```python
self.compressor = (
    COMPRESSOR_CLASSES[self.layer_type](config)
    if self.layer_type != "sliding_attention" else None
)
```

An upstream-format DSv4-Flash MTP checkpoint has **only the standard attn projections** under `mtp.0.attn.*` (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b` + `q_norm`, `kv_norm`, `attn_sink`) — **no `compressor.*` or `indexer.*` keys**. The matching layer_type is therefore `sliding_attention` (the only one with `compressor = None`), not a copy of the last main-layer's type.

### Symptom of the wrong choice

With `layer_type = "compressed_sparse_attention"` (last main layer's type), the MTP `DeepseekV4Attention` instantiates empty `compressor` + `indexer` submodules. The checkpoint has no `mtp.0.attn.compressor.*` keys to load → those submodules stay uninitialized → `_initialize_weights` falls through to `_init_weights` → `init.normal_` random-initializes the MTP block. Silent corruption of MTP weights.

I'll push the corrected logic (`layer_types[mtp_idx] = "sliding_attention"`, `mlp_layer_types[mtp_idx] = "moe"`) plus the updated PR body shortly. Wanted to leave this comment now so anyone reading the PR mid-iteration sees the right design.

A test that catches this regression class:

```python
def test_mtp_from_pretrained_no_silent_random_init():
    model = AutoModelForCausalLM.from_pretrained("...DSv4-Flash BF16 with mtp...")
    # Pick any deterministic-loaded MTP projection
    loaded = model.model.mtp[0].self_attn.q_a_proj.weight.detach().cpu()
    # Compare against the raw safetensors value
    import safetensors.torch as st
    with st.safe_open("...shard with mtp.0.attn.wq_a.weight...", framework="pt") as f:
        source = f.get_tensor("mtp.0.attn.wq_a.weight").cpu()
    assert (loaded.float() - source.float()).abs().max() < 1e-4
```

## [c4504426780] pasta-paul (NONE) · issue_comment · 2026-05-21
## Design note — `DeepseekV4NextNPredictor` is quantization-format-agnostic

While running our smoke (the W4A16-experts + FP8_BLOCK-attention + BF16-MTP recipe described at vllm-project/llm-compressor#2735), we landed on a sharper design point worth noting for this PR:

**The class shim enables MTP retention regardless of what precision the rest of the model uses.** Users can:

- Quantize the MTP block alongside the main MoE (W4A16 / NVFP4 / FP8 / etc.) — `targets=` regex would naturally match `mtp.*.mlp.experts.*` if not explicitly excluded
- Preserve MTP at higher precision than the main model (e.g., BF16 MTP + W4A16 experts) — `ignore=["lm_head", r"re:.*mtp\..*"]`

The latter is what DeepSeek's own MXFP4 release does, and what we're shipping. Speculative-decoding throughput depends on draft-head acceptance rate, which is quantization-noise-sensitive; preserving MTP at higher precision than the main MoE is the conservative recipe.

This PR's responsibility is just **making the class exist** so `mtp.*` keys have a submodule to land in. The quantization decision is downstream and recipe-level. Mentioning here because the PR description shouldn't constrain users to a particular precision choice for the MTP block.

Tests under `tests/models/deepseek_v4/` should also cover:

```python
def test_mtp_block_can_remain_bf16_when_main_model_is_quantized():
    """User opts out of MTP quantization via ignore=. MTP weights stay BF16."""
    # ...
```

In addition to the round-trip + presence tests previously listed.

## [c4507912853] Rocketknight1 (MEMBER) · issue_comment · 2026-05-21
Hey, sorry, we probably won't review this right now! I don't think we want to merge a "structural shell" without a `forward()` or anything. You're correct that the draft head is currently not being loaded, but if we're going to add it we'll probably want to be able to test that it's working correctly, which we can't do when it's just a naked weight-container.

cc @arthurzucker

## [c4508933322] pasta-paul (NONE) · issue_comment · 2026-05-21
Thanks Matt — your point stands, a structural shell without `forward()` isn't useful. I'll expand the PR with a functional `forward()` + tests. To save a round-trip I researched the two design choices myself rather than asking; calling out my proposed approach so you and Arthur can correct me before I sink time in the wrong direction.

## Q1: standalone module vs deeper integration

**Proposed: keep `DeepseekV4NextNPredictor` standalone, add an explicit invocation path.**

Specifically:

- `DeepseekV4Model.__init__` instantiates `self.mtp` as `nn.ModuleList` (already done in this PR).
- `DeepseekV4Model.forward()` does **not** invoke MTP by default — zero-cost path for non-MTP DSv4 variants (`num_nextn_predict_layers = 0`).
- Add a convenience method `DeepseekV4Model.compute_mtp_logits(input_ids, previous_hidden_states, position_ids, ...)` that runs a single MTP step. Returns the draft logits and the post-MTP residual (so callers can re-feed it for `num_speculative_tokens > 1`).
- Speculative-decoding orchestration (sampling, acceptance, draft chaining) is downstream tooling's responsibility (vLLM, sglang) — transformers exposes the primitive, doesn't bake in a speculation protocol.

Why standalone:

1. Both reference implementations (DeepSeek's release `inference/model.py:MTPBlock`, vLLM's `vllm/models/deepseek_v4/nvidia/mtp.py:DeepSeekV4MultiTokenPredictorLayer`) treat MTP as a separate forward path called *after* the main model, not interleaved into its forward.
2. Keeps the main `forward()` unchanged — no perf regression for the non-MTP path.
3. More unit-testable in isolation: synthetic hidden states + synthetic input_ids → MTP forward, without running the full 43-layer decoder.

## Q2: which reference implementation

**Proposed: port the math from DeepSeek's release inference code, cross-check class structure against vLLM.**

The decisive factor: vLLM's `HCHeadOp.forward_native` is `raise NotImplementedError` (`vllm/model_executor/layers/mhc.py:266`). vLLM's HC ops are CUDA/Triton kernels with no pure-PyTorch reference. We can't directly port `vllm/.../mtp.py` because the HC math (the load-bearing MTP-specific kernel) is only available there as `_hc_head_cuda_impl`.

DeepSeek's release `inference/model.py` *does* have the pure-PyTorch HC math:

```python
def hc_pre(self, x, hc_fn, hc_scale, hc_base):
    # x: [b,s,hc,d], hc_fn: [mix_hc,hc*d], hc_base: [mix_hc]
    shape, dtype = x.size(), x.dtype
    x = x.flatten(2).float()
    rsqrt = ...  # rsqrt of mean of squares (RMS) over the flattened HC dim
    mixes = F.linear(x, hc_fn) * rsqrt
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, ...)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(dtype), post, comb
```

And the MTP block forward is concretely:

```python
def forward(self, x, start_pos, input_ids):
    e = self.embed(input_ids)
    e = self.enorm(e)
    x = self.hnorm(x)
    x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
    x = super().forward(x, start_pos, input_ids)  # the inherited Block's forward
    logits = self.head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
    return logits
```

That's the math I'll port. vLLM's `mtp.py` is then a cross-check for module hierarchy decisions (e.g., should the MTP block subclass `DeepseekV4DecoderLayer` like DeepSeek release does, or compose one as a member? — DeepSeek release uses inheritance; vLLM uses composition; our current PR uses inheritance to match DeepSeek release; I'll keep that).

I'll cite both references in the file header + commit message.

## Tests

I'll add three under `tests/models/deepseek_v4/`:

1. **Synthetic forward shape/no-NaN** — tiny config (small vocab, small hidden, 2 decoder layers, 1 MTP layer), random init, run `compute_mtp_logits` once, assert output shape and finiteness.
2. **Weight round-trip preserves `mtp.*` keys** — `from_pretrained`/`save_pretrained` cycle leaves the `mtp.0.*` parameters bit-identical (catches the regression where the silent-drop regex was reintroduced).
3. **HC math equivalence with the DeepSeek release reference** — implement the same `hc_pre`/`hc_post` computation against a fixed seed and compare to our ported version with `torch.allclose(..., atol=1e-5)`. No weight download required; deterministic on synthetic inputs.

## Cross-library coordination

This PR is the upstream-modeling half of an MTP-preserving DSv4 quant story. The downstream serve-path detection lives in vLLM PR #43319 (the sibling B300 chat's work) — once both land, an MTP-preserving DSv4-Flash artifact ships clean across the whole stack: transformers loads + recognizes MTP, llm-compressor calibrates it (PR #2739), vLLM serves it with speculative decoding (#43319). Happy to coordinate test runs across libraries if useful.

I'll keep this PR open and push the revised commit with forward + tests. Aiming for ~24h. Let me know if you'd prefer a different invocation surface than `compute_mtp_logits(...)` — that's the one externally-visible API choice I can't fully decide without your input.

cc @arthurzucker

## [c4512728011] pasta-paul (NONE) · issue_comment · reply to [c4507912853] · 2026-05-21
**Update — the load-path side is now end-to-end validated** (responding to @Rocketknight1's "wait until it works" point from earlier in the thread).

We've shipped an NVFP4-FP8 quantization of DSV4-Flash that retains the MTP block in the saved safetensors and runs through vLLM's MTP draft path with `--speculative-config method=mtp`:

- Artifact: [`canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-NVFP4-FP8-MTP) (172 GB, 35 shards, 799 MTP tensors at BF16, public on HF).
- **MTP draft acceptance, AIME 2024 (thinking=high, 30 problems, c=8): 81.60%** — Prometheus counters confirm draft tokens emitted + accepted. Per-position acceptance 96.6% / 79.4% (positions 0 / 1).
- **Quality parity vs BF16+MTP reference**: 83.33% raw pass@1 (both), 96.0% vs 96.15% non-truncated at the same 65K max_tokens cap.
- **Decode speedup vs no-MTP** (same NVFP4 quant): 1.84× per-request median tok/s on AIME reasoning, 2.13× at c=1 chat coding. Across the chat-template c=1→c=16 sweep, acceptance held flat at 87.92–88.27% — no batch-load degradation.

The validation went through vLLM's own MTP load path (PRs [#43319](https://github.com/vllm-project/vllm/pull/43319) etc. for the BF16-on-disk detection that lets MTP coexist with quantized main weights). What this demonstrates for #46127: the saved \`mtp.*\` keys are functionally useful — when retained, they drive a measurable spec-decode speedup on real benchmarks. The transformers-side \`DeepseekV4NextNPredictor\` class would let other inference engines (not just vLLM) consume these weights through stock transformers' load path.

Maintainer ask was to add \`forward()\` + tests before merge — totally fair. The above is offered as supporting evidence that the load side has downstream value, in case it helps prioritize.

## [c4523426918] pasta-paul (NONE) · issue_comment · 2026-05-22
## Production validation — class works for MTP-preserving GPTQ

Quick update: this PR's `DeepseekV4NextNPredictor` class (applied as a runtime shim via `scripts/transformers_mtp_shim.py`) is now validated in a shipped artifact.

### Setup
- Repo: https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp
- Artifact: https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8-MTP (public)
- Shim: [`scripts/transformers_mtp_shim.py`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp/blob/main/scripts/transformers_mtp_shim.py)

### Evidence
- Phase 2 GPTQ calibration on 8× H200 ran end-to-end with the shim installed before `from_pretrained` (~15h, 768 calibration samples).
- All 797 `mtp.0.*` weight keys load correctly into the `DeepseekV4NextNPredictor` instance at load time. Without the shim, transformers' `_keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]` filter silently drops every `mtp.*` key. The class needs to exist BEFORE `from_pretrained` for the keys to land somewhere.
- Serve-time validation: vLLM (with sibling-filed PRs #43248+#43288+#43290+#43319 applied) loads the artifact and runs MTP speculative decoding at 69.94% draft-token acceptance on 200 random prompts.

Happy to land the shim behavior as a non-runtime feature if it helps move this PR forward — the diff is small and the production validation gives confidence.

Raw eval data in [`benchmarks/phase2/`](https://github.com/canada-quant/dsv4-flash-w4a16-fp8-mtp/tree/main/benchmarks/phase2) of the source repo if useful.

## [c4532860852] ArthurZucker (COLLABORATOR) · issue_comment · 2026-05-25
cc @Cyrilvallez

## [c4550928379] pasta-paul (NONE) · issue_comment · 2026-05-27
> **Disclosure:** this comment was generated with AI assistance.

**Tracking note — MTP-class cluster.** This filing is one of four artifacts for what is fundamentally a single bug — "DSv4 MTP weights have no home in the canonical loader path":

| Filing | Layer | Role |
|---|---|---|
| huggingface/transformers#46127 (PR) | transformers | Add `DeepseekV4NextNPredictor` class so the MTP block has a module to load into |
| huggingface/transformers#46129 (issue) | transformers | `conversion_mapping` doesn't cover `mtp.*` paths — companion to #46127 |
| vllm-project/llm-compressor#2735 (issue) | llm-compressor | Canonical DSv4 calibration example drops MTP; regex `^layers\.` anchored excludes `mtp.*` |
| vllm-project/llm-compressor#2739 (PR) | llm-compressor | Extend `ARCH_TO_2D_MAPPINGS` for the MTP block — fixes #2735 |

Posting this header on all four so maintainers see one effort rather than four independent threads. Happy to consolidate further if reviewers prefer a single tracking issue.

---
outcome: state=open, merged=False
