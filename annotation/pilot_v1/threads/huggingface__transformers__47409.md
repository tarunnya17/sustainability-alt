# #47409 [pull_request] fix: make num_experts BC fallback not override explicit n_routed_experts
https://github.com/huggingface/transformers/pull/47409  ·  opened by webtecnica on 2026-07-19

## [body] webtecnica (NONE) · opening post · 2026-07-19
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47409)

## Description

Fixes #47355

**Root cause:** In both `GlmMoeDsaConfig.__post_init__` and `DeepseekV32Config.__post_init__`, the backward-compatibility re-route from `num_experts` (legacy key) to `n_routed_experts` was **unconditional** — it could overwrite an explicit `n_routed_experts` value from the same `config.json`.

**Example:** A config with `{"n_routed_experts": 168, "num_experts": 256}` would silently use 256 experts, causing an OOM downstream (vLLM allocates for 256 experts instead of 168).

**Fix:** Only apply the BC fallback when `n_routed_experts` is still at its defining class's default value, which means it was NOT explicitly provided. Since `n_routed_experts` is a dataclass field consumed by `__init__` before `__post_init__` runs, we compare `self.n_routed_experts` against `type(self).n_routed_experts`.

**Files changed:**

| File | Change |
|------|--------|
| `modular_deepseek_v32.py` | Root fix — source that regenerates generated files |
| `configuration_deepseek_v32.py` | Generated copy of the same fix |
| `configuration_glm_moe_dsa.py` | Generated copy (unwrapped from modular) |
| `tests/models/glm_moe_dsa/test_config_glm_moe_dsa.py` | 3 test cases covering the fix + BC preservation |

**Verification (6 test cases, all pass):**

| Scenario | Expected | Result |
|----------|----------|--------|
| Both keys provided | `n_routed_experts=168` wins | ✅ |
| Only legacy `num_experts` | BC fallback applies: 128 | ✅ |
| Only modern `n_routed_experts` | Preserved: 42 | ✅ |
| DeepseekV32Config parent | Same fix: 64 | ✅ |
| Both keys same value (256) | No issue | ✅ |
| Not provided either | Class default (256) | ✅ |

---
outcome: state=closed, merged=False
