# alt-sustainability-developer-discussion

Find **sustainability-related trade-off conversations** in open-source LLM repositories: discussions where developers
weigh a resource concern (energy, compute, memory, cost, speed) against a competing objective (accuracy,
maintainability, reliability, compatibility, functionality, effort) **in the same development decision**.
Output: validated trade-off episodes for RQ1-RQ3 (types of trade-offs, justifications, negotiation/resolution).

This is the "alternative" pipeline to the sibling project `/student/scb052/Projects/sustainability-developer-discussion`.
The sibling collected the data and built an item-level keyword filter (tiers). It requires only that keywords
co-occur. This project requires a **link** between R and G (R ∧ G ∧ T), validates it against human labels, and
compares a rule baseline with a local LLM. Its tables are read **read-only**, never copied (`data/SOURCE.md`, MD5 checked).

## Design (decided by the user, 2026-11-02; don't reopen unless the user asks)

- **Data: Transformers only for now**: the sibling's 10,000 newest issues/PRs of `huggingface/transformers` created
  before 2026-09-01. Code is repo-agnostic (`--repo owner__name`); DeepSpeed and Megatron-LM come later.
  Leave-one-repo-out and per-repo comparisons wait for them.
- **Speed plays either role**: R next to a non-speed G ("2x faster but less accurate"), G next to a non-speed R
  ("saves memory but slower"). R and G must come from different groups; every link records `speed_role` (R/G/-).
  Speed alone (no G) puts an item in RESOURCE_ONLY with `speed_only=True`.
- **LLM screen: local open-weights model** with vLLM on one A100; default `Qwen/Qwen3-32B` (bf16), pinned to the HF
  revision sha; `--model Qwen/Qwen3-32B-AWQ --min-free 30000` when no GPU has ~75 GB free.
- Methodology (from the user's pasted plan): reconstruct conversations → R candidates → competing objectives →
  trade-off relationship → human gold standard (two raters, κ, adjudication) → P/R/F1 → episodes.

## Rules baseline (`src/lexicon.py` + `src/detect_rules.py`)

- Lexicon groups: R_env, R_compute, R_memory, R_technique (weak only), SPEED (R or G), G_accuracy, G_maintain,
  G_reliability, G_compat, G_function, G_effort, plus T_marker, T_contrast, T_alternative, T_response (ranking only).
  The R regexes are copied from the sibling's `filtering/keywords.py`.
- A strong term always counts; a weak term counts only in a sentence that also has an EFFECT word (reduce, increase,
  save, drop, worse …). A hit inside a longer hit is dropped ("overhead" inside "memory overhead").
- Link kinds: `sentence`, `comment` (title + body = opening post), `thread` (same inline review thread, or a reply via
  GitHub parent / quote / @mention), `adjacent` (within the next 2 comments).
- Groups: **HIGH_PRIORITY** = explicit link (sentence with R, G and a T_marker/T_contrast/T_alternative word, or
  a comment with R, G and a T_marker) · **POSSIBLE_TRADEOFF** = other link · **RESOURCE_ONLY** = R counts, no link ·
  **NO_MATCH**.
- Changed before v1 was frozen (no labels seen): "rather than / instead of" counts only within a sentence
  (`T_alternative`); comment-wide it made unrelated R and G "explicit" (HIGH 116 → 61).
- **v1 is frozen as the sampling frame.** Rule changes go to `--version v2`; evaluation still uses v1's design weights.

## Evaluation design

- Pilot sample: 50 per v1 group (HIGH has 61), seed 42, issue/PR proportional within each stratum. Bot-only items
  (138) are outside the frame. Design weight = N_stratum / n_stratum (HIGH 1.22, POSSIBLE 6.50, RESOURCE_ONLY 13.98,
  NO_MATCH 175.54); weighted P/R/F1 estimate all items.
- **Dev/test 50/50, fixed in `sample.csv`.** Tune rules and prompt on dev only. `evaluate.py --split test` runs once
  (it writes `reports/.test_evaluated` and refuses a second run without `--again`).
- Raters see `threads/*.md` (same rendering as the LLM), **no keyword highlighting and no stratum**. `sample.csv` is the
  hidden key.
- UNCERTAIN always goes to adjudication; a final UNCERTAIN is left out of the metrics (and counted).

## Where things are

| Path | What |
|---|---|
| `README.md` | Setup, run order with commands and status, rules, troubleshooting |
| `docs/pipeline.md` | What/how/why of every step, worked examples, weighted-metric example, glossary (for the user) |
| `docs/manual_evaluation.md` | Coordinator's process: rater packages, hidden files, checks, agreement, adjudication, dev/test, confirmation |
| `docs/rater_guide.md` | Handed to raters (no strata, rules or LLM details in it): thread format, columns, worked examples |
| `src/common.py` | Paths, `SEED`, `STRATA`, `LABELS`, `is_bot`/`KNOWN_BOTS`, `split_sentences`, `display_text`/`render_thread` (THE rendering for raters and LLM), `pick_gpu` (from each-bug-class-predictor-merge) |
| `src/clean_text.py` | Copied verbatim from the sibling (commit eb0a95b); cleaned text is byte-identical to the sibling's `build_texts` |
| `src/prepare_threads.py` | Step 1 → `data/threads/<repo>/turns.parquet` (title/body/comments, cleaned text, sentences, `thread_root`, `reply_to` + `reply_basis`) and `conversations.parquet` |
| `src/lexicon.py` | The vocabulary; **the only file to edit when tuning rules** |
| `src/detect_rules.py` | Steps 2-4 → `data/detect/rules_<v>/<repo>/` hits, links, items.csv (group, framing, score, top evidence, sibling tier), summary.json |
| `src/sample_pilot.py` | Step 6a → `annotation/pilot_v1/` sample.csv (hidden key), threads/, rater_1.xlsx, rater_2.xlsx (labels, episodes, codebook sheets), codebook.md. Refuses to overwrite rater sheets |
| `src/agreement.py` | Step 6b → agreement.json (κ), adjudication.xlsx; `--finalize` → gold.csv, gold_episodes.csv; `--selftest` on fake labels in `runs/selftest/` |
| `src/llm_screen.py` | Step 5 (LLM venv) → `runs/<time>_llm_<model>/` predictions.parquet (+ evidence check), prompt.txt, config.json, log.txt |
| `src/evaluate.py` | Step 6c → `reports/evaluation_<split>.json`, `errors_<split>.csv`: rules_high/any, llm_yes(/uncertain), hybrid; sample and design-weighted P/R/F1 with bootstrap CIs |
| `src/extract_episodes.py` | Step 7: `--confirm-sheet --detector …` → `annotation/confirm_v1/`; then → `data/episodes/episodes.csv`, `episode_turns.parquet` |
| `src/report.py` | `reports/pipeline_report.md`: funnel, κ, metrics, episodes |
| `tests/test_rules.py` | The methodology's examples + speed roles + P/R/F1 example (`.venv/bin/python -m pytest tests`) |

## Setup

- `.venv` (local `/u1`, small): `UV_CACHE_DIR=.cache/uv ~/.local/bin/uv venv --python ~/.local/bin/python3.11 .venv`
  then `~/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt`.
- **LLM venv and HF cache on `/u2`** (`/u1` is ~98% full; Qwen3-32B bf16 is ~65 GB):
  `/u2/users/scb052/alt-sustainability/venv-llm` (vLLM 0.33.0, torch 2.14 cu130; driver 595 / CUDA 13.2, so the
  standard wheel works), `HF_HOME=/u2/users/scb052/alt-sustainability/hf` (set by `llm_screen.py`).
  Rebuild: `UV_CACHE_DIR=/u2/users/scb052/alt-sustainability/uv-cache ~/.local/bin/uv venv --python ~/.local/bin/python3.11 venv-llm`
  and `uv pip install --python venv-llm/bin/python -r requirements-llm.txt`.

## Conventions

- Scripts run as `.venv/bin/python src/<script>.py` (LLM: `/u2/users/scb052/alt-sustainability/venv-llm/bin/python src/llm_screen.py`);
  `ROOT = Path(__file__).resolve().parents[1]`; `SEED = 42`.
- Shared lab machine: one GPU, chosen by `pick_gpu` (most free memory). Don't wait for an idle GPU. Long runs detached:
  `setsid nohup <python> src/llm_screen.py > /dev/null 2>&1 &` (log in the run folder; `kill $(cat runs/<run>/pid)`).
- Never overwrite human work: rater sheets, adjudication.xlsx and confirm_sheet.xlsx are only rebuilt with `--force`.
- Keep code and explanations simple; the user is learning.

## Status

- [x] Step 1: 10,000 items → 105,525 human turns; cleaning identical to the sibling (checked on all rows).
  Replies: 15,064 review parents, 7,025 @mentions, 1,596 quotes.
- [x] Steps 2-4 `rules_v1`: HIGH 61 · POSSIBLE 325 · RESOURCE_ONLY 699 (253 speed-only) · NO_MATCH 8,915.
  Of the sibling's 99 `tradeoff`-tier items, 74 are HIGH/POSSIBLE.
- [x] Step 6a: `annotation/pilot_v1/` (200 items, 25 dev + 25 test per stratum), rater sheets ready.
- [x] agreement / evaluate / extract_episodes / report pass on fake labels (`--selftest`, `runs/selftest/`).
- [x] LLM screen on the 200 pilot items: **use `runs/2026-11-02_10-40-39_llm_qwen3-32b/`** (7.5 min on one A100).
  100% parsed; yes 61 / no 139 (yes by stratum: HIGH 29, POSSIBLE 16, RESOURCE_ONLY 15, NO_MATCH 1);
  evidence spans found in the cited turns: mean share 0.92.
  - The earlier run `2026-11-02_10-15-44` (answer cap 1024 tokens) lost #43112 mid-JSON; the cap is now 2048.
  - Between the two runs one other answer flipped (#47874 yes → no) despite temperature 0: vLLM's batching is not
    bit-deterministic. Report LLM results from one fixed run; don't mix runs.
  - vLLM start-up needs `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer JIT needs ninja + nvcc); `llm_screen.py` sets it.
- [ ] **Waiting for people:** two raters label 200 items → `agreement.py` → adjudicate → `--finalize` →
  `evaluate.py` (dev) → revise rules (v2) / prompt → `evaluate.py --split test` once → LLM on `--items pool` →
  `extract_episodes.py --confirm-sheet` → confirm → `extract_episodes.py` → `report.py`.
