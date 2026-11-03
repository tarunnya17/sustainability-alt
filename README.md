# alt-sustainability-developer-discussion

This project finds **trade-off conversations** in open-source LLM repositories. These are GitHub issues and pull requests where
developers weigh a **resource** (energy, compute, memory, cost, speed) against **another goal** (accuracy,
maintainability, reliability, compatibility, functionality, development effort) **in the same decision**.

Example: *"int4 quantization halves GPU memory, but generation quality drops noticeably. Let's make it opt-in."*

The final output is a set of **human-checked trade-off episodes** for the research questions:

- **RQ1**: what kinds of trade-offs developers discuss
- **RQ2**: how they justify them
- **RQ3**: how they negotiate and resolve them

The current data is `huggingface/transformers`: the 10,000 newest issues and PRs created before 2026-09-01. The input
tables come from the sibling project `../sustainability-developer-discussion`. This project only reads them and never copies them
(see [data/SOURCE.md](data/SOURCE.md)).

## Documents

| Read | When |
|---|---|
| [docs/pipeline.md](docs/pipeline.md) | To understand what each step does, how it works and why |
| [docs/manual_evaluation.md](docs/manual_evaluation.md) | When you coordinate the human labelling: hand-out, agreement, adjudication, confirmation |
| [docs/rater_guide.md](docs/rater_guide.md) | Give this to the raters. It covers how to label one conversation |
| [annotation/pilot_v1/codebook.md](annotation/pilot_v1/codebook.md) | The label definitions (each rater sheet also has a copy) |
| [reports/pipeline_report.md](reports/pipeline_report.md) | The results so far |

## Setup (once)

There are two Python environments. The small one runs everything except the LLM. The large one runs the LLM and lives on `/u2`,
because `/u1` is nearly full and the model alone is about 65 GB.

```bash
# from the project folder
export UV_CACHE_DIR=.cache/uv
~/.local/bin/uv venv --python ~/.local/bin/python3.11 .venv
~/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt

# LLM environment (vLLM + PyTorch), on /u2
cd /u2/users/scb052/alt-sustainability
export UV_CACHE_DIR=/u2/users/scb052/alt-sustainability/uv-cache
~/.local/bin/uv venv --python ~/.local/bin/python3.11 venv-llm
~/.local/bin/uv pip install --python venv-llm/bin/python -r /u1/users/scb052/Projects/alt-sustainability-developer-discussion/requirements-llm.txt
```

`llm_screen.py` sets `HF_HOME=/u2/users/scb052/alt-sustainability/hf` itself, so the model download goes to `/u2`.

Shortcuts used below:

```bash
PY=.venv/bin/python
LLM=/u2/users/scb052/alt-sustainability/venv-llm/bin/python
```

## Run the pipeline

Run every command from the project folder. ✅ means the step is done and its output exists. ⏳ means it waits for people.

| # | Step | Command | Writes | Status |
|---|---|---|---|---|
| 0 | Tests for the rules | `$PY -m pytest tests` | — | ✅ |
| 1 | Rebuild conversations | `$PY src/prepare_threads.py` | `data/threads/<repo>/` | ✅ |
| 2-4 | Rule-based detection | `$PY src/detect_rules.py` | `data/detect/rules_v1/<repo>/` | ✅ |
| 6a | Draw the 200-item pilot sample and the rater sheets | `$PY src/sample_pilot.py` | `annotation/pilot_v1/` | ✅ |
| 5 | LLM screen of the 200 pilot items | `setsid nohup $LLM src/llm_screen.py > /dev/null 2>&1 &` | `runs/<time>_llm_qwen3-32b/` | ✅ (use `runs/2026-11-02_10-40-39_llm_qwen3-32b`) |
| — | **Two raters label the 200 items** | see [docs/manual_evaluation.md](docs/manual_evaluation.md) | `rater_1.xlsx`, `rater_2.xlsx` | ⏳ |
| 6b | Agreement (κ) and adjudication sheet | `$PY src/agreement.py` | `agreement.json`, `adjudication.xlsx` | |
| — | **Adjudicate the disagreements** | fill the yellow cells in `adjudication.xlsx` | | ⏳ |
| 6b | Final gold labels | `$PY src/agreement.py --finalize` | `gold.csv`, `gold_episodes.csv` | |
| 6c | Evaluate on **dev** (as often as you like) | `$PY src/evaluate.py --llm runs/2026-11-02_10-40-39_llm_qwen3-32b` | `reports/evaluation_dev.json`, `errors_dev.csv` | |
| — | Improve the rules or the prompt (dev only) | edit `src/lexicon.py`, then `$PY src/detect_rules.py --version v2` | `data/detect/rules_v2/` | |
| 6c | Evaluate on **test**, **once** | `$PY src/evaluate.py --split test --rules v1 v2 --llm runs/<final run>` | `reports/evaluation_test.json` | |
| 7 | LLM screen of all other candidates | `setsid nohup $LLM src/llm_screen.py --items pool > /dev/null 2>&1 &` | `runs/<time>_llm_qwen3-32b/` | |
| 7 | Confirmation sheet for the chosen detector | `$PY src/extract_episodes.py --confirm-sheet --detector hybrid --llm runs/<pool run>` | `annotation/confirm_v1/` | |
| — | **A person confirms the flagged items** | fill `confirm_sheet.xlsx` | | ⏳ |
| 7 | Episodes | `$PY src/extract_episodes.py` | `data/episodes/episodes.csv`, `episode_turns.parquet` | |
| — | Report | `$PY src/report.py --llm runs/<run> ...` | `reports/pipeline_report.md` | can run any time |

Each script has `--help`, and its first lines explain what it reads and writes.

### Check that the scripts work, without real labels

```bash
$PY src/agreement.py --selftest          # fake labels -> runs/selftest/annotation/ (gold.csv etc.)
$PY src/evaluate.py --selftest           # -> runs/selftest/evaluation_dev.json
$PY src/extract_episodes.py --pilot runs/selftest/annotation --out runs/selftest/episodes
```

The self-test only writes under `runs/selftest/`. It never touches `annotation/` or `reports/`.

### LLM runs

```bash
$LLM src/llm_screen.py --dry-run             # print the first prompt; no GPU needed
$LLM src/llm_screen.py --limit 5             # a quick test on 5 items
setsid nohup $LLM src/llm_screen.py > /dev/null 2>&1 &    # the real run, keeps running after you log out
tail -f runs/<newest folder>/log.txt         # follow progress
kill $(cat runs/<run>/pid)                   # stop it
```

- The machine is shared. `pick_gpu` takes the GPU with the most free memory and stops if it has less than `--min-free` MiB
  (default 75,000). If no GPU is that free, use the quantized model:
  `--model Qwen/Qwen3-32B-AWQ --min-free 30000`.
- The 200 pilot items take about 7.5 minutes on one A100.
- Report results from **one** run. vLLM is not bit-for-bit repeatable, so two runs can differ on an item or two.

## Where things are

```
src/                      the scripts (one per step) + common.py, lexicon.py, clean_text.py
tests/test_rules.py       the rule examples from the methodology
data/SOURCE.md            where the input tables are, with their MD5 checksums
data/threads/<repo>/      Step 1: turns.parquet, conversations.parquet
data/detect/rules_v1/     Steps 2-4: hits, links, items.csv, summary.json   (v1 = the frozen sampling frame)
annotation/pilot_v1/      Step 6: sample.csv (HIDDEN KEY), threads/*.md, rater_1.xlsx, rater_2.xlsx, codebook.md
                          later: agreement.json, adjudication.xlsx, gold.csv, gold_episodes.csv
annotation/confirm_v1/    Step 7: confirm_sheet.xlsx + threads/
runs/                     LLM runs (one folder each) and the self-test
reports/                  evaluation_<split>.json, errors_<split>.csv, pipeline_report.md
data/episodes/            Step 7: episodes.csv, episode_turns.parquet, summary.json
```

## Rules that protect the results

1. **Never overwrite human work.** The scripts refuse to rebuild rater sheets, `adjudication.xlsx` or `confirm_sheet.xlsx`
   unless you pass `--force`. Only use `--force` when you are sure nothing has been typed into the file yet.
2. **Raters never see `sample.csv`** (it holds each item's detector group and split), the rule hits, or the LLM answers.
3. **`rules_v1` is frozen.** It defines the sample and the design weights. After you change the rules, write `--version v2`.
4. **Tune on dev, test once.** `evaluate.py --split test` refuses a second run unless you pass `--again`.
5. **Do not mix LLM runs.** Pick one run per result.

## Troubleshooting

| Problem | Fix |
|---|---|
| `prepare_threads.py`: "changed since data/SOURCE.md was written" | The sibling's tables changed. Find out why before you update the MD5 in `src/common.py` and `data/SOURCE.md`. |
| `No GPU with 75000 MiB free` | Try again later, or use `--model Qwen/Qwen3-32B-AWQ --min-free 30000`. |
| vLLM start-up fails in FlashInfer / ninja / nvcc | `VLLM_USE_FLASHINFER_SAMPLER=0` must be set (`llm_screen.py` sets it). |
| Disk full on `/u1` | Model files and caches belong on `/u2` (see Setup). |
| `agreement.py` says `adjudication.xlsx exists; not overwritten` | This is expected once adjudication has started. Use `--force` only if nobody has typed into it yet. |
