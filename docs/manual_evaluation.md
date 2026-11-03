# Manual evaluation: how we run it

This document is for **the person who coordinates the labelling**. It covers what to hand out, what to keep hidden,
how to check the returned sheets, how to adjudicate, and what to run after each stage. The raters get
[rater_guide.md](rater_guide.md) instead.

Why we need people: the rules and the LLM are only *detectors*. Their precision and recall mean nothing without
a trusted reference. That reference is the **gold standard**: 200 conversations labelled by two people independently,
with disagreements resolved.

## Overview

There are three human tasks, at different times:

| Task | Who | Items | File | When |
|---|---|---|---|---|
| **A. Pilot labelling** | Rater 1 and Rater 2, independently | the 200 pilot items | `annotation/pilot_v1/rater_1.xlsx`, `rater_2.xlsx` | now |
| **B. Adjudication** | both raters together (+ a third person for deadlocks) | the disagreements and UNCERTAINs | `annotation/pilot_v1/adjudication.xlsx` | after A |
| **C. Confirmation** | one trained rater | the pool items the chosen detector flags | `annotation/confirm_v1/confirm_sheet.xlsx` | after the test evaluation |

Commands run between the tasks:

```
A ─► agreement.py ─► B ─► agreement.py --finalize ─► evaluate.py (dev, tune) ─► evaluate.py --split test (once)
  ─► llm_screen.py --items pool ─► extract_episodes.py --confirm-sheet ─► C ─► extract_episodes.py ─► report.py
```

## What raters may and may not see

The raters' labels must be independent of the detectors and of each other. Otherwise κ and the
evaluation measure nothing.

| Raters get | Raters must **not** see |
|---|---|
| `threads/*.md` (the 200 conversations) | `sample.csv`: it holds each item's detector group and split |
| their own `rater_N.xlsx` | the other rater's sheet, until adjudication |
| `codebook.md` | anything in `data/` (rule hits, `items.csv`), `runs/` (LLM answers), `reports/` |
| `docs/rater_guide.md` | `src/lexicon.py` (the keyword lists) and `docs/pipeline.md` |
| the GitHub link of each item | how many items of each kind the sample contains |

**If you are a rater yourself**, don't open the files on the right until you have finished labelling. Two of them
already exist: the LLM run and `sample.csv`. If you have already looked at them in detail, a colleague may be the
better choice for Rater 1 or 2.

## Task A: Pilot labelling

### A1. Before you start

1. **Pick two raters.** Both should know enough about ML engineering to follow a `transformers` PR discussion
   (what quantization, KV cache and attention implementations are).
2. **Optional practice round** (recommended). Both raters label the same 5-6 conversations **that are not in the pilot**,
   then compare and discuss the codebook. This catches misunderstandings before they cost κ. Practice items must
   come from outside the sample, or the pilot labels are no longer independent. To render 6 random candidates
   outside the pilot into `annotation/practice/threads/`:

   ```bash
   .venv/bin/python - <<'EOF'
   import sys; sys.path.insert(0, "src")
   import pandas as pd
   from common import ANNOTATION, DETECT, REPO, load_conversations, load_turns, render_thread
   out = ANNOTATION / "practice"
   items = pd.read_csv(DETECT / "rules_v1" / REPO / "items.csv")
   pilot = pd.read_csv(ANNOTATION / "pilot_v1" / "sample.csv")
   pool = items[~items["item_number"].isin(pilot["item_number"]) & items["group"].ne("NO_MATCH")]
   pick = pool.sample(6, random_state=1)["item_number"]
   conv = load_conversations().set_index("item_number")
   turns = load_turns()
   (out / "threads").mkdir(parents=True, exist_ok=True)
   for number, g in turns[turns["item_number"].isin(pick)].groupby("item_number"):
       row = conv.loc[number].copy()
       row["item_number"] = number
       (out / f"threads/{REPO}__{number}.md").write_text(render_thread(row, g) + "\n")
   print(sorted(pick))
   EOF
   ```

   The raters can write practice answers on paper or in a copy of a sheet. Practice labels are not used anywhere.
3. **Make one package per rater.** Keep `rater_N.xlsx` next to the `threads/` folder, because its links are relative:

   ```bash
   cd annotation/pilot_v1
   for i in 1 2; do
     zip -r ~/rater_${i}_pilot_v1.zip threads codebook.md rater_${i}.xlsx
     zip -j ~/rater_${i}_pilot_v1.zip ../../docs/rater_guide.md
   done
   unzip -l ~/rater_1_pilot_v1.zip | grep -c sample.csv     # must print 0
   cd ../..
   ```

4. **Agree on a schedule.** All 200 threads add up to about 245,000 words. The median is 560 words per
   conversation, one in ten has more than 2,400, and the longest has 21,000. Expect roughly 20 hours per rater. Plan
   sessions of 20-30 conversations, and set a return date.

### A2. While they label

- **Questions about the codebook:** answer in writing, **to both raters**, and keep a list of these clarifications
  (for example in `annotation/pilot_v1/clarifications.md`). Never answer a question about a specific item ("is #12345
  a trade-off?"). Ask the rater to mark it UNCERTAIN with a note instead.
- **Do not change the codebook halfway.** If something is really wrong, add a clarification, tell both raters, and
  write down from which item order number it applies.
- The raters must not discuss items with each other until both are done.

### A3. When the sheets come back

1. **Keep the originals** before anything else:

   ```bash
   mkdir -p annotation/pilot_v1/returned
   cp <download>/rater_1.xlsx annotation/pilot_v1/returned/rater_1_$(date +%F).xlsx
   cp <download>/rater_1.xlsx annotation/pilot_v1/rater_1.xlsx       # replaces the blank sheet
   ```

   Do the same for rater 2.
2. **Check each sheet** in Excel or LibreOffice with filters on the `labels` sheet:

   | Check | How |
   |---|---|
   | every row has a `label` | filter `label` = (blanks) |
   | label matches Q1-Q3 | TRADEOFF needs Q1, Q2, Q3 = yes. RESOURCE_ONLY needs Q1 = yes and Q3 = no. UNRELATED needs Q1 = no |
   | every UNCERTAIN has a `notes` entry | filter `label` = UNCERTAIN |
   | ids are written like `body;c3184836411` | no brackets `[ ]`, separated by `;` |
   | every TRADEOFF item has at least one row in the `episodes` sheet, with the same `item` number | compare the item numbers |
   | the sheet structure is unchanged | same column names, still 200 rows, nothing sorted away |

   If something is missing, send it back to **that rater** to fix. Only fix gaps (a missing label, a missing episode
   row). The rater should not change answers after seeing anything from the other rater.

## Agreement

Run this **only when both sheets are complete**:

```bash
.venv/bin/python src/agreement.py
```

It prints how many items each rater labelled and a κ table, and writes:

- `annotation/pilot_v1/agreement.json`: Cohen's κ and % agreement for Q1, Q2, Q3, Q5, label, multi_developer, and
  `label_binary_tradeoff` (TRADEOFF vs everything else, the one that matters most for the evaluation)
- `annotation/pilot_v1/adjudication.xlsx`: the sheet for Task B

How to read κ: 1 is perfect, 0 is chance. As a rough guide, 0.41-0.60 is moderate, 0.61-0.80 substantial, and above 0.80 almost
perfect. Report κ **before** adjudication (this file). If `label_binary_tradeoff` is below about 0.6, look at the
disagreements first. They usually point to one codebook rule that the raters read differently. Write that up as a finding
and clarify the rule before Task C.

> If you run `agreement.py` too early, it builds `adjudication.xlsx` from incomplete sheets, and later runs will
> not replace it. Rerun with `--force` once the sheets are complete. That is only safe while nobody has typed into
> `adjudication.xlsx`.

## Task B: Adjudication

Open `annotation/pilot_v1/adjudication.xlsx`. It has three sheets:

| Sheet | What it holds |
|---|---|
| `labels` | one row per item: `r1_*` and `r2_*` (both raters' answers, notes, evidence), then `final_*` (the decision), `final_Q4_supporting_ids`, `adjudication_note` |
| `episodes_r1_r2` | both raters' episode rows, with a `rater` column |
| `episodes_final` | **the episode list that will be used**. It is pre-filled with rater 1's episodes for items both raters called TRADEOFF, as a starting point |

**Pre-filled vs yellow.** Where both raters gave the same answer, `final_*` is already filled. Where they differ,
or either said UNCERTAIN, the cell is **empty and yellow**.

**Procedure** (both raters sit together and work through the yellow cells, in `order`):

1. Open the thread file, and read both raters' `evidence` and `notes`.
2. Discuss against the codebook's decision rules, not against what seems "obvious".
3. Fill every yellow cell in the row. Keep `final_label` consistent with `final_Q1`-`Q3`.
4. Edit `final_Q4_supporting_ids` (pre-filled with the union of both raters' ids) down to the turns that really
   support the label.
5. Write one line in `adjudication_note`: why, and which rule decided it. If the two cannot agree, a third person
   decides, and the note says so.
6. Use a final UNCERTAIN only if the conversation really doesn't let anyone decide. It is left out of the metrics, and
   the count is reported.

**Then fix `episodes_final`:**

- Every item whose `final_label` is TRADEOFF needs at least one row. Add rows for items that became TRADEOFF only during
  adjudication, using `episodes_r1_r2` as the source.
- Where the raters described the same episode differently, merge their descriptions into one row.
- If a conversation has two separate trade-offs, give it two rows (`episode_no` 1 and 2).
- Delete rows for items that are no longer TRADEOFF.

Do **not** edit `rater_1.xlsx` or `rater_2.xlsx` during adjudication. κ is computed from them, so they must stay as the
raters left them.

**Finalize:**

```bash
.venv/bin/python src/agreement.py --finalize
```

It refuses if any `final_label` is empty. It writes `annotation/pilot_v1/gold.csv` (the 200 items with their stratum,
weight, split and final answers) and `gold_episodes.csv` (from `episodes_final`).

## Using the gold labels: dev, then test once

```bash
.venv/bin/python src/evaluate.py --llm runs/2026-11-02_10-40-39_llm_qwen3-32b         # dev split
```

Open `reports/errors_dev.csv`. It has one row per dev item where at least one detector disagrees with the gold label,
with the rule evidence and the LLM explanation. Read the thread and ask *why* each one went wrong:

- **Rules** flag something that isn't a trade-off, or miss one. Edit `src/lexicon.py` (only that file), then:

  ```bash
  .venv/bin/python src/detect_rules.py --version v2
  .venv/bin/python -m pytest tests
  .venv/bin/python src/evaluate.py --rules v1 v2 --llm runs/2026-11-02_10-40-39_llm_qwen3-32b
  ```

- **LLM** is wrong. Edit `INSTRUCTIONS` in `src/llm_screen.py`, make a new run on the pilot items, and evaluate with that
  run. Use one run per reported result; don't mix runs.

Only look at **dev** errors. When the rules and the prompt are final:

```bash
.venv/bin/python src/evaluate.py --split test --rules v1 v2 --llm runs/<final pilot run>      # ONCE
```

This run is the result you report. It cannot be repeated without `--again`, and `--again` must be explained in the paper.

## Task C: Confirmation of new candidates

Choose the detector with the best **weighted** test numbers (usually `hybrid`: rules for recall, LLM as a filter).
Then:

```bash
setsid nohup /u2/users/scb052/alt-sustainability/venv-llm/bin/python src/llm_screen.py --items pool > /dev/null 2>&1 &
.venv/bin/python src/extract_episodes.py --confirm-sheet --detector hybrid --llm runs/<pool run>
```

This writes `annotation/confirm_v1/confirm_sheet.xlsx` and `threads/` for every pool item the detector flags. The sheet
has the same columns as the rater sheets, and the confirming rater follows the same [rater_guide.md](rater_guide.md):

- Fill `label` (and Q1-Q5) for every row. Rows that turn out not to be trade-offs get RESOURCE_ONLY or UNRELATED.
  These are the detector's false positives, and they are simply dropped.
- For every TRADEOFF, add the episode rows.
- Rows left without a label count as **not confirmed**. If any are left, the episode set is a validated *sample*, not a
  census, and `data/episodes/summary.json` says so.
- The confirming rater sees the same thread files, which also have no stratum or detector information. Still, they
  know every item was flagged, so judge strictly by the codebook.

Then:

```bash
.venv/bin/python src/extract_episodes.py                 # -> data/episodes/
.venv/bin/python src/report.py --llm runs/<pilot run> runs/<pool run>
```

## Checklist

- [ ] Raters chosen. Practice round done (optional)
- [ ] Packages sent (no `sample.csv` inside). Return date agreed
- [ ] Clarifications logged and sent to both raters
- [ ] Sheets returned, originals kept in `returned/`, checked
- [ ] `agreement.py`: κ recorded
- [ ] Adjudication done, `episodes_final` complete, `agreement.py --finalize`
- [ ] `evaluate.py` on dev, error analysis, v2 rules / prompt (dev only)
- [ ] `evaluate.py --split test`, **once**
- [ ] LLM on the pool, confirmation sheet built, filled
- [ ] `extract_episodes.py`, `report.py`
