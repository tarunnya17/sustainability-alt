# How the pipeline works

This document explains each step: **what** it does, **how** it does it, and **why**. The commands are in the
[README](../README.md). The human labelling is in [manual_evaluation.md](manual_evaluation.md).

## 1. The question in one picture

We want conversations where developers say, in effect, *"we gain X but lose Y"*, and X or Y is a resource.

| Element | Meaning | Examples |
|---|---|---|
| **R** resource concern | something that uses up energy or hardware | energy, carbon, compute, FLOPs, GPU hours, memory, VRAM, OOM, cloud cost, quantization, offloading |
| **G** competing objective | something else developers want | accuracy, maintainability, reliability, compatibility, functionality, development effort |
| **T** trade-off link | R and G compete **in the same decision** | "but", "at the cost of", "trade-off", or an objection in a reply, or a compromise ("make it optional") |

A trade-off needs all three: **R ∧ G ∧ T**.

**Speed is special.** It can be the resource or the competing goal:

- "2x faster **but less accurate**": speed is R (weighed against accuracy).
- "saves memory **but slower**": speed is G (weighed against memory).
- "lower latency but lower throughput": speed vs speed is **not** counted.

**How this differs from the sibling project.** The sibling flags a conversation when resource words and goal words
both *occur* somewhere in it. Here R and G must be **linked**: in the same sentence, the same comment, or two related
comments. We also check the result against human labels.

## 2. The whole pipeline

```
 sibling project (read-only): items.csv + comments.csv for huggingface/transformers
        │
 [1]  prepare_threads.py     each issue/PR becomes a conversation of "turns"
        │                    → data/threads/<repo>/turns.parquet
        ▼
 [2-4] detect_rules.py       keyword rules: find R, find G, find a link between them
        │  (lexicon.py)      → data/detect/rules_v1/<repo>/items.csv
        │                       every item gets one group:
        │                       HIGH_PRIORITY · POSSIBLE_TRADEOFF · RESOURCE_ONLY · NO_MATCH
        ▼
 [6a] sample_pilot.py        50 items from each group = 200 items, split 50/50 into dev and test
        │                    → annotation/pilot_v1/ (thread files + rater sheets)
        ├──────────────────────────────────────────┐
        ▼                                          ▼
   two raters label all 200 by hand         [5] llm_screen.py    a local LLM (Qwen3-32B) answers
        │                                          │              "is there a trade-off?" for the same 200
 [6b] agreement.py           κ, adjudication       │              → runs/<time>_llm_qwen3-32b/
        │                    → gold.csv            │
        ▼                                          │
 [6c] evaluate.py  ◄───────────────────────────────┘  how good are the rules, the LLM, and both together?
        │          dev: improve rules (v2) / prompt.   test: once, at the end.
        ▼
 [7]  llm_screen.py --items pool → extract_episodes.py --confirm-sheet → a person confirms
        │                        → extract_episodes.py → data/episodes/episodes.csv
        ▼
      report.py                  → reports/pipeline_report.md
```

The steps are numbered as in the methodology. Step 6a runs before Step 5 because the LLM screens the pilot sample first.
It would be too slow and too noisy to check all 10,000 items with the LLM before we know how good it is.

## 3. The steps

### Step 1: Rebuild the conversations (`src/prepare_threads.py`)

**What.** It turns the sibling's two tables into one ordered conversation per issue or PR.

**How.**

1. It checks the MD5 checksum of `items.csv` and `comments.csv` and stops if either changed (see `data/SOURCE.md`).
2. It removes bots. GitHub marks most bots, and `KNOWN_BOTS` in `common.py` lists the rest. A bot's opening post is dropped,
   but human comments under it stay. 138 items have no human text at all, and they are left out of the sample later.
3. It makes **turns**. A turn is one piece of text by one person, and each turn has a **unit id**:

   | Kind | Unit id | What it is |
   |---|---|---|
   | `title` | `title` | the issue/PR title |
   | `body` | `body` | the opening post |
   | `issue_comment` | `c<comment id>` | a normal comment |
   | `review` | `c<comment id>` | a PR review (APPROVED / CHANGES_REQUESTED / COMMENTED) |
   | `review_comment` | `c<comment id>` | an inline comment on a line of code |

   Turns are sorted by time within each item.
4. It **cleans** the text for keyword matching with `clean_text.py`, a byte-identical copy of the sibling's code. Cleaning
   lowercases the text and removes code blocks, inline code, tracebacks, quoted lines (`> ...`), URLs, @mentions, HTML, and
   **boilerplate**: any line that appears in 20 or more different texts, such as PR templates and reviewer lists. Then it
   splits the text into sentences.
5. It finds the **reply structure**. Cleaning removes quotes and @mentions, so this uses the raw text:
   - `thread_root`: the first comment of an inline review thread (GitHub's `in_reply_to`).
   - `reply_to` + `reply_basis`: the turn this one answers. The basis is one of:
     `review_parent` (GitHub says so), `quote` (it quotes a line of an earlier turn), `mention` (it @mentions someone, and
     we take that person's latest earlier turn), or `none`.

**Output.** `turns.parquet` holds one row per turn: raw text, cleaned text, sentences, and the reply fields.
`conversations.parquet` holds one row per item: state, merged, labels, number of turns, number of human authors.

**Numbers.** 10,000 items give 105,525 human turns. Of these, 15,064 replies come from review threads, 7,025 from
@mentions and 1,596 from quotes.

> Two texts, two jobs. The **cleaned** text is only used by the keyword rules. Raters and the LLM see a **readable**
> version of the raw text (`display_text` / `render_thread` in `common.py`): code is kept but long blocks are shortened,
> images become `[image]`, and template comments are removed. Raters and the LLM always see exactly the same text.

---

### Steps 2-4: Rule-based detection (`src/detect_rules.py` + `src/lexicon.py`)

**What.** These steps give every item one of four groups by looking for R words, G words and a link between them. This is the
cheap, transparent baseline.

**How. Step 2 and 3: find R and G words (the "hits").**

`lexicon.py` holds all the vocabulary as regular expressions, sorted into groups:

| Role | Groups |
|---|---|
| R | `R_env` (energy, carbon, sustainability), `R_compute` (compute cost, FLOPs, GPU hours, cloud bills), `R_memory` (memory, VRAM, OOM), `R_technique` (quantization, pruning, LoRA, flash attention, batching, caching) |
| R or G | `SPEED` (latency, throughput, speedup, overhead, faster/slower) |
| G | `G_accuracy`, `G_maintain`, `G_reliability`, `G_compat`, `G_function`, `G_effort` |
| T | `T_marker` (trade-off, at the cost of, downside, compromise, worth it …), `T_contrast` (but, however, although …), `T_alternative` (rather than, instead of), `T_response` (concern, optional, revert, alternative: used for ranking only) |

Each term is either **strong** or **weak**:

- A **strong** term always counts: "OOM", "peak memory", "perplexity", "maintenance burden".
- A **weak** term counts only if its sentence also has an **EFFECT word** (reduce, increase, save, drop, worse, better,
  less, more, improve, hurt …). "We load the model into memory" does not count. "This **reduces** memory" does.
- `R_technique` has only weak terms. In `transformers`, quantization and LoRA are everyday features. Mentioning them
  is not a resource concern unless an effect is claimed.
- A hit inside a longer hit is dropped. "overhead" inside "memory overhead" is not a separate SPEED hit, so "the memory
  overhead is large" is not memory vs speed.

Each sentence of a turn is scanned, and every hit is saved with an `active` flag (does it count?).

**How. Step 4: find a link (T).** R and G must come from **different groups**. A link records its kind:

| Link kind | R and G are … | Example |
|---|---|---|
| `sentence` | in the same sentence | "Reduces memory, but accuracy decreases." |
| `comment` | in the same turn (title + body count as one turn) | "This reduces peak memory. The trade-off is added complexity." |
| `thread` | in two turns of the same inline review thread, or one turn replies to the other | review comment: "uses more memory"; reply: "but it's more readable" |
| `adjacent` | in two turns at most 2 positions apart | A: "quantize to reduce GPU memory" → B: "that will reduce model quality" |

A link is **explicit** when the trade-off is spelled out:

- a `sentence` link whose sentence also has a `T_marker`, `T_contrast` or `T_alternative` word, or
- a `comment` link whose turn has a `T_marker` word somewhere.

`thread` and `adjacent` links are never explicit. They are where *implicit* trade-offs live: a proposal, then an
objection from someone else.

**The four groups** (one per item):

| Group | Rule | v1 count |
|---|---|---|
| `HIGH_PRIORITY` | at least one explicit link | 61 |
| `POSSIBLE_TRADEOFF` | some link, none explicit | 325 |
| `RESOURCE_ONLY` | an R (or speed) term counts, but no link to a G | 699 (253 of them speed only) |
| `NO_MATCH` | nothing | 8,915 |

Every item also gets:

- `score`, used only to sort items: 3 per explicit link, 2 per sentence link, 1 per other link, +2 if two people are
  involved, plus up to 5 `T_response` words in the linked turns.
- `framing`: `direct_environmental` if the link uses `R_env` (energy, carbon …), otherwise `resource_proxy`.
- `speed_role` on every link: `R`, `G` or `-`, and `speed_only` for items whose only resource is speed.
- `top_evidence`, the sentence(s) behind the best link, and `sibling_tier`, the sibling project's tier, for comparison.

**Examples.** These are from `tests/test_rules.py`, which you can run with `.venv/bin/python -m pytest tests`:

| Text | Group | Why |
|---|---|---|
| "Quantization reduces GPU memory by 35%, but decreases accuracy by 1.5%." | HIGH | R_memory + G_accuracy + "but", in one sentence |
| "This is 2× faster but less accurate." | HIGH | speed is R |
| "It saves memory but slower." | HIGH | speed is G |
| "This reduces peak memory. The trade-off is added complexity in the loader." | HIGH | comment link + T_marker "trade-off" |
| "Quantization increases inference latency." | POSSIBLE | link, no trade-off word |
| A: "I propose aggressive quantization to reduce GPU memory." → B: "That will reduce model quality considerably." | POSSIBLE | `adjacent` link, two developers |
| "Fix GPU memory allocation error." | RESOURCE_ONLY | R, no G |
| "Changing the attention implementation reduces memory requirements." | RESOURCE_ONLY | an optimization with no cost named |
| "The memory overhead is large." | RESOURCE_ONLY | "overhead" is inside "memory overhead" |
| "Latency is better but throughput is lower." | RESOURCE_ONLY | speed cannot trade against speed |
| "The model has excellent accuracy." | NO_MATCH | G alone is nothing |

**Output** (`data/detect/rules_v1/<repo>/`):

- `hits.parquet`: every keyword hit (turn, sentence, group, term, matched text, active)
- `links.parquet`: every R-G link (kind, explicit, groups, speed role, turns, authors, evidence sentences)
- `items.csv`: one row per item with group, framing, score and top evidence, sorted best first
- `summary.json`: counts, the most common R × G pairs, and a cross-table with the sibling's tiers

Of the sibling's 99 items in the `tradeoff` tier, 74 are HIGH or POSSIBLE here.

**Versions.** `rules_v1` is **frozen**. It defined the sample, so changing it would break the design weights (below).
To improve the rules, edit only `lexicon.py` and run `detect_rules.py --version v2`. The evaluation can compare v1 and v2
on the same gold labels.

---

### Step 5: LLM screen (`src/llm_screen.py`)

**What.** A local open-weights LLM reads each conversation and answers whether there is a trade-off, with evidence. Rules
miss implicit trade-offs and are fooled by coincidences. An LLM can read context, but it can also invent things. So we
measure both against human labels.

**How.**

1. **Model.** `Qwen/Qwen3-32B` (bf16) on one A100, run with vLLM. The exact model revision (sha) is saved in
   `config.json`. Temperature is 0 and Qwen's "thinking" mode is off.
2. **Input.** The conversation rendered by `render_thread`, the same text raters see, after the instructions from
   the methodology:
   - include a trade-off only if (1) a resource concern exists, (2) a second objective exists, (3) they compete in the
     same decision
   - do not count bug reports, unrelated mentions, or optimizations with no downside
   - do not infer energy savings from memory or speed optimizations; do not infer downsides nobody discussed
3. **Long conversations.** If a conversation has more than 24,000 tokens, it keeps the opening post, the turns with rule
   hits, their neighbours and their review-thread roots. The rest becomes `[… k comments omitted …]`. If that is still
   too long, the end is cut. In the pilot, 195 of 200 needed no trimming.
4. **Output format.** vLLM is forced to return JSON that matches a schema:

   ```json
   {"tradeoff_present": "yes | no | uncertain",
    "resource_concern": "...", "competing_objective": "...", "tradeoff_relationship": "...",
    "supporting_comment_ids": ["body", "c3184836411"],
    "exact_evidence_spans": ["verbatim quote", "..."],
    "explicit_or_implicit": "explicit | implicit | none",
    "explanation": "..."}
   ```

5. **Evidence check.** For each quote the model gives, the script checks that the words really appear in the turns it
   cites. Case, punctuation and Markdown are ignored. `evidence_verified` is the share of quotes found in the cited
   turns. A low value means the model made up or misplaced its evidence. `unknown_ids` lists cited ids that do not exist.

**Output** (`runs/<time>_llm_<model>/`): `predictions.parquet` (one row per item: the answer, the checks and the raw
output), `prompt.txt`, `config.json` and `log.txt`.

**Pilot run** `runs/2026-11-02_10-40-39_llm_qwen3-32b` (7.5 minutes): all 200 answers parsed. 61 yes, 139 no.
The yes answers by group were HIGH 29, POSSIBLE 16, RESOURCE_ONLY 15, NO_MATCH 1. On average, 92% of the quotes were found
in the cited turns.

**Caveats.**

- vLLM's batching is not bit-for-bit repeatable. Between two runs one answer flipped. Always report one fixed run.
- The answer limit is 2,048 tokens. With 1,024, one long answer was cut off mid-JSON.

**Later** (`--items pool`): the same screen runs on all R candidates outside the pilot (HIGH + POSSIBLE + RESOURCE_ONLY,
minus the 150 of them in the pilot: 935 items). This feeds Step 7.

---

### Step 6a: The pilot sample (`src/sample_pilot.py`)

**What.** It draws the 200 conversations that people will label, and writes everything the raters need.

**Why a stratified sample.** Trade-offs are rare. A simple random sample of 200 out of 9,862 would contain about one
HIGH item, so we would learn almost nothing. Instead we take **50 from each rule group** (all 4 groups, called
**strata**). This oversamples the likely trade-offs and still includes NO_MATCH items, so we can see what the
rules miss.

**How.**

1. The frame is the 9,862 items with human text, in their `rules_v1` group.
2. 50 items per group, with seed 42. Within each group, the issue/PR mix matches the group's.
3. **Design weight** = how many items each sampled item stands for = N_stratum / n_stratum:

   | Stratum | Items in the frame (N) | Sampled (n) | Weight |
   |---|---|---|---|
   | HIGH_PRIORITY | 61 | 50 | 1.22 |
   | POSSIBLE_TRADEOFF | 325 | 50 | 6.50 |
   | RESOURCE_ONLY | 699 | 50 | 13.98 |
   | NO_MATCH | 8,777 | 50 | 175.54 |

   So one NO_MATCH item in the sample stands for about 176 items in the repository.
4. A **dev/test split** of 25 + 25 per stratum is fixed now, in `sample.csv`. Rules and prompt may be tuned
   only by looking at dev. Test is used once at the end, so the final numbers are not tuned to the test items.
5. The 200 items are shuffled, so raters cannot guess the group from the position.
6. It writes:
   - `threads/<repo>__<number>.md`: each conversation, exactly as the LLM sees it, with **no keyword highlighting
     and no group**
   - `rater_1.xlsx`, `rater_2.xlsx`: identical label sheets (`labels`, `episodes`, `codebook`) with drop-down lists
   - `codebook.md`: definitions and decision rules
   - `sample.csv`: **the hidden key** (group, weight and split for each item). Raters must never see it.

The script refuses to overwrite rater sheets.

---

### Step 6b: Agreement, adjudication, gold labels (`src/agreement.py`)

**What.** It turns two people's labels into one trusted **gold standard**.

**How.**

1. After both raters finish, `agreement.py` computes **Cohen's κ** (kappa) for each question (Q1, Q2, Q3, Q5,
   label, multi_developer) and for TRADEOFF-vs-not. κ measures agreement beyond chance:
   1 means perfect, 0 means no better than chance. As a rough guide, 0.41-0.60 is moderate, 0.61-0.80 substantial, and above
   0.80 almost perfect.
2. It writes `adjudication.xlsx`. The two raters' answers are side by side, and a `final_*` column is pre-filled where
   they agree. Disagreements and every UNCERTAIN are left empty and yellow, for a person to decide.
3. `agreement.py --finalize` reads the decided `adjudication.xlsx` and writes `gold.csv` (one final label per item) and
   `gold_episodes.csv`.

The human side of this is in [manual_evaluation.md](manual_evaluation.md).

---

### Step 6c: Evaluation (`src/evaluate.py`)

**What.** It measures how well each detector finds the conversations that humans labelled TRADEOFF.

**The detectors compared** (a detector "says yes" when …):

| Detector | says yes when |
|---|---|
| `rules_v1_high` | the item is HIGH_PRIORITY |
| `rules_v1_any` | HIGH_PRIORITY or POSSIBLE_TRADEOFF |
| `llm_yes` | the LLM answered yes |
| `llm_yes_or_uncertain` | the LLM answered yes or uncertain |
| `hybrid_v1` | `rules_v1_any` **and** `llm_yes`: rules for recall, the LLM as a filter |

The same set exists for v2 if you pass `--rules v1 v2`.

**The metrics.** Compared with gold TRADEOFF:

- **Precision**: of the items the detector flags, how many are real trade-offs? = TP / (TP + FP)
- **Recall**: of the real trade-offs, how many does it flag? = TP / (TP + FN)
- **F1**: the balance of the two = 2·P·R / (P + R)

Example: 60 correct flags, 20 wrong flags, 40 missed: P = 0.75, R = 0.60, F1 = 0.67.

**Sample vs weighted numbers.** Each metric is reported twice:

- plain (`precision`, `recall`, `f1`): every sampled item counts once. This describes the 200 items.
- **design-weighted** (`w_precision`, `w_recall`, `w_f1`): every item counts as often as its weight. This
  **estimates the whole repository**. These are the numbers to report.

A made-up example shows why this matters. Suppose raters find TRADEOFF in 30 of 50 HIGH, 10 of 50 POSSIBLE, 3 of 50
RESOURCE_ONLY and 1 of 50 NO_MATCH items. `rules_v1_any` flags HIGH and POSSIBLE:

| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| sample (each item = 1) | 40 | 60 | 4 | 0.40 | 0.91 |
| weighted | 30×1.22 + 10×6.5 = 101.6 | 20×1.22 + 40×6.5 = 284.4 | 3×13.98 + 1×175.54 = 217.5 | 0.26 | 0.32 |

The one missed NO_MATCH item stands for about 176 missed trade-offs, so the real recall is far lower than the sample suggests.

Each weighted metric gets a **95% confidence interval** from a bootstrap: resample the items 2,000 times and see how
much the number moves. With 25 items per stratum in a split, expect wide intervals.

**Rules of use.**

- A final UNCERTAIN is left out and counted (`uncertain_left_out`).
- `--split dev` (default): use it as often as you like while improving rules or prompt.
- `--split test`: **once**. It writes `reports/.test_evaluated` and refuses a second run without `--again`.

**Output.** `reports/evaluation_<split>.json` has the metrics. `reports/errors_<split>.csv` has every item where at least
one detector is wrong, with the rule evidence and the LLM's explanation side by side. That file is where you learn
what to fix.

---

### Step 7: Episodes (`src/extract_episodes.py`)

**What.** It produces the final dataset for RQ1-RQ3, and only human-confirmed trade-offs go in.

An **episode** is one trade-off inside a conversation. A long PR can contain two episodes, for example memory vs accuracy in
comments 8-14 and memory vs maintainability in comments 27-35.

**How.**

1. After the test evaluation, choose the detector to trust (for example `hybrid`).
2. Run the LLM on the pool (`llm_screen.py --items pool`).
3. `extract_episodes.py --confirm-sheet --detector hybrid --llm runs/<pool run>` writes
   `annotation/confirm_v1/confirm_sheet.xlsx` (same columns as the rater sheets) and thread files for every pool item the
   detector flags.
4. A person labels those items and fills the episodes sheet. Detector flags are never accepted unchecked.
5. `extract_episodes.py` collects the episodes from the pilot gold (`gold_episodes.csv`) and from the confirmation sheet.

**Output** (`data/episodes/`):

- `episodes.csv`, one row per episode: `episode_id`, `repository`, `issue_pr_id`, `comment_ids`, `resource_concern`,
  `competing_objective`, `tradeoff_statement`, `tradeoff_framing`, `explicit_implicit`, `multi_developer_exchange`,
  `human_validated`, `evidence_spans`, `speed_role`, `source` (pilot / confirmed) and `url`
- `episode_turns.parquet`: **every** turn of each episode's issue/PR, in order, with `in_episode` marking the episode's
  turns. RQ3 (how trade-offs are negotiated and resolved) needs the whole conversation, not just the quotes.
- `summary.json`, including `coverage`. If not every flagged item was confirmed, the episodes are a **validated sample, not
  a census**.

**What the episodes miss.** Trade-offs in NO_MATCH items are never screened, so they are missing. The weighted
recall from Step 6c tells you roughly how many that is.

---

### Report (`src/report.py`)

It writes `reports/pipeline_report.md` from whatever exists so far: the funnel (10,000 items → … → episodes), the
detector groups, κ, the evaluation tables and the episode summary. Sections whose inputs don't exist yet say so. You can
run it at any time.

## 4. Glossary

| Term | Meaning |
|---|---|
| item | one GitHub issue or pull request |
| turn | one piece of text by one person: title, opening post, comment, review, inline review comment |
| unit id | a turn's id: `body` (title + opening post), or `c<comment id>`. Raters and the LLM cite these |
| hit | one keyword match in one sentence |
| strong / weak term | strong always counts; weak needs an EFFECT word in the same sentence |
| link | R and G (different groups) in the same sentence, turn, review thread/reply, or two turns apart |
| explicit link | a link spelled out with trade-off language (see Steps 2-4) |
| stratum / strata | the four rule groups, used to draw the sample |
| frame | the set the sample is drawn from: the 9,862 items with human text, grouped by `rules_v1` |
| design weight | how many repository items one sampled item stands for (N / n) |
| dev / test | the two halves of the sample: dev for tuning, test for the final measurement |
| gold standard | the final, adjudicated human labels (`gold.csv`) |
| κ (Cohen's kappa) | agreement between two raters, corrected for chance |
| adjudication | deciding the final label where raters disagree or were uncertain |
| episode | one trade-off within one conversation, confirmed by a person |
| TP / FP / FN | flagged and right / flagged but wrong / missed |
