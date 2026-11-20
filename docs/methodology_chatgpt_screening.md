# Methodology (current): ChatGPT screening of whole conversations in the HIGH_PRIORITY group

*Version of 2026-11-11. This document describes the approach we follow **now**. It replaces the evaluation path
described in `docs/pipeline.md` §6–§9 and `docs/methodology_explained.md` §6–§10 (local LLM on a 200-item pilot,
two raters, κ, weighted P/R/F1). The data collection, the conversation reconstruction and the rule-based grouping are
kept unchanged.*

---

## 0. Summary in one paragraph

From the 10,000 newest issues and pull requests (PRs) of `huggingface/transformers`, a keyword-rule detector places
each item in one of four groups. The **HIGH_PRIORITY** group (61 items) contains the items where a resource concern and
a competing objective appear **together with explicit trade-off language**. For each of these 61 items we export the
**whole conversation** (opening post, all human comments and review comments, final outcome) and give it with one fixed
prompt to **ChatGPT**. ChatGPT first decides whether the conversation really contains a sustainability-related
trade-off (screening) and lists each distinct trade-off it finds. For each trade-off it then proposes open codes for
what each research question needs: the **type** of trade-off (RQ1), the **justifications** for prioritizing one side
(RQ2), and the **disagreement, negotiation and resolution** (RQ3), each backed by exact quotes. The researcher checks
these answers against the conversation. Then the open codes are consolidated into a **taxonomy** (open → axial
coding) for each research question, and counted for the subsets of trade-offs that support them.

---

## 1. What changed, what stays

| Part | Status | Notes |
|---|---|---|
| Data: 10,000 Transformers issues/PRs (sibling project, read-only) | **kept** | `data/SOURCE.md` |
| Step 1: conversations rebuilt as ordered human turns | **kept** | 105,525 human turns, bots removed |
| Steps 2–4: rule detector `rules_v1` → 4 groups | **kept** | HIGH 61 · POSSIBLE 325 · RESOURCE_ONLY 699 · NO_MATCH 8,915 |
| Export of whole conversations for the HIGH group | **new (done)** | `data/chatgpt/huggingface__transformers/high_priority_v1/` |
| ChatGPT screening + RQ extraction with one prompt | **new (in progress)** | `prompt.txt`, answers go into `results.xlsx` |
| Researcher check of ChatGPT answers | **new** | see §8 |
| Open coding → RQ1 taxonomy; coding of RQ2 and RQ3 | **new** | see §9 |
| Local LLM (Qwen3-32B) on the 200-item pilot | set aside | run exists (`runs/2026-11-02_10-40-39_llm_qwen3-32b/`); may be used as a cross-check (§8.4) |
| 200-item stratified pilot, two raters, κ, weighted P/R/F1, dev/test | set aside | files remain in `annotation/pilot_v1/`; not part of the current path |

The main conceptual change: before, the goal was to **build and validate a detector** over all 10,000 items, then
extract episodes. Now the rules are used only as a **candidate selector** that finds the most promising items. The
analysis (RQ1–RQ3) is done in depth on those items, with ChatGPT as an assistant that pre-reads and pre-codes each
whole conversation and a researcher who checks it.

---

## 2. Research questions and the definition of a trade-off

### 2.1 Research questions

- **RQ1:** What types of sustainability-related trade-offs emerge in open-source LLM development discussions?
- **RQ2:** How do developers justify prioritizing sustainability or competing goals such as performance, accuracy,
  maintainability, cost, or development effort?
- **RQ3:** How are disagreements about sustainability-related trade-offs discussed and resolved?

### 2.2 What counts as a trade-off

A conversation contains a trade-off only if **all three** conditions hold (these are the three screening criteria in
Step 1 of the prompt):

1. **Resource concern (R):** an identifiable environmental or computational-resource concern: energy, power,
   emissions, compute, memory, hardware use, cost of compute, an efficiency technique (quantization, offloading,
   pruning, distillation …), or speed when it is weighed against a non-speed objective.
2. **Competing objective (G):** a second desirable objective or alternative: accuracy/model quality, speed,
   maintainability/complexity, reliability/correctness, compatibility, functionality, development effort.
3. **Same decision (T):** the conversation shows that R and G **compete in the same development decision**: improving
   one costs the other, and someone has to choose, compromise or mitigate.

Not trade-offs:

- ordinary bug reports, even when they mention memory or speed;
- optimizations with **no identified downside** ("this PR reduces memory by 40%");
- unrelated mentions of R and G in the same thread;
- a downside that the participants never mention (we do not infer costs ourselves).

### 2.3 Speed can be on either side

- "2× faster **but less accurate**": speed is the resource side, accuracy the competing side.
- "saves memory **but slower**": memory is the resource side, speed the competing side.
- "lower latency but lower throughput": speed vs speed. This is **not** a trade-off type for this study (both sides
  must come from different categories).

### 2.4 "Sustainability" here mostly means resource efficiency

Developers in this repository rarely mention energy or carbon. They talk about memory, speed and compute, which are
**resource proxies** for energy use. The prompt forbids inferring energy or carbon savings from memory or runtime
alone. In the analysis we therefore separate:

- **direct environmental** trade-offs: energy, power, emissions or sustainability are named;
- **resource-proxy** trade-offs: only memory, compute, speed, cost or efficiency techniques are named.

This separation must be reported. It is a finding in itself, and it is what keeps the study from overclaiming.

---

## 3. Overview of the pipeline

```
 GitHub huggingface/transformers: 10,000 newest issues + PRs created before 2026-09-01   (sibling project)
        │  Phase A  data collection (read-only)
        ▼
 Phase B  conversations rebuilt: 105,525 human turns, ordered, with reply structure
        │
        ▼
 Phase C  rule detector rules_v1 (keywords for R, G, T + link) → 4 groups
        │     HIGH_PRIORITY 61 · POSSIBLE_TRADEOFF 325 · RESOURCE_ONLY 699 · NO_MATCH 8,915
        ▼
 Phase D  export the WHOLE conversation of each of the 61 HIGH items (one Markdown file each)
        │
        ▼
 Phase E  ChatGPT, one new chat per item, one fixed prompt:
        │     Step 1  screening: trade-off present?  yes / no / uncertain; list each distinct trade-off
        │     per trade-off → Step 2 RQ1 open code · Step 3 RQ2 justifications · Step 4 RQ3 disagreement,
        │                     negotiation, resolution                         (one JSON answer, exact quotes)
        ▼
 Phase F  researcher checks every answer against the conversation (accept / correct)
        │
        ▼
 Phase G  analysis
              confirmed trade-offs (N_T items)        → RQ1: open coding → taxonomy of trade-off types
              ⊇ items with a justification (N_J)      → RQ2: kinds of justification, which side wins
              ⊇ items with a disagreement (N_D)       → RQ3: positions, kinds of resolution
```

The RQ subsets are **nested**: every RQ2 item and every RQ3 item is also an RQ1 item, but an item can support RQ2
without RQ3, RQ3 without RQ2, both, or neither.

---

## 4. Phase A: Data (kept)

- **Source:** the sibling project `sustainability-developer-discussion` collected issues, PRs, issue comments, review
  comments and reviews from GitHub. This project reads its tables read-only (MD5 recorded in `data/SOURCE.md`).
- **Sample frame:** the 10,000 newest issues and PRs of `huggingface/transformers` created before 2026-09-01.
- **Why Transformers:** the most widely used open-source library for LLM development, with a large and active
  maintainer community. Discussions there cover model loading, quantization, memory use, attention kernels and
  generation speed, which are exactly where resource decisions are made. DeepSpeed and Megatron-LM may follow; the
  code is repository-agnostic.

## 5. Phase B: Conversation reconstruction (kept)

Each issue or PR is rebuilt as an **ordered conversation** of human turns: title + opening post, issue comments,
review comments (grouped by review thread) and review bodies. Bot accounts are removed. Text is cleaned exactly like
in the sibling project (checked byte-for-byte). Replies are linked where GitHub records them (review threads) or where
they can be recovered (quotes, @mentions).

Result: 105,525 human turns in 10,000 conversations.

## 6. Phase C: Rule-based candidate selection (kept)

A keyword lexicon (`src/lexicon.py`) marks, sentence by sentence, words for resource concerns (R), competing objectives
(G) and trade-off language (T: "trade-off", "at the cost of", "but", "rather than" …). The detector then looks for a
**link** between an R and a G (same sentence, same comment, same review thread / reply, or adjacent comments).

| Group | Rule | Items |
|---|---|---|
| **HIGH_PRIORITY** | at least one **explicit** link: a sentence with R, G and a trade-off marker / contrast / alternative word, or one comment with R, G and a trade-off marker | **61** (32 issues, 29 PRs) |
| POSSIBLE_TRADEOFF | R and G linked, but not explicitly | 325 |
| RESOURCE_ONLY | R present, no link to a G | 699 |
| NO_MATCH | nothing | 8,915 |

Funnel over all 10,000 items: 1,085 have a resource concern → 407 also have a competing objective → 386 have them
linked → 61 have explicit trade-off language.

The rule version `v1` is **frozen**; the 61 items are fixed.

### Why only the HIGH_PRIORITY group

- It has the **highest density** of likely trade-offs, so the in-depth reading effort is spent where it pays off.
- 61 conversations are few enough for one-by-one manual screening in the ChatGPT app **and** a full check by the
  researcher.
- The cost: trade-offs written without explicit trade-off words (many are probably in POSSIBLE_TRADEOFF) are not
  covered. The study therefore describes **what kinds** of trade-offs exist and how they are argued and resolved. It
  does **not** estimate how often trade-offs occur in the whole repository (see §11). The same procedure can later be
  applied to POSSIBLE_TRADEOFF (`export_chatgpt.py --group POSSIBLE_TRADEOFF`).

---

## 7. Phase D and E: Whole conversations to ChatGPT

### 7.1 Why whole conversations

The rule detector only sees sentences and nearby comments. A trade-off, its justification and especially its
resolution are often spread over a whole thread: a contributor proposes a faster kernel, a maintainer objects about
accuracy twenty comments later, a benchmark is posted, and the PR is merged with a flag. RQ2 and RQ3 can only be
answered from the **whole** conversation, so each item is given in full.

### 7.2 What ChatGPT sees (`threads/<order>_<item>.md`)

Each file is one conversation, rendered with the same function (`render_thread`) used before for raters and the
local LLM:

```
# #41867 [issue] [RFC] Automatic CPU dtype fallback and thread optimization
https://github.com/huggingface/transformers/issues/41867  ·  opened by <author> on 2025-10-26

## [body] <author> (NONE) · opening post · 2025-10-26
<text of the opening post>

## [c3456379870] <author> (MEMBER) · comment · <date>
<text>
...
---
outcome: state=closed
```

- Every turn starts with an **id in square brackets** (`[body]`, `[c…]`). ChatGPT must cite these ids, so every claim
  can be traced back to a turn.
- The author's role (MEMBER, CONTRIBUTOR, NONE …) is visible. This matters for RQ3 (maintainer decisions).
- Long code blocks, `<details>` sections and images are collapsed; template comments are removed.
- The last line gives the **outcome** (state, merged). This helps RQ3, but only together with what people wrote.
- No keyword highlighting and no group label are shown, so ChatGPT is not told that the rules found a trade-off.

Sizes of the 61 conversations: median 9 turns (≈ 2,100 tokens), mean ≈ 4,500 tokens. Four are long:
#46419 (847 turns, ≈ 43,500 tokens), #40962 (≈ 33,800), #43838 (≈ 25,500), #45246 (≈ 22,200).

`index.csv` lists, for each item, the order, number, type, title, URL, file, size, and the **first and last turn
id**.

### 7.3 The prompt (`prompt.txt`, version 2 of 2026-11-18)

One fixed prompt is used for all 61 items. The first version (2026-11-09, one trade-off per item, fixed category
lists, "field: value" answers) is kept as `prompt_v1_2026-11-09.txt` and was never run. Version 2 differs in four
ways: it allows **several trade-offs per conversation**; RQ1 to RQ3 are coded **openly** (no fixed category lists);
RQ2 is recorded **per argument** and RQ3 includes **negotiation strategies**; and the answer is **one JSON object**
with a fixed template.

**Ground rules.** Use only the given text (no browsing of the GitHub link). Cite turn ids. Quotes are exact excerpts.
Do not infer motivations, reasons, downsides or environmental benefits. Open codes are short (2 to 6 words) and
grounded in the wording of the discussion; the examples in the prompt show the style and are not categories to pick.

**Step 1: Detect trade-offs.** The three conditions of §2.2 (resource concern, competing objective, link to the same
decision), plus the rules: speed can be on either side but speed vs speed does not count; a technique's name
(quantization, offloading …) is not a resource concern unless its effect is stated; resource savings are not
environmental benefits unless a participant says so; exclude unrelated mentions, bug reports without weighing, and
optimizations without a stated downside. **Distinct trade-off** = one development decision × one pair of objectives;
repeated mentions are one trade-off with several quotes. Each listed trade-off gets `tradeoff_status` yes / uncertain;
the item gets `tradeoff_present` yes (at least one "yes"), uncertain (only "uncertain" candidates) or no
(`tradeoffs` = []). `screening_explanation` says why.

**Step 2: RQ1, trade-off type** (per trade-off):

| Field | Meaning |
|---|---|
| `decision` | the development decision, in a few words |
| `resource_concern`, `competing_objective` | each side in the discussion's own terms |
| `tradeoff_relationship` | how gaining one side costs the other |
| `tradeoff_open_code` | "<resource side> vs <competing side>", e.g. "GPU memory vs accuracy" |
| `speed_role` | resource / competing / none |
| `environmental_framing` | direct (energy, power, emissions, carbon, sustainability named) / resource_proxy |
| `explicit_or_implicit` | explicit (one turn states both sides and the conflict) / implicit (emerges across turns) |
| `rq1_evidence` | list of {comment_id, quote} |

**Step 3: RQ2, justifications** (per trade-off, a list; [] if none). One entry per distinct argument that favours a
side **and gives a reason** (a bare preference or "LGTM" is not one): `participant`, `participant_role` (MEMBER,
CONTRIBUTOR, NONE …), `prioritized_side` (resource / competing / balanced), `prioritized_objective`, `reason`,
`justification_open_code` (what the reason is about, e.g. "users with small GPUs", "maintenance burden"),
`evidence_type` (how it is supported, e.g. benchmark or measurement, user or hardware constraint, technical
reasoning, project convention or policy, prior issue or experience, none), `evidence` ({comment_id, quote}).

**Step 4: RQ3, disagreement and resolution** (per trade-off). `rq3_disagreement` yes / no / uncertain: "yes" needs at
least two participants with conflicting positions on this trade-off (questions, clarifications, or one participant
weighing options alone are "no"). Then `rq3_positions` ({participant, participant_role, position}),
`rq3_negotiation_strategies` (open codes, e.g. "requesting benchmark", "proposing opt-in flag"),
`rq3_resolution_status` (resolved / deferred / unresolved / unclear), `rq3_resolution`, `rq3_resolution_open_code`
(e.g. "made optional via flag", "maintainer decision"), `rq3_evidence`. Without a disagreement these are [] / null.
`rq3_observable_outcome` is always filled: what visibly happened to the decision (turns + outcome line). Merging or
closing alone does not count as resolving a disagreement.

**Output.** One JSON object with a fixed template (keys and order given in the prompt), starting with
`last_turn_id`, `tradeoff_present`, `screening_explanation`, `tradeoffs`. Every field is filled for every listed
trade-off, including "uncertain" ones, so the researcher has everything when confirming a candidate.

### 7.4 How the prompt was designed (for the methodology chapter)

- **Grounded in the definition:** Step 1 is the R ∧ G ∧ T definition used throughout the project, written as
  checkable conditions, with explicit rules for the hard cases (speed, technique names, environmental claims).
- **Evidence required:** every claim needs a turn id and an exact quote. This makes the answers **verifiable** and
  reduces hallucination, because a quote that is not in the thread is easy to detect automatically.
- **Several trade-offs per conversation**, with a rule for what counts as distinct (decision × pair of objectives),
  so long threads are not reduced to one trade-off and the same trade-off is not counted twice.
- **Open coding with an LLM first pass:** ChatGPT proposes open codes for trade-off types (RQ1), justifications (RQ2)
  and negotiation strategies and resolutions (RQ3). Each chat is isolated, so the same idea will be worded
  differently across items; consolidating the codes into a taxonomy is the researcher's job (§9).
- **Separate dimensions instead of one label:** reason vs evidence type (RQ2); status vs resolution vs observable
  outcome (RQ3); environmental framing and speed role (RQ1). These keep different questions apart during coding.
- **Strict thresholds for RQ2/RQ3:** a reason must be written; two participants must disagree; closing or merging is
  not a resolution. These keep the RQ2/RQ3 subsets conservative.
- **Fixed JSON template:** the 61 answers come from 61 separate chats; a fixed template keeps the keys identical, so
  all answers can be parsed and checked by one script.

### 7.5 Procedure per item (manual, in the ChatGPT app)

Before starting:

1. **Freeze the prompt.** Keep a dated copy (for example `prompt_v2_2026-11-18.txt`). Do not change it during the
   61 runs. If it has to change, start again from item 1.
   *Note: `src/export_chatgpt.py` writes `prompt.txt` from the code, which still holds an older one-step prompt.
   Re-running the export would overwrite the current prompt.*
2. **Fix the model** (one ChatGPT model for all 61 items) and record its exact name.
3. **Isolate the chats:** turn off ChatGPT memory and custom instructions (or use a temporary chat), and turn off web
   search. Otherwise an earlier item could influence a later one.

For each item, in `index.csv` order:

1. Open a **new chat**.
2. Paste `prompt.txt`, then the content of `threads/<order>_<item>.md` right after "DISCUSSION:". If it is too long to
   paste, attach the file and note "attached".
3. Send. Copy the full answer into `results.xlsx` (column `chatgpt_json`), and fill in `chatgpt_model`, `date` and
   `pasted_or_attached`.
4. **Completeness check:** compare ChatGPT's `last_turn_id` with the `last_turn_id` in the sheet. If they differ,
   ChatGPT did not read to the end: rerun (split or attach) and write this in `notes`.

Practical notes:

- If ChatGPT adds text around the JSON, or the JSON is cut off (very long answers), ask it in the same chat to
  "return the complete JSON object only" and note this in `notes`.
- When a file is **attached** instead of pasted, ChatGPT may search the file instead of reading all of it. For the
  four long threads (§7.2), check the evidence with extra care.

---

## 8. Phase F: Checking ChatGPT's answers

ChatGPT is an **assistant**, not the final judge. Its answers are checked before they enter the analysis. Because
there are only 61 items, all of them can be checked.

### 8.1 Automatic checks (cheap, objective)

- **Completeness:** `last_turn_id` matches `index.csv`.
- **Format:** the answer parses as JSON with the template's keys; values are from the allowed lists; `tradeoffs` is
  [] exactly when `tradeoff_present` is "no"; RQ3 fields are [] / null when `rq3_disagreement` is "no".
- **Evidence is real:** every quote occurs (word for word, after whitespace normalization) in the conversation,
  and every cited `comment_id` exists **and** contains its quote. A quote that is not found is a red flag for
  hallucination.
- **Consistency:** `speed_role` agrees with the two sides (no speed vs speed); `environmental_framing` is "direct" only
  if an energy/emissions word appears in the quotes.

### 8.2 Researcher review (every item)

The researcher reads each conversation together with ChatGPT's answer and records, per item and per listed
trade-off:

| Decision | Values |
|---|---|
| Step 1 | each listed trade-off confirmed / rejected; missed trade-offs added; final item label yes / no |
| RQ1 | open code, sides, framing accepted / corrected |
| RQ2 | each justification accepted / corrected / removed; missed ones added |
| RQ3 | disagreement, positions, strategies, resolution accepted / corrected |
| note | why, when something was corrected |

Rules:

- Every ChatGPT **yes** is read in full: it must be confirmed before it counts.
- Every **uncertain** is decided by the researcher. If it remains undecidable, it is excluded and counted.
- Every **no** is also read, at least quickly: ChatGPT can miss trade-offs, and with 61 items this is affordable.
- The researcher's **final** labels are what the analysis uses. ChatGPT's raw labels are kept for the agreement
  report (§8.3).

### 8.3 Reporting the quality of the ChatGPT step

- **Agreement ChatGPT ↔ researcher** on Step 1 (percentage and Cohen's κ), and per RQ field (share of answers
  accepted without correction).
- Number of hallucinated or unfound evidence spans.
- Number of incomplete reads (`last_turn_id` mismatch) and how they were fixed.

### 8.4 Optional additional checks (decide before analysis starts)

- **Second human coder** on a subset (for example all confirmed trade-offs, or 20 random items) for κ on Step 1 and on
  the RQ1/RQ2/RQ3 codes. This is the usual expectation of reviewers for qualitative coding.
- **Cross-model check:** 50 of the 61 HIGH items were already screened by the local Qwen3-32B model (pilot run
  `runs/2026-11-02_10-40-39_llm_qwen3-32b/`, 29 yes among those 50, Step 1 only). Agreement between the two models
  on Step 1 can be reported as a robustness check. It is **not** a measure of accuracy.
- **Stability check:** re-run ChatGPT on a few items in a new chat to see whether Step 1 flips. ChatGPT is not
  deterministic.

---

## 9. Phase G: Analysis per research question

### 9.1 Unit of analysis and nested subsets

The **unit of analysis is one trade-off** (one development decision × one pair of objectives) inside one item. An
item can contain several trade-offs (prompt version 2). Report both counts: items with at least one confirmed
trade-off, and confirmed trade-offs.

```
 61 HIGH_PRIORITY items
   └── N_I items with at least one confirmed trade-off
         └── N_T confirmed trade-offs                                → RQ1 (all N_T)
               ├── N_J ⊆ N_T with at least one written justification → RQ2 (each justification is coded)
               ├── N_D ⊆ N_T with a disagreement between participants → RQ3 (discussion, negotiation)
               │      └── N_R ⊆ N_D with status resolved or deferred  → RQ3 (resolution)
               └── trade-offs with neither: still count for RQ1
```

**Illustration (made-up numbers).** Suppose ChatGPT and the researcher confirm **20** trade-offs. All 20 enter RQ1.
Perhaps 14 of them contain a written justification (RQ2 set, 14/20), and 8 contain a disagreement, of which 6 show how
it ended (RQ3 sets, 8/20 and 6/8). Some items are in both RQ2 and RQ3; some are in neither. Always report the
denominator: "14 of the 20 trade-offs (70%) contain an explicit justification", not "14 items".

### 9.2 RQ1: Taxonomy of trade-off types (open coding)

Input: for each confirmed trade-off, ChatGPT's (corrected) `tradeoff_open_code`, `resource_concern`,
`competing_objective`, `decision` and `rq1_evidence`.

Procedure (inductive, in the style of open → axial coding; ChatGPT's codes are the first pass):

1. **Initial codes.** Start from the free-text labels ("GPU memory vs accuracy", "VRAM vs generation quality",
   "inference latency vs code complexity"). Read the evidence again and adjust the label so it describes what the
   developers actually traded.
2. **Constant comparison.** Compare each new code with the existing ones. Merge synonyms ("VRAM vs generation quality"
   → "GPU memory vs accuracy/model quality"); split codes that turn out to mean different things. Keep a short
   **codebook**: code name, definition, inclusion/exclusion, one example quote with link.
3. **Grouping (axial).** Group the codes into higher-level categories along the two dimensions (resource side ×
   competing side). The categories come from the data; the definition in §2.2 (energy, compute, memory, cost,
   speed on the resource side) can serve as a sensitizing frame.
4. **Context attributes** recorded per trade-off, useful for describing the taxonomy:
   - framing: direct environmental vs resource proxy (§2.4);
   - the decision context: e.g. quantization, attention kernel, model loading/offloading, generation, training,
     serialization, dtype / precision, caching;
   - setting: training vs inference;
   - item type: issue vs PR;
   - explicit vs implicit (from Step 1);
   - speed role: speed as resource side or competing side.
5. **Result:** a taxonomy table: category → codes → number of items → one representative quote each. A resource ×
   competing **matrix** (counts per cell) gives a compact overview.

### 9.3 RQ2: Justifications

Input: every confirmed entry of `rq2_justifications` (after review). One trade-off can have several justifications,
for both sides.

1. Consolidate `justification_open_code` (what the reason is about) into categories by open → axial coding, as for
   RQ1 (e.g. user/hardware constraints, maintenance burden, backward compatibility, regression risk, scope).
2. Code `evidence_type` (how the reason is supported: benchmark/measurement, user constraint, technical reasoning,
   policy, prior experience, none) as a second dimension.
3. Record **which side was prioritized** (`prioritized_side`: resource / competing / balanced).
4. Record **who** justified it (`participant_role`: MEMBER/maintainer, CONTRIBUTOR, outside user NONE).
5. Analyse:
   - frequency of each justification category;
   - prioritized side overall and **per trade-off category from RQ1** (for example: is accuracy usually defended with
     benchmarks? is maintainability usually defended by maintainers?);
   - evidence-based (benchmarks, measurements) vs argument-based (policy, effort, scope) justifications.

### 9.4 RQ3: Disagreement and resolution

Input: confirmed trade-offs with `rq3_disagreement = yes` (discussion), and the subset with
`rq3_resolution_status` resolved or deferred (resolution).

1. Describe the **positions** (`rq3_positions`): who held which side, and their roles (e.g. contributor wants the
   faster kernel, maintainer objects on maintainability).
2. Consolidate the **negotiation strategies** (`rq3_negotiation_strategies`, e.g. requesting evidence, challenging,
   proposing alternatives) into categories by open → axial coding.
3. Consolidate the **resolution codes** (`rq3_resolution_open_code`, e.g. made optional, settled by benchmark,
   maintainer decision, compromise, moved to follow-up) the same way, and count the **status** (resolved / deferred /
   unresolved / unclear).
4. Compare with the **observable outcome** (`rq3_observable_outcome`), but interpret it only together with what
   participants wrote: a PR can be merged or closed for reasons unrelated to the trade-off.
5. Analyse:
   - how often trade-offs lead to a disagreement at all (N_D / N_T);
   - negotiation strategies, resolution categories and their frequency;
   - who decides (maintainer vs consensus vs evidence);
   - resolution kind per trade-off category (RQ1) and per justification kind (RQ2), for example whether
     benchmark-backed arguments end disputes more often.

### 9.5 What to report

| Table / figure | Content |
|---|---|
| Funnel | 10,000 → 1,085 R → 407 R+G → 386 linked → 61 HIGH → N_T confirmed → N_J / N_D / N_R |
| Screening quality | ChatGPT vs researcher on Step 1 (agreement, κ), corrections per RQ field, unfound evidence |
| RQ1 taxonomy | categories, codes, counts, example quotes; resource × competing matrix; direct vs proxy framing |
| RQ2 | justification categories × counts; evidence types; prioritized side; who; cross-tab with RQ1 categories |
| RQ3 | disagreement rate; negotiation strategies; resolution status and categories × counts; who decided; outcome |
| Case vignettes | 2–4 short worked examples that show a whole episode: trade-off → justification → resolution |

Quotes are cited with the item number and link, so readers can check them.

---

## 10. A worked example of one item (made up, to show the flow)

> **[body]** contributor: "This PR adds int4 quantization for the KV cache. It halves the cache memory on long
> prompts."
> **[c1]** maintainer: "Our tests show perplexity goes up by 3% on Llama-3-8B. That is too much to make it the
> default."
> **[c2]** contributor: "Users with 24 GB cards can't run 32k context otherwise. Benchmarks attached."
> **[c3]** maintainer: "OK, let's add it behind `cache_implementation="quantized"`, off by default, with a warning in
> the docs."
> outcome: state=closed, merged

| Step | Answer |
|---|---|
| 1 | one trade-off, status **yes**: memory (KV cache) vs accuracy (perplexity); decision: "make int4 KV cache the default" |
| RQ1 | open code "KV-cache memory vs accuracy" · speed_role none · resource_proxy · explicit |
| RQ2 | two justifications: contributor → resource side, "users with 24 GB cards", evidence: user/hardware constraint + benchmark; maintainer → competing side, "perplexity loss too large", evidence: measurement |
| RQ3 | disagreement yes; strategies: "presenting benchmark", "proposing opt-in flag"; status resolved; resolution code "made optional via flag"; outcome: "PR merged with the flag off by default" |

---

## 11. Threats to validity and limitations

**Construct validity**

- "Sustainability" is mostly operationalised through **resource proxies** (memory, speed, compute). We report the
  direct-environmental vs proxy split and do not claim energy savings that are not stated.
- Several trade-offs per item are allowed; deciding what counts as "distinct" (decision × pair of objectives) is a
  judgement, checked by the researcher.

**Internal validity (the ChatGPT step)**

- **Hallucination and misreading:** mitigated by mandatory turn ids and verbatim evidence, the automatic evidence
  check, and researcher review of every item.
- **Long conversations:** the four longest threads (22k–43k tokens) risk partial reading ("lost in the middle") or
  file search instead of full reading. The `last_turn_id` check shows only that the end was reached, not that the
  middle was read. Extra manual checking is needed here.
- **Non-determinism and model drift:** ChatGPT answers can change between runs and model versions. Mitigated by one
  fixed model, a frozen prompt, recorded model name and date, isolated chats; optionally a stability re-run.
- **Proprietary model:** results cannot be exactly reproduced by others. The prompt, the threads, the raw answers and
  the researcher's final labels are archived. The earlier local open-weights run (Qwen3-32B) offers a reproducible
  comparison point.
- **Single coder:** if no second coder is used, the coding depends on one person. A second coder on a subset (§8.4)
  addresses this.
- **Prompt anchoring:** the example codes in the prompt may steer ChatGPT toward them. The prompt says they are
  examples of style, not categories, and the researcher's open coding works from the evidence, not only the codes.
- **Inconsistent wording across chats:** each item is coded in an isolated chat, so the same idea gets different
  codes. This is expected; consolidation is done by the researcher (§9).

**External validity**

- **Selection by the rules:** only items with explicit trade-off language (HIGH) are analysed. Trade-offs written
  implicitly, or with words outside the lexicon, are missed. The results describe **types, justifications and
  resolutions** that occur. They do not estimate the **prevalence** of trade-offs in the repository. Recall of the rule
  filter is not measured in this approach.
- **One repository, one time window:** `huggingface/transformers`, 10,000 items before 2026-09-01. Other LLM projects
  (DeepSpeed, Megatron-LM) may differ.
- **Small N:** with 61 candidates and fewer confirmed trade-offs, the counts are descriptive. No statistical
  generalisation, and no claim of theoretical saturation unless more groups or repositories are added.

**Reliability (repeatability of the procedure)**

- Fixed inputs (frozen rules v1, fixed 61 items, rendered threads), frozen prompt, full log in `results.xlsx`, and a
  codebook for the open coding.

---

## 12. Open decisions (to settle before or during analysis)

1. **Second coder:** none, a subset, or all confirmed trade-offs? (Recommended: at least the confirmed trade-offs, κ on
   Step 1 and on the RQ codes.)
2. **Use of the Qwen3-32B run:** report ChatGPT–Qwen agreement on the 50 overlapping items, or leave it out?
3. **Extension:** apply the same procedure to (a sample of) POSSIBLE_TRADEOFF items to see what HIGH misses.

---

## 13. Status and files

| Step | Status | Where |
|---|---|---|
| Data, conversations, rules v1, 4 groups | done | `data/threads/…`, `data/detect/rules_v1/huggingface__transformers/items.csv` |
| Export of the 61 HIGH conversations | done | `data/chatgpt/huggingface__transformers/high_priority_v1/threads/`, `index.csv` |
| Prompt v2 (several trade-offs, open codes for RQ1–RQ3, JSON) | done (2026-11-18; v1 never run) | `…/high_priority_v1/prompt.txt` (freeze a dated copy); v1 in `prompt_v1_2026-11-09.txt` |
| ChatGPT runs, 61 items | **in progress** | `…/high_priority_v1/results.xlsx` |
| Automatic checks + researcher review | to do | |
| Open coding (RQ1 taxonomy), RQ2/RQ3 coding | to do | |
| Report tables and vignettes | to do | |
