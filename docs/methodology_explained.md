# How this project works, step by step (in plain language)

This document explains the whole project, from collecting the data to the final trade-off episodes. It is
meant to help you **understand** every step well enough to write the methodology chapter **in your own words**.

Each step has the same parts:

- **What** we did
- **How** we did it
- **Why** we did it this way (this is what reviewers ask about)
- **Points to mention in your methodology** (a checklist, not text to copy)

> **Done and not done yet.** Steps A, 1, 2-4, 5 (on the pilot) and 6a are finished. Steps 6b, 6c and 7 are
> designed and tested on fake labels, but they wait for the two human raters. In your methodology you can describe the
> finished steps in the past tense ("we collected …") and the waiting steps as the procedure ("two raters label …").
> Fill in the real numbers (κ, precision, recall) once they exist. Section 13 lists what is done.

---

## 1. The research problem in simple words

### The research questions

- **RQ1:** What kinds of sustainability-related trade-offs do developers discuss in open-source LLM projects?
- **RQ2:** How do they justify their choice (why do they prefer one side)?
- **RQ3:** How do they negotiate and resolve the trade-off (who argues, how does it end)?

To answer these questions we first need to **find** the conversations where such trade-offs happen. Most of this
project is about finding them reliably and proving that the finding method works.

### What counts as a trade-off

A trade-off has **three parts**. All three must be present:

| Part | Name | Simple meaning | Examples |
|---|---|---|---|
| **R** | resource concern | something that uses energy or hardware | energy, carbon, compute, FLOPs, GPU memory, out-of-memory (OOM), cloud cost, quantization, speed |
| **G** | competing objective | another thing developers want | accuracy, maintainability, reliability, compatibility, features, development effort |
| **T** | trade-off link | R and G compete **in the same decision** | "but", "at the cost of", "trade-off", an objection in a reply, a compromise ("make it optional") |

In short: **trade-off = R and G and T**.

Example of a trade-off:
*"int4 quantization halves GPU memory, but generation quality drops noticeably. Let's make it opt-in."*
R = GPU memory, G = generation quality, T = "but" plus the compromise "opt-in".

Example of **not** a trade-off:
*"This PR reduces memory usage by 40%."* There is R, but nobody mentions a cost. We call this **RESOURCE_ONLY**.

### Speed is special

Speed can be on either side:

- "2x faster **but less accurate**": speed is the **resource** (R), accuracy is the goal (G).
- "saves memory **but slower**": memory is the resource (R), speed is the goal (G).
- "lower latency but lower throughput": speed against speed. We do **not** count this as a trade-off.

For every link we record whether speed played the R role, the G role, or no role (`speed_role` = R / G / -).

### Why "sustainability" mostly means "resources" here

Developers in this repository almost never talk about energy or carbon directly (see Step A and Steps 2-4). They talk
about memory, speed and compute instead. These are **resource proxies**: using less memory or compute usually also
means using less energy, but the developers do not say so. We therefore record a **framing** for each trade-off:

- `direct_environmental`: energy, power, emissions or environmental sustainability is named
- `resource_proxy`: only memory, compute, speed, cost or efficiency techniques are named

We **never assume** energy savings when only memory or speed is mentioned.

---

## 2. The big picture

There are **two projects**:

1. The **sibling project** (`sustainability-developer-discussion`) **collected the data** from GitHub and tried a
   first, simple keyword filter.
2. **This project** reads the sibling's data (without changing or copying it) and builds a **better detector**: it
   requires a *link* between R and G, it compares a rule-based method with a local AI model (LLM), and it checks
   both against **human labels**.

```
 GitHub (huggingface/transformers)
        │  Step A: download issues, PRs, comments, reviews     (sibling project)
        ▼
 tables: items.csv + comments.csv
        │  Step 1: rebuild each issue/PR as an ordered conversation
        ▼
 105,525 human "turns" in 10,000 conversations
        │  Steps 2-4: keyword rules find R, G and a link between them
        ▼
 4 groups: HIGH_PRIORITY 61 · POSSIBLE_TRADEOFF 325 · RESOURCE_ONLY 699 · NO_MATCH 8,915
        │  Step 6a: draw 200 conversations (50 per group) for humans to label
        ▼
 200-item pilot sample ───────────────┬──────────────────────────────┐
        │                             │                              │
        │  Step 6b: two people        │  Step 5: local LLM           │
        │  label all 200 (gold)       │  (Qwen3-32B) labels the 200  │
        ▼                             ▼                              │
 gold standard  ──────────►  Step 6c: compare rules and LLM with gold (precision, recall, F1)
                                      │
                                      ▼  Step 7: run the best detector on the rest,
                                         a person confirms every flag
                              human-validated trade-off EPISODES  →  RQ1, RQ2, RQ3
```

The step numbers follow your methodology plan. Step 6a (the sample) happens before Step 5 (the LLM) because we first run
the LLM only on the 200 sample items. It makes no sense to run it on everything before we know how good it is.

---

## 3. Step A: Data collection (done in the sibling project)

### What

We downloaded the **10,000 newest issues and pull requests** of the GitHub repository **`huggingface/transformers`**
that were **created before 1 September 2026**, together with **all their comments, reviews and inline code-review
comments**.

| Fact | Value |
|---|---|
| Source | GitHub, repository `huggingface/transformers` |
| Items | 10,000 (7,837 pull requests, 2,163 issues) |
| Time window (creation date of the items) | 21 May 2025 to 31 August 2026 |
| Collection date | 4 October 2026 (the run took about 41 minutes) |
| Comment rows (all three kinds, including bots) | 102,783 |

### Why this repository and this window

- **Transformers** is one of the most important open-source LLM libraries. Many efficiency features
  (quantization, KV cache, attention kernels, offloading) are discussed and built there.
- **The newest items** are from the "LLM era" (mid-2025 onward). Older items (2018-2020) are mostly about BERT-style models.
- **One month of settling time**: items opened in September 2026 were left out, so most discussions had time to
  finish (merged, closed or rejected). This matters for RQ3, which is about how trade-offs are resolved.
- **Comments after the cutoff are kept**. If an August issue got replies in September, those replies are included.
- DeepSpeed and Megatron-LM are planned for later. The code already works for any repository.

### What kinds of data GitHub has

On GitHub, issues and pull requests share one number sequence (#1234 is either an issue or a PR). We call each one an
**item**. There are **three kinds of comments**, stored in different places:

| Kind | Where you see it on github.com |
|---|---|
| `issue_comment` | the normal conversation below an issue or PR |
| `review` | "X approved these changes" / "X requested changes", with an optional text |
| `review_comment` | an inline comment on a specific line of code in a PR |

We collected **all three**, because trade-offs are often argued in inline code reviews ("this doubles memory usage,
can we avoid it?").

We also kept: the author's **role** (MEMBER = Hugging Face staff, CONTRIBUTOR, NONE …), the **outcome** (open/closed,
merged or not, review decision), labels, timestamps and links between PRs and the issues they close.

### How

- A Python script used the **official GitHub API**: the REST API for issues and comments, and the **GraphQL API** for
  PR details (merged?, size, reviews, linked issues). A personal GitHub token gave 5,000 requests per hour.
- It used **repository-wide list endpoints** (100 records per request) instead of asking item by item. This needs a
  few hundred requests instead of tens of thousands.
- GitHub limits how deep you can page through long lists. The script handled this with **cursor-based paging** and
  **time windows** (start a new query from the last timestamp), so no comments were silently missed.
- It **waited** when the rate limit was nearly used, **retried** on network errors, and saved **checkpoints**, so a
  stopped run could continue.
- The exact API answers were saved unchanged as **raw files** (JSON Lines). Then they were turned into flat
  **tables** (`items.csv`, `comments.csv`). A **manifest** file records when the data was collected and with which
  settings and code version.

**Alternatives rejected:** the GitHub Search API (max 1,000 results), scraping the website (fragile and against
GitHub's terms), GH Archive/BigQuery (overkill and may cost money).

### The sibling's first keyword filter (background)

The sibling project also built a simple keyword filter. It marked a conversation as a trade-off candidate when a
resource word and a competing-goal word **both appear somewhere** in it (99 items in its `tradeoff` tier). Its
keyword list was made by merging two candidate lists (one drafted with ChatGPT, one written by hand) and then
**measuring every term on the data**. Terms that mostly matched unrelated text were dropped or made "weak".

It found something important: **explicit environmental words are almost absent**. Only 7 items used strong
environmental terms, and most of those used "sustainable" to mean "maintainable". This is a finding in itself:
developers talk about sustainability through resource proxies.

The weakness of that filter: two words that only *co-occur* (say, "OOM" in one comment and "accuracy" in an unrelated
comment) do not make a trade-off. **That is why this project exists.**

### How this project uses the data

This project **only reads** the sibling's two tables (`items.csv`, `comments.csv`). It never copies or changes them.
Before it starts, it checks their **MD5 checksums** (a fingerprint of the file). If a file changed, it stops. This
guarantees that every result comes from the same, fixed snapshot.

### Points to mention in your methodology

- repository, number of items, issues vs PRs, time window, collection date
- GitHub REST + GraphQL API; all three comment kinds; why inline review comments matter
- why the newest items and why the one-month cutoff
- raw data kept unchanged; tables rebuilt from it; checksums
- known limits of the data (see Section 12)

---

## 4. Step 1: Rebuilding the conversations

File: `src/prepare_threads.py`

### What

We turned the two tables into **one ordered conversation per item**. Each piece of text by one person is a **turn**.

| Turn kind | Its id | What it is |
|---|---|---|
| title | `title` | the issue/PR title (counted together with the opening post) |
| body | `body` | the opening post |
| issue_comment | `c<number>` | a normal comment |
| review | `c<number>` | a PR review |
| review_comment | `c<number>` | an inline comment on a line of code |

Turns are sorted by time inside each item. The ids (`body`, `c3184836411`) are what raters and the LLM cite as evidence.

### How

1. **Remove bots.** GitHub marks most bot accounts. One extra account (`HuggingFaceDocBuilderDev`, which posts the same
   "docs for this PR" message thousands of times) is on a manual list. We removed **16,854 bot comments** and the
   opening posts of **202 bot-opened items** (human comments under them stay). **138 items** have no human text at all.
   They stay in the data but are left out of the sample later.
2. **Clean the text for keyword matching.** We used exactly the same cleaning code as the sibling project (checked:
   the result is identical on every row). It:
   - lowercases the text and normalizes special characters (so "trade‑off" with an unusual hyphen still matches)
   - removes code blocks, inline code, error tracebacks, quoted lines (`> …`, so quoted text is not counted twice),
     links, @mentions, HTML and images
   - removes **boilerplate**: any line that appears in 20 or more different texts (PR/issue templates, reviewer lists).
     This found 603 such lines, with no hand-made template list.
3. **Split into sentences** (at `.`, `!`, `?` followed by a space, and at line breaks).
4. **Find who replies to whom.** GitHub only records replies inside inline review threads. For normal comments,
   people reply by quoting or by @mentioning someone. We used the raw text to find:
   - **review parent**: GitHub says this comment answers that one (15,064 turns)
   - **quote**: the comment quotes a line from an earlier turn (1,596 turns)
   - **mention**: the comment @mentions someone; we take that person's latest earlier turn (7,025 turns)
   - no detectable reply (62,244 turns)

### Result

**10,000 items → 105,525 human turns**: 9,798 titles, 9,798 opening posts, 23,205 normal comments, 24,673 reviews and
38,051 inline review comments.

### Two versions of each text, for two jobs

- The **cleaned** text is used **only by the keyword rules**.
- Humans and the LLM read a **readable** version of the original text: code is kept but long blocks are shortened,
  long log sections are collapsed, images become `[image]`, template comments are hidden. **Raters and the LLM always
  see exactly the same text.** This is important for a fair comparison.

### Why

- A trade-off is often built across several turns (a proposal, then an objection). We need the conversation order and
  the reply structure to see that.
- Bots are not developers, and their repeated messages would create false matches.
- Code, logs and templates contain many technical words ("memory", "quantization") that are not discussion.

### Points to mention in your methodology

- the unit of analysis (item = conversation; turn = one contribution by one person)
- bot removal (how many)
- text cleaning and boilerplate rule (reused from the sibling project for comparability)
- reply detection (review threads, quotes, mentions)
- raters and LLM see the same rendering

---

## 5. Steps 2-4: The rule-based detector (keywords)

Files: `src/lexicon.py` (the word lists) and `src/detect_rules.py`

This is the cheap, transparent **baseline** method. Everyone can see exactly why an item was flagged.

### Step 2 and 3: Find R words and G words

We wrote lists of words and phrases (as regular expressions) for each group:

| Role | Group | Example terms |
|---|---|---|
| R | `R_env` energy / environment | energy consumption, power draw, carbon footprint, CO2, kWh, sustainability |
| R | `R_compute` compute / cost | compute cost, FLOPs, GPU hours, GPU utilization, cloud cost, inference cost |
| R | `R_memory` memory | memory usage, peak memory, GPU memory, VRAM, OOM, out of memory, memory leak |
| R | `R_technique` efficiency techniques | quantization, int4/int8/fp8, pruning, distillation, LoRA, flash attention, offloading, batching |
| R or G | `SPEED` | latency, throughput, tokens/s, speedup, slowdown, overhead, faster/slower |
| G | `G_accuracy` | accuracy drop, perplexity, output quality, lower accuracy |
| G | `G_maintain` | maintainability, maintenance burden, technical debt, readability, added complexity |
| G | `G_reliability` | numerical stability, correctness, wrong results, NaN |
| G | `G_compat` | backward compatibility, breaking change, hardware support, portability |
| G | `G_function` | lose functionality, feature parity, unsupported, limitation |
| G | `G_effort` | development effort, engineering time, hard to implement, a lot of work |
| T | `T_marker` | trade-off, at the cost of, downside, drawback, compromise, worth it, sacrifice |
| T | `T_contrast` | but, however, although, whereas, unfortunately |
| T | `T_alternative` | rather than, instead of |
| T | `T_response` (ranking only) | concern, worried, optional, opt-in, configurable, revert, alternative |

Where the words came from: **most R terms were copied from the sibling project**, where every term had already been
measured on this data. A few were added from the methodology plan (GPU/cloud cost, "more/extra compute", batching,
caching). The G and T lists were written for this project from the competing objectives named in the plan, building on
the sibling's accuracy, maintainability and trade-off-language groups. Some terms were left out after looking at the
matches (for example "precision", which in this repository nearly always means the number format like "half
precision", not accuracy).

**No stemming.** Word endings, plurals, spellings and separators are written into the patterns by hand
(`quantiz*` matches quantize, quantized, quantization; `trade-off`, `trade off` and `tradeoff` all match). Automatic
stemmers break technical words like *bf16*, *tok/s* or *OOM*.

**Strong and weak terms.** This rule removes a lot of noise:

- A **strong** term always counts: "OOM", "peak memory", "perplexity", "maintenance burden".
- A **weak** term counts **only if the same sentence also has an effect word** (reduce, increase, save, drop, worse,
  better, less, more, improve, hurt …).
  "We load the model into **memory**" → does not count.
  "This **reduces** memory" → counts.
- Efficiency techniques (`R_technique`) are **weak only**. In Transformers, quantization and LoRA are everyday features.
  Just mentioning them is not a resource concern unless an effect is claimed.
- A match inside a longer match is dropped: "overhead" inside "memory overhead" is not a separate speed match.
  Otherwise "the memory overhead is large" would look like memory vs speed.

### Step 4: Find a link between R and G

R and G must come from **different groups**, and they must be **close together**. There are four kinds of link:

| Link kind | R and G are … | Example |
|---|---|---|
| `sentence` | in the same sentence | "Reduces memory, but accuracy decreases." |
| `comment` | in the same turn (title + opening post count as one) | "This reduces peak memory. The trade-off is added complexity." |
| `thread` | in two turns of the same inline review thread, or one turn replies to the other | review: "uses more memory" → reply: "but it's more readable" |
| `adjacent` | in two turns at most 2 positions apart | A: "quantize to reduce GPU memory" → B: "that will reduce model quality" |

A link is **explicit** when the trade-off is spelled out:

- a `sentence` link whose sentence also contains a T_marker, T_contrast or T_alternative word, or
- a `comment` link whose turn contains a T_marker word ("trade-off", "downside" …).

`thread` and `adjacent` links are never explicit. They catch **implicit** trade-offs: one person proposes, another objects.

### The four groups

Each item gets exactly one group:

| Group | Rule | Items |
|---|---|---|
| **HIGH_PRIORITY** | at least one explicit link | 61 |
| **POSSIBLE_TRADEOFF** | a link, but none explicit | 325 |
| **RESOURCE_ONLY** | an R (or speed) word counts, but no link to a G | 699 (253 of them only about speed) |
| **NO_MATCH** | nothing | 8,915 |

The funnel: 1,085 items have a resource concern → 407 also have a competing objective somewhere → **386 have R and G
linked** (HIGH + POSSIBLE) → 61 have explicit trade-off language.

Each item also gets: a **score** (only used for sorting: explicit links count most, two people involved adds points),
a **framing** (`direct_environmental` if the link uses an energy/environment word, otherwise `resource_proxy`), the
**speed role**, and the **best evidence sentence**.

The most common R × G pairs (number of items): memory × speed 102, speed × compatibility 53, speed × reliability 51,
memory × compatibility 46, memory × reliability 44.

### Examples (these are automated tests in `tests/test_rules.py`)

| Text | Group | Why |
|---|---|---|
| "Quantization reduces GPU memory by 35%, but decreases accuracy by 1.5%." | HIGH | memory + accuracy + "but" in one sentence |
| "This is 2× faster but less accurate." | HIGH | speed as R |
| "It saves memory but slower." | HIGH | speed as G |
| "Quantization increases inference latency." | POSSIBLE | a link, no trade-off word |
| A: "aggressive quantization to reduce GPU memory" → B: "That will reduce model quality considerably." | POSSIBLE | adjacent link, two developers |
| "Fix GPU memory allocation error." | RESOURCE_ONLY | R, no G |
| "Changing the attention implementation reduces memory requirements." | RESOURCE_ONLY | optimization, no cost named |
| "Latency is better but throughput is lower." | RESOURCE_ONLY | speed cannot trade against speed |
| "The model has excellent accuracy." | NO_MATCH | G alone means nothing |

### Freezing version 1

The rules were adjusted once **before any human label existed**: "rather than / instead of" first counted anywhere in a
comment, which made many unrelated R and G look "explicit". Now it only counts inside a sentence (HIGH went from 116 to 61).

Then the rules were **frozen as `v1`**. Version 1 defines the sample (Step 6a). If we changed it later, the sample and its
weights would no longer match. Any improvement after seeing labels goes into a **new version (`v2`)**, and both versions
are evaluated on the same human labels.

### Comparison with the sibling's filter

Of the sibling's 99 `tradeoff`-tier items, 74 are HIGH or POSSIBLE here. The other 25 have R and G only in unrelated
places, so the new rules call them RESOURCE_ONLY. Whether this is right is checked by the human labels.

### Why rules at all?

- Transparent and repeatable: same input, same output, and you can see the reason.
- Cheap: all 10,000 items in seconds.
- A **baseline** to compare the LLM with.
- The groups are used to build a smart sample (Step 6a).

Weakness: rules cannot understand meaning. They miss trade-offs written without the listed words, and they flag
coincidences.

### Points to mention in your methodology

- the R / G / T operationalization (with a table of groups and examples)
- strong vs weak terms and the effect-word rule; no stemming
- the four link kinds and the definition of "explicit"
- the four groups and their counts
- speed in either role
- v1 frozen before labelling; later changes as v2

---

## 6. Step 5: The LLM screen

File: `src/llm_screen.py`

### What

A large language model reads each conversation and answers: *is there a trade-off here, and where is the evidence?*
So far it has read the **200 pilot items**.

### Why an LLM, and why a local open model

- Rules miss **implicit** trade-offs and are fooled by coincidences. An LLM can read context.
- But an LLM can also **invent** things. That is why we measure it against human labels and check its evidence.
- We used a **local open-weights model** (not ChatGPT or another online service):
  - **reproducible**: the exact model version is fixed, anyone can download and re-run it
  - **free** to run on the lab's GPU, and no data is sent to a company
  - the model does not change behind our back

### How

1. **Model:** `Qwen/Qwen3-32B` (32 billion parameters, bf16 precision), run with the **vLLM** library on **one NVIDIA A100
   GPU**. The exact model revision is pinned (`9216db5781…`) and saved with the run.
2. **Settings:** temperature 0 (always pick the most likely answer, no randomness on purpose), Qwen's "thinking" mode off,
   seed 42, answers up to 2,048 tokens.
3. **Input:** the instructions, followed by the conversation **exactly as the raters see it**.
4. **The instructions** (from the methodology plan), in short:
   - include a trade-off only if (1) a resource concern exists, (2) a second objective exists, and (3) they compete in
     the same development decision
   - do not count ordinary bug reports, unrelated mentions, or optimizations with no downside
   - do not infer energy savings from memory or speed optimizations alone
   - do not infer disadvantages that the participants never discussed
   - cite the turn ids and copy evidence word for word
5. **Fixed answer format:** vLLM forces the model to answer in a fixed JSON structure: `tradeoff_present`
   (yes/no/uncertain), resource concern, competing objective, the relationship, supporting turn ids, exact evidence
   quotes, explicit or implicit, and a short explanation.
6. **Long conversations:** if a conversation is longer than 24,000 tokens, we keep the opening post, the turns with
   keyword matches, their neighbours and their review-thread starts, and replace the rest with
   "[… k comments omitted …]". Only **5 of the 200** needed this. The median conversation was about 1,300 tokens.
7. **Evidence check:** for every quote the model gives, the script checks that the words really appear in the turns it
   cites (ignoring case, punctuation and formatting). The share of quotes found is saved as `evidence_verified`.
   A low value means the model made up or misplaced its evidence.

### Result on the 200 pilot items

| Fact | Value |
|---|---|
| Run used for all results | `runs/2026-11-02_10-40-39_llm_qwen3-32b` |
| Time | about 7.5 minutes on one A100 |
| Answers in the correct format | 200 of 200 (100%) |
| Answers | 61 yes, 139 no, 0 uncertain |
| "yes" per rule group | HIGH 29 of 50, POSSIBLE 16 of 50, RESOURCE_ONLY 15 of 50, NO_MATCH 1 of 50 |
| Evidence quotes found in the cited turns | 92% on average |

These numbers say how often the LLM *says* yes. They do **not** say whether it is *right*. That needs the human labels (Step 6c).

### Lessons from the runs (worth reporting honestly)

- An earlier run with a 1,024-token answer limit cut one long answer in the middle. The limit is now 2,048.
- Between two runs one answer changed (yes → no), even at temperature 0. vLLM processes many conversations in
  parallel, and this is not perfectly repeatable at the bit level. Therefore **all results come from one fixed run**,
  and runs are never mixed.

### Points to mention in your methodology

- model name, size, exact revision, library, hardware
- temperature 0, thinking off, structured JSON output
- the prompt criteria (put the full prompt in an appendix; it is saved in the run's `prompt.txt`)
- how long conversations were handled (and how many)
- the evidence check
- non-determinism and the "one fixed run" rule

---

## 7. Step 6a: The pilot sample for human labelling

File: `src/sample_pilot.py`

### What

We selected **200 conversations** for people to label by hand. These labels become the **gold standard**: the
trusted reference that the rules and the LLM are measured against.

### Why not just a random sample?

Trade-offs are **rare**. A simple random sample of 200 from 9,862 conversations would contain about **one** HIGH item.
We would learn almost nothing. Instead we used a **stratified sample**: **50 items from each of the four rule groups**
(the groups are called **strata**). This gives many likely trade-offs, but also 50 NO_MATCH items, so we can see what
the rules **miss**.

### How

1. **Sampling frame:** the 9,862 items that have human-written text, each in its v1 group.
2. **50 items per group**, random seed 42. Within each group, the mix of issues and PRs matches the group's mix.
3. **Design weight:** each sampled item "stands for" a number of items in the whole repository:
   weight = (items in the group) / (items sampled from it).

   | Group | Items in frame (N) | Sampled (n) | Weight (N/n) |
   |---|---|---|---|
   | HIGH_PRIORITY | 61 | 50 | 1.22 |
   | POSSIBLE_TRADEOFF | 325 | 50 | 6.50 |
   | RESOURCE_ONLY | 699 | 50 | 13.98 |
   | NO_MATCH | 8,777 | 50 | 175.54 |

   So one NO_MATCH item in the sample represents about 176 items. These weights let us estimate results for the
   **whole repository**, not only for the sample (Step 6c).
4. **Dev/test split, fixed now:** in each group, 25 items are **dev** and 25 are **test**. We may look at dev as often
   as we like to improve the rules or the prompt. Test is used **once**, at the very end. This prevents us from tuning
   the method to the items we report on (like a student who sees the exam questions beforehand).
5. **Shuffled order:** the 200 items are mixed, so raters cannot guess the group from the position.
6. **Blind raters:** raters receive the conversations as plain text files, **without keyword highlighting and without the
   group**. The file with the groups and the split (`sample.csv`) is the **hidden key**. Raters never see it, and they
   never see the rule results or the LLM answers.

What raters receive: 200 thread files, their own spreadsheet (with drop-down lists), the codebook and the rater guide.

### Points to mention in your methodology

- why stratified (rarity of trade-offs); 50 per stratum; seed
- the design weights (table)
- dev/test split decided before labelling
- blinding (no strata, no highlights, no detector output)

---

## 8. Step 6b: Human labelling, agreement and adjudication

File: `src/agreement.py`. Guides: `docs/rater_guide.md` (for raters), `docs/manual_evaluation.md` (for the coordinator).
**Status: waiting for the raters.**

### What

**Two people** label all 200 conversations **independently**. Then we measure how much they agree, and they resolve
their disagreements together. The result is the **gold standard**.

### How one conversation is labelled

The rater reads the whole conversation and answers questions **in order**:

```
Q1  Is there a resource concern?                    no  → label UNRELATED
Q2  Is there a competing objective?                 no  → label RESOURCE_ONLY
Q3  Do R and G compete in the same decision?        no  → label RESOURCE_ONLY
                                                    yes → label TRADEOFF
Cannot decide from the text?                              label UNCERTAIN (with a note)
```

For a TRADEOFF they also fill in:

- **Q4**: which turns show it (ids like `body;c1001;c1003`)
- **Q5**: explicit (stated in one place) or implicit (built across turns)
- **multi_developer**: did at least two people exchange views on it?
- **evidence**: short quotes

And for every trade-off they add a row in the **episodes** sheet (one conversation can have two separate trade-offs):
which turns, the resource side (energy/environment, compute/cost, memory, efficiency technique, speed, other), the
competing objective (accuracy/quality, speed, maintainability, reliability, compatibility, functionality, development
effort, other), a one-sentence statement, the framing, explicit/implicit, multi-developer, and evidence quotes.

### The codebook's main decision rules

1. Don't add downsides nobody mentioned.
2. Memory or speed savings are not energy savings unless energy, power or emissions are named.
3. R and G must be about the **same** decision.
4. "Reduce memory by 40%" is an optimization claim, not a trade-off. "…but lose 2% accuracy" makes it one.
5. A plain resource bug (OOM, memory leak) is RESOURCE_ONLY unless a fix is weighed against another goal.
6. A benchmark showing both sides counts only if someone treats the second number as a cost or risk.
7. Use UNCERTAIN rarely, always with a note.

Raters label **what people write**, not what they could infer from the code. They must not use AI tools, because AI
is what their labels will be compared against.

### Agreement: Cohen's kappa (κ)

Simple "percent agreement" can be misleading. If 90% of conversations are "no trade-off", two raters who both
say "no" most of the time agree a lot **by chance**. **Cohen's κ corrects for chance agreement**:
κ = 1 means perfect agreement, κ = 0 means no better than chance. A commonly used rough guide: 0.41-0.60 moderate,
0.61-0.80 substantial, above 0.80 almost perfect.

κ is computed for Q1, Q2, Q3, Q5, the label, multi_developer, and most importantly for **TRADEOFF vs not TRADEOFF**.
It is reported **before** the disagreements are resolved, because that shows how clear the definitions are.

### Adjudication

The script builds an **adjudication sheet**: both raters' answers side by side. Where they agree, the final answer is
filled in automatically. Where they disagree, or either said UNCERTAIN, the cell is left empty for a decision. The two
raters discuss these cases using the codebook rules and write a one-line reason. If they still disagree, a third
person decides. A final UNCERTAIN is allowed but is left out of the metrics (and counted).

The final output is `gold.csv` (one final label per item) and `gold_episodes.csv`.

### Why

- One person's judgment could be biased. Two independent raters + κ show the labels are **reliable**.
- Adjudication gives one trusted answer per item.
- The estimated effort is high (about 245,000 words in total, roughly 20 hours per rater), which is why the sample is 200.

### Points to mention in your methodology

- two independent raters, their background (ML engineering knowledge)
- the codebook (Q1-Q5, labels, decision rules); put the codebook in an appendix
- optional practice round on items outside the sample
- κ before adjudication; the binary TRADEOFF κ
- adjudication procedure; how UNCERTAIN is handled

---

## 9. Step 6c: Evaluation (how good are the detectors?)

File: `src/evaluate.py`. **Status: waiting for the gold labels.**

### What

We compare each detector with the gold labels. A detector "says yes" in these cases:

| Detector | says yes when |
|---|---|
| `rules_v1_high` | the item is HIGH_PRIORITY |
| `rules_v1_any` | the item is HIGH_PRIORITY or POSSIBLE_TRADEOFF |
| `llm_yes` | the LLM answered yes |
| `llm_yes_or_uncertain` | the LLM answered yes or uncertain |
| `hybrid_v1` | `rules_v1_any` **and** `llm_yes` (rules find candidates, the LLM filters them) |

### The metrics, in simple words

Compared with the human TRADEOFF label:

- **TP** (true positive): flagged and really a trade-off. **FP** (false positive): flagged but not a trade-off.
  **FN** (false negative): a real trade-off that was missed.
- **Precision** = TP / (TP + FP): of the flagged items, how many are real trade-offs?
- **Recall** = TP / (TP + FN): of the real trade-offs, how many were found?
- **F1** = 2 × P × R / (P + R): one number that balances both.

Example: 60 correct flags, 20 wrong flags, 40 missed → precision 0.75, recall 0.60, F1 0.67.

### Why weighted metrics matter

Because the sample has 50 items from each group, but the groups are very different in size, the plain sample numbers
do **not** describe the repository. Each item must count as often as its design weight.

A made-up example. Suppose raters find TRADEOFF in 30 of 50 HIGH, 10 of 50 POSSIBLE, 3 of 50 RESOURCE_ONLY and 1 of
50 NO_MATCH items, and the detector flags HIGH and POSSIBLE:

| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| sample (each item counts 1) | 40 | 60 | 4 | 0.40 | 0.91 |
| weighted (each item counts its weight) | 101.6 | 284.4 | 217.5 | 0.26 | 0.32 |

The one missed NO_MATCH item stands for about 176 missed trade-offs. So the **real** recall is much lower than the
sample suggests. **The weighted numbers are the ones to report**, because they estimate the whole repository.

**Uncertainty:** every weighted number gets a **95% confidence interval** by bootstrapping: the items are resampled
2,000 times and we see how much the number changes. With 25 items per group per split, the intervals will be wide.
Say so in your limitations.

### Dev first, test once

1. Run the evaluation on **dev**. Read every error (`errors_dev.csv` shows the rule evidence and the LLM explanation
   side by side) and ask why it went wrong.
2. Improve the rules (as **v2**) and/or the LLM prompt, looking **only at dev**.
3. When everything is final, run on **test exactly once**. The script refuses a second test run unless forced, and a
   forced rerun must be explained in the paper.

### Why

- Without a human reference, the detectors' output means nothing.
- The test split gives an honest final number, not tuned to the items it is measured on.

### Points to mention in your methodology

- the detectors compared (including the hybrid)
- precision, recall, F1; design-weighted versions; bootstrap CIs
- dev for tuning, test once; error analysis on dev
- UNCERTAIN left out and counted

---

## 10. Step 7: The final trade-off episodes

File: `src/extract_episodes.py`. **Status: after the test evaluation.**

### What

The final dataset for RQ1-RQ3: **human-confirmed trade-off episodes**. An **episode** is one trade-off inside one
conversation. A long PR can contain two (for example memory vs accuracy in comments 8-14, memory vs maintainability in
comments 27-35).

### How

1. Choose the detector with the best **weighted test** results (probably `hybrid`).
2. Run the LLM on all **other** resource candidates outside the pilot (HIGH + POSSIBLE + RESOURCE_ONLY minus the 150
   already in the sample = **935 items**).
3. Build a **confirmation sheet** with every item the chosen detector flags.
4. **A trained person checks every flagged item** with the same codebook and adds the episode rows. Items that turn out
   not to be trade-offs are simply dropped. **No detector output enters the dataset without a human check.**
5. Collect the episodes from the pilot gold labels and from the confirmation sheet.

### Output

- `episodes.csv`: one row per episode, with repository, issue/PR number, turn ids, resource concern, competing
  objective, a one-sentence trade-off statement, framing, explicit/implicit, multi-developer exchange, evidence quotes,
  speed role, source (pilot or confirmed) and link.
- `episode_turns.parquet`: **every** turn of each episode's conversation, in order, with the episode's turns marked.
  RQ3 needs the whole conversation (who argued, how it ended), not only the quotes.
- `summary.json`, including **coverage**: if not every flagged item was confirmed, the episodes are a validated
  **sample**, not a complete list.

### How the episodes connect to the research questions

The pipeline produces the episodes. The analysis of RQ1-RQ3 comes after it, and its method is still yours to decide.
What the data gives you:

| RQ | What the episodes provide |
|---|---|
| RQ1 types of trade-offs | resource concern × competing objective for every episode, framing, explicit/implicit, speed role |
| RQ2 justifications | the evidence quotes and the full conversation, to code *why* people prefer one side |
| RQ3 negotiation and resolution | the full ordered conversation, author roles (maintainer vs outside contributor), multi-developer flag, and the outcome (merged, closed, opt-in compromise …) |

### Points to mention in your methodology

- definition of an episode
- detector chosen from the test results; LLM run on the pool
- every flagged item confirmed by a person
- what is missing: trade-offs in NO_MATCH items are never screened. The weighted recall from Step 6c estimates how many.

---

## 11. Safeguards for the quality of the results

These choices make the study trustworthy. They belong in your methodology or in a "validity" section.

| Safeguard | What it protects against |
|---|---|
| Input tables read-only, checked with MD5 checksums | results coming from a changed or different dataset |
| Same text cleaning as the sibling project | differences caused by preprocessing, not by the method |
| Rules v1 frozen before any label; changes go to v2 | a sample that no longer matches its weights |
| Stratified sample with design weights | learning nothing about rare trade-offs; misleading sample-only numbers |
| Dev/test split fixed in advance; test used once | tuning the method to the reported items (over-optimistic results) |
| Raters blind to strata, keyword hits and LLM answers | raters being influenced by the detectors |
| Two independent raters, κ, adjudication | one person's subjective judgment |
| Raters and LLM see exactly the same text | an unfair comparison |
| LLM: local, pinned revision, temperature 0, one fixed run | results that cannot be reproduced |
| Evidence check of the LLM's quotes | invented evidence |
| Every final episode confirmed by a person | detector errors entering the dataset |
| Scripts refuse to overwrite human work | losing labels by accident |
| Fixed random seed (42) everywhere | a different sample each time |

---

## 12. Limitations (be honest about these)

**Data**

- Only **one repository** so far (Transformers). Results may not hold for other projects. DeepSpeed and Megatron-LM are planned.
- About **15 months** of activity (May 2025 to August 2026).
- A **snapshot**: edited comments show only the latest text; deleted comments are gone; GitHub's list hid 5 PRs (0.05%).
- Only GitHub issues and PRs. Not included: commit messages, code changes, forums, Discord.
- Bot detection relies on GitHub's flag plus one manually listed account; a few automated accounts may remain.

**Method**

- The rules only see the listed words, in English. Trade-offs written differently are missed.
- Code is removed before matching. A trade-off visible only in the code (not in the discussion) is not counted. This is
  on purpose: we study what developers *say*.
- Reply detection (quotes, @mentions) is a heuristic and can be wrong or missing.
- The LLM can make mistakes or invent evidence (mitigated by the evidence check and human confirmation). Its output is
  not perfectly repeatable between runs.
- The pilot is small (25 items per group per split), so confidence intervals will be wide.
- Human labelling is a judgment, even with a codebook (mitigated by two raters and κ).
- Trade-offs inside NO_MATCH items never reach the final dataset; we can only estimate how many are missed.
- Speed against speed (latency vs throughput) is excluded by design.

**A finding, not a limitation:** explicit environmental language (energy, carbon) is almost absent. Developers
express sustainability concerns through **resource proxies** (memory, speed, compute).

---

## 13. Where the project stands now

| Step | Status | Key numbers |
|---|---|---|
| A. Data collection (sibling) | done | 10,000 items (7,837 PRs, 2,163 issues), 21 May 2025 - 31 Aug 2026, collected 4 Oct 2026 |
| 1. Rebuild conversations | done | 105,525 human turns; 16,854 bot comments removed; 138 bot-only items |
| 2-4. Rule detector v1 | done | HIGH 61 · POSSIBLE 325 · RESOURCE_ONLY 699 · NO_MATCH 8,915 |
| 6a. Pilot sample | done | 200 items (50 per group; 25 dev + 25 test each); sheets ready |
| 5. LLM on the pilot | done | 61 yes / 139 no; 100% valid answers; 92% evidence found |
| 6b. Two raters, κ, adjudication | **waiting for raters** | |
| 6c. Evaluation (dev, then test once) | after 6b | |
| 7. LLM on 935 pool items, confirmation, episodes | after 6c | |
| Report | can run any time | `reports/pipeline_report.md` |

All analysis scripts were already tested end to end on **fake labels** (a self-test), so they are ready when the real
labels arrive.

---

## 14. Tips for writing your methodology chapter

### A possible structure

1. **Research design overview**: the goal, the R ∧ G ∧ T definition, a figure of the pipeline (Section 2)
2. **Data source and collection**: repository, selection, window, API, what was collected (Section 3)
3. **Data preparation**: conversations, turns, bots, cleaning, reply structure (Section 4)
4. **Operationalizing trade-offs**: R, G, T groups, speed in either role, framing (Sections 1 and 5)
5. **Rule-based detection**: strong/weak terms, link kinds, explicit links, the four groups (Section 5)
6. **LLM-based screening**: model, settings, prompt, evidence check (Section 6)
7. **Gold standard**: stratified sample and weights, blinding, codebook, two raters, κ, adjudication (Sections 7-8)
8. **Evaluation**: detectors, weighted P/R/F1, bootstrap CIs, dev/test (Section 9)
9. **Episode extraction**: confirmation, the final dataset (Section 10)
10. **Threats to validity**: Sections 11 and 12

### How to write it in your own words

- For each step, answer three questions in your own sentences: **What did we do? How? Why this way?**
- Use the "Points to mention" lists above as a checklist, not as text.
- Give the **numbers** (they make the method concrete and checkable).
- Explain every choice that a reader could question (why stratified, why weighted, why a local LLM, why dev/test).
- Put long material in an **appendix**: the full keyword lists (`src/lexicon.py`), the LLM prompt
  (`runs/2026-11-02_10-40-39_llm_qwen3-32b/prompt.txt`), and the codebook (`annotation/pilot_v1/codebook.md`).
- Past tense for what is done, present or future for the planned steps. Update when the labels are in.

### Numbers you will probably cite

| What | Number |
|---|---|
| Items collected | 10,000 (7,837 PRs, 2,163 issues) |
| Creation window | 2025-05-21 to 2026-08-31 |
| Collection date | 2026-10-04 |
| Human turns | 105,525 |
| Bot comments removed / bot-opened items | 16,854 / 202 |
| Items with human text (sampling frame) | 9,862 |
| Boilerplate lines removed | 603 |
| Replies found: review threads / mentions / quotes | 15,064 / 7,025 / 1,596 |
| R-G links found / explicit | 1,047 / 114 |
| Rule groups v1 | HIGH 61, POSSIBLE 325, RESOURCE_ONLY 699 (253 speed only), NO_MATCH 8,915 |
| Sibling `tradeoff` items also HIGH/POSSIBLE here | 74 of 99 |
| Pilot sample | 200 (50 per group, 25 dev + 25 test each), seed 42 |
| Design weights | 1.22 / 6.50 / 13.98 / 175.54 |
| LLM | Qwen3-32B, bf16, vLLM 0.33.0, one A100, temperature 0, revision 9216db5781… |
| LLM pilot answers | 61 yes / 139 no; 5 of 200 conversations trimmed; 92% evidence verified |
| Pool for Step 7 | 935 items |

---

## 15. Glossary

| Term | Meaning |
|---|---|
| item | one GitHub issue or pull request |
| turn | one piece of text by one person (opening post, comment, review, inline review comment) |
| R / G / T | resource concern / competing objective / trade-off link |
| lexicon | the keyword lists used by the rules |
| strong / weak term | strong always counts; weak needs an effect word in the same sentence |
| link | R and G (different groups) close together: same sentence, same turn, related turns, or turns 2 apart |
| explicit / implicit | stated in one place with trade-off language / built across turns (proposal, then objection) |
| stratum (plural strata) | one of the four rule groups, used to draw the sample |
| sampling frame | the set the sample is drawn from (9,862 items with human text) |
| design weight | how many repository items one sampled item stands for (N / n) |
| dev / test | half of the sample for tuning / half for the one final measurement |
| gold standard | the final human labels after adjudication |
| Cohen's κ | agreement between two raters, corrected for chance |
| adjudication | deciding the final label where raters disagree or were unsure |
| precision / recall / F1 | share of flags that are right / share of real cases found / their balance |
| bootstrap CI | a range for a number, found by resampling the items many times |
| LLM | large language model (here Qwen3-32B, run locally) |
| temperature 0 | the model always picks its most likely answer |
| episode | one human-confirmed trade-off within one conversation |
| resource proxy | memory, speed or compute used as a stand-in for environmental cost |
