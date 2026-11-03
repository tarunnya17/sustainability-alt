# Rater guide: labelling trade-off conversations

Thank you for rating. This guide explains how to label one conversation. The definitions are in `codebook.md`, which is
also the third sheet of your spreadsheet. **Read the codebook first, then this guide.** If they ever seem to disagree,
the codebook wins. Tell the coordinator.

## What we are looking for

You will read 200 GitHub conversations (issues and pull requests) from the `huggingface/transformers` repository. For
each one, decide whether developers discuss a **trade-off** between:

- **R**, a **resource concern**: energy, power, carbon, compute, memory, hardware or cloud cost, efficiency techniques
  (quantization, offloading, KV cache …), or speed, **and**
- **G**, a **competing objective**: accuracy/quality, speed, maintainability/complexity, reliability/correctness,
  compatibility, functionality, development effort,

where **the developers themselves connect the two in the same decision** (**T**): one improves at the cost of the
other, someone objects to a change because of a cost, alternatives are compared, or a compromise is proposed.

Many conversations contain no trade-off. That is expected, so don't go looking for one. Your labels are the reference
that automatic methods will be measured against, so **your own careful judgment is exactly what we need.**

## What you received

```
rater_N.xlsx     your spreadsheet: sheets "labels", "episodes", "codebook"
threads/         200 files, one conversation each (.md = plain text)
codebook.md      the definitions and decision rules
rater_guide.md   this guide
```

Keep `rater_N.xlsx` and `threads/` in the same folder. Then the `thread_file` link in the sheet opens the conversation.
The `github` link opens the original page, which is useful for seeing the code changes of a PR.

## How to read a thread file

A made-up example:

```
# #99999 [pull_request] Load the vision tower in int4
https://github.com/huggingface/transformers/pull/99999  ·  opened by alice on 2026-05-02

## [body] alice (CONTRIBUTOR) · opening post · 2026-05-02
This loads the vision tower in int4. Peak GPU memory goes from 22 GB to 14 GB.

## [c1001] bob (MEMBER) · review CHANGES_REQUESTED · 2026-05-03
I'm worried about quality: on our captioning eval the int4 tower loses ~2 CIDEr points.

## [c1002] bob (MEMBER) · review_comment on src/transformers/models/x/modeling_x.py:210 · 2026-05-03
Nit: this import is unused.

## [c1003] alice (CONTRIBUTOR) · issue_comment · reply to [c1001] · 2026-05-04
Fair. I'll keep bf16 as the default and make int4 opt-in.

---
outcome: state=closed, merged=True
```

| Part | Meaning |
|---|---|
| `# #99999 [pull_request] …` | number, type (issue or pull_request) and title |
| `[body]`, `[c1001]` | the **id** of each turn. You cite these ids. The title belongs to `[body]` |
| `alice (CONTRIBUTOR)` | author and their role in the repository: MEMBER (Hugging Face organization), COLLABORATOR, CONTRIBUTOR (has contributed before), NONE (no previous contribution) |
| `opening post` | the first post of the issue/PR |
| `issue_comment` | a normal comment |
| `review APPROVED / CHANGES_REQUESTED / COMMENTED` | an overall review of a PR. `(no text)` means a review with no text |
| `review_comment on <file>:<line>` | an inline comment on a line of code |
| `reply to [c1001]` | this turn answers that one. It is found automatically, so it can be missing; read the order and quotes yourself |
| `[… 40 lines of code omitted]` | a long code block was shortened |
| `[collapsed block 'logs', 25 lines omitted]` | a collapsed section (logs, system info) was hidden |
| `[image]` | an image was here |
| `outcome:` | how it ended: open/closed, and for PRs whether it was merged |

Comments by bots (CI, doc builders) have been removed. Everything people wrote is in the file.

## Labelling one conversation, step by step

1. **Read the whole conversation.** You can skim code, but read all the prose, including inline review comments
   near the end. Trade-offs often show up late, in a review.
2. **Answer Q1 → Q2 → Q3 in order:**

   ```
   Q1  Is there a resource concern?                        no  → Q2 = no, Q3 = no → label UNRELATED
   Q2  Is there a competing objective (a second goal)?     no  → Q3 = no          → label RESOURCE_ONLY
   Q3  Do R and G compete in the same decision?            no  →                    label RESOURCE_ONLY
                                                           yes →                    label TRADEOFF
   Cannot decide from the text?                                                    label UNCERTAIN (+ notes)
   ```

3. **For TRADEOFF:** fill Q4 (which turns), Q5 (explicit/implicit), `multi_developer` and `evidence`, then add
   **one row per trade-off episode** in the `episodes` sheet.
4. **Save.**

## The `labels` sheet, column by column

The first six columns (`order` … `github`) are filled in already. Don't change them.

| Column | What to enter |
|---|---|
| `Q1_resource_concern` | `yes` / `no`. Speed or runtime performance alone (a slowdown, a faster kernel) counts as a resource concern |
| `Q2_competing_objective` | `yes` / `no`. Is a second desirable goal discussed? (`no` whenever Q1 is `no`) |
| `Q3_tradeoff_same_decision` | `yes` / `no`. Do R and G compete in the **same** decision? (`no` whenever Q1 or Q2 is `no`) |
| `Q4_supporting_ids` | the ids of the turns that show it, without brackets, separated by `;`, e.g. `body;c1001;c1003`. Required for TRADEOFF. For RESOURCE_ONLY, the turn(s) with the resource concern. Empty for UNRELATED |
| `Q5_explicit_implicit` | only for TRADEOFF. `explicit`: stated in one place ("saves memory but loses accuracy"). `implicit`: built across turns (a proposal, then an objection). `unclear` |
| `label` | `TRADEOFF` / `RESOURCE_ONLY` / `UNRELATED` / `UNCERTAIN`. It must match Q1-Q3 (see the steps above) |
| `multi_developer` | only for TRADEOFF. `yes` if at least two different people exchange views on the trade-off |
| `evidence` | short quote(s) copied from the thread that support your label |
| `notes` | anything useful. **Required for UNCERTAIN**: say what is unclear |

Use the drop-down lists. Don't type variants such as "Yes", "trade-off" or "y".

## The `episodes` sheet (TRADEOFF only)

An **episode** is one trade-off. Most TRADEOFF conversations have one. A long PR can have two separate ones, for example memory vs
accuracy in some comments and memory vs maintainability in others. That is two rows.

| Column | What to enter |
|---|---|
| `item` | the item number from the `labels` sheet (column `item`, e.g. `99999`) |
| `episode_no` | `1`, `2`, … within that item |
| `comment_ids` | all turns of this episode, where it is raised, argued and settled, e.g. `body;c1001;c1003` |
| `resource_concern` | the resource side: `energy/environment`, `compute/cost`, `memory`, `efficiency technique`, `speed`, `other`. Choose the resource that is saved or spent (int4 that saves memory → `memory`). Use `efficiency technique` only when no resource is named |
| `competing_objective` | `accuracy/quality`, `speed`, `maintainability`, `reliability`, `compatibility`, `functionality`, `development effort`, `other` |
| `tradeoff_statement` | one sentence in your own words: "X improves R but costs G (resolved by …)" |
| `tradeoff_framing` | `direct_environmental` **only** if energy, power, emissions or (environmental) sustainability is named. Otherwise `resource_proxy` |
| `explicit_implicit` | as Q5, for this episode |
| `multi_developer_exchange` | `yes` if two or more people discuss this episode |
| `evidence_spans` | the key quotes, copied word for word from the thread |
| `notes` | optional |

`speed` can be either side, but not both. "2x faster but less accurate" has resource `speed` and objective `accuracy/quality`. "Saves memory
but slower" has resource `memory` and objective `speed`.

For the thread example above:

| item | episode_no | comment_ids | resource_concern | competing_objective | tradeoff_statement | framing | explicit_implicit | multi_dev |
|---|---|---|---|---|---|---|---|---|
| 99999 | 1 | body;c1001;c1003 | memory | accuracy/quality | int4 vision tower cuts peak memory 22→14 GB but loses ~2 CIDEr; int4 made opt-in | resource_proxy | implicit | yes |

## Decision rules, short version

The full rules are in the codebook. These are the ones raters most often forget:

1. **Don't add downsides nobody mentioned.** An optimization with no cost discussed is RESOURCE_ONLY, even if you can
   think of a cost.
2. **Memory or speed savings are not energy savings** unless someone says energy/power/emissions.
3. **R and G must be about the same decision.** OOM in one comment and an unrelated accuracy question in another →
   RESOURCE_ONLY.
4. **"Reduce memory by 40%" is an optimization claim, not a trade-off.** "…but lose 2% accuracy" makes it one.
5. **A plain resource bug** (OOM, memory leak, slow kernel) is RESOURCE_ONLY unless a fix is weighed against another goal.
6. **Benchmarks showing both sides** count only if the discussion treats the second number as a considered cost or
   risk. "No accuracy loss" in answer to someone's concern counts. A table that nobody discusses does not.
7. **UNCERTAIN** sparingly, always with a note.

Label from **what people write in the conversation**. Use the GitHub page to understand context, for example what a
review comment refers to. A trade-off has to be stated or argued by the participants.

## Worked examples

All made up, and not from your sample.

| Conversation (short) | Label | Why |
|---|---|---|
| "Quantization reduces GPU memory by 35%, but decreases accuracy by 1.5%." | TRADEOFF, explicit, multi_developer no | R, G and "but" in one place, by one author |
| The int4 example above | TRADEOFF, implicit, multi_developer yes | proposal → objection → compromise |
| "This PR switches attention to SDPA, which reduces memory requirements." Approved, no other discussion. | RESOURCE_ONLY | no cost discussed (rule 1) |
| Issue: "OOM when loading a 70B model on 2×A100." Replies explain `device_map`. | RESOURCE_ONLY | resource bug, nothing weighed (rule 5) |
| "With torch.compile this is 2× faster, but outputs no longer match eager to 1e-5." | TRADEOFF | speed (R) vs correctness (G) |
| "Gradient checkpointing saves memory but training gets ~30% slower, so only enable it for the large config." | TRADEOFF | memory (R) vs speed (G) |
| "The new kernel lowers latency but throughput drops for large batches." | RESOURCE_ONLY | speed against speed is not a trade-off in this study |
| "Not a sustainable approach: every new model would need its own copy of this code." Nothing about resources. | UNRELATED | "sustainable" means maintainable here (rule 2) |
| A PR shows a table: "2× faster, perplexity 5.12 → 5.12". Nobody comments on it. | RESOURCE_ONLY | the second number is never treated as a risk (rule 6) |
| Reviewer: "Does the fused kernel hurt perplexity?" Author: "No, identical on wikitext, see table." | TRADEOFF, implicit, multi_developer yes | a concern raised and answered in the same decision (rule 6) |
| "Running in fp8 cuts energy per token by 30%, but needs H100s, so older GPUs lose support." | TRADEOFF, framing `direct_environmental` | energy is named; compatibility is the cost |
| An issue about a model's accuracy regression; resources never come up. | UNRELATED | no R |

## Good practice

- **Work alone.** Don't discuss items with the other rater until the coordinator says so.
- **Don't use AI tools** (ChatGPT, Copilot, Claude, …) to read, summarise or label the threads. AI is what your labels
  will be compared against.
- Questions about the codebook go to the **coordinator**, who answers both raters in writing. If a question is about one
  specific conversation, label it UNCERTAIN and write your question in `notes`.
- **Don't change the sheet's structure.** Don't add, delete, rename or reorder columns, don't delete rows, and don't
  rename the sheets. Filtering is fine; avoid sorting.
- Save often and keep a backup copy.
- Work in sessions of 20-30 conversations. The full set is roughly 20 hours: most threads are short, a few are very long.

## FAQ

**The thread is very long.** Skim the code, read the prose. Inline review comments (`review_comment on …`) are where
many trade-offs are argued, so don't skip them.

**The trade-off is only in the code diff, not in the conversation.** Then the conversation does not show a trade-off.
Label what people write.

**Where is the title?** In the first line. It belongs to `[body]`, so cite `body`.

**One conversation, two different trade-offs?** One row in `labels` (TRADEOFF), two rows in `episodes`.

**The same trade-off comes back much later.** Same episode. Add the later turns to its `comment_ids`.

**Not sure whether something is "the same decision"?** UNCERTAIN, with a note saying what you are unsure about.

## When you are done

Check that every row has a `label` and every TRADEOFF has an episode row. Save, and send **only `rater_N.xlsx`** back to the
coordinator. Keep a copy for yourself.
