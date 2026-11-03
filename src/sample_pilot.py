"""
Step 6a: draw the pilot gold-standard sample and write the annotation sheets.

    .venv/bin/python src/sample_pilot.py                    # 50 per detector group, from rules_v1

Input : data/detect/rules_<frame>/<repo>/items.csv (the sampling frame; v1 is frozen)
Output: annotation/pilot_<frame>/
          sample.csv        HIDDEN KEY (stratum, design weight, dev/test split). Raters must not open it.
          threads/*.md      each conversation as raters see it (no keyword highlighting, no stratum)
          rater_1.xlsx      label sheet for rater 1 (sheets: labels, episodes, codebook)
          rater_2.xlsx      the same for rater 2
          codebook.md       definitions and decision rules

Design weight = N_stratum / n_stratum, so precision and recall can be estimated for all items, not just the
sample. The dev/test split is fixed here: rules and prompt are tuned on dev; test is used once at the end.
"""

import argparse

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation
from sklearn.model_selection import train_test_split

from common import ANNOTATION, DETECT, LABELS, REPO, ROOT, SEED, STRATA, load_conversations, load_turns, render_thread

YES_NO = ["yes", "no"]
EXPLICIT = ["explicit", "implicit", "unclear"]
RESOURCE = ["energy/environment", "compute/cost", "memory", "efficiency technique", "speed", "other"]
OBJECTIVE = ["accuracy/quality", "speed", "maintainability", "reliability", "compatibility", "functionality",
             "development effort", "other"]
FRAMING = ["direct_environmental", "resource_proxy"]

LABEL_COLUMNS = [  # (column, width, allowed values or None)
    ("order", 6, None), ("item", 8, None), ("type", 12, None), ("title", 50, None), ("thread_file", 28, None),
    ("github", 10, None),
    ("Q1_resource_concern", 12, YES_NO), ("Q2_competing_objective", 12, YES_NO),
    ("Q3_tradeoff_same_decision", 12, YES_NO), ("Q4_supporting_ids", 22, None), ("Q5_explicit_implicit", 12, EXPLICIT),
    ("label", 16, LABELS), ("multi_developer", 10, YES_NO), ("evidence", 50, None), ("notes", 40, None),
]
EPISODE_COLUMNS = [
    ("item", 8, None), ("episode_no", 8, None), ("comment_ids", 24, None), ("resource_concern", 18, RESOURCE),
    ("competing_objective", 18, OBJECTIVE), ("tradeoff_statement", 50, None), ("tradeoff_framing", 18, FRAMING),
    ("explicit_implicit", 12, EXPLICIT), ("multi_developer_exchange", 10, YES_NO), ("evidence_spans", 60, None),
    ("notes", 30, None),
]

CODEBOOK = """# Codebook: sustainability-related trade-off conversations

Read the whole conversation (threads/<file>.md; the GitHub link has the code changes) before labelling.
Every turn starts with its id in brackets, e.g. `[body]` or `[c3184836411]`. Cite these ids in Q4 and in the episodes sheet.

## The three elements

- **R, resource concern**: environmental sustainability (energy, power, carbon, emissions) or computational resources
  (compute, FLOPs, GPU hours, memory, VRAM, OOM, cost of hardware/cloud, efficiency techniques such as quantization,
  pruning, offloading, KV cache, gradient checkpointing). **Speed** (latency, throughput, slowdown) counts as R when it
  is weighed against a non-speed objective, and as G when it is weighed against a non-speed resource.
- **G, competing objective**: a second desirable outcome: accuracy/model quality, speed, maintainability/complexity,
  reliability/correctness/numerical stability, compatibility/portability, functionality/features, development effort.
- **T, trade-off relationship**: the developers themselves connect R and G **in the same development decision**: one is
  improved at the cost of the other, an objection names a downside of a resource-motivated change (or the reverse),
  alternatives are compared, or a compromise is proposed (make it optional, a flag, both modes).

## Questions

| Column | Question |
|---|---|
| Q1 | Does the conversation concern environmental sustainability or computational resource efficiency? |
| Q2 | Is there an identifiable competing objective? |
| Q3 | Is there evidence that R and G compete **in the same decision** (proposed change, design choice, alternative)? |
| Q4 | Which turns provide the evidence? (ids separated by `;`) |
| Q5 | Is the relationship explicit (stated in one place, e.g. "saves memory but loses accuracy"), implicit (built across turns: a proposal, then an objection), or unclear? |
| label | TRADEOFF = Q1, Q2, Q3 all yes · RESOURCE_ONLY = Q1 yes but no competing objective or no relationship · UNRELATED = Q1 no · UNCERTAIN = not enough evidence; goes to adjudication |
| multi_developer | Do at least two different people actually exchange views on the trade-off? (A single author stating it is still TRADEOFF, but not multi-developer.) |
| evidence | Short quote(s) that support the label |

For every TRADEOFF conversation, add one row per **episode** in the `episodes` sheet. A long PR can hold two separate
trade-offs (e.g. memory vs accuracy in comments 8-14, memory vs maintainability in 27-35): that is two rows.

## Decision rules

1. **Do not infer disadvantages the developers never discussed.** "Changing the attention implementation reduces memory
   requirements" alone is RESOURCE_ONLY, even if you can imagine a cost.
2. **Do not infer energy or carbon savings from memory or runtime optimization.** Use `tradeoff_framing =
   direct_environmental` only when energy, power, emissions or sustainability (in the environmental sense) is named.
   "Not a sustainable approach" meaning "not maintainable" is not environmental.
3. **R and G in unrelated places are not a trade-off.** Accuracy in one bug report and memory in an unrelated comment
   of the same thread: RESOURCE_ONLY.
4. **An optimization claim is not a trade-off claim.** "Reduce GPU memory by 40%" is RESOURCE_ONLY;
   "reduce GPU memory by 40%, but lose 2% accuracy" is TRADEOFF.
5. **A plain bug report about a resource** ("OOM when loading model X", "fix GPU memory leak") is RESOURCE_ONLY unless
   a fix is weighed against another objective.
6. **Benchmarks that show both sides** ("2x faster, perplexity unchanged") are a trade-off only if the discussion treats
   the second number as a cost or risk that was considered; "no accuracy loss" as a reassurance against a raised
   concern counts, a table with no discussion does not.
7. Use **UNCERTAIN** sparingly, and always write a note: it is resolved in adjudication, never counted as yes or no.
"""


def write_sheet(path, rows):
    wb = Workbook()
    for name, columns, data in (("labels", LABEL_COLUMNS, rows), ("episodes", EPISODE_COLUMNS, [])):
        ws = wb.active if name == "labels" else wb.create_sheet(name)
        ws.title = name
        ws.append([c for c, _, _ in columns])
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for i, (col, width, allowed) in enumerate(columns):
            letter = ws.cell(row=1, column=i + 1).column_letter
            ws.column_dimensions[letter].width = width
            if allowed:
                dv = DataValidation(type="list", formula1='"' + ",".join(allowed) + '"', allow_blank=True)
                ws.add_data_validation(dv)
                dv.add(f"{letter}2:{letter}2000")
        for r in data:
            ws.append([r.get(c, "") for c, _, _ in columns])
        ws.freeze_panes = "B2"
    labels = wb["labels"]
    for row in labels.iter_rows(min_row=2):
        row[4].hyperlink = row[4].value
        row[5].hyperlink, row[5].value = row[5].value, "open"
        row[3].alignment = Alignment(wrap_text=True, vertical="top")
    cb = wb.create_sheet("codebook")
    cb.column_dimensions["A"].width = 140
    for line in CODEBOOK.split("\n"):
        cb.append([line])
    wb.save(path)


def main():
    parser = argparse.ArgumentParser(description="Pilot sample and annotation sheets (Step 6a).")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--frame", default="v1", help="detector version that defines the strata")
    parser.add_argument("--per-stratum", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="overwrite existing rater sheets (labels would be lost)")
    args = parser.parse_args()

    out = ANNOTATION / f"pilot_{args.frame}"
    if (out / "rater_1.xlsx").exists() and not args.force:
        raise SystemExit(f"{out} already has rater sheets; refusing to overwrite them (use --force only if they are empty).")

    items = pd.read_csv(DETECT / f"rules_{args.frame}" / args.repo / "items.csv")
    turns = load_turns(args.repo)
    human = items["item_number"].isin(turns["item_number"])
    print(f"frame: {human.sum()} items with human text ({(~human).sum()} bot-only items left out)")
    items = items[human]
    parts = []
    for stratum in STRATA:
        pool = items[items["group"] == stratum]
        n = min(args.per_stratum, len(pool))
        pick = pool if n == len(pool) else train_test_split(
            pool, train_size=n, stratify=pool["item_type"], random_state=SEED)[0]
        parts.append(pick.assign(stratum=stratum, N_stratum=len(pool), n_stratum=n, weight=len(pool) / n))
    sample = pd.concat(parts, ignore_index=True)
    dev, _ = train_test_split(sample.index, test_size=0.5, stratify=sample["stratum"], random_state=SEED)
    sample["split"] = "test"
    sample.loc[dev, "split"] = "dev"
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)  # raters see the strata mixed
    sample.insert(0, "order", range(1, len(sample) + 1))
    sample["thread_file"] = [f"threads/{r}__{n}.md" for r, n in zip(sample["repo"], sample["item_number"])]

    (out / "threads").mkdir(parents=True, exist_ok=True)
    conv = load_conversations(args.repo).set_index("item_number")
    turns = turns[turns["item_number"].isin(sample["item_number"])]
    for number, g in turns.groupby("item_number"):
        conv_row = conv.loc[number].copy()
        conv_row["item_number"] = number
        (out / f"threads/{args.repo}__{number}.md").write_text(render_thread(conv_row, g) + "\n")

    sample[["order", "repo", "item_number", "item_type", "stratum", "N_stratum", "n_stratum", "weight", "split",
            "thread_file", "url"]].to_csv(out / "sample.csv", index=False)
    rows = [{"order": r.order, "item": r.item_number, "type": r.item_type, "title": r.title,
             "thread_file": r.thread_file, "github": r.url} for r in sample.itertuples()]
    for rater in (1, 2):
        write_sheet(out / f"rater_{rater}.xlsx", rows)
    (out / "codebook.md").write_text(CODEBOOK)

    print(sample.groupby(["stratum", "split"]).size().unstack().reindex(STRATA).to_string())
    print(f"\n{len(sample)} items; weights: " +
          ", ".join(f"{s} {w:.2f}" for s, w in sample.groupby("stratum")["weight"].first().reindex(STRATA).items()))
    print(f"Output written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
