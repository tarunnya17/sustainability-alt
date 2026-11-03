"""
The pipeline in numbers: detection funnel, rater agreement, detection quality, episodes.

    .venv/bin/python src/report.py
    .venv/bin/python src/report.py --llm runs/<time>_llm_qwen3-32b      # adds the LLM answers to the funnel

Reads whatever exists so far (detector summary, agreement.json, evaluation_*.json, episodes/summary.json)
and writes reports/pipeline_report.md. Sections whose inputs do not exist yet say so.
"""

import argparse
import json
from datetime import date

import pandas as pd

from common import ANNOTATION, DETECT, REPO, REPORTS, ROOT, STRATA, load_turns


def table(df):
    return df.to_markdown(index=False)


def main():
    parser = argparse.ArgumentParser(description="Write reports/pipeline_report.md.")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--rules", default="v1")
    parser.add_argument("--llm", nargs="*", default=[])
    args = parser.parse_args()

    items = pd.read_csv(DETECT / f"rules_{args.rules}" / args.repo / "items.csv")
    human = items["item_number"].isin(load_turns(args.repo)["item_number"])
    r_items = items["group"].isin(STRATA[:3])
    rg_items = r_items & items["g_groups"].fillna("").ne("")
    funnel = [("issues and PRs collected", len(items)),
              ("with human-written text", int(human.sum())),
              ("resource concern counts (R or speed)", int(r_items.sum())),
              ("... and a competing objective somewhere (R + G)", int(rg_items.sum())),
              ("R and G linked: trade-off candidates (HIGH + POSSIBLE)", int(items["group"].isin(STRATA[:2]).sum())),
              ("... with explicit trade-off language (HIGH)", int(items["group"].eq("HIGH_PRIORITY").sum()))]
    for run in args.llm:
        pred = pd.read_parquet(ROOT / run / "predictions.parquet")
        funnel.append((f"LLM 'yes' among {len(pred)} screened ({run.rstrip('/').split('/')[-1]})",
                       int(pred["tradeoff_present"].eq("yes").sum())))
    pilot = ANNOTATION / f"pilot_{args.rules}"
    if (pilot / "gold.csv").exists():
        gold = pd.read_csv(pilot / "gold.csv")
        funnel.append(("human-validated TRADEOFF in the 200-item pilot sample", int(gold["label"].eq("TRADEOFF").sum())))
    episodes = ROOT / "data" / "episodes" / "summary.json"
    if episodes.exists():
        e = json.loads(episodes.read_text())
        funnel.append(("human-validated trade-off episodes", e["episodes"]))

    lines = [f"# Trade-off detection pipeline: report ({date.today()})", "",
             f"Repository: `{args.repo}`; detector `rules_{args.rules}` (the frozen sampling frame).", "",
             "## Funnel", "", table(pd.DataFrame(funnel, columns=["step", "items"])), ""]

    groups = items.groupby(["group", "item_type"]).size().unstack(fill_value=0).reindex(STRATA)
    lines += ["### Detector groups", "", groups.reset_index().to_markdown(index=False), ""]

    lines += ["## Rater agreement (Cohen's kappa)", ""]
    if (pilot / "agreement.json").exists():
        k = pd.DataFrame(json.loads((pilot / "agreement.json").read_text())).T.reset_index(names="question")
        lines += [table(k), ""]
    else:
        lines += ["Not yet: both raters have to finish `rater_1.xlsx` and `rater_2.xlsx`, then run `src/agreement.py`.", ""]

    lines += ["## Detection quality", ""]
    found = False
    for split in ("dev", "test", "all"):
        path = REPORTS / f"evaluation_{split}.json"
        if not path.exists():
            continue
        found = True
        r = json.loads(path.read_text())
        m = pd.DataFrame(r["overall"])[["detector", "tp", "fp", "fn", "precision", "recall", "f1",
                                         "w_precision", "w_recall", "w_f1", "w_f1_ci"]]
        lines += [f"### {split} split: {r['n']} items, {r['n_tradeoff']} TRADEOFF "
                  f"({r['uncertain_left_out']} UNCERTAIN left out)", "",
                  "`w_` = design-weighted (estimates all items, not just the sample).", "", table(m), ""]
    if not found:
        lines += ["Not yet: needs `gold.csv` (after adjudication), then `src/evaluate.py`.", ""]

    lines += ["## Episodes", ""]
    lines += [f"```json\n{episodes.read_text()}\n```" if episodes.exists() else
              "Not yet: needs the pilot gold labels and/or a filled confirmation sheet, then `src/extract_episodes.py`.", ""]
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "pipeline_report.md").write_text("\n".join(lines))
    print("\n".join(lines[:30]))
    print(f"\nwritten to {(REPORTS / 'pipeline_report.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
