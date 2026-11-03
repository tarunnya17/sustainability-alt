"""
Step 6c: how well do the detectors find trade-off conversations? Compared with the adjudicated gold labels.

    .venv/bin/python src/evaluate.py                                    # dev split, rules_v1 (+ LLM runs if given)
    .venv/bin/python src/evaluate.py --llm runs/<time>_llm_qwen3-32b
    .venv/bin/python src/evaluate.py --split test --llm ...             # ONCE, at the very end
    .venv/bin/python src/evaluate.py --selftest                         # on the fake gold of agreement.py --selftest

Input : <annotation dir>/gold.csv, data/detect/rules_<v>/<repo>/items.csv, <llm run>/predictions.parquet
Output: reports/evaluation_<split>.json, reports/errors_<split>.csv   (runs/selftest/ for --selftest)

Detectors (positive = predicts TRADEOFF):
  rules_<v>_high  HIGH_PRIORITY            rules_<v>_any  HIGH_PRIORITY or POSSIBLE_TRADEOFF
  llm_yes         LLM says yes             llm_yes_or_uncertain
  hybrid_<v>      rules_<v>_any and llm_yes
Metrics: precision, recall, F1 on the sample, and design-weighted (each item counts N_stratum / n_stratum times,
so the numbers estimate all items, not just the stratified sample), with bootstrap 95% CIs.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

from common import ANNOTATION, DETECT, REPORTS, ROOT, RUNS, SEED


def prf(tp, fp, fn):
    """Precision, recall, F1 from (possibly weighted) counts; nan where undefined."""
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p + r else float("nan")
    return p, r, f


def weighted_prf(y, pred, w):
    y, pred, w = np.asarray(y, bool), np.asarray(pred, bool), np.asarray(w, float)  # bootstrap passes floats
    return prf(float(w[y & pred].sum()), float(w[~y & pred].sum()), float(w[y & ~pred].sum()))


def ci(y, pred, w, which):
    """Bootstrap 95% CI (percentile) of one weighted metric; items resampled with their weights."""
    idx = {"precision": 0, "recall": 1, "f1": 2}[which]
    if y.sum() == 0 or pred.sum() == 0:
        return [float("nan"), float("nan")]
    res = bootstrap((y, pred, w), lambda a, b, c: weighted_prf(a, b, c)[idx], paired=True, vectorized=False,
                    n_resamples=2000, method="percentile", random_state=SEED)
    return [round(float(res.confidence_interval.low), 3), round(float(res.confidence_interval.high), 3)]


def detector_columns(gold, rules_versions, llm_dirs):
    """One boolean column per detector, plus evidence columns for the error sheet."""
    df = gold.copy()
    names = []
    for v in rules_versions:
        items = pd.concat(pd.read_csv(p) for p in (DETECT / f"rules_{v}").glob("*/items.csv"))
        items = items[["repo", "item_number", "group", "top_evidence"]].rename(
            columns={"group": f"rules_{v}_group", "top_evidence": f"rules_{v}_evidence"})
        df = df.merge(items, on=["repo", "item_number"], how="left")
        df[f"rules_{v}_high"] = df[f"rules_{v}_group"].eq("HIGH_PRIORITY")
        df[f"rules_{v}_any"] = df[f"rules_{v}_group"].isin(["HIGH_PRIORITY", "POSSIBLE_TRADEOFF"])
        names += [f"rules_{v}_high", f"rules_{v}_any"]
    for d in llm_dirs:
        tag = "" if len(llm_dirs) == 1 else "_" + Path(d).name
        pred = pd.read_parquet(Path(d) / "predictions.parquet")
        pred = pred[["repo", "item_number", "tradeoff_present", "explanation", "evidence_verified"]].rename(
            columns={"tradeoff_present": f"llm{tag}_answer", "explanation": f"llm{tag}_explanation",
                     "evidence_verified": f"llm{tag}_evidence_verified"})
        df = df.merge(pred, on=["repo", "item_number"], how="left")
        missing = df[f"llm{tag}_answer"].isna().sum()
        if missing:
            print(f"warning: {missing} gold items have no prediction in {d} (counted as 'no')")
        df[f"llm{tag}_yes"] = df[f"llm{tag}_answer"].eq("yes")
        df[f"llm{tag}_yes_or_uncertain"] = df[f"llm{tag}_answer"].isin(["yes", "uncertain"])
        names += [f"llm{tag}_yes", f"llm{tag}_yes_or_uncertain"]
        for v in rules_versions:
            df[f"hybrid{tag}_{v}"] = df[f"rules_{v}_any"] & df[f"llm{tag}_yes"]
            names.append(f"hybrid{tag}_{v}")
    return df, names


def evaluate(df, names):
    rows = []
    y = df["label"].eq("TRADEOFF").to_numpy()
    w = df["weight"].to_numpy(float)
    for name in names:
        pred = df[name].fillna(False).to_numpy(bool)
        tp, fp, fn = int((y & pred).sum()), int((~y & pred).sum()), int((y & ~pred).sum())
        p, r, f = prf(tp, fp, fn)
        wp, wr, wf = weighted_prf(y, pred, w)
        rows.append({"detector": name, "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
                     "w_precision": round(wp, 3), "w_recall": round(wr, 3), "w_f1": round(wf, 3),
                     "w_precision_ci": ci(y, pred, w, "precision"), "w_recall_ci": ci(y, pred, w, "recall"),
                     "w_f1_ci": ci(y, pred, w, "f1")})
    return rows


def main():
    parser = argparse.ArgumentParser(description="Detection quality against the gold labels (Step 6c).")
    parser.add_argument("--annotation", default=str(ANNOTATION / "pilot_v1"))
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--rules", nargs="+", default=["v1"], help="detector versions, e.g. v1 v2")
    parser.add_argument("--llm", nargs="*", default=[], help="LLM run folders (with predictions.parquet)")
    parser.add_argument("--again", action="store_true", help="allow a second evaluation on the test split")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    annotation = RUNS / "selftest" / "annotation" if args.selftest else Path(args.annotation)
    out = RUNS / "selftest" if args.selftest else REPORTS
    out.mkdir(parents=True, exist_ok=True)
    marker = out / ".test_evaluated"
    if args.split == "test" and not args.selftest:
        if marker.exists() and not args.again:
            raise SystemExit(f"The test split was already evaluated ({marker.read_text().strip()}). "
                             "Tuning must use dev; pass --again only if you know why.")
        marker.write_text(f"{datetime.now():%Y-%m-%d %H:%M} rules={args.rules} llm={args.llm}\n")

    gold = pd.read_csv(annotation / "gold.csv")
    uncertain = gold["label"].eq("UNCERTAIN")
    if args.split != "all":
        gold = gold[gold["split"] == args.split]
        uncertain = uncertain[gold.index]
    print(f"{args.split}: {len(gold)} items, {int(uncertain.sum())} final UNCERTAIN left out")
    gold = gold[~uncertain]

    llm_dirs = [ROOT / d if not str(d).startswith("/") else Path(d) for d in args.llm]
    df, names = detector_columns(gold, args.rules, llm_dirs)
    result = {"split": args.split, "n": len(df), "n_tradeoff": int(df["label"].eq("TRADEOFF").sum()),
              "uncertain_left_out": int(uncertain.sum()), "rules": args.rules, "llm": [str(d) for d in args.llm],
              "overall": evaluate(df, names)}
    if df["repo"].nunique() > 1:
        result["per_repo"] = {repo: evaluate(g, names) for repo, g in df.groupby("repo")}
    (out / f"evaluation_{args.split}.json").write_text(json.dumps(result, indent=2))

    wrong = pd.Series(False, index=df.index)
    for name in names:
        wrong |= df[name].fillna(False) != df["label"].eq("TRADEOFF")
    keep = ["order", "repo", "item_number", "stratum", "label", "thread_file", "url"] + names + \
           [c for c in df.columns if c.endswith(("_evidence", "_explanation", "_evidence_verified"))]
    df.loc[wrong, keep].to_csv(out / f"errors_{args.split}.csv", index=False)

    table = pd.DataFrame(result["overall"]).set_index("detector")
    print(table[["tp", "fp", "fn", "precision", "recall", "f1", "w_precision", "w_recall", "w_f1", "w_f1_ci"]].to_string())
    print(f"\n{int(wrong.sum())} items where at least one detector is wrong -> {out / f'errors_{args.split}.csv'}")


if __name__ == "__main__":
    main()
