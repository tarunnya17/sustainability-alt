"""
Step 6b: agreement between the two raters, the adjudication sheet, and the final gold labels.

    .venv/bin/python src/agreement.py              # kappas + adjudication.xlsx (after both raters are done)
    .venv/bin/python src/agreement.py --finalize   # adjudication.xlsx -> gold.csv, gold_episodes.csv
    .venv/bin/python src/agreement.py --selftest   # whole chain on fake labels, in runs/selftest/annotation/

Input : <dir>/sample.csv, <dir>/rater_1.xlsx, <dir>/rater_2.xlsx
Output: <dir>/agreement.json      Cohen's kappa and % agreement per question
        <dir>/adjudication.xlsx   both raters side by side; final_* columns pre-filled where they agree.
                                  Fill the empty final_* cells (and fix episodes_final), then run --finalize.
        <dir>/gold.csv, gold_episodes.csv   (--finalize)

UNCERTAIN is never pre-filled: it always goes to adjudication. A final UNCERTAIN is allowed but is left out
of the detection metrics (evaluate.py reports how many).
"""

import argparse
import json
import shutil

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill
from sklearn.metrics import cohen_kappa_score

from common import ANNOTATION, LABELS, ROOT, RUNS, SEED

FIELDS = ["Q1_resource_concern", "Q2_competing_objective", "Q3_tradeoff_same_decision", "Q5_explicit_implicit",
          "label", "multi_developer"]
TEXT_FIELDS = ["Q4_supporting_ids", "evidence", "notes"]


def read_sheet(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet, dtype=str).fillna("")
    return df.apply(lambda s: s.str.strip())


def kappas(r1, r2):
    """Cohen's kappa and % agreement per field, on items both raters answered."""
    out = {}
    pairs = {f: (r1[f], r2[f]) for f in FIELDS}
    pairs["label_binary_tradeoff"] = (r1["label"].where(r1["label"] == "", r1["label"].eq("TRADEOFF").map(str)),
                                      r2["label"].where(r2["label"] == "", r2["label"].eq("TRADEOFF").map(str)))
    for name, (a, b) in pairs.items():
        both = (a != "") & (b != "")
        n = int(both.sum())
        if n == 0:
            out[name] = {"n": 0}
            continue
        agree = float((a[both] == b[both]).mean())
        kappa = cohen_kappa_score(a[both], b[both]) if len(set(a[both]) | set(b[both])) > 1 else float("nan")
        out[name] = {"n": n, "kappa": round(float(kappa), 3), "agreement": round(agree, 3)}
    return out


def write_adjudication(path, merged, episodes):
    wb = Workbook()
    ws = wb.active
    ws.title = "labels"
    cols = ["order", "item", "title", "thread_file", "github"]
    for f in FIELDS + TEXT_FIELDS:
        cols += [f"r1_{f}", f"r2_{f}"]
    cols += [f"final_{f}" for f in FIELDS] + ["final_Q4_supporting_ids", "adjudication_note"]
    ws.append(cols)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    todo = PatternFill("solid", fgColor="FFF2CC")
    for _, r in merged.iterrows():
        row = [r.get(c.removeprefix("final_"), "") if not c.startswith(("r1_", "r2_", "final_")) else
               r.get(c, "") for c in cols]
        ws.append(row)
        for f in FIELDS:  # pre-fill only real agreement; UNCERTAIN always needs a decision
            a, b = r[f"r1_{f}"], r[f"r2_{f}"]
            cell = ws.cell(row=ws.max_row, column=cols.index(f"final_{f}") + 1)
            if a and a == b and a != "UNCERTAIN":
                cell.value = a
            else:
                cell.fill = todo
        ids = sorted(set(filter(None, (r["r1_Q4_supporting_ids"] + ";" + r["r2_Q4_supporting_ids"]).split(";"))))
        ws.cell(row=ws.max_row, column=cols.index("final_Q4_supporting_ids") + 1).value = ";".join(ids)
    ws.freeze_panes = "C2"
    for name, df in episodes.items():
        sheet = wb.create_sheet(name)
        sheet.append(list(df.columns))
        for row in df.itertuples(index=False):
            sheet.append(list(row))
    wb.save(path)


def agreement(folder, force=False):
    sample = pd.read_csv(folder / "sample.csv")
    r1, r2 = (read_sheet(folder / f"rater_{i}.xlsx", "labels") for i in (1, 2))
    for r in (r1, r2):
        r["item"] = r["item"].astype(int)
    done = [(r["label"] != "").sum() for r in (r1, r2)]
    print(f"labelled: rater 1 {done[0]}, rater 2 {done[1]} of {len(sample)}")
    merged = r1[["order", "item", "title", "thread_file", "github"]].copy()
    for i, r in ((1, r1), (2, r2)):
        r = r.set_index("item").reindex(merged["item"]).reset_index()
        for f in FIELDS + TEXT_FIELDS:
            merged[f"r{i}_{f}"] = r[f].values
    result = kappas(merged.rename(columns={f"r1_{f}": f for f in FIELDS}), merged.rename(columns={f"r2_{f}": f for f in FIELDS}))
    (folder / "agreement.json").write_text(json.dumps(result, indent=2))
    print(pd.DataFrame(result).T.to_string())

    out = folder / "adjudication.xlsx"
    if out.exists() and not force:
        print(f"\n{out} exists (it may hold adjudication work); not overwritten. Use --force to rebuild it.")
        return
    e1, e2 = (read_sheet(folder / f"rater_{i}.xlsx", "episodes") for i in (1, 2))
    both = pd.concat([e1.assign(rater=1), e2.assign(rater=2)], ignore_index=True)
    agreed_tradeoff = merged.loc[(merged["r1_label"] == "TRADEOFF") & (merged["r2_label"] == "TRADEOFF"), "item"].astype(str)
    final = e1[e1["item"].isin(agreed_tradeoff)]  # a starting point; edit for the final episode list
    write_adjudication(out, merged, {"episodes_r1_r2": both, "episodes_final": final})
    print(f"\nadjudication sheet: {out.relative_to(ROOT)} ({(merged['r1_label'] != merged['r2_label']).sum()} label disagreements)")


def finalize(folder):
    sample = pd.read_csv(folder / "sample.csv")
    adj = read_sheet(folder / "adjudication.xlsx", "labels")
    missing = adj[adj["final_label"] == ""]
    if len(missing):
        raise SystemExit(f"{len(missing)} items have no final_label (order: {missing['order'].tolist()[:20]} ...)")
    bad = set(adj["final_label"]) - set(LABELS)
    if bad:
        raise SystemExit(f"unknown labels: {bad}")
    gold = adj[["item"] + [f"final_{f}" for f in FIELDS] + ["final_Q4_supporting_ids"]].copy()
    gold.columns = ["item_number"] + FIELDS + ["Q4_supporting_ids"]
    gold["item_number"] = gold["item_number"].astype(int)
    gold = sample.merge(gold, on="item_number", how="left")
    gold.to_csv(folder / "gold.csv", index=False)
    episodes = read_sheet(folder / "adjudication.xlsx", "episodes_final")
    episodes.to_csv(folder / "gold_episodes.csv", index=False)
    print(gold["label"].value_counts().to_string())
    print(f"gold.csv: {len(gold)} items, gold_episodes.csv: {len(episodes)} episodes -> {folder.relative_to(ROOT)}")


def selftest(folder):
    """Fake labels: rater 1 follows the detector stratum, rater 2 = rater 1 with 10% of labels changed."""
    test = RUNS / "selftest" / "annotation"
    shutil.rmtree(test, ignore_errors=True)
    test.mkdir(parents=True)
    for name in ("sample.csv", "rater_1.xlsx", "rater_2.xlsx"):
        shutil.copy(folder / name, test / name)
    rng = np.random.default_rng(SEED)
    sample = pd.read_csv(test / "sample.csv").set_index("item_number")
    guess = {"HIGH_PRIORITY": "TRADEOFF", "POSSIBLE_TRADEOFF": "RESOURCE_ONLY", "RESOURCE_ONLY": "RESOURCE_ONLY",
             "NO_MATCH": "UNRELATED"}
    for rater in (1, 2):
        wb = load_workbook(test / f"rater_{rater}.xlsx")
        ws = wb["labels"]
        head = [c.value for c in ws[1]]
        for row in ws.iter_rows(min_row=2):
            item = row[head.index("item")].value
            label = guess[sample.loc[item, "stratum"]]
            if sample.loc[item, "stratum"] == "POSSIBLE_TRADEOFF" and item % 3 == 0:
                label = "TRADEOFF"
            if rater == 2 and rng.random() < 0.10:
                label = rng.choice([l for l in LABELS if l != label])
            yn = {"TRADEOFF": ("yes", "yes", "yes"), "RESOURCE_ONLY": ("yes", "no", "no"),
                  "UNRELATED": ("no", "no", "no"), "UNCERTAIN": ("yes", "yes", "no")}[label]
            for f, v in zip(FIELDS[:3], yn):
                row[head.index(f)].value = v
            row[head.index("label")].value = label
            row[head.index("Q5_explicit_implicit")].value = "explicit" if label == "TRADEOFF" else ""
            if label == "TRADEOFF":
                wb["episodes"].append([item, 1, "body", "memory", "accuracy/quality", "fake", "resource_proxy",
                                       "explicit", "no", "fake span", ""])
        wb.save(test / f"rater_{rater}.xlsx")
    agreement(test, force=True)
    # fake adjudication: take rater 1 wherever a final_* cell is empty
    wb = load_workbook(test / "adjudication.xlsx")
    ws = wb["labels"]
    head = [c.value for c in ws[1]]
    for row in ws.iter_rows(min_row=2):
        for f in FIELDS:
            cell = row[head.index(f"final_{f}")]
            if cell.value in (None, ""):
                cell.value = row[head.index(f"r1_{f}")].value
    wb.save(test / "adjudication.xlsx")
    finalize(test)


def main():
    parser = argparse.ArgumentParser(description="Rater agreement, adjudication and gold labels (Step 6b).")
    parser.add_argument("--dir", default=str(ANNOTATION / "pilot_v1"))
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--force", action="store_true", help="rebuild adjudication.xlsx even if it exists")
    args = parser.parse_args()
    folder = ROOT / args.dir if not args.dir.startswith("/") else type(ROOT)(args.dir)
    if args.selftest:
        selftest(folder)
    elif args.finalize:
        finalize(folder)
    else:
        agreement(folder, args.force)


if __name__ == "__main__":
    main()
