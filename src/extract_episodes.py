"""
Step 7: the validated trade-off episodes for RQ1-RQ3.

    # 1. after choosing a detector on dev/test: a confirmation sheet for its positives outside the pilot
    .venv/bin/python src/extract_episodes.py --confirm-sheet --detector hybrid --llm runs/<time>_llm_qwen3-32b
    # 2. after confirming (fill labels + episodes in annotation/confirm_v1/confirm_sheet.xlsx):
    .venv/bin/python src/extract_episodes.py

Detectors: rules_high, rules_any (rules only), llm_yes, hybrid (rules_any and llm_yes; needs --llm on the pool).

Output: annotation/confirm_<v>/confirm_sheet.xlsx + threads/   (--confirm-sheet)
        data/episodes/episodes.csv          one row per human-validated episode
        data/episodes/episode_turns.parquet every turn of each parent issue/PR, in order, with in_episode
Only human-confirmed episodes enter. If not every detector positive was confirmed, the result is a
validated SAMPLE, not a census (episodes.csv says which in `source`, summary.json in `coverage`).
"""

import argparse
import json

import pandas as pd

from common import ANNOTATION, DETECT, REPO, ROOT, load_conversations, load_turns, render_thread
from sample_pilot import write_sheet

COLUMNS = ["episode_id", "repository", "issue_pr_id", "comment_ids", "resource_concern", "competing_objective",
           "tradeoff_statement", "tradeoff_framing", "explicit_implicit", "multi_developer_exchange",
           "human_validated", "evidence_spans", "speed_role", "source", "url"]


def confirm_sheet(args):
    out = ANNOTATION / f"confirm_{args.rules}"
    if (out / "confirm_sheet.xlsx").exists() and not args.force:
        raise SystemExit(f"{out / 'confirm_sheet.xlsx'} exists (it may hold confirmations); use --force to rebuild.")
    items = pd.read_csv(DETECT / f"rules_{args.rules}" / args.repo / "items.csv")
    pilot = pd.read_csv(ANNOTATION / f"pilot_{args.rules}" / "sample.csv")
    pool = items[~items["item_number"].isin(pilot["item_number"])].copy()
    pool["rules_high"] = pool["group"].eq("HIGH_PRIORITY")
    pool["rules_any"] = pool["group"].isin(["HIGH_PRIORITY", "POSSIBLE_TRADEOFF"])
    if args.llm:
        pred = pd.read_parquet(ROOT / args.llm / "predictions.parquet")[["repo", "item_number", "tradeoff_present"]]
        pool = pool.merge(pred, on=["repo", "item_number"], how="left")
        pool["llm_yes"] = pool["tradeoff_present"].eq("yes")
        pool["hybrid"] = pool["rules_any"] & pool["llm_yes"]
    if args.detector not in pool:
        raise SystemExit(f"detector {args.detector} needs --llm (a run on --items pool)")
    chosen = pool[pool[args.detector]].sort_values("item_number")

    (out / "threads").mkdir(parents=True, exist_ok=True)
    conv = load_conversations(args.repo).set_index("item_number")
    turns = load_turns(args.repo)
    for number, g in turns[turns["item_number"].isin(chosen["item_number"])].groupby("item_number"):
        row = conv.loc[number].copy()
        row["item_number"] = number
        (out / f"threads/{args.repo}__{number}.md").write_text(render_thread(row, g) + "\n")
    rows = [{"order": i + 1, "item": r.item_number, "type": r.item_type, "title": r.title,
             "thread_file": f"threads/{args.repo}__{r.item_number}.md", "github": r.url}
            for i, r in enumerate(chosen.itertuples())]
    write_sheet(out / "confirm_sheet.xlsx", rows)
    (out / "detector.json").write_text(json.dumps({"detector": args.detector, "rules": args.rules, "llm": args.llm,
                                                   "pool": len(pool), "positives": len(chosen)}, indent=2))
    print(f"{len(chosen)} of {len(pool)} pool items flagged by {args.detector} -> {(out / 'confirm_sheet.xlsx').relative_to(ROOT)}")


def read(path, sheet):
    return pd.read_excel(path, sheet_name=sheet, dtype=str).fillna("").apply(lambda s: s.str.strip())


def episodes_from(labels, episodes, source, repo):
    """labels: item + label (+ Q4/evidence/multi_developer); episodes: the episodes sheet. TRADEOFF items only."""
    tradeoff = labels[labels["label"] == "TRADEOFF"].copy()
    for col in ("Q4_supporting_ids", "Q5_explicit_implicit", "multi_developer", "evidence"):
        if col not in tradeoff:  # gold.csv has no evidence column
            tradeoff[col] = ""
    episodes = episodes[episodes["item"].isin(tradeoff["item"])].copy()
    missing = tradeoff[~tradeoff["item"].isin(episodes["item"])]
    if len(missing):  # a TRADEOFF label without an episode row: one episode from the label row
        print(f"warning ({source}): {len(missing)} TRADEOFF items have no episode row; using their label row")
        episodes = pd.concat([episodes, pd.DataFrame({
            "item": missing["item"], "episode_no": "1", "comment_ids": missing["Q4_supporting_ids"],
            "explicit_implicit": missing["Q5_explicit_implicit"], "multi_developer_exchange": missing["multi_developer"],
            "evidence_spans": missing["evidence"]})], ignore_index=True).fillna("")
    episodes["episode_no"] = episodes["episode_no"].replace("", "1")
    out = pd.DataFrame({
        "episode_id": repo + "#" + episodes["item"] + "-" + episodes["episode_no"],
        "repository": repo.replace("__", "/"), "issue_pr_id": episodes["item"].astype(int),
        **{c: episodes.get(c, "") for c in ["comment_ids", "resource_concern", "competing_objective",
                                            "tradeoff_statement", "tradeoff_framing", "explicit_implicit",
                                            "multi_developer_exchange", "evidence_spans"]},
        "human_validated": True, "source": source})
    out["speed_role"] = out["resource_concern"].eq("speed").map({True: "R", False: ""}) + \
        out["competing_objective"].eq("speed").map({True: "G", False: ""})
    return out


def build(args):
    parts, coverage = [], {}
    pilot = ROOT / args.pilot if args.pilot else ANNOTATION / f"pilot_{args.rules}"
    if (pilot / "gold.csv").exists():
        gold = pd.read_csv(pilot / "gold.csv", dtype=str).fillna("").rename(columns={"item_number": "item"})
        parts.append(episodes_from(gold, pd.read_csv(pilot / "gold_episodes.csv", dtype=str).fillna(""),
                                   "pilot", args.repo))
        coverage["pilot"] = "all 200 sampled items adjudicated"
    else:
        print("no pilot gold.csv yet (run agreement.py --finalize)")
    sheet = ANNOTATION / f"confirm_{args.rules}" / "confirm_sheet.xlsx"
    if sheet.exists():
        labels = read(sheet, "labels")
        done = labels["label"] != ""
        parts.append(episodes_from(labels[done], read(sheet, "episodes"), "confirmed", args.repo))
        coverage["confirmed"] = f"{int(done.sum())} of {len(labels)} detector positives confirmed"
        if not done.all():
            coverage["note"] = "not every detector positive was confirmed: a validated sample, not a census"
    if not parts:
        raise SystemExit("nothing to extract yet")
    episodes = pd.concat(parts, ignore_index=True)
    conv = load_conversations(args.repo).set_index("item_number")
    episodes["url"] = episodes["issue_pr_id"].map(conv["url"])
    episodes["multi_developer_exchange"] = episodes["multi_developer_exchange"].eq("yes")

    turns = load_turns(args.repo)
    turns = turns[turns["item_number"].isin(episodes["issue_pr_id"])].copy()
    ids = episodes.groupby("issue_pr_id")["comment_ids"].apply(lambda s: {u.strip() for x in s for u in x.split(";") if u.strip()})
    turns["in_episode"] = [u in ids.get(n, set()) or (u == "title" and "body" in ids.get(n, set()))
                           for n, u in zip(turns["item_number"], turns["unit_id"])]
    turns["episode_ids"] = turns["item_number"].map(episodes.groupby("issue_pr_id")["episode_id"].apply(";".join))
    keep = ["repo", "item_number", "episode_ids", "seq", "unit_id", "kind", "author", "author_association",
            "created_at", "review_state", "thread_root", "reply_to", "reply_basis", "file_path", "line", "url",
            "in_episode", "raw"]

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    episodes[COLUMNS].to_csv(out / "episodes.csv", index=False)
    turns[keep].to_parquet(out / "episode_turns.parquet", index=False)
    summary = {"episodes": len(episodes), "items": int(episodes["issue_pr_id"].nunique()),
               "by_source": episodes["source"].value_counts().to_dict(),
               "multi_developer": int(episodes["multi_developer_exchange"].sum()), "coverage": coverage}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Output written to {out.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Validated trade-off episodes (Step 7).")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--rules", default="v1", help="the frozen sampling frame version")
    parser.add_argument("--confirm-sheet", action="store_true")
    parser.add_argument("--detector", default="hybrid", choices=["rules_high", "rules_any", "llm_yes", "hybrid"])
    parser.add_argument("--llm", default="", help="LLM run folder on --items pool")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pilot", default="", help="pilot annotation folder (default annotation/pilot_<rules>)")
    parser.add_argument("--out", default="data/episodes")
    args = parser.parse_args()
    confirm_sheet(args) if args.confirm_sheet else build(args)


if __name__ == "__main__":
    main()
