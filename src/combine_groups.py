"""
One view of the rule groups over all repos of a dataset (after detect_rules.py has run for each repo).

    .venv/bin/python src/combine_groups.py --dataset full
    .venv/bin/python src/combine_groups.py --dataset full --bot-candidates

Input : <detect>/rules_<version>/<repo>/items.csv for every repo, <threads>/<repo>/turns.parquet
Output: <detect>/rules_<version>/all_repos/items.csv     the repos' items.csv stacked (column `repo`)
        <detect>/rules_<version>/all_repos/summary.json  groups per repo and in total, by item type,
                                                         speed-only items, bot-only items
--bot-candidates: data/full/bot_candidates.csv, accounts with >= 50 non-empty comments that GitHub does not mark
        as bots (the sibling's rule for a manual look). `distinct_share` = share of distinct comment openings: a bot
        that posts the same message again and again has a low value. Confirmed bots go into KNOWN_BOTS (common.py).
"""

import argparse
import json

import pandas as pd

from common import DATASETS, FULL, FULL_REPOS, REPO, STRATA, detect_dir, is_bot, load_turns, source_tables

MIN_COMMENTS = 50


def bot_candidates(dataset, repos):
    rows = []
    for repo in repos:
        comments = pd.read_csv(source_tables(repo, dataset) / "comments.csv",
                               usecols=["author", "author_is_bot", "body"])
        human = comments[~is_bot(comments)]
        human = human[human["body"].fillna("").str.strip() != ""]  # empty review bodies (plain approvals)
        for author, g in human.groupby("author"):
            if len(g) < MIN_COMMENTS:
                continue
            openings = g["body"].fillna("").str.strip().str[:80]
            rows.append({"repo": repo, "author": author, "n_comments": len(g),
                         "distinct_share": round(openings.nunique() / len(g), 3),
                         "most_common_opening": openings.value_counts().index[0]})
    out = pd.DataFrame(rows).sort_values(["distinct_share", "n_comments"], ascending=[True, False])
    path = FULL / "bot_candidates.csv"
    out.to_csv(path, index=False)
    print(out.head(40).to_string(index=False))
    print(f"\n{len(out)} accounts written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Rule groups over all repos of a dataset.")
    parser.add_argument("--dataset", default="full", choices=DATASETS)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--bot-candidates", action="store_true")
    args = parser.parse_args()
    repos = FULL_REPOS if args.dataset == "full" else [REPO]
    if args.bot_candidates:
        return bot_candidates(args.dataset, repos)

    base = detect_dir(args.dataset) / f"rules_{args.version}"
    parts = []
    for repo in repos:
        items = pd.read_csv(base / repo / "items.csv", low_memory=False)
        human = set(load_turns(repo, args.dataset)["item_number"])
        items["bot_only"] = ~items["item_number"].isin(human)  # no human text at all: counted in NO_MATCH
        parts.append(items)
    items = pd.concat(parts, ignore_index=True)

    groups = pd.crosstab(items["repo"], items["group"]).reindex(index=repos, columns=STRATA, fill_value=0)
    groups["total"] = groups.sum(axis=1)
    groups.loc["all_repos"] = groups.sum()
    by_type = items.groupby(["repo", "group", "item_type"]).size().unstack(fill_value=0)
    summary = {
        "dataset": args.dataset, "version": args.version, "repos": repos, "items": len(items),
        "groups": {r: {g: int(n) for g, n in row.items()} for r, row in groups.iterrows()},
        "groups_by_type": {r: {g: by_type.loc[(r, g)].to_dict() for g in STRATA if (r, g) in by_type.index}
                           for r in repos},
        "speed_only_resource_items": {r: int(((items["repo"] == r) & (items["group"] == "RESOURCE_ONLY")
                                              & items["speed_only"].fillna(False).astype(bool)).sum()) for r in repos},
        "bot_only_items": {r: int(items.loc[items["repo"] == r, "bot_only"].sum()) for r in repos},
    }
    out = base / "all_repos"
    out.mkdir(parents=True, exist_ok=True)
    items.to_csv(out / "items.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(groups.to_string())
    print("\nbot-only items (in NO_MATCH):", summary["bot_only_items"])
    print("RESOURCE_ONLY with speed as the only concern:", summary["speed_only_resource_items"])
    print(f"\nOutput written to {out}")


if __name__ == "__main__":
    main()
