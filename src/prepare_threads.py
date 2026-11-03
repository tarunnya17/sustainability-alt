"""
Step 1: rebuild every issue/PR as a conversation of "turns" (title, opening post, comments in time order).

    .venv/bin/python src/prepare_threads.py
    .venv/bin/python src/prepare_threads.py --repo huggingface__transformers

Input : the sibling project's tables (read-only, data/SOURCE.md)
Output: data/threads/<repo>/turns.parquet          one row per human title / body / comment / review
        data/threads/<repo>/conversations.parquet  one row per issue/PR (metadata, participants)
        data/threads/<repo>/summary.json

Reply structure is taken from the RAW text, because cleaning removes quotes and @mentions:
- thread_root: first comment of an inline review thread (follows parent_id); otherwise the turn itself
- reply_to:    the turn this one answers, with reply_basis =
               review_parent (GitHub's in_reply_to), quote (a "> line" found in an earlier turn),
               mention (@login -> that person's latest earlier turn) or none
"""

import argparse
import hashlib
import json
import re

import pandas as pd

from clean_text import clean, find_boilerplate, normalize
from common import SOURCE_MD5, THREADS, is_bot, sibling_tables, split_sentences

MENTION = re.compile(r"(?<![\w.])@([\w-]+)")
FENCED = re.compile(r"```.*?(```|$)|~~~.*?(~~~|$)", re.S)
MIN_QUOTE = 15  # a quoted line shorter than this ("> yes") is too vague to find its source


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def quoted_lines(raw):
    lines = [normalize(l.lstrip()[1:]).strip() for l in raw.split("\n") if l.lstrip().startswith(">")]
    return [l for l in lines if len(l) >= MIN_QUOTE]


def find_replies(turns):
    """Fill thread_root, reply_to and reply_basis for the turns of ONE item (already in seq order)."""
    ids = set(turns["unit_id"])
    parent = {u: f"c{p}" for u, p in zip(turns["unit_id"], turns["parent_id"]) if pd.notna(p)}
    roots, reply_to, basis = [], [], []
    seen = []  # (unit_id, author, normalized raw) of earlier turns
    for t in turns.itertuples(index=False):
        root = t.unit_id
        while parent.get(root) in ids:
            root = parent[root]
        roots.append(root)

        target, why = None, "none"
        if t.kind in ("title", "body"):
            pass
        elif parent.get(t.unit_id) in ids:
            target, why = parent[t.unit_id], "review_parent"
        else:
            raw = t.raw if isinstance(t.raw, str) else ""
            for q in quoted_lines(raw):
                match = next((u for u, _, text in reversed(seen) if q in text), None)
                if match:
                    target, why = match, "quote"
                    break
            if target is None:
                logins = {m for m in MENTION.findall(FENCED.sub(" ", raw)) if m != t.author}
                latest = [next((u for u, a, _ in reversed(seen) if a == login), None) for login in logins]
                latest = [u for u in latest if u]
                if latest:  # several mentions: answer the most recent of those turns
                    order = {u: i for i, (u, _, _) in enumerate(seen)}
                    target, why = max(latest, key=order.get), "mention"
        reply_to.append(target)
        basis.append(why)
        seen.append((t.unit_id, t.author, normalize(t.raw) if isinstance(t.raw, str) else ""))
    return roots, reply_to, basis


def main():
    parser = argparse.ArgumentParser(description="Rebuild conversations (Step 1).")
    parser.add_argument("--repo", default="huggingface__transformers")
    parser.add_argument("--boilerplate-min", type=int, default=20,
                        help="a line in this many different texts is boilerplate (sibling's setting)")
    args = parser.parse_args()

    src = sibling_tables(args.repo)
    for name, expected in SOURCE_MD5.get(args.repo, {}).items():
        if md5(src / name) != expected:
            raise SystemExit(f"{src / name} changed since data/SOURCE.md was written; check it, then update the MD5.")
    items = pd.read_csv(src / "items.csv")
    comments = pd.read_csv(src / "comments.csv", dtype={"parent_id": "Int64", "review_id": "Int64", "line": "Int64"})

    # ---- turns: human title/body + human comments, in time order within each item ----
    human_items = items[~is_bot(items)]
    human_comments = comments[~is_bot(comments)]
    opening = []
    for kind, seq in (("title", 0), ("body", 1)):
        opening.append(pd.DataFrame({
            "item_number": human_items["item_number"], "unit_id": kind, "seq": seq, "kind": kind,
            "author": human_items["author"], "author_association": human_items["author_association"],
            "created_at": human_items["created_at"], "url": human_items["url"], "raw": human_items[kind]}))
    c = human_comments.sort_values(["item_number", "created_at", "comment_id"])
    c = pd.DataFrame({
        "item_number": c["item_number"], "unit_id": "c" + c["comment_id"].astype(str),
        "seq": c.groupby("item_number").cumcount() + 2, "kind": c["comment_kind"],
        "author": c["author"], "author_association": c["author_association"], "created_at": c["created_at"],
        "url": c["url"], "raw": c["body"], "comment_id": c["comment_id"], "review_id": c["review_id"],
        "review_state": c["review_state"], "parent_id": c["parent_id"], "file_path": c["file_path"], "line": c["line"]})
    turns = pd.concat(opening + [c], ignore_index=True).sort_values(["item_number", "seq"], ignore_index=True)
    turns.insert(0, "repo", args.repo)

    # ---- cleaning: exactly the sibling's (same boilerplate rule over the same texts) ----
    boilerplate = find_boilerplate(turns["raw"], args.boilerplate_min)
    turns["text"] = [clean(t, boilerplate) for t in turns["raw"]]
    turns["sentences"] = [split_sentences(t) for t in turns["text"]]

    # ---- reply structure ----
    parts = []
    for _, g in turns.groupby("item_number", sort=False):
        roots, reply_to, basis = find_replies(g)
        parts.append(g.assign(thread_root=roots, reply_to=reply_to, reply_basis=basis))
    turns = pd.concat(parts, ignore_index=True)

    # ---- conversations ----
    comment_turns = turns[~turns["kind"].isin(["title", "body"])]
    authors = turns.groupby("item_number")["author"].nunique()
    keep = ["item_number", "item_type", "title", "url", "author", "author_association", "author_is_bot", "state",
            "state_reason", "merged", "review_decision", "labels", "comments_count", "created_at", "closed_at",
            "closed_by", "linked_issue_numbers", "linked_pr_numbers"]
    conv = items[keep].copy()
    conv.insert(0, "repo", args.repo)
    conv["n_turns"] = conv["item_number"].map(comment_turns.groupby("item_number").size()).fillna(0).astype(int)
    conv["n_human_authors"] = conv["item_number"].map(authors).fillna(0).astype(int)

    out = THREADS / args.repo
    out.mkdir(parents=True, exist_ok=True)
    turns.to_parquet(out / "turns.parquet", index=False)
    conv.to_parquet(out / "conversations.parquet", index=False)
    summary = {
        "items": len(items), "turns": len(turns), "comment_turns": len(comment_turns),
        "bot_items_skipped": len(items) - len(human_items), "bot_comments_skipped": len(comments) - len(human_comments),
        "boilerplate_lines": len(boilerplate),
        "reply_basis": comment_turns["reply_basis"].value_counts().to_dict(),
        "turns_in_review_threads": int((comment_turns["thread_root"] != comment_turns["unit_id"]).sum()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Output written to {out}")


if __name__ == "__main__":
    main()
