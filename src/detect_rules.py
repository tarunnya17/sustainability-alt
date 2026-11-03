"""
Steps 2-4: rule-based trade-off detection. A conversation is a candidate when a resource concern (R) and a
competing objective (G) from DIFFERENT groups are linked (T) in the same sentence, the same comment, or
related comments.

    .venv/bin/python src/detect_rules.py                 # writes data/detect/rules_v1/
    .venv/bin/python src/detect_rules.py --version v2    # after editing lexicon.py

Input : data/threads/<repo>/turns.parquet (Step 1)
Output: data/detect/rules_<version>/<repo>/
          hits.parquet   one row per keyword hit (unit, sentence, group, term, matched text, active?)
          links.parquet  one row per R-G link (kind, explicit?, groups, speed role, units, authors, evidence)
          items.csv      one row per issue/PR: group, framing, link counts, score, top evidence
          summary.json   counts per group, term and link kind; cross-tab with the sibling's tiers

Rules (see CLAUDE.md):
- a strong hit always counts; a weak hit counts only in a sentence that also has an EFFECT word
- a hit inside a longer hit is dropped ("overhead" inside "memory overhead")
- speed (SPEED) is R next to a non-speed G, and G next to a non-speed R
- link kinds: sentence (same sentence), comment (same comment; title + body = the opening post),
  thread (same inline review thread, or one replies to the other), adjacent (within the next 2 comments)
- item groups: HIGH_PRIORITY  an explicit link: a sentence with R, G and a T_marker/T_contrast/T_alternative
                              word, or a comment with R, G and a T_marker
               POSSIBLE_TRADEOFF  some other link
               RESOURCE_ONLY  an R or speed hit counts somewhere, but there is no link
               NO_MATCH       nothing
"""

import argparse
import json
from collections import Counter
from itertools import product

import pandas as pd

from common import DETECT, REPO, STRATA, SIBLING, load_conversations, load_turns
from lexicon import EFFECT, GROUPS, compiled_terms

TERMS = compiled_terms()
KIND_RANK = {"sentence": 0, "comment": 1, "thread": 2, "adjacent": 3}
ADJACENT = 2  # a comment is linked to the next 2 comments


def sentence_hits(sentence):
    """Hits in one sentence, each with `active` (does it count?). Nested and weak-in-strong hits are dropped."""
    found = []
    for group, role, strength, term, regex in TERMS:
        for m in regex.finditer(sentence):
            found.append({"group": group, "role": role, "strength": strength, "term": term,
                          "start": m.start(), "end": m.end(), "matched_text": m.group(0)})
    kept = []
    for h in found:
        inside_longer = any(o is not h and o["start"] <= h["start"] and h["end"] <= o["end"]
                            and o["end"] - o["start"] > h["end"] - h["start"] for o in found)
        weak_in_strong = h["strength"] == "weak" and any(
            o["strength"] == "strong" and o["start"] < h["end"] and h["start"] < o["end"] for o in found)
        if not (inside_longer or weak_in_strong):
            kept.append(h)
    effects = [(m.start(), m.end()) for m in EFFECT.finditer(sentence)]
    for h in kept:
        h["active"] = h["strength"] == "strong" or any(not (h["start"] <= s and e <= h["end"]) for s, e in effects)
    return kept


def sides(hits):
    """Groups available as R and as G among active hits (speed is on both sides)."""
    active = {h["group"] for h in hits if h["active"]}
    r = {g for g in active if GROUPS[g]["role"] in ("R", "RG")}
    g = {g for g in active if GROUPS[g]["role"] in ("G", "RG")}
    return r, g


def pairs(r_side, g_side):
    """All (r_group, g_group, speed_role) with r != g; speed cannot pair with itself."""
    return [(r, g, "R" if r == "SPEED" else "G" if g == "SPEED" else "-")
            for r, g in product(sorted(r_side), sorted(g_side)) if r != g]


def first_sentence(unit, group, side):
    for s in unit["sentences"]:
        if group in s[side]:
            return s["text"]
    return ""


def detect_item(turns):
    """turns: one item's turns in seq order. Returns (hit_rows, link_rows, item_info)."""
    # ---- units: the title joins the opening post ----
    units, order = {}, []
    for t in turns.itertuples(index=False):
        key = "body" if t.kind == "title" else t.unit_id
        if key not in units:
            units[key] = {"unit_id": key, "author": t.author, "thread_root": t.thread_root if key != "body" else "body",
                          "reply_to": "body" if t.reply_to == "title" else t.reply_to, "sentences": [], "has_text": False}
            order.append(key)
        u = units[key]
        for s in t.sentences:
            u["has_text"] = True
            u["sentences"].append({"text": s, "hits": sentence_hits(s)})

    hit_rows, links = [], []
    for key in order:
        u = units[key]
        for i, s in enumerate(u["sentences"]):
            s["R"], s["G"] = sides(s["hits"])
            s["marker"] = next((h["term"] for h in s["hits"] if h["group"] == "T_marker"), None)
            s["contrast"] = next((h["matched_text"] for h in s["hits"] if h["group"] in ("T_contrast", "T_alternative")), None)
            for h in s["hits"]:
                hit_rows.append({"unit_id": key, "sentence_idx": i, **{k: h[k] for k in
                                 ("group", "role", "strength", "term", "matched_text", "active")}})
        all_hits = [h for s in u["sentences"] for h in s["hits"]]
        u["R"], u["G"] = sides(all_hits)
        u["marker"] = next((s["marker"] for s in u["sentences"] if s["marker"]), None)
        u["responses"] = sum(h["group"] == "T_response" for h in all_hits)

        # ---- within one sentence, then within one comment ----
        seen = set()
        for s in u["sentences"]:
            for r, g, role in pairs(s["R"], s["G"]):
                seen.add((r, g))
                marker = s["marker"] or s["contrast"]
                links.append({"kind": "sentence", "explicit": bool(marker), "marker": marker, "r_group": r,
                              "g_group": g, "speed_role": role, "unit_ids": key, "authors": u["author"],
                              "multi_developer": False, "r_evidence": s["text"], "g_evidence": s["text"]})
        for r, g, role in pairs(u["R"], u["G"]):
            if (r, g) in seen:
                continue
            links.append({"kind": "comment", "explicit": bool(u["marker"]), "marker": u["marker"], "r_group": r,
                          "g_group": g, "speed_role": role, "unit_ids": key, "authors": u["author"],
                          "multi_developer": False, "r_evidence": first_sentence(u, r, "R"),
                          "g_evidence": first_sentence(u, g, "G")})

    # ---- across related comments ----
    text_units = [k for k in order if units[k]["has_text"]]
    for i, a_key in enumerate(text_units):
        a = units[a_key]
        for j in range(i + 1, len(text_units)):
            b_key = text_units[j]
            b = units[b_key]
            same_thread = a["thread_root"] == b["thread_root"] and a_key != "body"
            replies = b["reply_to"] == a_key or a["reply_to"] == b_key
            near = j - i <= ADJACENT
            if not (same_thread or replies or near):
                continue
            kind = "thread" if same_thread or replies else "adjacent"
            for x, y in ((a, b), (b, a)):  # R in one comment, G in the other
                for r, g, role in pairs(x["R"], y["G"]):
                    links.append({"kind": kind, "explicit": False, "marker": None, "r_group": r, "g_group": g,
                                  "speed_role": role, "unit_ids": f"{a_key};{b_key}",
                                  "authors": f"{a['author']};{b['author']}",
                                  "multi_developer": a["author"] != b["author"],
                                  "r_evidence": first_sentence(x, r, "R"), "g_evidence": first_sentence(y, g, "G")})

    # ---- item group ----
    active = [h for h in hit_rows if h["active"]]
    r_groups = sorted({h["group"] for h in active if h["role"] in ("R", "RG")})
    g_groups = sorted({h["group"] for h in active if h["role"] == "G"})
    explicit = [l for l in links if l["explicit"]]
    if explicit:
        group = "HIGH_PRIORITY"
    elif links:
        group = "POSSIBLE_TRADEOFF"
    elif r_groups:
        group = "RESOURCE_ONLY"
    else:
        group = "NO_MATCH"
    linked_units = {k for l in links for k in l["unit_ids"].split(";")}
    responses = sum(units[k]["responses"] for k in linked_units)
    multi = any(l["multi_developer"] for l in links)
    score = (3 * len(explicit) + 2 * sum(l["kind"] == "sentence" and not l["explicit"] for l in links)
             + sum(l["kind"] != "sentence" and not l["explicit"] for l in links) + 2 * multi + min(responses, 5))
    best = min(links, key=lambda l: (not l["explicit"], KIND_RANK[l["kind"]]), default=None)
    top = ""
    if best:
        top = best["r_evidence"] if best["kind"] == "sentence" else f"{best['r_evidence']} || {best['g_evidence']}"
    env_r = any(l["r_group"] == "R_env" for l in links) or (not links and "R_env" in r_groups)
    info = {
        "group": group, "framing": "direct_environmental" if env_r else ("resource_proxy" if r_groups else ""),
        "explicit_env": "R_env" in r_groups, "speed_only": r_groups == ["SPEED"],
        "r_groups": ";".join(r_groups), "g_groups": ";".join(g_groups),
        "n_links": len(links), "n_explicit": len(explicit),
        "link_kinds": ";".join(sorted({l["kind"] for l in links}, key=KIND_RANK.get)),
        "speed_roles": ";".join(sorted({l["speed_role"] for l in links} - {"-"})),
        "multi_developer": multi, "score": score, "top_evidence": top[:400],
    }
    return hit_rows, links, info


def main():
    parser = argparse.ArgumentParser(description="Rule-based trade-off detection (Steps 2-4).")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    turns = load_turns(args.repo)
    conv = load_conversations(args.repo)
    all_hits, all_links, rows = [], [], []
    for number, g in turns.groupby("item_number", sort=True):
        hits, links, info = detect_item(g)
        all_hits += [{"item_number": number, **h} for h in hits]
        all_links += [{"item_number": number, **l} for l in links]
        rows.append({"item_number": number, **info})

    items = conv[["repo", "item_number", "item_type", "title", "url"]].merge(pd.DataFrame(rows), on="item_number", how="left")
    items["group"] = items["group"].fillna("NO_MATCH")  # items with no human text at all
    sibling_scored = SIBLING / "data" / "filtered" / args.repo / "items_scored.csv"
    if sibling_scored.exists():
        tiers = pd.read_csv(sibling_scored, usecols=["item_number", "tier"])
        items = items.merge(tiers.rename(columns={"tier": "sibling_tier"}), on="item_number", how="left")
        items["sibling_tier"] = items["sibling_tier"].fillna("none")
    hits = pd.DataFrame(all_hits)
    links = pd.DataFrame(all_links)
    for df in (hits, links):
        df.insert(0, "repo", args.repo)

    out = DETECT / f"rules_{args.version}" / args.repo
    out.mkdir(parents=True, exist_ok=True)
    hits.to_parquet(out / "hits.parquet", index=False)
    links.to_parquet(out / "links.parquet", index=False)
    items.sort_values(["group", "score"], key=lambda s: s.map(STRATA.index) if s.name == "group" else -s) \
         .to_csv(out / "items.csv", index=False)

    active = hits[hits["active"]].drop_duplicates(["item_number", "group", "term"])
    summary = {
        "version": args.version, "items": len(items),
        "groups": {g: int((items["group"] == g).sum()) for g in STRATA},
        "groups_by_type": items.groupby(["group", "item_type"]).size().unstack(fill_value=0).to_dict("index"),
        "speed_only_resource_items": int(((items["group"] == "RESOURCE_ONLY") & items["speed_only"]).sum()),
        "explicit_env_items": int(items["explicit_env"].fillna(False).sum()),
        "links": len(links), "links_by_kind": links["kind"].value_counts().to_dict(),
        "explicit_links": int(links["explicit"].sum()),
        "speed_role_in_links": links["speed_role"].value_counts().to_dict(),
        "items_per_link_pair": {f"{r} x {g}": int(n) for (r, g), n in
                                links.drop_duplicates(["item_number", "r_group", "g_group"])
                                .groupby(["r_group", "g_group"]).size().sort_values(ascending=False).head(25).items()},
        "items_per_active_term": {f"{g}/{t}": int(n) for (g, t), n in
                                  active.groupby(["group", "term"]).size().sort_values(ascending=False).items()},
    }
    if "sibling_tier" in items:
        summary["crosstab_sibling_tier"] = pd.crosstab(items["group"], items["sibling_tier"]).to_dict("index")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"items: {len(items)}   links: {len(links)} ({summary['explicit_links']} explicit)")
    for g in STRATA:
        print(f"  {g:18} {summary['groups'][g]:>5}")
    print(f"  (RESOURCE_ONLY with speed as the only concern: {summary['speed_only_resource_items']})")
    if "sibling_tier" in items:
        print("\ncross-tab with the sibling's tiers:")
        print(pd.crosstab(items["group"], items["sibling_tier"]).reindex(STRATA).to_string())
    print(f"\nOutput written to {out}")


if __name__ == "__main__":
    main()
