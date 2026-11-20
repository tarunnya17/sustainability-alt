"""
Export one rule group's conversations for screening by hand in the ChatGPT app, in items.csv order.

    .venv/bin/python src/export_chatgpt.py                              # HIGH_PRIORITY of rules_v1
    .venv/bin/python src/export_chatgpt.py --group POSSIBLE_TRADEOFF

Output: data/chatgpt/<repo>/<group>_<rules>/
    threads/<order>_<item>.md   the whole conversation (opening post + all human comments and review comments),
                                rendered exactly as for the raters and the local LLM (render_thread)
    prompt.txt                  the local LLM's instructions, plus a check that ChatGPT read to the end
    index.csv                   order, item, type, title, url, file, size, first and last turn id
    results.xlsx                one row per item: paste ChatGPT's JSON answer there (never overwritten)

How to use: one new chat per item → paste prompt.txt → paste the thread file right after it (or attach the
file when it is too long to paste) → send → paste the answer into results.xlsx.
"""

import argparse

import pandas as pd

from common import DETECT, REPO, ROOT, load_conversations, load_turns, render_thread
from llm_screen import INSTRUCTIONS, SYSTEM

# The local LLM's prompt, with two additions for the app: use only the given text (the app can browse),
# and name the last turn id, so a partly read thread shows up (last_turn_id differs from index.csv)
PROMPT = (SYSTEM + "\n\n"
          + INSTRUCTIONS.replace(
              "- explanation: one or two sentences\n",
              "- explanation: one or two sentences\n"
              "- last_turn_id: the id of the last turn in the discussion (to confirm you read all of it)\n\n"
              "Use only the discussion text given here. Do not open the GitHub link or use other sources. "
              "Return only the JSON object.\n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--rules", default="v1")
    ap.add_argument("--group", default="HIGH_PRIORITY")
    args = ap.parse_args()

    items = pd.read_csv(DETECT / f"rules_{args.rules}" / args.repo / "items.csv")
    items = items[items["group"] == args.group].reset_index(drop=True)  # keep items.csv order
    turns = load_turns(args.repo).sort_values(["item_number", "seq"])
    conv = load_conversations(args.repo).set_index("item_number")

    out = ROOT / "data" / "chatgpt" / args.repo / f"{args.group.lower()}_{args.rules}"
    (out / "threads").mkdir(parents=True, exist_ok=True)
    (out / "prompt.txt").write_text(PROMPT)

    width = len(str(len(items)))
    rows = []
    for order, r in enumerate(items.itertuples(), start=1):
        g = turns[turns["item_number"] == r.item_number]
        if g.empty:
            print(f"#{r.item_number}: no human text, skipped")
            continue
        conv_row = conv.loc[r.item_number].copy()
        conv_row["item_number"] = r.item_number
        text = render_thread(conv_row, g) + "\n"
        name = f"threads/{order:0{width}d}_{r.item_number}.md"
        (out / name).write_text(text)
        ids = [u for u in g["unit_id"] if u != "title"]
        rows.append({"order": order, "item_number": r.item_number, "item_type": r.item_type, "title": r.title,
                     "url": r.url, "thread_file": name, "n_turns": len(ids), "n_chars": len(text),
                     "approx_tokens": len(text) // 4, "first_turn_id": ids[0], "last_turn_id": ids[-1]})

    index = pd.DataFrame(rows)
    index.to_csv(out / "index.csv", index=False)

    results = out / "results.xlsx"
    if results.exists():
        print(f"{results} exists; left as it is (it may hold your answers)")
    else:
        sheet = index[["order", "item_number", "url", "thread_file", "approx_tokens", "last_turn_id"]].assign(
            chatgpt_model="", date="", pasted_or_attached="", chatgpt_json="", notes="")
        sheet.to_excel(results, index=False)

    print(f"{len(index)} {args.group} items → {out.relative_to(ROOT)}")
    print(f"size (approx tokens): median {index['approx_tokens'].median():.0f}, max {index['approx_tokens'].max()}")


if __name__ == "__main__":
    main()
