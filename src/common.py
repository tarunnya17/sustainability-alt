"""
Shared paths, constants and helpers for every script in src/.

- the sibling project's tables are the (read-only) input, see data/SOURCE.md
- `render_thread` is the ONE way a conversation is shown, to raters and to the LLM
- `pick_gpu` is copied from ../each-bug-class-predictor-merge/src/common.py
"""

import os
import re
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
REPO = "huggingface__transformers"  # <owner>__<repo>, as in the sibling's folder names

SIBLING = Path("/student/scb052/Projects/sustainability-developer-discussion")
SOURCE_MD5 = {  # data/SOURCE.md; prepare_threads.py stops if a table changed
    "huggingface__transformers": {"items.csv": "9d89b15a7c04325b43770a6dfc6367f8",
                                  "comments.csv": "545ea278b7b2143c55077ed2fd6b26fa"},
}

THREADS = ROOT / "data" / "threads"
DETECT = ROOT / "data" / "detect"
ANNOTATION = ROOT / "annotation"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"

STRATA = ["HIGH_PRIORITY", "POSSIBLE_TRADEOFF", "RESOURCE_ONLY", "NO_MATCH"]
LABELS = ["TRADEOFF", "RESOURCE_ONLY", "UNRELATED", "UNCERTAIN"]

# Bot accounts that GitHub does not mark as bots (from the sibling's filter_discussions.py)
KNOWN_BOTS = {"HuggingFaceDocBuilderDev"}


def is_bot(df):
    return df["author_is_bot"].astype(str).eq("True") | df["author"].isin(KNOWN_BOTS)


def sibling_tables(repo=REPO):
    return SIBLING / "data" / "tables" / repo


def load_turns(repo=REPO):
    return pd.read_parquet(THREADS / repo / "turns.parquet")


def load_conversations(repo=REPO):
    return pd.read_parquet(THREADS / repo / "conversations.parquet")


# ---------- sentences ----------

SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text):
    """Simple splitter on cleaned text: after . ! ? followed by space, and at line breaks."""
    return [s.strip() for s in SENTENCE_END.split(text or "") if s and s.strip()]


# ---------- rendering (raters and LLM see the same text) ----------

HTML_COMMENT = re.compile(r"<!--.*?(-->|$)", re.S)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img[^>]*>", re.I)
FENCE = re.compile(r"^(\s*)(```|~~~).*?^\s*\2[ \t]*$|^(\s*)(```|~~~).*\Z", re.S | re.M)
DETAILS = re.compile(r"<details>(.*?)(</details>|\Z)", re.S | re.I)
SUMMARY = re.compile(r"<summary>(.*?)</summary>", re.S | re.I)
BLANKS = re.compile(r"\n{3,}")


def _collapse_code(match, keep=6, max_lines=15):
    lines = match.group(0).split("\n")
    if len(lines) <= max_lines:
        return match.group(0)
    fence = lines[0].strip()[:3]
    return "\n".join(lines[:keep] + [f"[… {len(lines) - keep - 1} lines of code omitted]", fence])


def _collapse_details(match):
    summary = SUMMARY.search(match.group(1))
    title = re.sub(r"\s+", " ", summary.group(1)).strip() if summary else "details"
    n = match.group(1).count("\n")
    return f"[collapsed block '{title}', {n} lines omitted]"


def display_text(raw):
    """Raw Markdown made readable: template comments and images removed, long code and <details> collapsed."""
    if not isinstance(raw, str) or not raw.strip():
        return ""
    text = HTML_COMMENT.sub("", raw)
    text = DETAILS.sub(_collapse_details, text)
    text = FENCE.sub(_collapse_code, text)
    text = IMAGE.sub("[image]", text)
    return BLANKS.sub("\n\n", text).strip()


def turn_header(t):
    """One line naming a turn: its unit id (what raters and the LLM cite), author and kind."""
    when = str(t["created_at"])[:10]
    who = f"{t['author']} ({t['author_association']})"
    if t["kind"] in ("title", "body"):
        return f"[{t['unit_id']}] {who} · opening post · {when}"
    extra = ""
    if t["kind"] == "review":
        extra = f" {t['review_state']}"
    elif t["kind"] == "review_comment":
        extra = f" on {t['file_path']}" + (f":{int(t['line'])}" if pd.notna(t["line"]) else "")
    if isinstance(t["reply_to"], str):
        extra += f" · reply to [{t['reply_to']}]"
    return f"[{t['unit_id']}] {who} · {t['kind']}{extra} · {when}"


def render_thread(conv, turns, keep_units=None):
    """Markdown of one conversation. `conv`: a conversations row; `turns`: its turns in seq order.
    With `keep_units`, other comments are replaced by '[… k comments omitted …]' (the opening post is always kept)."""
    out = [f"# #{conv['item_number']} [{conv['item_type']}] {conv['title']}",
           f"{conv['url']}  ·  opened by {conv['author']} on {str(conv['created_at'])[:10]}", ""]
    skipped = 0
    for _, t in turns.iterrows():
        if t["kind"] == "title":
            continue
        if keep_units is not None and t["kind"] != "body" and t["unit_id"] not in keep_units:
            skipped += 1
            continue
        if skipped:
            out += [f"[… {skipped} comments omitted …]", ""]
            skipped = 0
        body = display_text(t["raw"])
        if t["kind"] == "review" and not body:
            out += [f"## {turn_header(t)} (no text)", ""]
            continue
        out += [f"## {turn_header(t)}", body or "(empty)", ""]
    if skipped:
        out += [f"[… {skipped} comments omitted …]", ""]
    outcome = f"state={conv['state']}"
    if conv["item_type"] == "pull_request":
        outcome += f", merged={conv['merged']}"
    out.append(f"---\noutcome: {outcome}")
    return "\n".join(out)


# ---------- GPU (from ../each-bug-class-predictor-merge/src/common.py) ----------

def pick_gpu(min_free: int = 20000) -> None:
    """Shared lab machine: use ONE GPU, the one with the most free memory (then the lowest load),
    unless one is set already. Refuses if it has less than min_free MiB free."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    gpu, free, _ = min((tuple(map(int, line.split(","))) for line in out.strip().splitlines()),
                       key=lambda t: (-t[1], t[2]))
    if free < min_free:
        raise SystemExit(f"No GPU with {min_free} MiB free (best: GPU {gpu}, {free} MiB free).")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
