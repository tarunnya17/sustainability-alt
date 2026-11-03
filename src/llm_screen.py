"""
Step 5: screen conversations with a local open-weights LLM (vLLM, one A100). Runs in the separate LLM venv.

    PY=/u2/users/scb052/alt-sustainability/venv-llm/bin/python
    $PY src/llm_screen.py --dry-run                     # print the first prompt, no GPU
    $PY src/llm_screen.py --limit 5                     # smoke run
    setsid nohup $PY src/llm_screen.py > /dev/null 2>&1 &            # the 200 pilot items
    $PY src/llm_screen.py --items pool                  # all R candidates not in the pilot (Step 7)
    $PY src/llm_screen.py --model Qwen/Qwen3-32B-AWQ --min-free 30000   # when no GPU has ~75 GB free

Output: runs/<time>_llm_<model>/predictions.parquet   one row per item: the model's JSON answer + checks
        runs/<time>_llm_<model>/prompt.txt, config.json, log.txt

The prompt is the one from the methodology (criteria 1-3, "do not infer energy savings ..."), and it asks for
traceable evidence. Every evidence span is checked against the turns the model cites (evidence_verified =
share of spans found there), so invented quotes or relations show up.
Long conversations (over --budget tokens) keep the opening post, the turns with keyword hits (rules_v1) and
their neighbours and review-thread roots; the rest becomes "[... k comments omitted ...]".
"""

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import unicodedata
from datetime import datetime

import pandas as pd

from common import (ANNOTATION, DETECT, REPO, RUNS, SEED, display_text, load_conversations, load_turns, pick_gpu,
                    render_thread)

LLM_HOME = "/u2/users/scb052/alt-sustainability"  # HF cache lives here: /u1 is nearly full
os.environ.setdefault("HF_HOME", f"{LLM_HOME}/hf")
# FlashInfer's sampler is JIT-compiled at start-up and needs ninja + nvcc (no CUDA toolkit here); vLLM's own
# PyTorch sampler gives the same greedy output
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]  # venv tools, unactivated

SYSTEM = "You are classifying GitHub development discussions for an empirical software engineering study."

INSTRUCTIONS = """Determine whether the discussion below contains a sustainability-related or computational-resource-efficiency trade-off.

Include a trade-off only if:
1. An identifiable environmental or computational-resource concern exists (for example energy, power, emissions, compute, memory, hardware use, cost of compute, efficiency techniques such as quantization or offloading, or speed when it is weighed against a non-speed objective).
2. A second desirable development objective or alternative exists (for example accuracy or model quality, speed, maintainability or complexity, reliability or correctness, compatibility, functionality, development effort).
3. The discussion establishes that the objectives compete in the context of the same development decision.

Do not classify ordinary bug reports, unrelated mentions, or optimizations with no identified downside as trade-offs.
Do not infer energy or carbon savings from memory or runtime optimization alone.
Do not infer disadvantages that the participants never discussed.

Every turn of the discussion starts with its id in square brackets, for example [body] or [c3184836411]. Use these ids in supporting_comment_ids. Copy each evidence span word for word from the discussion.

Return a JSON object with:
- tradeoff_present: "yes", "no" or "uncertain"
- resource_concern: the environmental or computational-resource concern ("" if none)
- competing_objective: the competing objective or alternative ("" if none)
- tradeoff_relationship: how the two compete in the same decision ("" if none)
- supporting_comment_ids: the ids of the turns that show the trade-off
- exact_evidence_spans: short verbatim quotes that show it
- explicit_or_implicit: "explicit" (stated in one place), "implicit" (built up across turns) or "none"
- explanation: one or two sentences

Discussion:
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "tradeoff_present": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "resource_concern": {"type": "string"},
        "competing_objective": {"type": "string"},
        "tradeoff_relationship": {"type": "string"},
        "supporting_comment_ids": {"type": "array", "items": {"type": "string"}},
        "exact_evidence_spans": {"type": "array", "items": {"type": "string"}},
        "explicit_or_implicit": {"type": "string", "enum": ["explicit", "implicit", "none"]},
        "explanation": {"type": "string"},
    },
    "required": ["tradeoff_present", "resource_concern", "competing_objective", "tradeoff_relationship",
                 "supporting_comment_ids", "exact_evidence_spans", "explicit_or_implicit", "explanation"],
    "additionalProperties": False,
}

SPACE = re.compile(r"\s+")
ELLIPSIS = re.compile(r"\.\.\.|…|\[\.\.\.\]")
TAG = re.compile(r"<[^>]+>")
PUNCT = re.compile(r"[^\w\s]")


def norm(text):
    """Compare words, not formatting: HTML tags, Markdown and punctuation are ignored (the model often
    flattens lists or drops ** and backticks); the wording itself must match."""
    text = TAG.sub(" ", unicodedata.normalize("NFKC", text or "").lower())
    return SPACE.sub(" ", PUNCT.sub(" ", text)).strip()


def select_items(args):
    if args.items == "pilot":
        return pd.read_csv(ANNOTATION / f"pilot_{args.rules}" / "sample.csv")[["repo", "item_number"]]
    if args.items == "pool":
        items = pd.read_csv(DETECT / f"rules_{args.rules}" / args.repo / "items.csv")
        pilot = pd.read_csv(ANNOTATION / f"pilot_{args.rules}" / "sample.csv")
        pool = items[items["group"].isin(["HIGH_PRIORITY", "POSSIBLE_TRADEOFF", "RESOURCE_ONLY"])
                     & ~items["item_number"].isin(pilot["item_number"])]
        return pool[["repo", "item_number"]]
    return pd.read_csv(args.items)[["repo", "item_number"]]


def hit_units(repo, rules):
    """item_number -> set of unit ids with an active R or G hit (title counts as the opening post)."""
    hits = pd.read_parquet(DETECT / f"rules_{rules}" / repo / "hits.parquet",
                           columns=["item_number", "unit_id", "role", "active"])
    hits = hits[hits["active"] & hits["role"].isin(["R", "G", "RG"])]
    return hits.groupby("item_number")["unit_id"].apply(set).to_dict()


def trimmed_units(turns, hits):
    """Turns with hits, their neighbours and their review-thread roots."""
    ids = [u for u in turns["unit_id"] if u != "title"]
    keep = set()
    for i, u in enumerate(ids):
        if u in hits:
            keep.update(ids[max(0, i - 1):i + 2])
    roots = turns.set_index("unit_id")["thread_root"]
    keep |= {roots[u] for u in list(keep) if u in roots.index}
    return keep


def build_prompt(conv, turns, hits, tokenizer, budget):
    text = render_thread(conv, turns)
    n = len(tokenizer.encode(text))
    trim = "none"
    if n > budget:
        text = render_thread(conv, turns, keep_units=trimmed_units(turns, hits))
        n, trim = len(tokenizer.encode(text)), "selected"
        if n > budget:  # still too long: cut the end
            text = tokenizer.decode(tokenizer.encode(text)[:budget]) + "\n[… discussion truncated …]"
            n, trim = budget, "truncated"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": INSTRUCTIONS + text}], n, trim


def check_evidence(answer, turns):
    """Share of evidence spans found verbatim in the cited turns; ids that do not exist."""
    texts = {}
    for t in turns.itertuples(index=False):
        key = "body" if t.unit_id == "title" else t.unit_id
        texts[key] = texts.get(key, "") + " " + norm(display_text(t.raw) if t.unit_id != "title" else t.raw)
    cited = []
    for raw_id in answer.get("supporting_comment_ids", []):
        u = raw_id.strip().strip("[]#")
        u = "body" if u == "title" else (f"c{u}" if u.isdigit() else u)
        cited.append(u)
    unknown = [u for u in cited if u not in texts]
    scope = " ".join(texts[u] for u in cited if u in texts)
    everything = " ".join(texts.values())
    spans = [s for s in answer.get("exact_evidence_spans", []) if s.strip()]
    in_cited = in_any = 0
    for span in spans:
        pieces = [norm(p).strip(" \"'“”") for p in ELLIPSIS.split(span)]
        pieces = [p for p in pieces if len(p) >= 8] or [norm(span).strip(" \"'“”")]
        in_cited += all(p in scope for p in pieces)
        in_any += all(p in everything for p in pieces)
    share = (lambda k: round(k / len(spans), 3) if spans else None)
    return share(in_cited), share(in_any), ";".join(unknown)


def gpu_memory(index):
    out = subprocess.run(["nvidia-smi", "-i", str(index), "--query-gpu=memory.free,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    return tuple(map(int, out.strip().split(",")))


def main():
    parser = argparse.ArgumentParser(description="LLM screen of conversations (Step 5).")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--items", default="pilot", help="pilot | pool | a CSV with repo,item_number")
    parser.add_argument("--rules", default="v1", help="sampling frame / keyword hits used for trimming")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--min-free", type=int, default=75000, help="MiB a GPU needs free")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--budget", type=int, default=24000, help="max tokens of discussion text")
    # 1024 cut one of 200 pilot answers (#43112, many evidence spans) mid-JSON
    parser.add_argument("--max-tokens", type=int, default=2048, help="max tokens of the answer")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from huggingface_hub import HfApi

    revision = HfApi().model_info(args.model).sha
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    items = select_items(args)
    if args.limit:
        items = items.head(args.limit)
    conv = load_conversations(args.repo).set_index("item_number")
    turns = load_turns(args.repo)
    turns = turns[turns["item_number"].isin(items["item_number"])]
    hits = hit_units(args.repo, args.rules)

    prompts, meta = [], []
    for number, g in turns.groupby("item_number"):
        row = conv.loc[number].copy()
        row["item_number"] = number
        messages, n, trim = build_prompt(row, g, hits.get(number, set()), tokenizer, args.budget)
        prompts.append(messages)
        meta.append({"repo": args.repo, "item_number": number, "n_prompt_tokens": n, "trimmed": trim})
    if args.dry_run:
        print(prompts[0][1]["content"][:6000])
        m = pd.DataFrame(meta)
        print(f"\n{len(m)} prompts; tokens median {m['n_prompt_tokens'].median():.0f}, max {m['n_prompt_tokens'].max()}; "
              f"trimmed: {m['trimmed'].value_counts().to_dict()}")
        return

    out = RUNS / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_llm_{args.model.split('/')[-1].lower()}"
    out.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(out / "log.txt"), logging.StreamHandler()])
    log = logging.getLogger()
    (out / "pid").write_text(str(os.getpid()))
    (out / "prompt.txt").write_text(f"SYSTEM:\n{SYSTEM}\n\nUSER:\n{INSTRUCTIONS}<rendered discussion>\n\nSCHEMA:\n"
                                    + json.dumps(SCHEMA, indent=2))

    pick_gpu(args.min_free)
    free, total = gpu_memory(os.environ["CUDA_VISIBLE_DEVICES"])
    utilization = round(min(0.92, (free - 1500) / total), 3)
    log.info(f"{len(prompts)} prompts, model {args.model}@{revision[:10]}, GPU {os.environ['CUDA_VISIBLE_DEVICES']} "
             f"({free} MiB free, gpu_memory_utilization {utilization})")

    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model=args.model, revision=revision, max_model_len=args.max_model_len, seed=SEED,
              gpu_memory_utilization=utilization)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=SEED,
                            structured_outputs=StructuredOutputsParams(json=SCHEMA))
    config = {"model": args.model, "revision": revision, "vllm": vllm_version, "items": args.items,
              "rules": args.rules, "n": len(prompts), "budget": args.budget, "max_model_len": args.max_model_len,
              "sampling": {"temperature": 0, "max_tokens": args.max_tokens, "seed": SEED},
              "enable_thinking": False, "gpu": os.environ["CUDA_VISIBLE_DEVICES"],
              "prompt_sha1": hashlib.sha1((SYSTEM + INSTRUCTIONS + json.dumps(SCHEMA)).encode()).hexdigest(),
              "started": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    outputs = llm.chat(prompts, params, chat_template_kwargs={"enable_thinking": False})
    rows = []
    for m, o in zip(meta, outputs):
        text = o.outputs[0].text
        try:
            answer, ok = json.loads(text), True
        except json.JSONDecodeError:
            answer, ok = {}, False
        g = turns[turns["item_number"] == m["item_number"]]
        verified, anywhere, unknown = check_evidence(answer, g) if ok else (None, None, "")
        rows.append({**m, "parse_ok": ok,
                     **{k: answer.get(k) for k in ("tradeoff_present", "resource_concern", "competing_objective",
                                                   "tradeoff_relationship", "explicit_or_implicit", "explanation")},
                     "supporting_comment_ids": ";".join(answer.get("supporting_comment_ids", [])),
                     "exact_evidence_spans": json.dumps(answer.get("exact_evidence_spans", []), ensure_ascii=False),
                     "evidence_verified": verified, "evidence_found_anywhere": anywhere, "unknown_ids": unknown,
                     "finish_reason": o.outputs[0].finish_reason, "raw_output": text})
    pred = pd.DataFrame(rows)
    pred.to_parquet(out / "predictions.parquet", index=False)
    config["finished"] = f"{datetime.now():%Y-%m-%d %H:%M:%S}"
    (out / "config.json").write_text(json.dumps(config, indent=2))
    log.info(f"parse ok {pred['parse_ok'].mean():.1%}; answers {pred['tradeoff_present'].value_counts().to_dict()}")
    log.info(f"evidence verified in cited turns (mean share) {pred['evidence_verified'].mean():.3f}, "
             f"anywhere {pred['evidence_found_anywhere'].mean():.3f}; items citing unknown ids "
             f"{(pred['unknown_ids'] != '').sum()}")
    log.info(f"Output written to {out}")


if __name__ == "__main__":
    main()
