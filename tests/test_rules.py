"""The paste's illustrative examples, run through the rule-based detector on tiny synthetic conversations.

    .venv/bin/python -m pytest tests
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clean_text import clean  # noqa: E402
from common import split_sentences  # noqa: E402
from detect_rules import detect_item  # noqa: E402


def conversation(*comments, title="Some change"):
    """comments: (author, text) pairs; the first one is the opening post."""
    rows = [{"unit_id": "title", "kind": "title", "author": comments[0][0], "raw": title}]
    rows.append({"unit_id": "body", "kind": "body", "author": comments[0][0], "raw": comments[0][1]})
    rows += [{"unit_id": f"c{i}", "kind": "issue_comment", "author": a, "raw": t} for i, (a, t) in enumerate(comments[1:])]
    df = pd.DataFrame(rows)
    df["thread_root"] = df["unit_id"]
    df["reply_to"] = None
    df["sentences"] = [split_sentences(clean(t)) for t in df["raw"]]
    return df


def run(*comments, **kw):
    _, links, info = detect_item(conversation(*comments, **kw))
    return links, info


def test_explicit_memory_vs_accuracy():
    _, info = run(("a", "Quantization reduces GPU memory by 35%, but decreases accuracy by 1.5%."))
    assert info["group"] == "HIGH_PRIORITY"
    _, info = run(("a", "Reduce memory, but accuracy decreases."))
    assert info["group"] == "HIGH_PRIORITY"


def test_technique_with_latency_is_a_link_without_marker():
    links, info = run(("a", "Quantization increases inference latency."))
    assert info["group"] == "POSSIBLE_TRADEOFF"
    assert {l["speed_role"] for l in links} == {"G"}


def test_resource_without_competing_goal():
    assert run(("a", "Fix GPU memory allocation error."))[1]["group"] == "RESOURCE_ONLY"
    # paste 6.3: an optimization with no discussed cost is resource optimization only
    assert run(("a", "Changing the attention implementation reduces memory requirements."))[1]["group"] == "RESOURCE_ONLY"


def test_competing_goal_alone_is_no_match():
    assert run(("a", "The model has excellent accuracy."))[1]["group"] == "NO_MATCH"


def test_implicit_tradeoff_across_comments():
    links, info = run(("a", "I propose enabling aggressive quantization to reduce GPU memory."),
                      ("b", "That will reduce model quality considerably."),
                      ("a", "Perhaps we can provide both precision modes."))
    assert info["group"] == "POSSIBLE_TRADEOFF"
    assert info["multi_developer"]
    assert any(l["kind"] == "adjacent" and l["r_group"] == "R_memory" and l["g_group"] == "G_accuracy" for l in links)


def test_speed_on_either_side():
    links, info = run(("a", "This is 2× faster but less accurate."))
    assert info["group"] == "HIGH_PRIORITY" and {l["speed_role"] for l in links} == {"R"}
    links, info = run(("a", "It saves memory but slower."))
    assert info["group"] == "HIGH_PRIORITY" and {l["speed_role"] for l in links} == {"G"}


def test_speed_does_not_pair_with_itself_or_inside_a_phrase():
    # "overhead" inside "memory overhead" is dropped, so there is no memory x speed link
    assert run(("a", "The memory overhead is large."))[1]["group"] == "RESOURCE_ONLY"
    assert run(("a", "Latency is better but throughput is lower."))[1]["group"] == "RESOURCE_ONLY"


def test_comment_level_marker():
    _, info = run(("a", "This reduces peak memory. The trade-off is added complexity in the loader."))
    assert info["group"] == "HIGH_PRIORITY"


def test_metrics_on_the_paste_example():
    from evaluate import prf, weighted_prf
    p, r, f = prf(60, 20, 40)
    assert (round(p, 3), round(r, 3), round(f, 3)) == (0.75, 0.6, 0.667)
    # weights: an item sampled at 1-in-10 counts 10 times
    y = [True, True, False, True]
    pred = [True, False, True, True]
    assert weighted_prf(y, pred, [1, 10, 1, 1]) == prf(2, 1, 10)
