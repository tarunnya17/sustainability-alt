"""
The vocabulary of the rule-based detector: resource concerns (R), competing objectives (G),
speed (either side), and trade-off language (T).

This is the only file to edit when tuning the rules. Write a new detector version (--version v2)
after a change; v1 is the frozen sampling frame of the pilot.

Structure:  GROUPS[group] = {"role": ..., "label": ..., "strong": {term: regex}, "weak": {term: regex}}
- role:   R (resource concern), G (competing objective), RG (speed: either side), T (trade-off language)
- strong: the term alone counts
- weak:   counts only in a sentence that also has an EFFECT word (reduce, increase, save, drop, worse ...)
- term:   the canonical name stored in hits.parquet (the matched text is stored too)

Patterns are matched against cleaned, NFKC-normalized, lowercased text (clean_text.py), so they are
lowercase. No stemming: endings, plurals, separators and US/UK spellings are written in by hand, as in
the sibling's filtering/keywords.py, which most R patterns are copied from (groups A-E there).
"""

import re

SEP = r"[-_ ]?"  # "trade-off" / "trade off" / "tradeoff", "kv_cache" / "kv-cache" / "kv cache"

GROUPS = {
    # ---------------- R: resource concern ----------------
    "R_env": {
        "role": "R", "label": "direct environmental (energy, power, carbon, sustainability)",
        "strong": {  # sibling group A
            "sustainab*": r"\bsustainab\w*",
            "energy consumption/efficiency": rf"\benergy{SEP}(consumption|efficien\w*|usage|use|cost|aware|saving|reduction)",
            "power consumption/efficiency": rf"\bpower{SEP}(consumption|usage|efficien\w*|draw)",
            "watt / kwh / joule": r"\b(watts?|watt-hours?|kwh|joules?)\b",
            "carbon footprint/emissions": rf"\bcarbon{SEP}(footprint|emissions?|efficien\w*|intensity)",
            "co2 / ghg": r"\bco2e?\b|carbon dioxide|greenhouse|\bghg\b",
            "green ai/software/computing": r"\bgreen (ai|software|computing)\b",
            "environmental impact": r"\b(environmental(ly)?|ecological) (impact|cost|footprint|friendly)",
            "climate": r"\bclimate\b",
        },
        "weak": {
            "energy": r"\benergy\b",
            "electricity": r"\belectricity\b",
        },
    },
    "R_compute": {
        "role": "R", "label": "compute / operational cost",
        "strong": {  # sibling group B + operational cost and amounts of compute from the paste
            "compute cost": rf"\b(compute|computational){SEP}(cost|budget|resources?|efficien\w*|overhead|burden)",
            "gpu hours": rf"\bgpu{SEP}hours?\b",
            "gpu/cpu utilization": rf"\b(gpu|cpu){SEP}(utili[sz]ation|usage|consumption)",
            "resource usage/efficiency": rf"\bresources?{SEP}(cost|efficien\w*|usage|consumption|utili[sz]ation|saving|intensive|hungry)",
            "training/inference cost": rf"\b(training|inference){SEP}costs?\b",
            "flops": r"\b[gt]?flops\b",
            "wasteful": r"\bwasteful\b",
            "operational cost": rf"\b(gpu|cloud|infra(structure)?|hardware|serving|hosting){SEP}(costs?|bills?|spend\w*|budget)\b",
            "amount of compute": r"\b(more|less|extra|additional|wasted|redundant|unnecessary|duplicate[d]?) (compute|computations?)\b",
        },
        "weak": {
            "expensive/cheap": r"\b(expensive|costly|cheap|cheaper)\b",
            "training/inference time": rf"\b(training|inference){SEP}time\b",
        },
    },
    "R_memory": {
        "role": "R", "label": "memory",
        "strong": {  # sibling group C
            "memory usage/footprint": rf"\b(memory|vram){SEP}(footprint|usage|consumption|efficien\w*|overhead|savings?|reduction|requirements?|pressure|spikes?|bloat)",
            "peak memory": r"\bpeak memory\b",
            "gpu/cpu memory": r"\b(gpu|cpu) memory\b",
            "oom": r"\boom\b|\bout[- ]of[- ]memory\b",
            "vram": r"\bvram\b",
            "memory leak": r"\bmemory leaks?\b",
        },
        "weak": {
            "memory": r"\bmemory\b",
            "offload": r"\boffload\w*",
            "kv cache": rf"\bkv{SEP}cache\b",
        },
    },
    "R_technique": {
        "role": "R", "label": "efficiency technique",
        "strong": {},  # everyday features in transformers; never enough on their own (sibling's finding)
        "weak": {  # sibling group E + batching/caching from the paste
            "quantization": r"\bquantiz\w*",
            "low-bit dtype": r"\b(int8|int4|fp8|fp4|nf4)\b",
            "mixed precision": r"\bmixed precision\b",
            "pruning": r"\bprun(e|ed|ing)\b",
            "sparsity": r"\bsparsity\b",
            "distillation": r"\bdistill\w*",
            "lora/peft": r"\blora\b|\bpeft\b|\bparameter[- ]efficient\b",
            "flash attention": rf"\bflash{SEP}att\w*",
            "gradient checkpointing": rf"\b(gradient|activation){SEP}checkpointing\b",
            "model compression": r"\bmodel compression\b",
            "batching": r"\b(continuous |dynamic )?batching\b",
            "caching": r"\bcaching\b",
        },
    },
    # ---------------- speed: R next to a non-speed G, G next to a non-speed R ----------------
    "SPEED": {
        "role": "RG", "label": "speed / runtime performance",
        "strong": {  # sibling group D
            "latency": r"\blatency\b",
            "throughput": r"\bthroughput\b",
            "tokens/s": r"\btok(ens)?/s\b|\btokens per second\b",
            "speedup/slowdown": rf"\bspeed{SEP}ups?\b|\bslow{SEP}downs?\b|\b\d+(\.\d+)?\s?[x×] (faster|slower)\b",
            "performance regression": r"\b(perf|performance|speed) regressions?\b",
            "overhead": r"(?<!maintenance )\boverhead\b",
            "execution time": rf"\bexecution time\b|\bwall{SEP}clock\b|\b(inference|training) speed\b|\bruntime (cost|overhead)\b",
        },
        "weak": {
            "faster/slower": r"\b(faster|slower)\b",
            "benchmark": r"\bbenchmark\w*",
        },
    },
    # ---------------- G: competing objective ----------------
    "G_accuracy": {
        "role": "G", "label": "model accuracy / quality",
        # "precision" is left out: in this repo it is almost always the dtype (mixed/half/lower precision)
        "strong": {
            "accuracy drop": rf"\baccuracy{SEP}(drop|loss|degradation|regression|hit)s?\b|\b(quality|performance) degradation\b",
            "perplexity": r"\bperplexity\b",
            "model/output quality": r"\b(model|output|generation|prediction)s? quality\b",
            "worse accuracy/quality": r"\b(lower|worse|degraded|reduced|decreased|better|higher) (accuracy|quality)\b",
        },
        "weak": {
            "accuracy": r"\b(accuracy|accurate|inaccurate)\b",
            "quality": r"\bquality\b",
            "f1": r"\bf1\b",
            "logits": r"\blogits\b",
        },
    },
    "G_maintain": {
        "role": "G", "label": "maintainability / complexity",
        "strong": {
            "maintainability": r"\bmaintainab\w*",
            "maintenance burden": rf"\bmaintenance{SEP}(burden|cost|overhead)",
            "technical debt": r"\b(technical|tech) debt\b",
            "bloat": r"(?<!memory )(?<!vram )\bbloat\w*",
            "readability": r"\breadabilit\w*",
            "added complexity": r"\b(added|additional|extra|more|unnecessary|increased?) complexity\b|\bcomplexity (cost|overhead|burden)\b",
            "hard to maintain/debug": r"\b(hard|harder|difficult|more difficult) to (debug|maintain|read|understand)\b",
        },
        "weak": {
            "maintenance": r"\bmaintenance\b",
            "complexity": r"\bcomplexity\b",
            "code duplication": r"\bcode duplication\b",
            "readable": r"\breadable\b",
            "complicated/hacky": r"\b(complicated|hacky|convoluted)\b",
        },
    },
    "G_reliability": {
        "role": "G", "label": "reliability / correctness / numerical stability",
        "strong": {
            "numerical stability": r"\bnumerical(ly)? (instabilit\w*|stabilit\w*|precision|issues?|errors?|differences?)\b",
            "correctness": r"\bcorrectness\b",
            "training instability": r"\btraining instabilit\w*",
            "wrong results": r"\b(wrong|incorrect|garbage|gibberish|silently wrong) (results?|outputs?|generations?|logits|values)\b",
        },
        "weak": {
            "stability": r"\b(stabilit\w*|instabilit\w*|unstable)\b",
            "nan/inf": r"\b(nans?|infs?)\b",
            "failure": r"\bfailures?\b",
        },
    },
    "G_compat": {
        "role": "G", "label": "compatibility / portability",
        "strong": {
            "backward compatibility": r"\bbackwards?[- ]?(compat\w*|incompat\w*)|\bbc[- ]?break\w*",
            "breaking change": r"\bbreaking changes?\b",
            "hardware support": r"\bhardware support\b",
            "portability": r"\bportab\w*",
        },
        "weak": {
            "compatibility": r"\b(compatib\w*|incompatib\w*)",
        },
    },
    "G_function": {
        "role": "G", "label": "functionality / features",
        "strong": {
            "lose functionality": r"\b(loss|lose|losing|drop|dropping|remove|removing|without) (of )?(some |the )?(functionality|features?|support)\b",
            "feature parity": r"\bfeature parity\b",
        },
        "weak": {
            "functionality": r"\bfunctionalit\w*",
            "capability": r"\bcapabilit\w*",
            "unsupported": r"\bunsupported\b|\bnot supported\b",
            "limitation": r"\blimitations?\b",
            "flexibility": r"\bflexib\w*",
        },
    },
    "G_effort": {
        "role": "G", "label": "development effort",
        "strong": {
            "development effort": r"\b(development|implementation|engineering|developer) effort\b",
            "engineering time": r"\b(engineering|developer|dev) time\b",
            "hard to implement": r"\b(hard|difficult|complex|tricky) to (implement|support|integrate)\b",
            "waste of time": r"\bwast(e|ing) (of )?(\w+ ){0,2}time\b",
            "lot of work": r"\b(a lot|lots) of (work|effort)\b",
        },
        "weak": {
            "effort": r"\befforts?\b",
        },
    },
    # ---------------- T: trade-off language ----------------
    "T_marker": {
        "role": "T", "label": "explicit trade-off marker (sentence or comment level)",
        "strong": {
            "trade-off": rf"\btrade{SEP}offs?\b",
            "at the cost of": r"\bat the (cost|expense) of\b",
            "sacrifice": r"\bsacrific\w*",
            "worth": r"\b(not )?worth (it|the)\b",
            "downside/drawback": r"\b(downsides?|drawbacks?)\b",
            "compromise": r"\bcompromis\w*",
            "cost/benefit": r"\bcost[/ -]benefit\b|\bpros and cons\b",
            "price to pay": r"\bprice to pay\b|\b(comes?|coming) at (a|the|some) (cost|price)\b|\bin exchange for\b",
            "penalty": r"(?<!repetition )(?<!length )(?<!presence )(?<!frequency )\bpenalt(y|ies)\b",
            "balance/prioritize": r"\bbalance (between|against)\b|\bprioriti[sz]\w*",
        },
        "weak": {},
    },
    "T_contrast": {
        "role": "T", "label": "contrast connective (sentence level only)",
        "strong": {
            "contrast": r"\b(but|however|although|though|whereas|while|yet|nevertheless|unfortunately)\b|\bon the other hand\b",
        },
        "weak": {},
    },
    "T_alternative": {
        # sentence level only: comment-wide, these made unrelated R and G "explicit" (in 1,958 items)
        "role": "T", "label": "alternative connective (sentence level only)",
        "strong": {
            "rather than / instead of": r"\brather than\b|\binstead of\b",
        },
        "weak": {},
    },
    "T_response": {
        "role": "T", "label": "objection / alternative cue (ranking only)",
        "strong": {
            "concern": r"\b(concern(s|ed)?|worr(y|ied|ies)|afraid|hesitant|not sure|skeptical)\b",
            "negative effect": r"\b(degrad\w*|hurts?|hurting|regress\w*|breaks?|breaking)\b",
            "make it optional": r"\boptional\w*|\bopt[- ](in|out)\b|\b(behind|under) a flag\b|\bconfigurable\b|\bby default\b|\bdefault to\b",
            "alternative": r"\balternative(ly|s)?\b|\bboth (options|modes|paths|implementations|versions)\b|\brevert\w*",
        },
        "weak": {},
    },
}

# A weak term only counts in a sentence with one of these words (a claim about an effect).
EFFECT = re.compile(r"\b(reduc\w*|decreas\w*|increas\w*|sav(e|es|ed|ing|ings)|lower\w*|higher|drops?|dropp\w*|"
                    r"improv\w*|degrad\w*|hurts?|cuts?|cutting|less|more|fewer|worse|better|halv\w*|doubl\w*|"
                    r"lose|losing|gains?|boost\w*|regress\w*|impact\w*|affect\w*|significant\w*|noticeabl\w*)\b")

R_GROUPS = [g for g, s in GROUPS.items() if s["role"] == "R"]
G_GROUPS = [g for g, s in GROUPS.items() if s["role"] == "G"]


def compiled_terms():
    """[(group, role, strength, term, compiled_regex), ...] in a fixed order."""
    out = []
    for group, spec in GROUPS.items():
        for strength in ("strong", "weak"):
            for term, pattern in spec[strength].items():
                out.append((group, spec["role"], strength, term, re.compile(pattern)))
    return out
