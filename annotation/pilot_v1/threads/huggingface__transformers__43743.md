# #43743 [pull_request] Modular playground
https://github.com/huggingface/transformers/pull/43743  ·  opened by itazap on 2026-02-04

## [body] itazap (COLLABORATOR) · opening post · 2026-02-04
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=43743)

### Modular Detector Updates:

**Hub / index:** FAISS index. The new index is at: [itazap/transformers_code_embeddings_v3](https://huggingface.co/datasets/itazap/transformers_code_embeddings_v3) (old: [hf-internal-testing/transformers_code_embeddings](https://huggingface.co/datasets/hf-internal-testing/transformers_code_embeddings))

**Model:** [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B), now using last-token pooling instead of mean pooling

**Sanitization:** type hints, dtype save/cast boilerplate, and minor constructor kwargs (e.g. bias=) are stripped
Inheritance modular search: parse all modular_*.py imports to build themodel ancestry graph, then walks ancestor chains to parent-model classes that might not land in the raw top-k (e.g. a query matching Qwen3MLP will also score LlamaMLP via the Qwen3→Llama modular inheritance chain).

**Time filter:** models released after the query model are excluded (uses doc dates from docs/source/en/model_doc/*.md).

**Output / print changes:**

*Model match summary*: ranked table of candidate models with num_matched × mean_score² composite score, redundant ancestors filtered out (aka self-contained)
*Suggested modular inheritance:* greedy set-cover picks ≤3 parent models and assigns each class to its best parent, with --generate-prompt for an agent (WIP)


**new cli flags**
--ignore-models  # exclude specific models
--summary-only        # skip per-symbol tables
--generate-prompt [file]   # write an agent prompt for modular file creation
--push-only          # push index to Hub without rebuilding
--use_jaccard / --no-use_jaccard # now on by default

### Evaluation:

utils/run_modular_detector_eval.py: evaluates detector accuracy against ground-truth modular inheritance.
Eval dataset: [itazap/modeling-dataset](https://huggingface.co/datasets/itazap/modeling-dataset) - model_name, original_modeling_code (not available for all, but found via private GitHub repos, first commits, etc.), current_modular_code, bases (models inherited from in modular)
Runs: modular_model_detector on every model's “original” modeling code, checks whether the ground-truth parent models appear in the detector's top-1/3/5 suggestions, and writes per-model results to JSON.

**Model subsets:**
• FILTERED_MODELS - the ~30 models currently used for eval (the ones I could find original code for)
• LIGHT_MODELS - small subset of <=15 classes for quick testing (--light flag)

**CLI flags:**
--eval-source hub-original|json
--eval-hub-repo     # defaults to itazap/modeling-dataset
--output results.json
--light         # restrict to LIGHT_MODELS
--limit N        # first N entries only
--workers N       # parallel threads (OOM risk with >1)
--clear-cache-each-run  # on by default
--reload-analyzer-each-run # recreate analyzer per entry to free GPU memory (edited)

**Modular Inheritance - Eval Dataset**

Naiive but a dataset mapping our models to the model class(es) they inherit form in their modular file

https://huggingface.co/datasets/itazap/modular-model-eval/viewer


**Usage:**


```
 python utils/modular_model_detector.py --modeling-file src/transformers/models/sarvam/modeling_sarvam.py 
------------

[… 106 lines of code omitted]

## [c3883662340] itazap (COLLABORATOR) · issue_comment · 2026-02-11
run-slow: persimmon

---
outcome: state=open, merged=False
