# Codebook: sustainability-related trade-off conversations

Read the whole conversation (threads/<file>.md; the GitHub link has the code changes) before labelling.
Every turn starts with its id in brackets, e.g. `[body]` or `[c3184836411]`. Cite these ids in Q4 and in the episodes sheet.

## The three elements

- **R, resource concern**: environmental sustainability (energy, power, carbon, emissions) or computational resources
  (compute, FLOPs, GPU hours, memory, VRAM, OOM, cost of hardware/cloud, efficiency techniques such as quantization,
  pruning, offloading, KV cache, gradient checkpointing). **Speed** (latency, throughput, slowdown) counts as R when it
  is weighed against a non-speed objective, and as G when it is weighed against a non-speed resource.
- **G, competing objective**: a second desirable outcome: accuracy/model quality, speed, maintainability/complexity,
  reliability/correctness/numerical stability, compatibility/portability, functionality/features, development effort.
- **T, trade-off relationship**: the developers themselves connect R and G **in the same development decision**: one is
  improved at the cost of the other, an objection names a downside of a resource-motivated change (or the reverse),
  alternatives are compared, or a compromise is proposed (make it optional, a flag, both modes).

## Questions

| Column | Question |
|---|---|
| Q1 | Does the conversation concern environmental sustainability or computational resource efficiency? |
| Q2 | Is there an identifiable competing objective? |
| Q3 | Is there evidence that R and G compete **in the same decision** (proposed change, design choice, alternative)? |
| Q4 | Which turns provide the evidence? (ids separated by `;`) |
| Q5 | Is the relationship explicit (stated in one place, e.g. "saves memory but loses accuracy"), implicit (built across turns: a proposal, then an objection), or unclear? |
| label | TRADEOFF = Q1, Q2, Q3 all yes · RESOURCE_ONLY = Q1 yes but no competing objective or no relationship · UNRELATED = Q1 no · UNCERTAIN = not enough evidence; goes to adjudication |
| multi_developer | Do at least two different people actually exchange views on the trade-off? (A single author stating it is still TRADEOFF, but not multi-developer.) |
| evidence | Short quote(s) that support the label |

For every TRADEOFF conversation, add one row per **episode** in the `episodes` sheet. A long PR can hold two separate
trade-offs (e.g. memory vs accuracy in comments 8-14, memory vs maintainability in 27-35): that is two rows.

## Decision rules

1. **Do not infer disadvantages the developers never discussed.** "Changing the attention implementation reduces memory
   requirements" alone is RESOURCE_ONLY, even if you can imagine a cost.
2. **Do not infer energy or carbon savings from memory or runtime optimization.** Use `tradeoff_framing =
   direct_environmental` only when energy, power, emissions or sustainability (in the environmental sense) is named.
   "Not a sustainable approach" meaning "not maintainable" is not environmental.
3. **R and G in unrelated places are not a trade-off.** Accuracy in one bug report and memory in an unrelated comment
   of the same thread: RESOURCE_ONLY.
4. **An optimization claim is not a trade-off claim.** "Reduce GPU memory by 40%" is RESOURCE_ONLY;
   "reduce GPU memory by 40%, but lose 2% accuracy" is TRADEOFF.
5. **A plain bug report about a resource** ("OOM when loading model X", "fix GPU memory leak") is RESOURCE_ONLY unless
   a fix is weighed against another objective.
6. **Benchmarks that show both sides** ("2x faster, perplexity unchanged") are a trade-off only if the discussion treats
   the second number as a cost or risk that was considered; "no accuracy loss" as a reassurance against a raised
   concern counts, a table with no discussion does not.
7. Use **UNCERTAIN** sparingly, and always write a note: it is resolved in adjudication, never counted as yes or no.
