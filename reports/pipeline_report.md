# Trade-off detection pipeline: report (2026-11-02)

Repository: `huggingface__transformers`; detector `rules_v1` (the frozen sampling frame).

## Funnel

| step                                                             |   items |
|:-----------------------------------------------------------------|--------:|
| issues and PRs collected                                         |   10000 |
| with human-written text                                          |    9862 |
| resource concern counts (R or speed)                             |    1085 |
| ... and a competing objective somewhere (R + G)                  |     407 |
| R and G linked: trade-off candidates (HIGH + POSSIBLE)           |     386 |
| ... with explicit trade-off language (HIGH)                      |      61 |
| LLM 'yes' among 200 screened (2026-11-02_10-40-39_llm_qwen3-32b) |      61 |

### Detector groups

| group             |   issue |   pull_request |
|:------------------|--------:|---------------:|
| HIGH_PRIORITY     |      32 |             29 |
| POSSIBLE_TRADEOFF |      97 |            228 |
| RESOURCE_ONLY     |     192 |            507 |
| NO_MATCH          |    1842 |           7073 |

## Rater agreement (Cohen's kappa)

Not yet: both raters have to finish `rater_1.xlsx` and `rater_2.xlsx`, then run `src/agreement.py`.

## Detection quality

Not yet: needs `gold.csv` (after adjudication), then `src/evaluate.py`.

## Episodes

Not yet: needs the pilot gold labels and/or a filled confirmation sheet, then `src/extract_episodes.py`.
