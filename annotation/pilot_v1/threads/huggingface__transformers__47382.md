# #47382 [pull_request] ci: harden daily CI reporting against GitHub API rate limits
https://github.com/huggingface/transformers/pull/47382  ·  opened by tarekziade on 2026-07-17

## [body] tarekziade (COLLABORATOR) · opening post · 2026-07-17
[[image]](https://transformers-ci.lor-e.huggingface.cool/d/pytest-observability-by-pr/pytest-observability-branch?var-pr=47382)

# What does this PR do?

The daily model CI report crashed in the "Send message to Slack" step with `KeyError: 'jobs'` (get_ci_error_statistics.get_jobs) and `KeyError: 'created_at'` (get_previous_daily_ci.get_last_daily_ci_run) whenever GitHub returned an error payload instead of run/job data.

The large matrices (Model/Torch-pipeline/Example/Quantization/Kernels) paginate over 1000+ jobs and exhaust the GitHub API rate limit, so their reports never post to Slack, making the run look like it never ran even though every test executed.

This harden this step with retry loops and a clear error in case it failed.

## [c5002461093] tarekziade (COLLABORATOR) · issue_comment · 2026-07-17
slack step pushed again with this fix on a throw-away wf to verify https://github.com/huggingface/transformers/actions/runs/29575160394/job/87867850101

## [c5002773356] tarekziade (COLLABORATOR) · issue_comment · 2026-07-17
attempt 2 https://github.com/huggingface/transformers/actions/runs/29576459980/job/87871909474

## [c5002815005] tarekziade (COLLABORATOR) · issue_comment · 2026-07-17
(second commit will be removed before merging, it's only here to exercise the fix on yesterday's daily)

## [c3602983186] vasqu (COLLABORATOR) · review_comment on utils/get_ci_error_statistics.py:38 · 2026-07-17
Could we have some gh reference if possible for their API response? More of a nit, but I think it's easier if we can easily access here via a comment

## [c3602993083] vasqu (COLLABORATOR) · review_comment on utils/get_ci_error_statistics.py:61 · 2026-07-17
Max retries based on anything with 8?

## [c3603039025] vasqu (COLLABORATOR) · review_comment on utils/get_ci_error_statistics.py:87 · 2026-07-17
I'm a bit confused why we are manually doing it ourselves, doesn't requests provide max retries (with backoffs) or is this not working well enough / not responding to the status codes as we wished?

## [c3603054677] vasqu (COLLABORATOR) · review_comment on utils/get_ci_error_statistics.py:125 · 2026-07-17
Ok so we can guarantee that it exists?

## [c3603061592] vasqu (COLLABORATOR) · review_comment on utils/get_previous_daily_ci.py:46 · 2026-07-17
Ah ok so we guard here

## [c3603065601] vasqu (COLLABORATOR) · review_comment on .github/workflows/repost-daily-ci-report.yml:1 · 2026-07-17
Curiuous if we want this to be automatic? But maybe let's see how it fares and how much manual triggering is needed

## [c4722472387] vasqu (COLLABORATOR) · review COMMENTED · 2026-07-17
My main point is to maybe not be manually implementing the retry? Unsure because it's been a while but pretty sure requests supports backoffs by itself. Also ok to not to if there is a genuine reason for the manual version

## [c3603936616] tarekziade (COLLABORATOR) · review_comment on .github/workflows/repost-daily-ci-report.yml:1 · reply to [c3603065601] · 2026-07-17
this was just a temporary wf to re-post the data from yesterday without re-running. not being comited at the end as explained previously

## [c4723589794] tarekziade (COLLABORATOR) · review COMMENTED · 2026-07-17 (no text)

## [c3603951348] tarekziade (COLLABORATOR) · review_comment on utils/get_ci_error_statistics.py:87 · reply to [c3603039025] · 2026-07-17
we're doing custom headers lookups to drive the retries

## [c4723605933] tarekziade (COLLABORATOR) · review COMMENTED · 2026-07-17 (no text)

## [c3603960327] tarekziade (COLLABORATOR) · review_comment on utils/get_ci_error_statistics.py:125 · reply to [c3603054677] · 2026-07-17
if key does not exist we get a key error

## [c4723615509] tarekziade (COLLABORATOR) · review COMMENTED · 2026-07-17 (no text)

## [c3603965445] tarekziade (COLLABORATOR) · review_comment on utils/get_ci_error_statistics.py:38 · reply to [c3602983186] · 2026-07-17
sure I can point to https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api?apiVersion=2026-03-10

## [c4723621045] tarekziade (COLLABORATOR) · review COMMENTED · 2026-07-17 (no text)

## [c5004399126] tarekziade (COLLABORATOR) · issue_comment · 2026-07-17
attempt 2 worked :)

## [c3603996189] vasqu (COLLABORATOR) · review_comment on utils/get_ci_error_statistics.py:87 · reply to [c3603039025] · 2026-07-17
Gotcha!

## [c4723655164] vasqu (COLLABORATOR) · review COMMENTED · 2026-07-17 (no text)

## [c4723657348] vasqu (COLLABORATOR) · review APPROVED · 2026-07-17
Thanks 🫡

---
outcome: state=closed, merged=True
