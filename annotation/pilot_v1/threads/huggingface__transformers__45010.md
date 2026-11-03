# #45010 [pull_request] fix: pin 69 unpinned action(s),extract 2 unsafe expression(s) to env vars
https://github.com/huggingface/transformers/pull/45010  ·  opened by dagecko on 2026-03-26

## [body] dagecko (NONE) · opening post · 2026-03-26
## Fix: CI/CD Security Vulnerabilities in GitHub Actions

Hi! [Runner Guard](https://github.com/Vigilant-LLC/runner-guard), an open-source
CI/CD security scanner by [Vigilant Cyber Security](https://www.vigilantdefense.com),
identified security vulnerabilities in this repository's GitHub Actions workflows.

This PR applies automated fixes where possible and reports additional findings
for your review.

### Fixes applied (in this PR)

| Rule | Severity | File | Description |
|------|----------|------|-------------|
| RGS-002 | high | `.github/workflows/benchmark.yml` | Extracted 1 unsafe expression(s) to env vars |
| RGS-007 | high | `.github/workflows/build-ci-docker-images.yml` | Pinned 4 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/build-docker-images.yml` | Pinned 29 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/build-nightly-ci-docker-images.yml` | Pinned 6 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/build-past-ci-docker-images.yml` | Pinned 6 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/build_documentation.yml` | Pinned 2 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/build_pr_documentation.yml` | Pinned 1 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/check-workflow-permissions.yml` | Pinned 1 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/codeql.yml` | Pinned 1 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/model_jobs.yml` | Pinned 1 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/pr_build_doc_with_comment.yml` | Pinned 1 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/release-conda.yml` | Pinned 1 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/release.yml` | Pinned 1 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/self-scheduled-amd-mi250-caller.yml` | Pinned 4 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/self-scheduled-amd-mi325-caller.yml` | Pinned 4 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/self-scheduled-amd-mi355-caller.yml` | Pinned 4 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/ssh-runner.yml` | Pinned 1 third-party action(s) to commit SHA |
| RGS-007 | high | `.github/workflows/trufflehog.yml` | Pinned 1 third-party action(s) to commit SHA |
| RGS-002 | high | `.github/workflows/update_metdata.yml` | Extracted 1 unsafe expression(s) to env vars |
| RGS-007 | high | `.github/workflows/upload_pr_documentation.yml` | Pinned 1 third-party action(s) to commit SHA |

### Advisory: additional findings (manual review recommended)

| Rule | Severity | File | Description |
| RGS-008 | high | `.github/workflows/benchmark_v2.yml` | Secret Directly Interpolated in run Block |
| RGS-012 | high | `.github/workflows/circleci-failure-summary-comment.yml` | Secret Exfiltration via Outbound HTTP Request |
| RGS-004 | high | `.github/workflows/pr-repo-consistency-bot.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/pr-repo-consistency-bot.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/pr-repo-consistency-bot.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/pr-repo-consistency-bot.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/pr_build_doc_with_comment.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/pr_build_doc_with_comment.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/pr_build_doc_with_comment.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/pr_build_doc_with_comment.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/self-comment-ci.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/self-comment-ci.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/self-comment-ci.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/self-comment-ci.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/self-comment-ci.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/self-comment-ci.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/self-comment-ci.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/self-comment-ci.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/self-comment-ci.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/trl-ci-bot.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/trl-ci-bot.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/trl-ci-bot.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/trl-ci-bot.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/trl-ci-bot.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-004 | high | `.github/workflows/trl-ci-bot.yml` | Comment-Triggered Workflow Without Author Authorization Check |
| RGS-005 | medium | `.github/workflows/assign-reviewers.yml` | Excessive Permissions on Untrusted Trigger |
| RGS-005 | medium | `.github/workflows/circleci-failure-summary-comment.yml` | Excessive Permissions on Untrusted Trigger |
| RGS-005 | medium | `.github/workflows/pr-repo-consistency-bot.yml` | Excessive Permissions on Untrusted Trigger |
| RGS-005 | medium | `.github/workflows/pr-repo-consistency-bot.yml` | Excessive Permissions on Untrusted Trigger |
| RGS-005 | medium | `.github/workflows/pr_build_doc_with_comment.yml` | Excessive Permissions on Untrusted Trigger |
| RGS-005 | medium | `.github/workflows/pr_build_doc_with_comment.yml` | Excessive Permissions on Untrusted Trigger |
| RGS-005 | medium | `.github/workflows/pr_build_doc_with_comment.yml` | Excessive Permissions on Untrusted Trigger |
| RGS-005 | medium | `.github/workflows/pr_slow_ci_suggestion.yml` | Excessive Permissions on Untrusted Trigger |
| RGS-005 | medium | `.github/workflows/self-comment-ci.yml` | Excessive Permissions on Untrusted Trigger |
| RGS-005 | medium | `.github/workflows/self-comment-ci.yml` | Excessive Permissions on Untrusted Trigger |
| RGS-005 | medium | `.github/workflows/self-comment-ci.yml` | Excessive Permissions on Untrusted Trigger |
| RGS-005 | medium | `.github/workflows/self-comment-ci.yml` | Excessive Permissions on Untrusted Trigger |

### Why this matters

GitHub Actions workflows that use untrusted input in `run:` blocks, expose
secrets inline, or use unpinned third-party actions are vulnerable to
code injection, credential theft, and supply chain attacks. These are the same
vulnerability classes exploited in the [tj-actions/changed-files incident](https://www.vigilantdefense.com/resources/runner-guard)
and subsequent supply chain attacks, which compromised CI secrets across
thousands of repositories.

### How to verify

Review the diff — each change is mechanical and preserves workflow behavior:
- **Expression extraction** (RGS-002/008/014): Moves `${{ }}` expressions from
  `run:` blocks into `env:` mappings, preventing shell injection
- **SHA pinning** (RGS-007): Pins third-party actions to immutable commit SHAs
  (original version tag preserved as comment)
- **Debug env removal** (RGS-015): Removes `ACTIONS_RUNNER_DEBUG`/`ACTIONS_STEP_DEBUG`
  which leak secrets in workflow logs

Run `brew install Vigilant-LLC/tap/runner-guard && runner-guard scan .` or install from the
[repo](https://github.com/Vigilant-LLC/runner-guard) to verify.

---

Found by [Runner Guard](https://github.com/Vigilant-LLC/runner-guard) | Built by [Vigilant Cyber Security](https://www.vigilantdefense.com) | [Learn more](https://www.vigilantdefense.com/resources/runner-guard)

If this PR is not welcome, just close it -- we won't send another.

## [c4134165759] Rocketknight1 (MEMBER) · issue_comment · 2026-03-26
cc @tarekziade @ydshieh, feel free to close it if you're not interested

## [c4136670608] dagecko (NONE) · issue_comment · 2026-03-26
Hey @tarekziade @ydshieh — this PR pins third-party actions to commit SHAs and extracts expressions from run blocks to prevent shell injection. These are the same vulnerability classes being actively exploited in the tj-actions, Trivy, LiteLLM supply chain attack chain. We scanned the top 50K repos on GitHub and over 20,000 have the same vulns that TeamPCP used. The blast radius potential is pretty crazy. We're submitting PRs on as many of them as we can. More context on what we're finding: https://x.com/vigilance_one/status/2036581210663616729

Happy to answer any questions.

— Chris Nyhuis, Vigilant

## [c4145669953] dagecko (NONE) · issue_comment · 2026-03-27
Resubmitted as #45077. Had a problem with my fork, apologies for the noise.

---
outcome: state=closed, merged=False
