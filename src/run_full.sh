#!/usr/bin/env bash
# The "full" dataset: every issue/PR created before 2026-09-01 of the three repos, then Steps 1-4 (rule groups).
#
#   setsid nohup bash src/run_full.sh > /dev/null 2>&1 &
#   tail -f data/full/run_full.log      # progress
#   kill $(cat data/full/pid)           # stop; run the same command again to resume (checkpoints)
#
# Collection uses the sibling project's collector code unchanged (run from its folder, so it finds its .env
# with the GitHub token); the data is written here, to data/full/raw and data/full/tables.
# Repos run one after another (they share one token's rate limit), smallest first.
set -euo pipefail
ALT=/u1/users/scb052/Projects/alt-sustainability-developer-discussion
SIB=/student/scb052/Projects/sustainability-developer-discussion
OUT=$ALT/data/full
mkdir -p "$OUT"
echo $$ > "$OUT/pid"
exec >> "$OUT/run_full.log" 2>&1
echo "=== $(date -Is) start (pid $$)"

cd "$SIB"
for r in "deepspeedai DeepSpeed" "NVIDIA Megatron-LM" "huggingface transformers"; do
  set -- $r
  echo "=== $(date -Is) collect $1/$2"
  # GitHub sometimes answers with an empty page that the collector does not retry (seen in pr_details).
  # A rerun continues from the checkpoint, so retry as long as attempts make progress (the checkpoint file
  # changes); stop after 5 attempts in a row without progress.
  ckpt="$OUT/raw/$1__$2/_checkpoints.json"
  stuck=0
  while true; do
    before=$(md5sum "$ckpt" 2>/dev/null || true)
    .venv/bin/python -u -m collector.collect --owner "$1" --repo "$2" --max-items 0 --until 2026-09-01 --out "$OUT" \
      && break
    if [ "$(md5sum "$ckpt" 2>/dev/null || true)" = "$before" ]; then stuck=$((stuck + 1)); else stuck=0; fi
    [ "$stuck" -ge 5 ] && { echo "=== $(date -Is) FAILED: collect $1/$2, 5 attempts in a row without progress"; exit 1; }
    echo "=== $(date -Is) collect $1/$2 stopped (no progress $stuck/5), resuming in 2 min"
    sleep 120
  done
  .venv/bin/python -u -m collector.build_tables --owner "$1" --repo "$2" --out "$OUT"
done

cd "$ALT"
for r in deepspeedai__DeepSpeed NVIDIA__Megatron-LM huggingface__transformers; do
  echo "=== $(date -Is) Steps 1-4 $r"
  .venv/bin/python -u src/prepare_threads.py --dataset full --repo "$r"
  .venv/bin/python -u src/detect_rules.py --dataset full --repo "$r"
done
.venv/bin/python -u src/combine_groups.py --dataset full
.venv/bin/python -u src/combine_groups.py --dataset full --bot-candidates
echo "=== $(date -Is) done"
rm -f "$OUT/pid"
