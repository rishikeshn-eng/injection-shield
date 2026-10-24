#!/bin/bash
cd "$(dirname "$0")/.."
until grep -q "== phase2 done" results/finalize.log; do pgrep -f "scripts/finalize.sh" >/dev/null || break; sleep 30; done
../_shared/gpulock.sh bash scripts/phase3.sh
for t in t0.5 fpr1; do
  AGENT_MODEL=gemma-4-26b-a4b-it JUDGE_MODEL=gemma-4-31b-it AGENT_WORKERS=2 nice -n 15 .venv/bin/python -u scripts/agent_eval.py run --threshold $t > results/agent_$t.log 2>&1
done
echo "== after_phase2 done"
