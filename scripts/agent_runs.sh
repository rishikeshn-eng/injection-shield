#!/bin/bash
# Agent test runs (shield scores already in results/agent/cases.json). Agent + utility judge: Gemma-4-26B.
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
for t in t0.5 fpr1; do
  AGENT_MODEL=gemma-4-26b-a4b-it JUDGE_MODEL=gemma-4-26b-a4b-it AGENT_WORKERS=2 GEMINI_DEBUG=1 \
    .venv/bin/python -u scripts/agent_eval.py run --threshold $t > results/agent_$t.log 2>&1
done
echo "== agent done"
