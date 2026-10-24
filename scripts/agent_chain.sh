#!/bin/bash
# Score agent cases with the shipped ONNX model (CPU), then run the agent test at both thresholds.
set -e
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false
.venv/bin/python scripts/agent_eval.py score --model onnx
.venv/bin/python - <<'PY'
import json
d = json.load(open("results/agent/cases.json")); cs = d["cases"]
assert any(c.get("setting") == "base" for c in cs), "stale cases.json"
for name, t in d["thresholds"].items():
    for st in ["base", "enhanced"]:
        a = [c for c in cs if c["kind"] == "attack" and c["setting"] == st]
        print(name, st, "flagged", sum(c["score_full"] >= t for c in a), "/", len(a))
    b = [c for c in cs if c["kind"] == "benign"]
    print(name, "benign flagged", sum(c["score_full"] >= t for c in b), "/", len(b))
PY
for t in t0.5 fpr1; do
  AGENT_MODEL=gemma-4-26b-a4b-it JUDGE_MODEL=gemma-4-31b-it AGENT_WORKERS=2 GEMINI_DEBUG=1 \
    .venv/bin/python -u scripts/agent_eval.py run --threshold $t > results/agent_$t.log 2>&1
done
echo "== agent done"
