#!/bin/bash
# One-command reproduction. API responses are cached in cache/, so reruns cost nothing.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
export HF_HOME=$PWD/data/hf_home TOKENIZERS_PARALLELISM=false
LOCK=../_shared/gpulock.sh; [ -x "$LOCK" ] || LOCK=""   # shared-GPU lock when run inside the portfolio

# 1. data
mkdir -p data/gh data/hf
[ -d data/gh/BIPIA ] || git clone -q --depth 1 https://github.com/microsoft/BIPIA.git data/gh/BIPIA
[ -d data/gh/InjecAgent ] || git clone -q --depth 1 https://github.com/uiuc-kang-lab/InjecAgent.git data/gh/InjecAgent
[ -d data/gh/agentdojo ] || git clone -q --depth 1 https://github.com/ethz-spylab/agentdojo.git data/gh/agentdojo
mkdir -p data/gh/tensortrust && for f in extraction-robustness/v1/extraction_robustness_dataset.jsonl hijacking-robustness/v1/hijacking_robustness_dataset.jsonl; do
  [ -f "data/gh/tensortrust/$(basename $f)" ] || curl -sSL -o "data/gh/tensortrust/$(basename $f)" "https://raw.githubusercontent.com/HumanCompatibleAI/tensor-trust-data/main/benchmarks/$f"; done
$PY scripts/download_hf.py

# 2. synthetic data (Gemini / Gemma) and corpora
$PY scripts/synthesize.py --no-write
$PY scripts/synthesize.py --cached-only
$PY scripts/salvage_v1.py
$PY scripts/build_corpus.py
$PY scripts/build_external.py

# 3. train + evaluate (GPU)
$LOCK bash scripts/phase2.sh

# 4. LLM judge, agent test, report
$PY scripts/judge_baseline.py
AGENT_MODEL=gemma-4-26b-a4b-it $PY scripts/agent_eval.py run --threshold fpr1
AGENT_MODEL=gemini-3.5-flash-lite $PY scripts/agent_eval.py run --threshold fpr1
$PY scripts/report.py
