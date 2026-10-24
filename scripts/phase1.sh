#!/bin/bash
# Phase 1 (runs under the GPU lock): public-data baselines while synthetic data is still being generated.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export HF_HOME=$PWD/data/hf_home TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
echo "== smoke"; $PY scripts/train.py --model microsoft/MiniLM-L12-H384-uncased --name smoke --epochs 0.01 --no_synth
echo "== protectai"; $PY scripts/evaluate.py --kind protectai --name protectai-v2
echo "== deberta public-only"; $PY -u scripts/train.py --model microsoft/deberta-v3-small --name deberta-small-publiconly --no_synth --seed 0
$PY scripts/evaluate.py --kind hf --path models/deberta-small-publiconly-s0 --name deberta-small-publiconly-s0
echo "== phase1 done"
