#!/bin/bash
# Phase 3 (GPU lock): occlusion-based shield scores for the agent test + short-fragment diagnostic.
set -e
cd "$(dirname "$0")/.."
PY="nice -n 15 .venv/bin/python"
export HF_HOME=$PWD/data/hf_home TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
$PY scripts/agent_eval.py score --model models/distilroberta-s0
$PY scripts/diag_fragments.py distilroberta-s0 minilm-s0
echo "== phase3 done"
