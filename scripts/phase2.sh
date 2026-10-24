#!/bin/bash
# Phase 2 (run under the GPU lock): train on the full corpus, evaluate every detector on every split,
# precompute shield scores for the agent test, export the browser model.
# Main model: DistilRoBERTa (DeBERTa-v3-small diverged to NaN loss on MPS within 200 steps, see README).
set -e
cd "$(dirname "$0")/.."
PY="nice -n 15 .venv/bin/python"
export HF_HOME=$PWD/data/hf_home TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
SEEDS=${SEEDS:-"0 1 2"}
train() { [ -f models/$2-s$3/config.json ] || $PY -u scripts/train.py --model $1 --name $2 --seed $3 "${@:4}"; }
evalm() { $PY scripts/evaluate.py --kind hf --path models/$1 --name $1; }

$PY scripts/evaluate.py --kind keyword --name keyword
train distilbert/distilroberta-base distilroberta 0 && evalm distilroberta-s0
train distilbert/distilroberta-base distilroberta-publiconly 0 --no_synth && evalm distilroberta-publiconly-s0
train microsoft/MiniLM-L12-H384-uncased minilm 0 --lr 5e-5 --epochs 3 && evalm minilm-s0
$PY scripts/export_onnx.py --model models/minilm-s0
$PY scripts/agent_eval.py score --model models/distilroberta-s0
$PY scripts/evaluate.py --kind protectai --name protectai-v2
for s in $SEEDS; do [ "$s" = 0 ] || { train distilbert/distilroberta-base distilroberta $s && evalm distilroberta-s$s; }; done
echo "== phase2 done"
