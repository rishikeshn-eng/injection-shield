#!/bin/bash
# Release the GPU lock after seed 1 (instead of holding it for all of phase 2), then run seed 2 as separate
# short gpulock calls so other projects' jobs can interleave.
cd "$(dirname "$0")/.."
until [ -f results/metrics/distilroberta-s1.json ]; do sleep 15; done
sleep 2
for p in $(pgrep -f "scripts/train.py --model distilbert/distilroberta-base --name distilroberta --seed 2") 77868 56326; do
  ps -o command= -p $p | grep -qE "injection|phase2|train.py|gpulock" && kill $p
done
echo "released lock after seed 1"
# let phase 3 (agent scoring) take its turn first
until grep -q "== phase3 done" results/after_phase2.log 2>/dev/null; do sleep 30; done
export HF_HOME=$PWD/data/hf_home TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1
../_shared/gpulock.sh nice -n 15 .venv/bin/python -u scripts/train.py --model distilbert/distilroberta-base --name distilroberta --seed 2
../_shared/gpulock.sh nice -n 15 .venv/bin/python scripts/evaluate.py --kind hf --path models/distilroberta-s2 --name distilroberta-s2
echo "== seed2 done"
