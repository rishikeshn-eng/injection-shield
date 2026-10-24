#!/bin/bash
# Wait for synthesis workers, materialise synthetic data, rebuild corpora, run phase 2 under the GPU lock.
cd "$(dirname "$0")/.."
while pgrep -f "scripts/synthesize.py --" >/dev/null; do sleep 30; done
P="nice -n 15 /opt/homebrew/opt/python@3.11/bin/python3.11"
$P scripts/synthesize.py --cached-only && $P scripts/salvage_v1.py
nice -n 15 .venv/bin/python scripts/build_corpus.py && nice -n 15 .venv/bin/python scripts/build_external.py
../_shared/gpulock.sh bash scripts/phase2.sh
