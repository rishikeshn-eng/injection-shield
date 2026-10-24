"""Diagnostic: the window classifier is trained on whole windows, so scoring an isolated sentence is out of
distribution. Measure how often benign sentences, scored on their own, cross the 1%-FPR threshold, by length.
Writes results/diag_fragments.json. Run under the GPU lock."""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from agent_eval import segments  # noqa: E402
from common import RESULTS, ROOT, device, load, score_texts  # noqa: E402


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    rng = random.Random(0)
    docs = [r["text"] for r in load("test_hardneg")] + \
           [r["text"] for r in load("test_synth_id") if r["label"] == 0]
    frags = [s.strip() for d in docs for s in segments(d) if len(s.strip()) >= 3]
    frags = rng.sample(frags, min(3000, len(frags)))
    out = {}
    for name in sys.argv[1:] or ["distilroberta-s0", "minilm-s0"]:
        if name == "onnx":  # shipped int8 browser model on CPU
            from common import OnnxScorer
            sc = OnnxScorer()
            thr, tok, s, model = sc.thresholds["fpr1"], sc.tok, sc(frags), None
        else:
            path = os.path.join(ROOT, "models", name)
            thr = json.load(open(os.path.join(path, "thresholds.json")))["fpr1"]
            tok = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path).to(device())
            s = score_texts(model, tok, frags)
        n_tok = np.array([len(tok(f)["input_ids"]) for f in frags])
        res = {"threshold": thr, "n": len(frags), "flag_rate_all": float((s >= thr).mean())}
        for lo, hi in [(0, 8), (8, 16), (16, 32), (32, 64), (64, 10**6)]:
            m = (n_tok >= lo) & (n_tok < hi)
            res[f"flag_rate_tokens_{lo}_{hi}"] = float((s[m] >= thr).mean()) if m.any() else None
            res[f"n_tokens_{lo}_{hi}"] = int(m.sum())
        out[name] = res
        print(name, json.dumps(res))
        if model is not None:
            del model
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    json.dump(out, open(os.path.join(RESULTS, "diag_fragments.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
