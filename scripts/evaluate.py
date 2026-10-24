"""Score every eval split with one detector and write results/metrics/<name>.json.

  python scripts/evaluate.py --kind hf --path models/deberta-small-s0 --name deberta-small-s0
  python scripts/evaluate.py --kind protectai --name protectai-v2
  python scripts/evaluate.py --kind keyword --name keyword
Thresholds: "t0.5" (score >= 0.5; for keyword rules: >= 1 match) and "fpr1" (calibrated to 1% FPR on our val split).
"""
import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import EVAL_SPLITS, RESULTS, ROOT, device, load, metrics, save_scores, score_texts, threshold_at_fpr  # noqa

# classic heuristic filter (in the spirit of rebuff / LLM-guard "ban substrings" / vigil YARA rules)
KEYWORD_PATTERNS = [
    r"ignore (all |any |the |your )?(previous|prior|above|earlier|preceding) (instructions|prompts|directions|rules)",
    r"disregard (all |any |the |your )?(previous|prior|above|earlier)",
    r"forget (all |everything|your|the) (previous|prior|above|instructions)",
    r"(new|updated|revised) (instructions|directive|system prompt)",
    r"system prompt", r"\[\s*system\s*\]", r"<\|im_start\|>", r"###\s*(system|instruction)",
    r"you are now", r"act as (an? )?(unrestricted|dan|jailbroken)", r"\bdan\b.*do anything now", r"do anything now",
    r"jailbreak", r"developer mode", r"override (your|all|the) (instructions|rules|safety)",
    r"important!{2,}", r"<information>", r"(ai|assistant|language model)s?,? (must|should|are instructed to)",
    r"note to (the )?(ai|assistant|llm|language model)", r"reveal (your|the) (password|secret|instructions|prompt)",
    r"(print|repeat|output) (your|the) (instructions|prompt|password)",
]
KW = [re.compile(p, re.I | re.S) for p in KEYWORD_PATTERNS]


def keyword_scores(texts):
    return np.array([sum(bool(k.search(t)) for k in KW) for t in texts], dtype=np.float32)


def get_scorer(args):
    if args.kind == "keyword":
        return keyword_scores, {"t0.5": 1.0}
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if args.kind == "protectai":
        path = "protectai/deberta-v3-base-prompt-injection-v2"
        win, stride, pos = 512, 384, 1  # id2label {0: SAFE, 1: INJECTION}
    else:
        path = os.path.join(ROOT, args.path) if not os.path.isabs(args.path) else args.path
        win, stride, pos = 320, 224, 1
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device())
    if args.kind == "protectai":
        assert model.config.id2label[1].upper().startswith("INJ"), model.config.id2label
    fixed = {"t0.5": 0.5}
    info = os.path.join(path, "train_info.json") if args.kind == "hf" else None
    if info and os.path.exists(info):
        fixed = json.load(open(info))["thresholds"]
    return (lambda texts: score_texts(model, tok, texts, max_len=win, stride=stride, pos_index=pos, bs=16)), fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["hf", "protectai", "keyword"], required=True)
    ap.add_argument("--path")
    ap.add_argument("--name", required=True)
    ap.add_argument("--splits", default=",".join(EVAL_SPLITS))
    args = ap.parse_args()
    score, thr = get_scorer(args)
    # every detector is calibrated on the same (full) validation split: fpr1 = 1% FPR on val negatives
    if True:
        val = load("val")
        vs = score([r["text"] for r in val])
        thr["fpr1"] = threshold_at_fpr(np.array([r["label"] for r in val]), vs, 0.01)
    if args.kind == "hf":
        p = args.path if os.path.isabs(args.path) else os.path.join(ROOT, args.path)
        json.dump(thr, open(os.path.join(p, "thresholds.json"), "w"), indent=1)
    res = {"thresholds": thr, "splits": {}}
    from common import CORPUS
    for split in args.splits.split(","):
        if not os.path.exists(os.path.join(CORPUS, f"{split}.jsonl")):
            print("skip (missing)", split)
            continue
        rows = load(split)
        s = score([r["text"] for r in rows])
        save_scores(args.name, split, rows, s)
        y = np.array([r["label"] for r in rows])
        res["splits"][split] = metrics(y, s, thr)
        m = res["splits"][split]
        print(f"{split:26s} n={m['n']:5d} auroc={m['auroc']:.3f} "
              + " ".join(f"{k}: R={m[k]['recall']:.3f} FPR={m[k]['fpr']:.3f} P={m[k]['precision']:.3f}"
                         for k in thr), flush=True)
    os.makedirs(os.path.join(RESULTS, "metrics"), exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS, "metrics", f"{args.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
