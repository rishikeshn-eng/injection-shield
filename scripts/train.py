"""Fine-tune a small transformer as a window-level injection classifier.

Usage: python scripts/train.py --model microsoft/deberta-v3-small --name deberta-small --seed 0
Must run under the shared GPU lock (see RULES.md):  _shared/gpulock.sh .venv/bin/python scripts/train.py ...

Window labelling: documents are split into WIN-token windows. A window of an injected document whose injection span
is known (synthetic, BIPIA) is positive only if it covers >= 60% of the span, negative if it covers none of it, and
dropped if in between. Rows without a known span (public direct-injection sets) use their first window and label.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, ROOT, STRIDE, WIN, device, load, metrics, score_texts, threshold_at_fpr  # noqa: E402

MAX_NEG_WINDOWS_PER_DOC = 3


def window_examples(tok, rows, rng):
    texts = [r["text"] for r in rows]
    enc = tok(texts, truncation=True, max_length=WIN, stride=WIN - STRIDE, return_overflowing_tokens=True,
              return_offsets_mapping=True)
    by_doc = {}
    for i, d in enumerate(enc["overflow_to_sample_mapping"]):
        by_doc.setdefault(d, []).append(i)
    ex = []
    for d, idx in by_doc.items():
        r = rows[d]
        span = r.get("span")
        start = r["text"].find(span) if span else -1
        if r["label"] == 1 and start >= 0:
            end = start + len(span)
            for i in idx:
                offs = [o for o in enc["offset_mapping"][i] if o[1] > 0]
                lo, hi = offs[0][0], offs[-1][1]
                cover = max(0, min(hi, end) - max(lo, start)) / max(1, end - start)
                if cover >= 0.6:
                    ex.append((enc["input_ids"][i], 1))
                elif cover == 0:
                    ex.append((enc["input_ids"][i], 0))
        elif r["label"] == 1:
            ex.append((enc["input_ids"][idx[0]], 1))
        else:
            for i in (idx if len(idx) <= MAX_NEG_WINDOWS_PER_DOC else rng.sample(idx, MAX_NEG_WINDOWS_PER_DOC)):
                ex.append((enc["input_ids"][i], 0))
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--no_synth", action="store_true", help="ablation: train without synthetic indirect data")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = device()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(dev)

    train_rows, val_rows = load("train"), load("val")
    if args.no_synth:
        train_rows = [r for r in train_rows if not r["source"].startswith("synth")]
        val_rows = [r for r in val_rows if not r["source"].startswith("synth")]
    ex = window_examples(tok, train_rows, rng)
    n_pos = sum(l for _, l in ex)
    print(f"train docs={len(train_rows)} windows={len(ex)} pos={n_pos} neg={len(ex) - n_pos}", flush=True)

    # length-bucketed shuffled batches
    def batches():
        idx = list(range(len(ex)))
        rng.shuffle(idx)
        chunks = [idx[i:i + args.bs * 50] for i in range(0, len(idx), args.bs * 50)]
        out = []
        for c in chunks:
            c.sort(key=lambda i: len(ex[i][0]))
            out += [c[i:i + args.bs] for i in range(0, len(c), args.bs)]
        rng.shuffle(out)
        return out

    steps_per_epoch = math.ceil(len(ex) / args.bs)
    total = int(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    w = torch.tensor([1.0, (len(ex) - n_pos) / max(1, n_pos)], device=dev).clamp(max=3.0)
    lossf = torch.nn.CrossEntropyLoss(weight=w)

    step, t0 = 0, time.time()
    model.train()
    while step < total:
        for b in batches():
            if step >= total:
                break
            batch = tok.pad({"input_ids": [ex[i][0] for i in b]}, return_tensors="pt")
            batch = {k: v.to(dev) for k, v in batch.items()}
            y = torch.tensor([ex[i][1] for i in b], device=dev)
            loss = lossf(model(**batch).logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)

    out_dir = os.path.join(ROOT, "models", f"{args.name}-s{args.seed}")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    # thresholds from validation: default 0.5 and the threshold giving 1% FPR on val
    vs = score_texts(model, tok, [r["text"] for r in val_rows])
    vy = np.array([r["label"] for r in val_rows])
    thr = {"t0.5": 0.5, "fpr1": threshold_at_fpr(vy, vs, 0.01)}
    json.dump({"thresholds": thr, "val": metrics(vy, vs, thr, n_boot=200), "train_windows": len(ex),
               "seconds": time.time() - t0, "args": vars(args)},
              open(os.path.join(out_dir, "train_info.json"), "w"), indent=1)
    print("val", json.dumps(metrics(vy, vs, thr, n_boot=0)), flush=True)
    print("saved", out_dir)


if __name__ == "__main__":
    main()
