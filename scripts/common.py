"""Shared helpers: corpus loading, sliding-window scoring, metrics with bootstrap CIs."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "corpus")
RESULTS = os.path.join(ROOT, "results")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "hf_home"))

EVAL_SPLITS = ["test_public", "test_bipia", "test_promptshield", "test_synth_id", "test_synth_ood_style",
               "test_synth_ood_carrier", "test_synth_ood_both", "test_hardneg", "ext_injecagent", "ext_agentdojo"]
WIN, STRIDE = 320, 224  # tokens per window, step between window starts


def load(split):
    with open(os.path.join(CORPUS, f"{split}.jsonl")) as f:
        return [json.loads(l) for l in f]


def device():
    import torch
    torch.set_num_threads(4)  # shared laptop: see _shared/RULES.md
    return "mps" if torch.backends.mps.is_available() else "cpu"


def windows(tok, texts, max_len=WIN, stride=STRIDE):
    """Tokenize into overlapping windows. Returns (encodings, doc_index per window, char offsets)."""
    enc = tok(texts, truncation=True, max_length=max_len, stride=max_len - stride, return_overflowing_tokens=True,
              return_offsets_mapping=True, padding=False)
    return enc


def score_texts(model, tok, texts, max_len=WIN, stride=STRIDE, bs=32, pos_index=1, max_windows=12):
    """P(injection) per text = max over its windows (capped at max_windows per text, beginning + end kept)."""
    import torch
    dev = next(model.parameters()).device
    enc = windows(tok, texts, max_len, stride)
    doc = list(enc["overflow_to_sample_mapping"])
    ids = enc["input_ids"]
    # cap very long docs: keep first and last windows
    by_doc = {}
    for i, d in enumerate(doc):
        by_doc.setdefault(d, []).append(i)
    keep = []
    for d, idx in by_doc.items():
        if len(idx) > max_windows:
            idx = idx[: max_windows // 2] + idx[-(max_windows // 2):]
        keep += idx
    keep.sort(key=lambda i: len(ids[i]))  # length-sorted batches = less padding
    probs = np.zeros(len(ids), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for b in range(0, len(keep), bs):
            bi = keep[b: b + bs]
            batch = tok.pad({"input_ids": [ids[i] for i in bi]}, return_tensors="pt")
            batch = {k: v.to(dev) for k, v in batch.items()}
            logits = model(**batch).logits.float()
            p = torch.softmax(logits, -1)[:, pos_index].cpu().numpy()
            probs[bi] = p
    out = np.zeros(len(texts), dtype=np.float32)
    for i in keep:
        out[doc[i]] = max(out[doc[i]], probs[i])
    return out


def _rates(y, s, t):
    pred = s >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
    return dict(precision=prec, recall=rec, fpr=fpr, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) == 2 else float("nan")


def metrics(y, s, thresholds, n_boot=1000, seed=0):
    """Metrics at each named threshold, plus AUROC; 95% bootstrap CIs for recall, FPR and AUROC."""
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = {"n": int(len(y)), "pos": int(y.sum()), "auroc": auroc(y, s)}
    boots = [rng.integers(0, len(y), len(y)) for _ in range(n_boot)]
    out["auroc_ci"] = _ci([auroc(y[b], s[b]) for b in boots])
    for name, t in thresholds.items():
        m = _rates(y, s, t)
        m["threshold"] = float(t)
        bs = [_rates(y[b], s[b], t) for b in boots]
        m["recall_ci"] = _ci([b["recall"] for b in bs])
        m["fpr_ci"] = _ci([b["fpr"] for b in bs])
        out[name] = m
    return out


def _ci(v):
    v = np.array([x for x in v if x == x])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2


def threshold_at_fpr(y, s, target):
    """Smallest threshold whose FPR on (y, s) is <= target."""
    neg = np.sort(np.asarray(s)[np.asarray(y) == 0])[::-1]
    k = int(np.floor(target * len(neg)))
    return float(neg[k]) + 1e-6 if k < len(neg) else 0.0


def save_scores(name, split, rows, scores):
    d = os.path.join(RESULTS, "scores", name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{split}.jsonl"), "w") as f:
        for r, s in zip(rows, scores):
            f.write(json.dumps({"label": r["label"], "score": float(s), "source": r["source"],
                                "style": r.get("style"), "carrier": r.get("carrier")}) + "\n")


class OnnxScorer:
    """The shipped int8 ONNX model (site/model) on CPU via onnxruntime: same windowing / max rule as score_texts.
    Not torch, so it needs no GPU lock; 4 intra-op threads to keep the shared laptop responsive."""

    def __init__(self, path=os.path.join(ROOT, "site", "model", "injection-shield")):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(os.path.join(path, "onnx", "model_quantized.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.names = [i.name for i in self.sess.get_inputs()]
        self.tok = AutoTokenizer.from_pretrained(path)
        meta = json.load(open(os.path.join(os.path.dirname(path), "meta.json")))
        self.thresholds = meta["thresholds"]

    def __call__(self, texts, bs=16):
        enc = windows(self.tok, texts)
        doc, ids = list(enc["overflow_to_sample_mapping"]), enc["input_ids"]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        probs = np.zeros(len(ids), dtype=np.float32)
        for b in range(0, len(order), bs):
            bi = order[b: b + bs]
            batch = self.tok.pad({"input_ids": [ids[i] for i in bi]}, return_tensors="np")
            feed = {n: batch[n].astype(np.int64) if n in batch else np.zeros_like(batch["input_ids"], dtype=np.int64)
                    for n in self.names}
            lo = self.sess.run(None, feed)[0]
            e = np.exp(lo - lo.max(-1, keepdims=True))
            probs[bi] = e[:, 1] / e.sum(-1)
        out = np.zeros(len(texts), dtype=np.float32)
        for i, d in enumerate(doc):
            out[d] = max(out[d], probs[i])
        return out
