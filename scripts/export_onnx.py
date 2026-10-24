"""Export a fine-tuned classifier to ONNX, quantise to int8, check parity, and lay it out for transformers.js.

  python scripts/export_onnx.py --model models/minilm-s0 --out site/model/injection-shield
Writes config.json, tokenizer.json, tokenizer_config.json, onnx/model.onnx (fp32, not committed) and
onnx/model_quantized.onnx (int8), plus site/model/meta.json and results/onnx_parity.json.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, ROOT, load, metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="site/model/injection-shield")
    ap.add_argument("--n_check", type=int, default=600)
    args = ap.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    src, out = os.path.join(ROOT, args.model), os.path.join(ROOT, args.out)
    os.makedirs(os.path.join(out, "onnx"), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(src)
    assert tok.is_fast, "transformers.js needs tokenizer.json (fast tokenizer)"
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    model.config.id2label, model.config.label2id = {0: "SAFE", 1: "INJECTION"}, {"SAFE": 0, "INJECTION": 1}
    model.config.save_pretrained(out)
    tok.save_pretrained(out)
    for f in ["vocab.txt", "special_tokens_map.json"]:
        p = os.path.join(out, f)
        if os.path.exists(p) and f == "vocab.txt":
            os.remove(p)  # tokenizer.json carries the vocab

    names = ["input_ids", "attention_mask"] + (["token_type_ids"] if "token_type_ids" in tok.model_input_names else [])
    dummy = tok(["hello world", "a slightly longer second example"], padding=True, return_tensors="pt")
    fp32 = os.path.join(out, "onnx", "model.onnx")

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *xs):
            return self.m(**dict(zip(names, xs))).logits

    torch.onnx.export(Wrap(model), tuple(dummy[n] for n in names), fp32, input_names=names, output_names=["logits"],
                      dynamic_axes={**{n: {0: "batch", 1: "seq"} for n in names}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    q8 = os.path.join(out, "onnx", "model_quantized.onnx")
    quantize_dynamic(fp32, q8, weight_type=QuantType.QInt8)

    # parity: torch fp32 vs onnx int8 on a sample of the hardest eval splits (first window only)
    import onnxruntime as ort
    sess = ort.InferenceSession(q8, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    rows = []
    for split in ["test_synth_id", "test_synth_ood_style", "ext_injecagent", "test_public"]:
        try:
            rs = load(split)
        except FileNotFoundError:
            continue
        rows += [rs[i] for i in rng.choice(len(rs), min(len(rs), args.n_check // 4), replace=False)]
    pt, ox = [], []
    for i in range(0, len(rows), 16):
        b = tok([r["text"] for r in rows[i:i + 16]], truncation=True, max_length=320, padding=True,
                return_tensors="pt")
        with torch.no_grad():
            pt += torch.softmax(model(**b).logits, -1)[:, 1].tolist()
        lo = sess.run(None, {n: b[n].numpy().astype(np.int64) for n in names})[0]
        e = np.exp(lo - lo.max(-1, keepdims=True))
        ox += (e[:, 1] / e.sum(-1)).tolist()
    pt, ox, y = np.array(pt), np.array(ox), np.array([r["label"] for r in rows])
    thr = json.load(open(os.path.join(src, "train_info.json")))["thresholds"]
    parity = {"n": len(rows), "mean_abs_diff": float(np.abs(pt - ox).mean()), "max_abs_diff": float(np.abs(pt - ox).max()),
              "decision_agreement_fpr1": float(np.mean((pt >= thr["fpr1"]) == (ox >= thr["fpr1"]))),
              "torch_fp32": metrics(y, pt, thr, n_boot=0), "onnx_int8": metrics(y, ox, thr, n_boot=0),
              "size_mb_fp32": os.path.getsize(fp32) / 2**20, "size_mb_int8": os.path.getsize(q8) / 2**20}
    json.dump(parity, open(os.path.join(RESULTS, "onnx_parity.json"), "w"), indent=1)
    meta = {"base": json.load(open(os.path.join(src, "train_info.json")))["args"]["model"].split("/")[-1],
            "thresholds": thr, "size_mb": parity["size_mb_int8"], "window": 320, "stride": 224}
    json.dump(meta, open(os.path.join(os.path.dirname(out), "meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in parity.items() if not isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
