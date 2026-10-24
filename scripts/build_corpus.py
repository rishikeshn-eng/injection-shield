"""Assemble the training corpus and every evaluation split into data/corpus/*.jsonl.

Each row: {"text", "label" (1 = injection/jailbreak), "source", "family", "split", ...metadata}.
family: direct (user-turn jailbreak/injection), indirect (instruction hidden in data/tool output), benign.

Splits
  train / val                 training pool (public train splits + BIPIA train + synthetic train styles/carriers)
  test_public                 held-out test splits of the public direct-injection datasets
  test_synth_id               synthetic pairs: training styles x training carriers, unseen documents
  test_synth_ood_style        synthetic pairs: 4 attack styles never seen in training
  test_synth_ood_carrier      synthetic pairs: training styles in 3 carrier formats never seen in training
  test_synth_ood_both         unseen style AND unseen carrier
  test_hardneg                benign hard negatives (imperative docs, security articles, ...)
  test_bipia                  BIPIA test contexts x test attacks (+ clean test contexts)
  test_promptshield           PromptShield test split (sampled)
  ext_injecagent / ext_agentdojo   external indirect-injection benchmarks never used in training (built elsewhere)
"""
import glob
import json
import os
import random
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
OUT = os.path.join(D, "corpus")
rng = random.Random(0)
rows = []


def add(text, label, source, family, split, **meta):
    if isinstance(text, str) and len(text.strip()) >= 8:
        rows.append({"text": text.strip(), "label": int(label), "source": source, "family": family,
                     "split": split, **meta})


def pq(pattern):
    return pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(D, "hf", pattern)))])


def split_of(f):
    return "test" if "test" in f else ("val" if "validation" in f else "train")


# ---- public direct-injection / jailbreak datasets ----
for f in glob.glob(os.path.join(D, "hf/deepset/*.parquet")):
    for r in pd.read_parquet(f).itertuples():
        add(r.text, r.label, "deepset", "direct" if r.label else "benign", split_of(f))

for name in ["gandalf_ignore_instructions", "gandalf_summarization"]:
    for f in glob.glob(os.path.join(D, f"hf/{name}/*.parquet")):
        for r in pd.read_parquet(f).itertuples():
            add(r.text, 1, name, "direct", split_of(f))

for f in glob.glob(os.path.join(D, "hf/mosscap_prompt_injection/*.parquet")):
    df = pd.read_parquet(f, columns=["prompt"]).drop_duplicates("prompt")
    df = df.sample(n=min(len(df), 2500 if "train" in f else 400), random_state=0)
    for t in df.prompt:
        add(t, 1, "mosscap", "direct", split_of(f))

for f in glob.glob(os.path.join(D, "hf/safe-guard-prompt-injection/*.parquet")):
    for r in pd.read_parquet(f).itertuples():
        add(r.text, r.label, "safeguard", "direct" if r.label else "benign", split_of(f))

for f in ["train", "test"]:
    df = pd.read_csv(os.path.join(D, f"hf/jailbreak-classification/default_jailbreak_dataset_{f}.csv"))
    for r in df.itertuples():
        lab = int(r.type == "jailbreak")
        add(r.prompt, lab, "jackhhao", "direct" if lab else "benign", f)

tt = []
for f in glob.glob(os.path.join(D, "gh/tensortrust/*.jsonl")):
    for line in open(f):
        tt.append(json.loads(line)["attack"])
tt = sorted(set(tt))
rng.shuffle(tt)
for i, t in enumerate(tt):
    add(t, 1, "tensortrust", "direct", "test" if i % 5 == 0 else "train")

# PromptShield: train subsample for training, test subsample as its own benchmark
ps_tr = json.load(open(os.path.join(D, "hf/PromptShield/train.json")))
ps_te = json.load(open(os.path.join(D, "hf/PromptShield/test.json")))
for r in rng.sample(ps_tr, 6000):
    add(r["prompt"], r["label"], "promptshield", "mixed" if r["label"] else "benign", "train")
for r in rng.sample(ps_te, 4000):
    add(r["prompt"], r["label"], "promptshield", "mixed" if r["label"] else "benign", "test_promptshield")

# public "test" -> test_public, "val" -> val
for r in rows:
    if r["split"] == "test":
        r["split"] = "test_public"

# ---- BIPIA: insert attack strings into email/table/code contexts (start / middle / end), as in the paper ----
bip = os.path.join(D, "gh/BIPIA/benchmark")


def flat_attacks(path):
    j = json.load(open(path))
    return [(cat, a) for cat, lst in j.items() for a in lst]


def insert(ctx, attack, pos):
    if pos == "start":
        return attack + "\n" + ctx
    if pos == "end":
        return ctx + "\n" + attack
    sents = ctx.split(". ")
    k = len(sents) // 2
    return ". ".join(sents[:k]) + ". " + attack + " " + ". ".join(sents[k:])


for part in ["train", "test"]:
    text_att = flat_attacks(f"{bip}/text_attack_{part}.json")
    code_att = flat_attacks(f"{bip}/code_attack_{part}.json")
    split = "train" if part == "train" else "test_bipia"
    for task in ["email", "table", "code"]:
        ctxs = [json.loads(l)["context"] for l in open(f"{bip}/{task}/{part}.jsonl")]
        ctxs = ["\n".join(c) if isinstance(c, list) else c for c in ctxs]
        ctxs = list(dict.fromkeys(c for c in ctxs if isinstance(c, str)))
        atts = code_att if task == "code" else text_att
        n_inj = 4 if part == "train" else 3
        for c in ctxs[: (150 if task == "table" else 50)]:
            add(c, 0, f"bipia_{task}", "benign", split)
            for cat, a in rng.sample(atts, n_inj):
                add(insert(c, a, rng.choice(["start", "middle", "end"])), 1, f"bipia_{task}", "indirect", split,
                    style=cat, span=a.strip())

# ---- synthetic pairs ----
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from synthesize import HELDOUT_CARRIERS, HELDOUT_STYLES  # noqa: E402

pairs = [json.loads(l) for f in ["pairs.jsonl", "pairs_v1.jsonl"] if os.path.exists(os.path.join(D, "synth", f))
         for l in open(os.path.join(D, "synth", f))]
groups = sorted({p["group"] for p in pairs})
rng.shuffle(groups)
gsplit = {}
for i, g in enumerate(groups):
    gsplit[g] = "train" if i % 10 < 8 else ("val" if i % 10 == 8 else "test_synth_id")
for p in pairs:
    hs, hc = p["style"] in HELDOUT_STYLES, p["carrier"] in HELDOUT_CARRIERS
    if hs and hc:
        split = "test_synth_ood_both"
    elif hs:
        split = "test_synth_ood_style"
    elif hc:
        split = "test_synth_ood_carrier"
    else:
        split = gsplit[p["group"]]
    meta = dict(carrier=p["carrier"], style=p["style"], pair=p["pair"])
    add(p["clean"], 0, "synth", "benign", split, **meta)
    add(p["injected"], 1, "synth", "indirect", split, **meta,
        span=p["injection"].strip() if p["span_verbatim"] else None)

ben = [json.loads(l) for f in ["benign_extra.jsonl", "benign_v1.jsonl"] if os.path.exists(os.path.join(D, "synth", f))
       for l in open(os.path.join(D, "synth", f))]
for b in ben:
    split = "test_hardneg" if (b["carrier"] in HELDOUT_CARRIERS or int(b["group"].split("|")[1]) % 5 == 0) else "train"
    add(b["text"], 0, "synth_hardneg", "benign", split, carrier=b["carrier"], flavour=b["flavour"])

# ---- dedupe (exact text), test splits win over train ----
prio = lambda s: 0 if s.startswith("test") else (1 if s == "val" else 2)  # noqa: E731
rows.sort(key=lambda r: prio(r["split"]))
seen, final = set(), []
for r in rows:
    k = r["text"].lower()
    if k in seen:
        continue
    seen.add(k)
    final.append(r)

# carve a val split out of public train rows (synthetic val already assigned by group)
for r in final:
    if r["split"] == "train" and r["source"] != "synth" and rng.random() < 0.06:
        r["split"] = "val"

os.makedirs(OUT, exist_ok=True)
by = {}
for r in final:
    by.setdefault(r["split"], []).append(r)
stats = {}
for s, rs in sorted(by.items()):
    with open(os.path.join(OUT, f"{s}.jsonl"), "w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")
    c = Counter(r["label"] for r in rs)
    stats[s] = {"n": len(rs), "pos": c[1], "neg": c[0],
                "sources": dict(Counter(r["source"] for r in rs).most_common())}
    print(f"{s:28s} n={len(rs):6d} pos={c[1]:6d} neg={c[0]:6d}  {dict(Counter(r['source'] for r in rs).most_common(6))}")
json.dump(stats, open(os.path.join(ROOT, "results", "corpus_stats.json"), "w"), indent=1)
