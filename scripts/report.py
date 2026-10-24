"""Aggregate results/metrics/*.json (+ agent summaries) into results/tables.md, figures and site/results.json."""
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, ROOT, metrics  # noqa: E402

SITE = os.path.join(ROOT, "site")
SPLIT_LABELS = {
    "test_public": "Public direct (test)", "test_promptshield": "PromptShield", "test_bipia": "BIPIA (test)",
    "test_synth_id": "Synthetic ID", "test_synth_ood_style": "OOD style", "test_synth_ood_carrier": "OOD carrier",
    "test_synth_ood_both": "OOD style+carrier", "ext_injecagent": "InjecAgent", "ext_agentdojo": "AgentDojo",
    "test_hardneg": "Hard negatives",
}
RECALL_SPLITS = ["test_public", "test_bipia", "test_synth_id", "test_synth_ood_style", "test_synth_ood_carrier",
                 "test_synth_ood_both", "ext_injecagent", "ext_agentdojo"]
FPR_SPLITS = ["test_public", "test_promptshield", "test_synth_id", "test_synth_ood_style", "test_hardneg",
              "ext_injecagent", "ext_agentdojo"]
DETECTORS = [  # (metrics file stem or glob for seeds, label)
    ("keyword", "Keyword rules"),
    ("protectai-v2", "ProtectAI deberta-v3-base-injection-v2 (184M)"),
    ("llm-judge", "LLM judge (Gemma-4-31B via Gemini API)*"),
    ("distilroberta-publiconly-s*", "DistilRoBERTa, public data only"),
    ("distilroberta-s*", "DistilRoBERTa, + synthetic (ours)"),
    ("minilm-s*", "MiniLM-L12, + synthetic (ours, browser model)"),
]


def load_detector(stem):
    files = sorted(glob.glob(os.path.join(RESULTS, "metrics", stem + ".json")))
    return [json.load(open(f)) for f in files]


def agg(runs, split, thr, key):
    """Mean over seeds; CI = bootstrap CI of the single seed, or min/max-of-CI envelope across seeds."""
    ms = [r["splits"][split][thr] for r in runs if split in r["splits"] and thr in r["splits"][split]]
    if not ms:
        return None
    vals = [m[key] for m in ms]
    cis = [m.get(key + "_ci", [np.nan, np.nan]) for m in ms]
    out = {key: float(np.mean(vals)), key + "_ci": [float(np.nanmin([c[0] for c in cis])),
                                                     float(np.nanmax([c[1] for c in cis]))], "seeds": len(ms)}
    if len(ms) > 1:
        out[key + "_sd"] = float(np.std(vals, ddof=1))
    return out


def pct(x):
    return "–" if x is None or x != x else f"{100 * x:.1f}"


def main():
    rows, site_det = [], []
    table = {}
    for stem, label in DETECTORS:
        runs = load_detector(stem)
        if not runs:
            continue
        thr = "fpr1" if "fpr1" in runs[0]["thresholds"] else "t0.5"
        splits = {}
        for s in set(RECALL_SPLITS + FPR_SPLITS):
            r, f = agg(runs, s, thr, "recall"), agg(runs, s, thr, "fpr")
            au = [x["splits"][s]["auroc"] for x in runs if s in x["splits"]]
            if r or f:
                splits[s] = {**(r or {}), **(f or {}), "auroc": float(np.mean(au)) if au else None}
        table[label] = (thr, splits, len(runs))
        site_det.append({"label": label + (f" ({len(runs)} seeds)" if len(runs) > 1 else ""), "splits": splits,
                         "threshold": thr})

    def md_table(splits, key):
        head = "| Detector | " + " | ".join(SPLIT_LABELS[s] for s in splits) + " |"
        sep = "|---|" + "---:|" * len(splits)
        lines = [head, sep]
        for label, (thr, sp, n) in table.items():
            cells = []
            for s in splits:
                m = sp.get(s)
                if not m or key not in m:
                    cells.append("–")
                    continue
                c = m.get(key + "_ci")
                sd = f" ±{100 * m[key + '_sd']:.1f}" if key + "_sd" in m else ""
                cells.append(f"{pct(m[key])}{sd} [{pct(c[0])}–{pct(c[1])}]" if c and c[0] == c[0] else pct(m[key]))
            lines.append(f"| {label} ({thr}) | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    md = ["## Recall (TPR, %) at the 1%-FPR-on-validation threshold\n", md_table(RECALL_SPLITS, "recall"),
          "\n\n## False-positive rate (%) on benign documents at the same threshold\n", md_table(FPR_SPLITS, "fpr")]
    # AUROC table
    head = "| Detector | " + " | ".join(SPLIT_LABELS[s] for s in RECALL_SPLITS) + " |"
    md += ["\n\n## AUROC\n", head, "|---|" + "---:|" * len(RECALL_SPLITS)]
    for label, (thr, sp, n) in table.items():
        md.append(f"| {label} | " + " | ".join(
            "–" if not sp.get(s) or sp[s].get("auroc") is None else f"{sp[s]['auroc']:.3f}" for s in RECALL_SPLITS) + " |")

    # paired comparison with the LLM judge on its subsample
    sub_dir = os.path.join(RESULTS, "judge_subsample")
    figs = []
    os.makedirs(os.path.join(SITE, "fig"), exist_ok=True)

    # ---- figure 1: recall at 1% FPR, grouped bars ----
    show = [l for l in table if "LLM judge" not in l]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.8 / max(1, len(show))
    colors = ["#9aa0a6", "#e8710a", "#1a73e8", "#a142f4", "#12b5cb", "#f9ab00", "#d01884"]
    for i, l in enumerate(show):
        sp = table[l][1]
        v = [sp.get(s, {}).get("recall", np.nan) for s in RECALL_SPLITS]
        ax.bar(np.arange(len(RECALL_SPLITS)) + i * w, v, w, label=l, color=colors[i % len(colors)])
    ax.set_xticks(np.arange(len(RECALL_SPLITS)) + w * (len(show) - 1) / 2)
    ax.set_xticklabels([SPLIT_LABELS[s] for s in RECALL_SPLITS], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Recall at 1% val FPR")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.5, ncol=2, loc="lower left", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(SITE, "fig", "recall.png"), dpi=150)
    figs.append({"title": "Recall by evaluation set", "src": "fig/recall.png",
                 "caption": "Same operating point for every detector: the threshold that gives 1% false positives "
                            "on the validation split."})

    # ---- figure 2: per-style recall of the main model vs the public-only model (synthetic test splits) ----
    def style_recall(name, thr):
        d = defaultdict(lambda: [0, 0])
        for s in ["test_synth_id", "test_synth_ood_style", "test_synth_ood_carrier", "test_synth_ood_both"]:
            p = os.path.join(RESULTS, "scores", name, s + ".jsonl")
            if not os.path.exists(p):
                continue
            for line in open(p):
                r = json.loads(line)
                if r["label"] == 1:
                    d[r["style"]][0] += r["score"] >= thr
                    d[r["style"]][1] += 1
        return {k: a / b for k, (a, b) in d.items() if b}

    per_style = {}
    for stem, label in [("keyword", "Keyword"), ("protectai-v2", "ProtectAI"),
                        ("distilroberta-publiconly-s0", "DistilRoBERTa public-only"), ("distilroberta-s0", "DistilRoBERTa + synth"),
                        ("minilm-s0", "MiniLM + synth")]:
        f = os.path.join(RESULTS, "metrics", stem + ".json")
        if os.path.exists(f):
            m = json.load(open(f))
            per_style[label] = style_recall(stem, m["thresholds"].get("fpr1", m["thresholds"]["t0.5"]))
    if per_style:
        from synthesize import HELDOUT_STYLES, STYLES
        styles = [s for s in STYLES if s not in HELDOUT_STYLES] + [s for s in STYLES if s in HELDOUT_STYLES]
        M = np.array([[per_style[k].get(s, np.nan) for s in styles] for k in per_style])
        fig, ax = plt.subplots(figsize=(11, 0.55 * len(per_style) + 1.6))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(styles)))
        ax.set_xticklabels([s + (" (held-out)" if s in HELDOUT_STYLES else "") for s in styles], rotation=30,
                           ha="right", fontsize=8.5)
        ax.set_yticks(range(len(per_style)))
        ax.set_yticklabels(list(per_style), fontsize=9)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] == M[i, j]:
                    ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.02)
        fig.tight_layout()
        fig.savefig(os.path.join(SITE, "fig", "styles.png"), dpi=150)
        figs.append({"title": "Recall per attack style (synthetic test sets)", "src": "fig/styles.png",
                     "caption": "Right-hand four styles are never seen in training. Threshold: 1% val FPR."})
        md += ["\n\n## Recall per attack style (synthetic test sets, %)\n",
               "| Detector | " + " | ".join(styles) + " |", "|---|" + "---:|" * len(styles)]
        for k, v in per_style.items():
            md.append(f"| {k} | " + " | ".join(pct(v.get(s)) for s in styles) + " |")

    # ---- judge paired comparison ----
    if os.path.isdir(sub_dir) and os.path.exists(os.path.join(RESULTS, "metrics", "llm-judge.json")):
        md += ["\n\n## Paired comparison on the LLM-judge subsample (recall / FPR, %)\n",
               "The judge scores 8 documents per request on <=160 documents per split (free-tier quota). Ours and "
               "the judge are evaluated on the identical documents. Judge threshold 0.5; ours: 1% val FPR.\n",
               "| Split | n | Judge recall | Judge FPR | Ours recall | Ours FPR |", "|---|---:|---:|---:|---:|---:|"]
        ours = os.path.join(RESULTS, "scores", "distilroberta-s0")
        thr_ours = json.load(open(os.path.join(RESULTS, "metrics", "distilroberta-s0.json")))["thresholds"]["fpr1"] \
            if os.path.exists(os.path.join(RESULTS, "metrics", "distilroberta-s0.json")) else None
        judge_rows = {}
        for f in sorted(glob.glob(os.path.join(sub_dir, "*.jsonl"))):
            split = os.path.basename(f)[:-6]
            sub = [json.loads(l) for l in open(f)]
            sub = [r for r in sub if r["judge"] == r["judge"]]
            y = np.array([r["label"] for r in sub])
            js = np.array([r["judge"] for r in sub])
            jm = metrics(y, js, {"t": 0.5}, n_boot=0)["t"]
            om = {"recall": None, "fpr": None}
            sf = os.path.join(ours, split + ".jsonl")
            if thr_ours is not None and os.path.exists(sf):
                from common import load as load_split
                txt2s = {r["text"]: json.loads(l)["score"] for r, l in zip(load_split(split), open(sf))}
                os_ = np.array([txt2s.get(r["text"], np.nan) for r in sub])
                ok = ~np.isnan(os_)
                om = metrics(y[ok], os_[ok], {"t": thr_ours}, n_boot=0)["t"]
            judge_rows[split] = {"n": len(sub), "judge": jm, "ours": om}
            md.append(f"| {SPLIT_LABELS.get(split, split)} | {len(sub)} | {pct(jm['recall'])} | {pct(jm['fpr'])} | "
                      f"{pct(om.get('recall'))} | {pct(om.get('fpr'))} |")

    # ---- agent ----
    agent = None
    for f in sorted(glob.glob(os.path.join(RESULTS, "agent", "summary_*.json"))):
        a = json.load(open(f))
        md += [f"\n\n## End-to-end agent ({a['agent']}, shield {os.path.basename(a['model'])}, threshold "
               f"{a['threshold']} = {a['threshold_value']:.3f})\n",
               "| Shield | Attack success ↓ | ASR direct-harm | ASR data-stealing | Utility benign ↑ | "
               "Utility under attack ↑ | Benign flagged |", "|---|---:|---:|---:|---:|---:|---:|"]
        rows_a = []
        for mode, s in a["modes"].items():
            def f2(x):
                return f"{pct(x['rate'])} [{pct(max(0, x['ci95'][0]))}–{pct(min(1, x['ci95'][1]))}] (n={x['n']})"
            md.append(f"| {mode} | {f2(s['attack_success'])} | {pct(s['attack_success_dh']['rate'])} | "
                      f"{pct(s['attack_success_ds']['rate'])} | {f2(s['utility_benign'])} | "
                      f"{f2(s['utility_under_attack'])} | {pct(s['benign_flag_rate']['rate'])} |")
            rows_a.append({"mode": f"{mode} ({a['agent']})", "asr": pct(s["attack_success"]["rate"]),
                           "ub": pct(s["utility_benign"]["rate"]), "ua": pct(s["utility_under_attack"]["rate"]),
                           "flag": pct(s["benign_flag_rate"]["rate"])})
        if agent is None:
            agent = {"rows": [], "note": ""}
        agent["rows"] += rows_a
        agent["note"] += (f"{a['agent']}: {a['modes']['none']['attack_success']['n']} InjecAgent attack cases "
                          f"(enhanced setting, half direct-harm, half data-stealing) and "
                          f"{a['modes']['none']['utility_benign']['n']} benign tool outputs. ")

    onnx = os.path.join(RESULTS, "onnx_parity.json")
    if os.path.exists(onnx):
        o = json.load(open(onnx))
        md += [f"\n\n## ONNX int8 export\n\nSize {o['size_mb_int8']:.1f} MB (fp32 {o['size_mb_fp32']:.1f} MB). "
               f"Mean |Δp| vs PyTorch fp32 = {o['mean_abs_diff']:.4f}, max {o['max_abs_diff']:.3f}; decision "
               f"agreement at the 1%-FPR threshold {100 * o['decision_agreement_fpr1']:.1f}% (n={o['n']})."]

    open(os.path.join(RESULTS, "tables.md"), "w").write("\n".join(md) + "\n")
    stats = json.load(open(os.path.join(RESULTS, "corpus_stats.json")))
    n_train = stats.get("train", {}).get("n", 0)
    n_synth = sum(v["sources"].get("synth", 0) for v in stats.values())
    headline = []
    main_lab = next((l for l in table if l.startswith("DistilRoBERTa, +")), None)
    pub_lab = next((l for l in table if "public data only" in l), None)
    if main_lab:
        sp = table[main_lab][1]
        headline.append({"v": pct(sp.get("test_synth_ood_style", {}).get("recall")) + "%",
                         "k": "recall on held-out attack styles at 1% FPR (ours)"})
        if pub_lab:
            headline.append({"v": pct(table[pub_lab][1].get("test_synth_ood_style", {}).get("recall")) + "%",
                             "k": "same, classifier trained on public (mostly direct) data only"})
        headline.append({"v": pct(sp.get("ext_injecagent", {}).get("recall")) + "%",
                         "k": "recall on InjecAgent (never trained on)"})
    if agent:
        a0 = json.load(open(sorted(glob.glob(os.path.join(RESULTS, "agent", "summary_*.json")))[0]))
        headline.append({"v": f"{pct(a0['modes']['none']['attack_success']['rate'])}% → "
                              f"{pct(a0['modes']['redact']['attack_success']['rate'])}%",
                         "k": f"agent attack success, no shield → redact mode ({a0['agent']})"})
    site = {"setup": f"Training set {n_train:,} documents; {n_synth:,} synthetic documents across all splits "
                     f"(paired clean/injected). Thresholds are fixed before looking at test data.",
            "headline": headline,
            "recall_cols": [{"key": s, "label": SPLIT_LABELS[s]} for s in RECALL_SPLITS],
            "fpr_cols": [{"key": s, "label": SPLIT_LABELS[s]} for s in FPR_SPLITS],
            "detectors": site_det, "figures": figs, "agent": agent}
    json.dump(site, open(os.path.join(SITE, "results.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
