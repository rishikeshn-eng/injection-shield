"""Recover the cached responses of the first synthesis configuration (3 pairs per call, unshuffled 2112-job plan)
so no paid-for generation is wasted. Writes data/synth/pairs_v1.jsonl / benign_v1.jsonl in the same schema."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import synthesize as S  # noqa: E402
from gemini import BULK, generate  # noqa: E402


def v1_jobs(train_calls=11, heldout_calls=6, benign_calls=320, seed=0):
    rng = random.Random(seed)
    jobs = []
    for carrier in S.CARRIERS:
        for style in S.STYLES:
            k = heldout_calls if style in S.HELDOUT_STYLES else train_calls
            for i in range(k):
                jobs.append({"kind": "pair", "id": f"v1|{carrier}|{style}|{i}", "carrier": carrier, "style": style,
                             "goal": rng.choice(S.GOALS), "domain": rng.choice(S.DOMAINS),
                             "position": rng.choice(S.POSITIONS), "seed": rng.randrange(10**6)})
    carriers = list(S.CARRIERS)
    for i in range(benign_calls):
        jobs.append({"kind": "benign", "id": f"v1benign|{i}", "carrier": carriers[i % len(carriers)],
                     "flavour": S.BENIGN_FLAVOURS[i % len(S.BENIGN_FLAVOURS)], "domain": rng.choice(S.DOMAINS),
                     "seed": rng.randrange(10**6)})
    return jobs


def main():
    out = os.path.join(S.OUT)
    n_p = n_b = 0
    with open(os.path.join(out, "pairs_v1.jsonl"), "w") as fp, open(os.path.join(out, "benign_v1.jsonl"), "w") as fb:
        for job in v1_jobs():
            if job["kind"] == "pair":
                p = S.PROMPT.format(carrier_desc=S.CARRIERS[job["carrier"]], domain=job["domain"],
                                    style_desc=S.STYLES[job["style"]], goal=job["goal"], n=3, position=job["position"])
            else:
                p = S.BENIGN_PROMPT.format(carrier_desc=S.CARRIERS[job["carrier"]], domain=job["domain"],
                                           flavour=job["flavour"], n=5)
            try:
                items = S.items_of(generate(p, model=BULK, temperature=1.0, json_mode=True, seed=job["seed"],
                                            cache_only=True))
            except (KeyError, ValueError):
                continue
            for j, it in enumerate(items):
                if job["kind"] == "benign":
                    if isinstance(it, str) and len(it) > 80:
                        fb.write(json.dumps({"text": it, "group": job["id"], "carrier": job["carrier"],
                                             "flavour": job["flavour"]}) + "\n")
                        n_b += 1
                    continue
                if not isinstance(it, dict):
                    continue
                c, x, inj = it.get("clean"), it.get("injected"), it.get("injection")
                if not (isinstance(c, str) and isinstance(x, str) and isinstance(inj, str)) or len(c) < 100 \
                        or x.strip() == c.strip() or len(inj) < 15 or inj.strip() in c:
                    continue
                fp.write(json.dumps({"group": job["id"], "pair": f"{job['id']}|{j}", "carrier": job["carrier"],
                                     "style": job["style"], "goal": job["goal"], "domain": job["domain"],
                                     "position": job["position"], "generator": BULK, "clean": c, "injected": x,
                                     "injection": inj, "span_verbatim": inj in x}) + "\n")
                n_p += 1
    print("salvaged pairs", n_p, "benign", n_b)


if __name__ == "__main__":
    main()
