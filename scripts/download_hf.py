"""Download the public prompt-injection datasets (raw parquet/json/csv files) into data/hf/."""
import json
import os
import subprocess
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = ["deepset/prompt-injections", "Lakera/gandalf_ignore_instructions", "Lakera/gandalf_summarization",
            "Lakera/mosscap_prompt_injection", "xTRam1/safe-guard-prompt-injection", "hendzh/PromptShield",
            "jackhhao/jailbreak-classification"]

for d in DATASETS:
    j = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{d}", timeout=120))
    for s in j["siblings"]:
        f = s["rfilename"]
        if not f.endswith((".parquet", ".json", ".csv")) or f.startswith("balanced"):
            continue
        name = d.split("/")[1]
        out = os.path.join(ROOT, "data", "hf", name, os.path.basename(f) if name == "deepset" else f.replace("/", "_"))
        if name == "deepset":
            out = os.path.join(ROOT, "data", "hf", name, ("test" if "test" in f else "train") + ".parquet")
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run(["curl", "-sSL", "-m", "900", "-o", out, f"https://huggingface.co/datasets/{d}/resolve/main/{f}"],
                       check=True)
        print(out, os.path.getsize(out))
