"""Minimal Gemini REST client: disk cache (jsonl), <=4 concurrent requests, exponential backoff on 429/5xx.

Stdlib only so it runs anywhere. Every response is cached by sha256(model, payload) in cache/gemini_<model>.jsonl,
so re-running any script is free.
"""
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "cache")
BULK = "gemini-3.5-flash-lite"
JUDGE = "gemini-3.5-flash"
MAX_CONCURRENT = 4

_lock = threading.Lock()
_caches: dict[str, dict] = {}
_sem = threading.Semaphore(MAX_CONCURRENT)


_mtimes: dict[str, float] = {}


def refresh(model):
    """Reload a model's cache file if another process appended to it since we last read it."""
    path = os.path.join(CACHE_DIR, f"gemini_{model}.jsonl")
    if os.path.exists(path) and os.path.getmtime(path) != _mtimes.get(model):
        with _lock:
            _caches.pop(model, None)
            _cache(model)


def _cache(model):
    if model not in _caches:
        path = os.path.join(CACHE_DIR, f"gemini_{model}.jsonl")
        d = {}
        if os.path.exists(path):
            _mtimes[model] = os.path.getmtime(path)
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        d[r["k"]] = r["v"]
                    except Exception:
                        pass
        _caches[model] = d
    return _caches[model]


def _key(model, payload):
    return hashlib.sha256((model + json.dumps(payload, sort_keys=True)).encode()).hexdigest()


def call_raw(payload, model=BULK, retries=12, timeout=180, cache_only=False):
    """POST a generateContent payload; returns the response JSON (cached)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    k = _key(model, payload)
    with _lock:
        c = _cache(model)
        if k in c:
            return c[k]
    if cache_only:
        raise KeyError("not cached")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps(payload).encode()
    delay = 4.0
    last = None
    for _ in range(retries):
        req = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        try:
            with _sem:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    out = json.loads(r.read())
            with _lock:
                c[k] = out
                with open(os.path.join(CACHE_DIR, f"gemini_{model}.jsonl"), "a") as f:
                    f.write(json.dumps({"k": k, "v": out}) + "\n")
            return out
        except urllib.error.HTTPError as e:
            raw = e.read()
            last = f"HTTP {e.code}: {raw[:300]!r}"
            # empty-body 404s are transient edge errors for this key; real 404s carry a JSON error
            if e.code not in (429, 500, 502, 503, 504) and not (e.code == 404 and not raw.strip()):
                raise RuntimeError(last)
            m = re.search(rb"retry in ([0-9.]+)s", raw)
            if b"PerDay" in raw:  # daily cap: waiting hours is pointless, let the caller move on
                raise RuntimeError("daily quota exhausted for " + model)
            if m:
                time.sleep(min(float(m.group(1)), 120) + random.random() * 5)
                continue
        except Exception as e:  # timeouts, connection resets
            last = repr(e)
        time.sleep(delay + random.random() * delay)
        delay = min(delay * 2, 120)
    raise RuntimeError(f"gemini failed after retries: {last}")


def text_of(resp):
    try:
        parts = resp["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))
    except (KeyError, IndexError):
        return ""


def generate(prompt, model=BULK, temperature=1.0, json_mode=False, system=None, max_tokens=8192, seed=None,
             cache_only=False, thinking=None):
    gc = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if thinking is None and model.startswith("gemma"):
        thinking = "minimal"  # gemma-4 otherwise spends ~3.5k thought tokens per call
    if thinking:
        gc["thinkingConfig"] = {"thinkingLevel": thinking}
    if json_mode:
        gc["responseMimeType"] = "application/json"
    if seed is not None:
        gc["seed"] = seed
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gc}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    return text_of(call_raw(payload, model=model, cache_only=cache_only))


def pmap(fn, items, workers=MAX_CONCURRENT, desc=""):
    """Parallel map with progress; exceptions are returned in place of results."""
    results = [None] * len(items)
    done = [0]
    t0 = time.time()

    def run(i):
        try:
            results[i] = fn(items[i])
        except Exception as e:
            results[i] = e
        with _lock:
            done[0] += 1
            if done[0] % 50 == 0 or done[0] == len(items):
                print(f"[{desc}] {done[0]}/{len(items)} {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, range(len(items))))
    return results
