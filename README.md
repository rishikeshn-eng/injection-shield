# Injection Shield

**A small classifier for _indirect_ prompt injection, i.e. instructions hidden in the web pages, emails, API responses, code and memory that an AI agent reads. It runs in the browser (int8 ONNX) and is evaluated on attack styles and document formats it never saw in training.**

Live demo + results: **https://rishikeshn-eng.github.io/injection-shield/** (paste text → injection score; nothing leaves your browser).

<!--HEADLINE-->

## Problem

Prompt-injection attacks on agents have moved from "ignore previous instructions" typed by a user to **indirect injection**: the payload sits in data the agent fetches with a tool, and its job is to hijack the agent's *goal* (send an email, move money, leak a file, plant a memory).

- **OWASP Top 10 for Agentic Applications (Dec 2025)** ranks *Agent Goal Hijack* (ASI01) first. Its listed vectors include deceptive tool outputs and poisoned external data ([overview](https://www.promptfoo.dev/docs/red-team/owasp-agentic-ai/)).
- **EchoLeak (CVE-2025-32711, CVSS 9.3)** was a zero-click exfiltration from Microsoft 365 Copilot via a single crafted email. The exploit chain **evaded Microsoft's XPIA prompt-injection classifier** ([paper](https://arxiv.org/html/2509.10540v1), [report](https://thehackernews.com/2025/06/zero-click-ai-vulnerability-exposes.html)).
- Practitioners report that open injection classifiers, trained mostly on direct jailbreak prompts, do well on jailbreaks but poorly on tool-abuse/indirect cases ([StackOne](https://www.stackone.com/blog/training-22mb-prompt-injection-classifier/)), and that small detectors overfit to the surface form of their training set ([LessWrong](https://www.lesswrong.com/posts/uPEX2zDTPioFmieg2/your-prompt-injection-defense-metric-might-be-lying-to-you)).
- Google's Gemini team describes classifier-based defences against indirect injection as one layer that adaptive attacks erode ([Shi et al. 2025](https://arxiv.org/pdf/2505.14534)).

The questions here are: (1) how badly do existing detectors miss indirect injections, (2) does training on synthetic indirect injections fix that, and does the fix *generalise to attack styles and document formats held out from training* rather than only to more of the same, and (3) does a cheap classifier in front of an agent actually lower attack success without destroying utility?

## Related work

- Benchmarks: **BIPIA** (Yi et al. 2023, indirect injections in email/table/code contexts), **InjecAgent** (Zhan et al. 2024, 1,054 tool-integrated test cases, direct-harm and data-stealing), **AgentDojo** (Debenedetti et al. 2024, stateful agent environments with injection vectors), **TensorTrust** (Toyer et al. 2023), **PromptShield** (Jacob et al. 2024, low-FPR detection benchmark), Lakera Gandalf/Mosscap, deepset prompt-injections, jackhhao jailbreak-classification, safe-guard-prompt-injection.
- Detectors: **Meta Prompt Guard / Prompt Guard 2** (gated on Hugging Face; could not be downloaded without a token, so it is *not* evaluated here), **ProtectAI deberta-v3-base-prompt-injection-v2** (evaluated), LLM-as-judge guards (evaluated with Gemini), and rule/heuristic filters such as rebuff, vigil and LLM-Guard's substring bans (a keyword baseline in that spirit is evaluated).
- System-level defences (spotlighting, CaMeL, tool-result parsing, dual-LLM) are complementary: a detector like this is one layer, not a fix.

## Method

### Data (`scripts/build_corpus.py`, `scripts/build_external.py`, `scripts/synthesize.py`)

<!--DATA-->

**Synthetic indirect injections.** Gemini/Gemma generate *paired* documents. For each document the generator returns the clean version, which deliberately contains benign imperatives addressed to the human reader (*"please reply by Friday"*, *"ignore my previous email, wrong attachment"*, *"run npm install first"*) to act as hard negatives, plus **the same document with an injection woven in** and the exact injected span. Pairing forces the classifier to learn the injection rather than the topic, carrier or writing style. Each call is conditioned on a carrier format, an attack style, an attacker goal (14, e.g. exfiltrate email, transfer funds, grant smart-lock access, plant a phishing link), a domain (20) and an insertion position.

- **16 carrier formats**: web page HTML, Markdown blog, email, JSON API response, source code, GitHub issue, README/docs, chat export, product reviews, search results, PDF extract, agent memory store, MCP/tool manifest, plus **3 held-out carriers** (calendar invite, terminal/log output, CSV table).
- **12 attack styles**: 8 for training (instruction override, fake system message, authority impersonation, task-continuation hijack, hidden markup, data exfiltration via URL/image, tool-call injection, fake error/alert) and **4 held out entirely** (memory poisoning, obfuscation/encoding/other language, *persuasion without any imperative*, spoofed dialogue/tool boundaries).
- Splits are by generation call (group), so near-duplicate documents never straddle train and test. Every held-out style or carrier goes only to test.
- Extra hard negatives: benign documents with dense imperative language, security articles that *discuss* prompt injection, legitimate bot/CI configuration, urgent but genuine notices, and a user's own legitimate notes to their assistant.

**External benchmarks never used for training.**
- *InjecAgent*: all 2,108 base + enhanced tool responses (positives). Negatives are the same 17 tool-response templates with Gemini-written benign content in the attacker slot (340).
- *AgentDojo v1*: every injection vector of the banking/slack/travel/workspace suites, rendered inside the environment object that contains it (email, event, file, transaction, review, web page) and filled with each injection-task GOAL wrapped in AgentDojo's own attack templates (`important_instructions`, `ignore_previous`, `system_message`, `direct`, `injecagent`). Negatives: the same objects with benign defaults, plus every other environment object.

### Model (`scripts/train.py`)

Long documents are split into **320-token windows (stride 224)** and a document's score is the **max over its windows**, so an injection buried at the end of a long page is not truncated away. During training, a window of an injected document is positive only if it covers ≥60% of the known injection span. Windows containing none of it are used as *negatives*: clean text from the very same document. Windows in between are dropped. Rows without a known span (public direct-injection sets) use their first window. Fine-tuning: AdamW, lr 3e-5, 2 epochs, batch 16, linear warm-up/decay, class-weighted CE, MPS.

Models: `microsoft/deberta-v3-small` (main, 141M), `distilroberta-base` (82M) and `microsoft/MiniLM-L12-H384-uncased` (33M, the browser model). Ablation: DeBERTa-v3-small trained on the same pipeline **without any synthetic data**, which stands in for "a classifier trained on existing public data".

**Operating points are fixed before touching test data**: `t0.5` (score ≥ 0.5) and `fpr1`, the threshold giving 1% false positives on the validation split. All tables use `fpr1`. 95% CIs are percentile bootstraps (1,000 resamples). Seeds are reported where more than one was run.

### Baselines (`scripts/evaluate.py`, `scripts/judge_baseline.py`)

- **Keyword rules**: 25 regexes for classic injection phrasing (≥1 match = flag).
- **ProtectAI deberta-v3-base-prompt-injection-v2** (184M), the most-downloaded open injection classifier, scored with the same max-over-512-token-windows rule.
- **LLM-as-judge (Gemini)**: a security-classifier system prompt; 8 numbered documents per request; ≤160 documents per split because of the free-tier quota (500 requests/day/model shared by six projects). The fine-tuned model is compared on exactly the same subsample.
- **Meta Prompt Guard**: gated, not downloadable without an HF token; not evaluated.

### End-to-end agent test (`scripts/agent_eval.py`)

A tool-calling agent receives an InjecAgent case: the user's request, the agent's own call to the user tool, and the tool response containing the attacker instruction, with both the user tool and the attacker's tools available. **Attack success** = the agent's next step calls an attacker tool (InjecAgent's first-step criterion, enhanced setting, half direct-harm, half data-stealing). **Utility** = a Gemini judge decides whether the reply fulfils the user's request, on benign versions of the same tool outputs and on the attacked ones. Shield modes: `none`; `block` (withhold the whole tool output if flagged); `redact` (split the output into sentences/fields and remove only flagged segments).

### Browser export (`scripts/export_onnx.py`, `site/`)

`torch.onnx.export` (opset 17), onnxruntime dynamic int8 quantisation, and a parity check against PyTorch. The site runs it with transformers.js using the same windowing and threshold, and highlights the individual segments that score above threshold on their own.

## Results

<!--RESULTS-->

## Limitations

<!--LIMITATIONS-->

## Reproduce

```bash
uv venv && uv pip install torch transformers datasets scikit-learn sentencepiece protobuf pandas matplotlib onnx onnxruntime pyarrow pyyaml
export GEMINI_API_KEY=...            # synthesis, judge and agent (all responses cached in cache/)
bash scripts/reproduce.sh            # downloads data, builds corpora, trains, evaluates, exports, writes site/
```

`scripts/reproduce.sh` wraps every GPU step in `_shared/gpulock.sh` when that file exists. The Hugging Face upload was skipped (no token); the int8 model is in `site/model/`.
