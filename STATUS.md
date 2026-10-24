# STATUS — injection-shield

- **Repo:** https://github.com/rishikeshn-eng/injection-shield (public)
- **Pages:** https://rishikeshn-eng.github.io/injection-shield/ (deployed from `site/` by `.github/workflows/pages.yml`)
- **State:** complete except for items under "Left undone".

## Headline

- Training on 4,096 paired synthetic indirect injections raises recall on **attack styles never seen in training** from 13.5% (same DistilRoBERTa, public data only) to **76.2% ±2.1** (2 seeds), at a 1% validation false-positive budget. MiniLM gets 83.1%. On the same split ProtectAI's deberta-v3-base-injection-v2 gets 11.5%.
- AgentDojo (never trained on): 92.7% recall vs 52.2% public-only and 35.1% ProtectAI.
- InjecAgent (never trained on): no gain (62.7% vs 62.6% public-only). The detector catches 100% of the "enhanced" attacks but only 12–24% of bare polite requests. This is a fundamental limit of text-only detection, documented in the README.
- The shipped browser model is MiniLM-L12 int8 ONNX, **32.4 MB**, 98.7% decision agreement with PyTorch, running in-browser with transformers.js.
- End-to-end agent (InjecAgent, Gemma-4-26B tool-calling agent): attack success 0.8% → 0%. In-context redaction keeps 89.2% utility under attack vs 55.0% for blocking. The agent is already robust, so the ASR reduction is small.

## Left undone / follow-ups

- **≥5k synthetic injections:** stopped at 4,096. The free-tier Gemini key allows 500 requests/day per model (20 for 3.5 Flash), shared by six projects.
- **LLM-judge baseline:** ran on Gemma-4-31B, not Gemini. Gemma-4-31B returned frequent HTTP 500s, so the judge was still running at hand-off. Finished splits: public, BIPIA, PromptShield and all synthetic splits; hard negatives, InjecAgent and AgentDojo are pending. Re-run `JUDGE_MODEL=gemini-3.5-flash-lite python scripts/judge_baseline.py` after the quota resets, then `python scripts/report.py`.
- **Agent test with a more injection-susceptible agent** (Gemini Flash-Lite or a local Qwen2.5-1.5B), so the shield's ASR reduction becomes measurable: `AGENT_MODEL=gemini-3.5-flash-lite python scripts/agent_eval.py run --threshold fpr1`.
- **DistilRoBERTa seed 2:** cancelled because the shared GPU queue was saturated. Main results use 2 seeds.
- **Meta Prompt Guard 2:** not evaluated (gated on Hugging Face, no token). No Hugging Face upload (no token); the model ships in `site/model/`.
- **DeBERTa-v3-small:** diverged (NaN) on MPS, so it was replaced by DistilRoBERTa.
