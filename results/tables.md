## Recall (TPR, %) at the 1%-FPR-on-validation threshold

| Detector | Public direct (test) | BIPIA (test) | Synthetic ID | OOD style | OOD carrier | OOD style+carrier | InjecAgent | AgentDojo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Keyword rules (fpr1) | 18.9 [17.0–20.7] | 0.0 [0.0–0.0] | 34.8 [29.5–40.4] | 19.6 [16.6–22.5] | 34.2 [30.5–38.2] | 26.4 [18.8–33.6] | 50.0 [47.6–52.2] | 40.0 [36.7–43.4] |
| ProtectAI deberta-v3-base-injection-v2 (184M) (fpr1) | 66.8 [64.4–69.1] | 7.1 [5.1–9.3] | 16.3 [12.0–20.7] | 11.5 [9.2–14.0] | 7.9 [5.9–10.1] | 6.9 [3.3–11.3] | 51.1 [48.9–53.1] | 35.1 [32.1–38.2] |
| LLM judge (Gemma-4-31B via Gemini API)* (t0.5) | 75.0 [65.4–84.2] | 70.0 [60.0–79.6] | 98.8 [95.9–100.0] | 92.5 [86.7–97.6] | 100.0 [100.0–100.0] | 92.3 [86.1–97.5] | – | – |
| DistilRoBERTa, public data only (fpr1) | 93.8 [92.5–95.0] | 68.4 [64.4–72.2] | 21.0 [16.3–25.6] | 13.5 [11.0–16.1] | 23.8 [20.3–27.5] | 21.5 [15.1–28.5] | 62.6 [60.4–64.8] | 52.2 [48.9–55.6] |
| DistilRoBERTa, + synthetic (ours) (fpr1) | 97.5 ±0.5 [96.3–98.5] | 91.4 ±4.2 [85.5–96.3] | 96.9 ±1.3 [93.7–99.3] | 76.2 ±2.1 [71.6–81.0] | 98.7 ±0.1 [97.6–99.6] | 82.6 ±4.9 [72.9–91.8] | 62.7 ±5.2 [56.8–68.5] | 92.7 ±2.6 [89.0–96.1] |
| MiniLM-L12, + synthetic (ours, browser model) (fpr1) | 97.8 [97.1–98.5] | 94.6 [92.5–96.4] | 97.5 [95.6–99.2] | 83.1 [79.9–86.0] | 98.9 [98.0–99.7] | 87.5 [81.7–93.0] | 58.1 [55.8–60.2] | 80.4 [77.6–83.0] |


## False-positive rate (%) on benign documents at the same threshold

| Detector | Public direct (test) | PromptShield | Synthetic ID | OOD style | OOD carrier | Hard negatives | InjecAgent | AgentDojo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Keyword rules (fpr1) | 0.0 [0.0–0.0] | 1.2 [0.9–1.6] | 1.1 [0.0–2.6] | 0.0 [0.0–0.0] | 0.5 [0.0–1.2] | 4.2 [1.9–6.9] | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] |
| ProtectAI deberta-v3-base-injection-v2 (184M) (fpr1) | 0.2 [0.1–0.5] | 0.3 [0.1–0.5] | 1.4 [0.3–2.8] | 2.4 [1.3–3.6] | 0.5 [0.0–1.1] | 8.0 [5.0–11.5] | 0.9 [0.0–1.9] | 5.9 [2.8–9.5] |
| LLM judge (Gemma-4-31B via Gemini API)* (t0.5) | 2.5 [0.0–6.1] | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | 0.0 [0.0–0.0] | – | – | – |
| DistilRoBERTa, public data only (fpr1) | 0.4 [0.1–0.7] | 4.3 [3.6–5.0] | 3.6 [1.5–6.0] | 2.5 [1.4–3.7] | 6.8 [4.9–8.9] | 3.4 [1.5–5.7] | 1.2 [0.3–2.5] | 0.6 [0.0–2.0] |
| DistilRoBERTa, + synthetic (ours) (fpr1) | 0.9 ±0.2 [0.4–1.6] | 9.0 ±1.2 [7.2–11.0] | 0.4 ±0.0 [0.0–1.1] | 0.2 ±0.1 [0.0–0.8] | 6.3 ±5.7 [1.1–13.0] | 9.5 ±3.2 [4.2–15.6] | 3.8 ±1.2 [1.4–7.1] | 5.3 ±0.0 [2.1–8.9] |
| MiniLM-L12, + synthetic (ours, browser model) (fpr1) | 1.5 [0.9–2.1] | 12.0 [10.8–13.2] | 0.7 [0.0–1.8] | 0.1 [0.0–0.5] | 1.4 [0.5–2.5] | 7.3 [4.6–10.3] | 1.8 [0.6–3.2] | 3.6 [0.7–6.5] |


## AUROC

| Detector | Public direct (test) | BIPIA (test) | Synthetic ID | OOD style | OOD carrier | OOD style+carrier | InjecAgent | AgentDojo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Keyword rules | 0.594 | 0.500 | 0.669 | 0.598 | 0.668 | 0.632 | 0.750 | 0.700 |
| ProtectAI deberta-v3-base-injection-v2 (184M) | 0.974 | 0.407 | 0.681 | 0.678 | 0.678 | 0.693 | 0.832 | 0.801 |
| LLM judge (Gemma-4-31B via Gemini API)* | 0.893 | 0.900 | 1.000 | 0.963 | 1.000 | 0.974 | – | – |
| DistilRoBERTa, public data only | 0.998 | 0.954 | 0.762 | 0.705 | 0.666 | 0.635 | 0.926 | 0.914 |
| DistilRoBERTa, + synthetic (ours) | 0.997 | 0.974 | 0.998 | 0.969 | 0.995 | 0.944 | 0.898 | 0.985 |
| MiniLM-L12, + synthetic (ours, browser model) | 0.998 | 0.989 | 0.996 | 0.968 | 0.997 | 0.963 | 0.868 | 0.966 |


## Recall per attack style (synthetic test sets, %)

| Detector | override | fake_system | authority | task_hijack | hidden_markup | exfiltration | tool_call | fake_error | memory_poison | obfuscated | persuasion | fake_dialogue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Keyword | 57.3 | 43.8 | 12.4 | 87.3 | 16.0 | 10.7 | 6.2 | 34.3 | 11.1 | 3.1 | 48.1 | 17.4 |
| ProtectAI | 38.5 | 1.9 | 12.4 | 1.6 | 11.1 | 6.8 | 9.3 | 7.8 | 12.9 | 8.8 | 8.1 | 12.3 |
| DistilRoBERTa public-only | 74.0 | 25.7 | 21.9 | 5.6 | 22.2 | 15.5 | 15.5 | 11.8 | 15.1 | 16.4 | 6.7 | 21.5 |
| DistilRoBERTa + synth | 100.0 | 99.0 | 100.0 | 100.0 | 87.7 | 98.1 | 100.0 | 100.0 | 95.6 | 57.9 | 73.8 | 83.1 |
| MiniLM + synth | 100.0 | 97.1 | 99.0 | 100.0 | 91.4 | 98.1 | 100.0 | 100.0 | 97.3 | 52.8 | 81.9 | 94.5 |


## Paired comparison on the LLM-judge subsample (recall / FPR, %)

The judge scores 8 documents per request on <=160 documents per split (free-tier quota). Ours and the judge are evaluated on the identical documents. Judge threshold 0.5; ours: 1% val FPR.

| Split | n | Judge recall | Judge FPR | Ours recall | Ours FPR |
|---|---:|---:|---:|---:|---:|
| BIPIA (test) | 160 | 70.0 | 0.0 | 92.5 | 18.8 |
| PromptShield | 160 | 72.5 | 0.0 | 60.0 | 6.2 |
| Public direct (test) | 160 | 75.0 | 2.5 | 96.2 | 1.2 |
| Synthetic ID | 160 | 98.8 | 0.0 | 97.5 | 1.2 |
| OOD style+carrier | 152 | 92.3 | 0.0 | 89.7 | 13.5 |
| OOD carrier | 160 | 100.0 | 0.0 | 100.0 | 13.8 |
| OOD style | 160 | 92.5 | 0.0 | 73.8 | 0.0 |


## End-to-end agent (gemma-4-26b-a4b-it, shield model (MiniLM int8 ONNX), threshold fpr1 = 0.125)

| Shield | Attack success ↓ | ASR base attacks | ASR enhanced attacks | Utility benign ↑ | Utility under attack ↑ | Benign outputs flagged |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.8 [0.0–2.5] (n=120) | 0.0 | 1.7 | 90.0 [83.4–96.6] (n=80) | 93.3 [88.9–97.8] (n=120) | 0.0 |
| block | 0.8 [0.0–2.5] (n=120) | 0.0 | 1.7 | 90.0 [83.4–96.6] (n=80) | 55.0 [46.1–63.9] (n=120) | 2.5 |
| redact | 0.0 [0.0–0.0] (n=120) | 0.0 | 0.0 | 90.0 [83.4–96.6] (n=80) | 89.2 [83.6–94.7] (n=120) | 2.5 |


## End-to-end agent (gemma-4-26b-a4b-it, shield model (MiniLM int8 ONNX), threshold t0.5 = 0.500)

| Shield | Attack success ↓ | ASR base attacks | ASR enhanced attacks | Utility benign ↑ | Utility under attack ↑ | Benign outputs flagged |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.8 [0.0–2.5] (n=120) | 0.0 | 1.7 | 90.0 [83.4–96.6] (n=80) | 93.3 [88.9–97.8] (n=120) | 0.0 |
| block | 0.8 [0.0–2.5] (n=120) | 0.0 | 1.7 | 90.0 [83.4–96.6] (n=80) | 53.3 [44.4–62.3] (n=120) | 1.2 |
| redact | 0.0 [0.0–0.0] (n=120) | 0.0 | 0.0 | 90.0 [83.4–96.6] (n=80) | 89.2 [83.6–94.7] (n=120) | 1.2 |


## ONNX int8 export

Size 32.4 MB (fp32 127.5 MB). Mean |Δp| vs PyTorch fp32 = 0.0144, max 0.985; decision agreement at the 1%-FPR threshold 98.7% (n=600).
