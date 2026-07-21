# Appendix A — Statistical Tests

Fisher's exact tests are two-sided, computed directly from the hypergeometric
distribution with exact integer combinatorics (no scipy in the environment) and
validated against the tea-tasting table `[[3,1],[1,3]]`, p = 0.4857. Intervals
are Wilson score intervals, which remain correct at small *n* and at zero counts.
Reproduce with `python scripts/significance_stage3.py`.

---

### A.1 — Capture: does necessity suppress capture relative to propensity?

Referenced in §4.1. Nine contrasts; the **largest** p-value is 1.9 × 10⁻⁵.

| Model | Comparison | Capture (necessity) | Capture (propensity) | Odds ratio | *p* |
|:---|:---|---:|---:|---:|---:|
| **Qwen3-4B** | GSM8K vs MMLU | 2/500 (0.4%) | 32/800 (4.0%) | 0.096 | 1.9 × 10⁻⁵ \*\*\* |
| | GSM8K vs CommonsenseQA | 2/500 (0.4%) | 21/400 (5.2%) | 0.072 | 3.7 × 10⁻⁶ \*\*\* |
| | GSM8K vs pooled propensity | 2/500 (0.4%) | 53/1200 (4.4%) | 0.087 | 1.4 × 10⁻⁶ \*\*\* |
| **DeepSeek-R1-Distill-8B** | GSM8K vs MMLU | 11/500 (2.2%) | 115/800 (14.4%) | 0.134 | 6.0 × 10⁻¹⁵ \*\*\* |
| | GSM8K vs CommonsenseQA | 11/500 (2.2%) | 95/400 (23.8%) | 0.072 | 5.0 × 10⁻²⁵ \*\*\* |
| | GSM8K vs pooled propensity | 11/500 (2.2%) | 210/1200 (17.5%) | 0.106 | 4.2 × 10⁻²² \*\*\* |
| **Gemma-3-4B** | GSM8K vs MMLU | 0/500 (0.0%) | 329/800 (41.1%) | 0.001 | 1.1 × 10⁻⁸⁴ \*\*\* |
| | GSM8K vs CommonsenseQA | 0/500 (0.0%) | 160/400 (40.0%) | 0.001 | 1.1 × 10⁻⁶⁶ \*\*\* |
| | GSM8K vs pooled propensity | 0/500 (0.0%) | 489/1200 (40.8%) | 0.001 | 2.3 × 10⁻⁹¹ \*\*\* |

<sub>\*\*\* p < 0.001. Odds ratios below 1 mean capture is *less* likely on the necessity arm. Zero-cell odds ratios use the Haldane correction.</sub>

---

### A.2 — Disclosure on the propensity arms (Opus judge)

Referenced in §4.2.

| Model | Disclosed | Rate | 95% CI |
|:---|---:|---:|:---:|
| Qwen3-4B | 27/53 | 50.9% | [37.9%, 63.9%] |
| DeepSeek-R1-Distill-8B | 108/210 | 51.4% | [44.7%, 58.1%] |
| **Gemma-3-4B** | 1/489 | **0.2%** | [0.0%, 1.1%] |

| Comparison | Odds ratio | *p* | |
|:---|---:|---:|:---|
| Qwen3-4B vs DeepSeek-R1-Distill-8B | 0.98 | 1.00 | n.s. |
| Qwen3-4B vs Gemma-3-4B | 506.8 | 8.3 × 10⁻³⁰ | \*\*\* |
| DeepSeek-R1-Distill-8B vs Gemma-3-4B | 516.7 | 3.8 × 10⁻⁶⁶ | \*\*\* |

<sub>The two disclosing models are statistically indistinguishable from each other; both differ from Gemma by roughly thirty orders of magnitude in *p*. This is the basis for calling the difference categorical rather than a gradient.</sub>

---

### A.3 — Disclosure on the necessity arm

Referenced in §4.3.

| Model | Arm | Disclosed | Rate | 95% CI |
|:---|:---|---:|---:|:---:|
| DeepSeek-R1-Distill-8B | GSM8K (necessity) | 2/11 | 18.2% | [5.1%, 47.7%] |
| DeepSeek-R1-Distill-8B | Propensity (pooled) | 108/210 | 51.4% | [44.7%, 58.1%] |
| Qwen3-4B | GSM8K (necessity) | 2/2 | 100% | [34.2%, 100.0%] |
| Gemma-3-4B | GSM8K (necessity) | — | — | 0 captures |

<sub>DeepSeek necessity vs propensity: Fisher OR = 0.210, **p = 0.059, not significant**. Qwen's interval spans 34.2%–100% and is compatible with almost any true rate. Neither necessity cell constrains the hypothesis.</sub>
