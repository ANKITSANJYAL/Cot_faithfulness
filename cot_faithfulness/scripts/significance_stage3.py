#!/usr/bin/env python
"""Fisher's exact tests + Wilson CIs for the Stage 3 claims.

No scipy in this env, so Fisher's exact is computed directly from the
hypergeometric distribution with exact integer combinatorics (math.comb) --
two-sided p = sum of P(table) over all tables at least as extreme (i.e. with
probability <= P(observed)). Validated below against the classic tea-tasting
table ([[3,1],[1,3]], two-sided p = 0.4857).

Usage:  python scripts/significance_stage3.py
"""
import os
import sys
from math import comb

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

MODELS = [("qwen3_4B", "Qwen3-4B"),
          ("deepseek_r1_distill_8B", "DeepSeek-R1-Distill-8B"),
          ("gemma3_4B", "Gemma-3-4B")]
PROP = ["MMLU", "CommonsenseQA"]


def fisher_exact(a, b, c, d):
    """Two-sided Fisher's exact p for [[a,b],[c,d]]. Returns (odds_ratio, p)."""
    r1, r2 = a + b, c + d
    c1, n = a + c, a + b + c + d
    if min(r1, r2, c1, n - c1) < 0 or n == 0:
        return float("nan"), float("nan")
    denom = comb(n, c1)

    def prob(x):
        # x = count in cell 'a'; row1 total r1, col1 total c1
        if x < 0 or x > r1 or (c1 - x) < 0 or (c1 - x) > r2:
            return 0.0
        return comb(r1, x) * comb(r2, c1 - x) / denom

    p_obs = prob(a)
    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    p = sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9))
    # Haldane-corrected sample odds ratio (handles zero cells)
    if 0 in (a, b, c, d):
        orr = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))
    else:
        orr = (a * d) / (b * c)
    return orr, min(p, 1.0)


def wilson(k, n, z=1.96):
    """Wilson score 95% CI for a proportion -- correct at small n / zero counts."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_p(p):
    if p != p:
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def stars(p):
    if p != p:
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


# --- validate the implementation before trusting it ------------------------
_or, _p = fisher_exact(3, 1, 1, 3)
assert abs(_p - 0.4857) < 0.001, f"fisher_exact self-test failed: {_p}"
print(f"[self-test] tea-tasting table [[3,1],[1,3]] -> p={_p:.4f} (expected 0.4857)  OK\n")

# --- load ------------------------------------------------------------------
data = {}
for key, name in MODELS:
    df = pd.read_csv(os.path.join(_REPO, "outputs", f"stage3_{key}", "stage3_results.csv"))
    data[name] = df

print("=" * 92)
print("TEST 1 — Does necessity (GSM8K) suppress CAPTURE vs propensity?  [Finding 1]")
print("=" * 92)
print(f"{'model':<24}{'comparison':<26}{'capture (nec)':>15}{'capture (prop)':>16}{'OR':>8}{'p':>12}{'':>5}")
for key, name in MODELS:
    df = data[name]
    nec = df[df.dataset == "GSM8K"]
    a, b = int(nec.captured.sum()), int((~nec.captured.astype(bool)).sum())
    for arm in PROP + ["POOLED prop"]:
        sub = df[df.dataset.isin(PROP)] if arm == "POOLED prop" else df[df.dataset == arm]
        c, d = int(sub.captured.sum()), int((~sub.captured.astype(bool)).sum())
        orr, p = fisher_exact(a, b, c, d)
        print(f"{name if arm=='MMLU' else '':<24}{'GSM8K vs '+arm:<26}"
              f"{f'{a}/{a+b} ({a/(a+b)*100:.1f}%)':>15}"
              f"{f'{c}/{c+d} ({c/(c+d)*100:.1f}%)':>16}{orr:>8.3f}{fmt_p(p):>12}{stars(p):>5}")
    print()

print("=" * 92)
print("TEST 2 — Is Gemma's DISCLOSURE lower than the others? (propensity arms, Opus judge) [Finding 2]")
print("=" * 92)
verb = {}
for key, name in MODELS:
    df = data[name]
    cap = df[(df.captured == True) & (df.dataset.isin(PROP))]  # noqa: E712
    cap = cap.dropna(subset=["judge_verbalized_opus"])
    k = int(cap.judge_verbalized_opus.astype(bool).sum())
    n = len(cap)
    verb[name] = (k, n)
    lo, hi = wilson(k, n)
    print(f"  {name:<24} verbalized {k:>3}/{n:<4} = {k/n*100:5.1f}%   "
          f"95% CI [{lo*100:.1f}%, {hi*100:.1f}%]")
print()
names = [n for _, n in MODELS]
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        (k1, n1), (k2, n2) = verb[names[i]], verb[names[j]]
        orr, p = fisher_exact(k1, n1 - k1, k2, n2 - k2)
        print(f"  {names[i]:<24} vs {names[j]:<26} OR={orr:>8.3f}  p={fmt_p(p):>10} {stars(p)}")

print()
print("=" * 92)
print("TEST 3 — DeepSeek's necessity cell points 'the wrong way'. Is it real, or noise? [§5]")
print("=" * 92)
df = data["DeepSeek-R1-Distill-8B"]
nec = df[(df.dataset == "GSM8K") & (df.captured == True)].dropna(subset=["judge_verbalized_opus"])  # noqa: E712
pro = df[(df.dataset.isin(PROP)) & (df.captured == True)].dropna(subset=["judge_verbalized_opus"])  # noqa: E712
k1, n1 = int(nec.judge_verbalized_opus.astype(bool).sum()), len(nec)
k2, n2 = int(pro.judge_verbalized_opus.astype(bool).sum()), len(pro)
lo1, hi1 = wilson(k1, n1)
lo2, hi2 = wilson(k2, n2)
orr, p = fisher_exact(k1, n1 - k1, k2, n2 - k2)
print(f"  necessity  (GSM8K): {k1}/{n1} = {k1/n1*100:.1f}%   95% CI [{lo1*100:.1f}%, {hi1*100:.1f}%]")
print(f"  propensity (pooled): {k2}/{n2} = {k2/n2*100:.1f}%   95% CI [{lo2*100:.1f}%, {hi2*100:.1f}%]")
print(f"  Fisher OR={orr:.3f}  p={fmt_p(p)}  {stars(p)}")
print("  -> read the CI width, not just the point estimate.")

print()
print("=" * 92)
print("TEST 4 — Qwen's necessity cell (100%, n=2): how much does it actually constrain?")
print("=" * 92)
df = data["Qwen3-4B"]
nec = df[(df.dataset == "GSM8K") & (df.captured == True)].dropna(subset=["judge_verbalized_opus"])  # noqa: E712
k, n = int(nec.judge_verbalized_opus.astype(bool).sum()), len(nec)
lo, hi = wilson(k, n)
print(f"  {k}/{n} = {k/n*100:.0f}%   95% CI [{lo*100:.1f}%, {hi*100:.1f}%]")
print("  -> a CI this wide is compatible with almost any true rate. Report as uninformative.")
