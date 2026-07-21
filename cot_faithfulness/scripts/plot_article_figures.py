#!/usr/bin/env python
"""Article figures for the CoT-faithfulness write-up, in a Google-style theme.

Regenerates every figure referenced in the Findings section straight from the
committed stage3_results.csv files -- no GPU, no API key. Figures land in
writeup/figures/.

    python scripts/plot_article_figures.py
"""
import os
import sys
from math import comb

import matplotlib
matplotlib.use("Agg")          # headless: write files, don't display
import matplotlib.pyplot as plt
import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(_REPO, "writeup", "figures")

MODELS = [("qwen3_4B", "Qwen3-4B"),
          ("deepseek_r1_distill_8B", "DeepSeek-8B"),
          ("gemma3_4B", "Gemma-3-4B")]
PROP = ["MMLU", "CommonsenseQA"]

# --- Google palette ---------------------------------------------------------
BLUE, RED, YELLOW, GREEN = "#4285F4", "#EA4335", "#FBBC04", "#34A853"
GREY_TEXT, GREY_GRID, INK = "#5F6368", "#E8EAED", "#202124"


def google_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        # Roboto/Google Sans if the system has them; DejaVu is the safe fallback.
        "font.sans-serif": ["Roboto", "Google Sans", "Helvetica Neue",
                            "Arial", "DejaVu Sans"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GREY_GRID,
        "axes.labelcolor": GREY_TEXT,
        "axes.titlecolor": INK,
        "axes.titlesize": 15,
        "axes.titleweight": "medium",
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GREY_GRID,
        "grid.linewidth": 1.0,
        "xtick.color": GREY_TEXT,
        "ytick.color": GREY_TEXT,
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })


def despine(ax, left=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GREY_GRID)
    if left:
        ax.spines["left"].set_visible(False)
    else:
        ax.spines["left"].set_color(GREY_GRID)


# --- stats ------------------------------------------------------------------

def wilson(k, n, z=1.96):
    """Wilson score 95% CI -- correct at small n and at zero counts."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def fisher_exact(a, b, c, d):
    """Two-sided Fisher's exact p for [[a,b],[c,d]] (no scipy in this env)."""
    r1, r2 = a + b, c + d
    c1, n = a + c, a + b + c + d
    if n == 0:
        return float("nan")
    denom = comb(n, c1)

    def prob(x):
        if x < 0 or x > r1 or (c1 - x) < 0 or (c1 - x) > r2:
            return 0.0
        return comb(r1, x) * comb(r2, c1 - x) / denom

    p_obs = prob(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    return min(sum(prob(x) for x in range(lo, hi + 1)
                   if prob(x) <= p_obs * (1 + 1e-9)), 1.0)


def load():
    data = {}
    for key, name in MODELS:
        path = os.path.join(_REPO, "outputs", f"stage3_{key}", "stage3_results.csv")
        if not os.path.exists(path):
            sys.exit(f"missing {path} -- run Stage 3 first")
        data[name] = pd.read_csv(path)
    return data


# --- Figure 1: capture rate by arm -----------------------------------------

def fig1_capture(data):
    arms = [("GSM8K", "GSM8K\n(necessity)", BLUE),
            ("MMLU", "MMLU\n(propensity)", RED),
            ("CommonsenseQA", "CommonsenseQA\n(propensity)", YELLOW)]
    names = [n for _, n in MODELS]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    width, xs = 0.26, range(len(names))

    for i, (arm, label, colour) in enumerate(arms):
        vals, errs = [], [[], []]
        for name in names:
            df = data[name]
            sub = df[df.dataset == arm]
            k, n = int(sub.captured.sum()), len(sub)
            p = k / n if n else 0
            lo, hi = wilson(k, n)
            vals.append(p * 100)
            errs[0].append((p - lo) * 100)
            errs[1].append((hi - p) * 100)
        pos = [x + (i - 1) * width for x in xs]
        ax.bar(pos, vals, width * 0.92, label=label, color=colour, zorder=3)
        ax.errorbar(pos, vals, yerr=errs, fmt="none", ecolor=GREY_TEXT,
                    elinewidth=1.1, capsize=3, zorder=4)
        # Sit the value label above the CI whisker, not the bar top, or it
        # collides with the error bar on the taller arms.
        for x, v, e_hi in zip(pos, vals, errs[1]):
            ax.text(x, v + e_hi + 1.4, f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=9.5, color=INK)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(names)
    ax.set_ylabel("Capture rate (%)")
    ax.set_title("Necessity blocks the manipulation", pad=14)
    ax.set_ylim(0, 52)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.xaxis.grid(False)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02))
    despine(ax, left=True)
    ax.text(0, -0.20,
            "Cue flipped the answer to its wrong target. Bars are Wilson 95% CIs.\n"
            "Every GSM8K-vs-propensity contrast is significant (Fisher's exact, max p = 1.9×10⁻⁵).",
            transform=ax.transAxes, fontsize=9, color=GREY_TEXT, va="top")
    save(fig, "fig1_capture_rate.png")


# --- Figure 2: disclosure on propensity arms --------------------------------

def fig2_disclosure(data):
    names = [n for _, n in MODELS]
    vals, los, his, labels, colours = [], [], [], [], []
    for name in names:
        df = data[name]
        cap = df[(df.captured == True) & (df.dataset.isin(PROP))]  # noqa: E712
        cap = cap.dropna(subset=["judge_verbalized_opus"])
        k, n = int(cap.judge_verbalized_opus.astype(bool).sum()), len(cap)
        p = k / n if n else 0
        lo, hi = wilson(k, n)
        vals.append(p * 100)
        los.append((p - lo) * 100)
        his.append((hi - p) * 100)
        labels.append(f"{k}/{n}")
        # Gemma is the different regime -- colour encodes that, not a ranking.
        colours.append(RED if name.startswith("Gemma") else BLUE)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = range(len(names))
    ax.bar(xs, vals, 0.5, color=colours, zorder=3)
    ax.errorbar(xs, vals, yerr=[los, his], fmt="none", ecolor=GREY_TEXT,
                elinewidth=1.2, capsize=4, zorder=4)
    for x, v, e_hi in zip(xs, vals, his):
        ax.text(x, v + e_hi + 2.2, f"{v:.1f}%", ha="center",
                fontsize=11, color=INK)

    ax.set_xticks(list(xs))
    # Fold the k/n counts into the tick label -- as a separate ax.text below the
    # axis they landed on top of the model names.
    ax.set_xticklabels([f"{n}\n{lab}" for n, lab in zip(names, labels)])
    ax.set_ylabel("Disclosure rate (%)")
    ax.set_title("Disclosure is categorical, not a gradient", pad=14)
    ax.set_ylim(0, 78)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.xaxis.grid(False)
    despine(ax, left=True)
    ax.text(0, -0.22,
            "Of captured propensity traces, share whose CoT admits the cue (Opus judge).\n"
            "Qwen vs DeepSeek: p = 1.00 (n.s.).  Gemma vs each: p = 8×10⁻³⁰ and 4×10⁻⁶⁶.",
            transform=ax.transAxes, fontsize=9, color=GREY_TEXT, va="top")
    save(fig, "fig2_disclosure_rate.png")


# --- Figure 3: the necessity arm is unmeasurable (forest plot) ---------------

def fig3_necessity_forest(data):
    rows = []
    for _, name in MODELS:
        df = data[name]
        for arm, tag in (("GSM8K", "necessity"), ("PROP", "propensity")):
            sub = (df[(df.captured == True) & (df.dataset == "GSM8K")]  # noqa: E712
                   if arm == "GSM8K" else
                   df[(df.captured == True) & (df.dataset.isin(PROP))])  # noqa: E712
            sub = sub.dropna(subset=["judge_verbalized_opus"])
            k, n = int(sub.judge_verbalized_opus.astype(bool).sum()), len(sub)
            rows.append((f"{name} — {tag}", k, n, tag))

    rows = [r for r in rows if r[2] > 0]          # drop Gemma necessity (0 captures)
    rows.reverse()
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, (label, k, n, tag) in enumerate(rows):
        p = k / n
        lo, hi = wilson(k, n)
        colour = BLUE if tag == "necessity" else GREY_TEXT
        ax.plot([lo * 100, hi * 100], [i, i], color=colour, lw=2.4,
                solid_capstyle="round", zorder=3)
        ax.plot([p * 100], [i], "o", color=colour, ms=8, zorder=4)
        ax.text(101, i, f"{k}/{n}", va="center", fontsize=9, color=GREY_TEXT)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlabel("Disclosure rate (%), Wilson 95% CI")
    ax.set_title("A null result by construction: the necessity arm cannot be measured",
                 pad=14, fontsize=14)
    ax.set_xlim(0, 108)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%" if v <= 100 else "")
    ax.yaxis.grid(False)
    despine(ax, left=True)
    ax.text(0, -0.20,
            "Necessity cells (blue) rest on 2 and 11 captured traces; their intervals span\n"
            "almost the whole range. Gemma has 0 necessity captures and cannot appear.",
            transform=ax.transAxes, fontsize=9, color=GREY_TEXT, va="top")
    save(fig, "fig3_necessity_forest.png")


# --- Figure 4: the terseness confound ---------------------------------------

def fig4_cot_length(data):
    names = [n for _, n in MODELS]
    meds, colours = [], []
    for name in names:
        df = data[name]
        cap = df[(df.captured == True) & (df.dataset.isin(PROP))].copy()  # noqa: E712
        wc = cap.thinking_text.fillna("").str.split().str.len()
        meds.append(float(wc.median()))
        colours.append(RED if name.startswith("Gemma") else BLUE)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    xs = range(len(names))
    ax.bar(xs, meds, 0.5, color=colours, zorder=3)
    for x, v in zip(xs, meds):
        ax.text(x, v + 18, f"{v:.0f}", ha="center", fontsize=11, color=INK,
                fontweight="medium")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names)
    ax.set_ylabel("Median CoT length (words)")
    ax.set_title("Checking the confound: is Gemma just terser?", pad=14, fontsize=14)
    ax.set_ylim(0, max(meds) * 1.25)
    ax.xaxis.grid(False)
    despine(ax, left=True)
    ax.text(0, -0.24,
            "Captured propensity traces. Gemma reasons far more briefly — a real confound —\n"
            "but 150 words still leaves room for one clause of disclosure, and it discloses 1 in 489.",
            transform=ax.transAxes, fontsize=9, color=GREY_TEXT, va="top")
    save(fig, "fig4_cot_length.png")


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, _REPO)}")


def main():
    google_style()
    data = load()
    print("Writing article figures ->", os.path.relpath(OUTDIR, _REPO))
    fig1_capture(data)
    fig2_disclosure(data)
    fig3_necessity_forest(data)
    fig4_cot_length(data)
    print("done.")


if __name__ == "__main__":
    main()
