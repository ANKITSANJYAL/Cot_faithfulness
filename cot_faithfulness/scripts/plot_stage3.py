#!/usr/bin/env python
"""Plot Stage 3 faithfulness results.

    python scripts/plot_stage3.py --out outputs/stage3
"""
import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless cluster: write file, don't display
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cot_faith.config import Stage3Config
from cot_faith.faithfulness import compute_agreement


def _bar(ax, order, values, ylabel, title, ylim=None):
    x = np.arange(len(order))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    for i, v in enumerate(values):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/stage3")
    args = ap.parse_args()

    cfg = Stage3Config()
    summary = pd.read_csv(os.path.join(args.out, "stage3_summary.csv"), index_col=0)
    order = [d for d in cfg.datasets if d in summary.index]
    s = summary.reindex(order)

    # 1. Headline: LLM-judge verbalization rate per arm, in necessity order.
    fig, ax = plt.subplots(figsize=(7, 5))
    _bar(ax, order, s["judge_verbalization_rate"].values,
        "verbalization rate (LLM judge)",
        "Verbalization rate by necessity arm\n(higher = CoT more often admits the cue)",
        ylim=(0, 1))
    plt.tight_layout()
    path = os.path.join(args.out, "verbalization_rate.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print("saved", path)

    # 2. Capture rate per arm -- context for how much data backs (1).
    fig, ax = plt.subplots(figsize=(7, 5))
    _bar(ax, order, s["capture_rate"].values, "capture rate",
        "Cue capture rate by arm\n(fraction of problems the cue flipped to its target)",
        ylim=(0, 1))
    plt.tight_layout()
    path = os.path.join(args.out, "capture_rate.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print("saved", path)

    # 3. String-match vs LLM-judge agreement per arm, on captured rows only.
    results_path = os.path.join(args.out, "stage3_results.csv")
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        cap = df[df["captured"] == True].dropna(  # noqa: E712
            subset=["string_match_verbalized", "judge_verbalized"])
        agree_rates = []
        for d in order:
            g = cap[cap["dataset"] == d]
            if len(g):
                agree_rates.append(
                    (g["string_match_verbalized"].astype(bool) ==
                     g["judge_verbalized"].astype(bool)).mean())
            else:
                agree_rates.append(float("nan"))
        fig, ax = plt.subplots(figsize=(7, 5))
        _bar(ax, order, agree_rates, "agreement",
            "String-match vs LLM-judge agreement\n(sanity check on the cheap layer)",
            ylim=(0, 1))
        plt.tight_layout()
        path = os.path.join(args.out, "string_match_vs_judge_agreement.png")
        plt.savefig(path, dpi=120)
        plt.close(fig)
        print("saved", path)
    else:
        print(f"  {results_path} not found -- skipping string-match/judge agreement plot")

    # 4. Judge vs human agreement, only once the validation sample is labeled.
    val_path = os.path.join(args.out, "human_validation_sample.csv")
    if os.path.exists(val_path):
        val_df = pd.read_csv(val_path)
        has_labels = ("human_verbalized" in val_df.columns
                      and val_df["human_verbalized"].astype(str).str.strip().ne("").any())
        if has_labels:
            pct, kappa, n = compute_agreement(val_path)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.bar(["judge vs human"], [pct])
            ax.set_ylim(0, 1)
            ax.set_ylabel("agreement")
            ax.set_title(f"Judge vs human agreement (n={n})\nCohen's kappa = {kappa:.2f}")
            ax.text(0, pct + 0.02, f"{pct:.2f}", ha="center")
            plt.tight_layout()
            path = os.path.join(args.out, "judge_vs_human_agreement.png")
            plt.savefig(path, dpi=120)
            plt.close(fig)
            print("saved", path)
        else:
            print(f"  {val_path} has no human labels yet -- skipping judge-vs-human plot")
            print("  (fill in the human_verbalized column, then re-run this script)")
    else:
        print(f"  {val_path} not found -- skipping judge-vs-human plot")


if __name__ == "__main__":
    main()
