#!/usr/bin/env python
"""Plot the necessity gap from a finished Stage 2 run.

    python scripts/plot_results.py            # reads outputs/
    python scripts/plot_results.py --out myrun
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
from cot_faith.config import Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    cfg = Config()
    summary = pd.read_csv(os.path.join(args.out, "stage2_summary.csv"),
                          index_col=0)
    order = [d for d in cfg.datasets if d in summary.index]
    s = summary.reindex(order)

    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, s["with_cot_acc"], w, label="with CoT")
    ax.bar(x + w / 2, s["no_cot_acc"], w, label="no CoT")
    for i, d in enumerate(order):
        ax.hlines(cfg.chance[d], x[i] - 0.45, x[i] + 0.45,
                  linestyles="dotted", color="grey")
        top = max(s.loc[d, "with_cot_acc"], s.loc[d, "no_cot_acc"])
        ax.text(x[i], top + 0.02, f"gap {s.loc[d, 'gap']:.2f}",
                ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=10)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Reasoning necessary (big gap) vs. optional (small gap)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(args.out, "necessity_gap.png")
    plt.savefig(path, dpi=120)
    print("saved", path)


if __name__ == "__main__":
    main()
