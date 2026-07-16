#!/usr/bin/env python
"""Merge per-GPU Stage 3 outputs into one run and recompute the cross-arm summary.

Used after the SLURM job runs each arm-subset on its own GPU:

    python scripts/aggregate_stage3.py \\
        --inputs outputs/stage3/gpu0_gsm8k outputs/stage3/gpu1_csqa_mmlu \\
        --out outputs/stage3
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cot_faith.faithfulness import export_validation_sample


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset"):
        n = len(g)
        captured = g[g["captured"] == True]  # noqa: E712
        n_captured = len(captured)
        rows.append({
            "dataset": ds, "n": n, "n_captured": n_captured,
            "capture_rate": n_captured / n if n else 0.0,
            "string_match_verbalization_rate":
                captured["string_match_verbalized"].mean() if n_captured else float("nan"),
            "judge_verbalization_rate":
                captured["judge_verbalized"].mean() if n_captured else float("nan"),
        })
    return pd.DataFrame(rows).set_index("dataset").round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="per-GPU output dirs")
    ap.add_argument("--out", default="outputs/stage3")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dfs, configs = [], []
    for d in args.inputs:
        path = os.path.join(d, "stage3_results.csv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        dfs.append(pd.read_csv(path))
        cfg_path = os.path.join(d, "run_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                configs.append(json.load(f))

    if not dfs:
        raise SystemExit("No stage3_results.csv found in any --inputs dir")

    df = pd.concat(dfs, ignore_index=True)
    dup = df["id"].duplicated()
    if dup.any():
        print(f"  WARNING: {dup.sum()} duplicate problem ids across inputs -- "
              f"check --inputs don't overlap (e.g. the same arm run on two GPUs)")

    summary = summarise(df)

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "stage3_results.csv"), index=False)
    summary.to_csv(os.path.join(args.out, "stage3_summary.csv"))
    with open(os.path.join(args.out, "run_config.json"), "w") as f:
        json.dump({"aggregated_from": args.inputs, "per_job_configs": configs}, f, indent=2)

    print("\n=== Stage 3 aggregated summary ===")
    print(summary.to_string())

    export_validation_sample(df, os.path.join(args.out, "human_validation_sample.csv"),
                             n=50, seed=args.seed)
    print(f"\nSaved merged results, summary, and a 50-row human validation sample to {args.out}/")


if __name__ == "__main__":
    main()
