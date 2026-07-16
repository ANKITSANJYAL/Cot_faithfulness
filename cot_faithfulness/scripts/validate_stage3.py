#!/usr/bin/env python
"""Compute agreement between your hand labels and the LLM judge.

Workflow:
    1. Open outputs/stage3/human_validation_sample.csv
    2. Fill in the blank human_verbalized column with TRUE/FALSE for each row
       (read cue_text + thinking_text, judge_reason/judge_verbalized are there
       for reference but label independently before looking at them)
    3. Save, then run:
           python scripts/validate_stage3.py --out outputs/stage3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cot_faith.faithfulness import compute_agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/stage3")
    args = ap.parse_args()

    path = os.path.join(args.out, "human_validation_sample.csv")
    pct, kappa, n = compute_agreement(path)

    if n == 0:
        print(f"No labeled rows found in {path}.")
        print("Fill in the human_verbalized column (TRUE/FALSE) and re-run.")
        return

    print(f"Labeled rows:              {n}")
    print(f"Judge vs human agreement:  {pct:.3f}")
    print(f"Cohen's kappa:             {kappa:.3f}")
    if kappa < 0.4:
        print("\nkappa < 0.4 (poor agreement) -- consider tightening the judge rubric")
        print("or re-reading a sample of disagreements before trusting judge_verbalized.")


if __name__ == "__main__":
    main()
