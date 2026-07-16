#!/usr/bin/env python
"""Re-score Stage 3 verbalization with a stronger LLM judge, ADDING columns
rather than overwriting the original judge -- so Haiku-vs-<new> agreement can
be reported as a robustness check.

Pure Anthropic API; no GPU / vLLM. Run on a login node. Needs ANTHROPIC_API_KEY
in the repo's .env (same as run_stage3.py).

For each model it loads outputs/stage3_<model>/stage3_results.csv, re-judges the
captured rows with the chosen model, writes judge_verbalized_<tag> /
judge_reason_<tag> back into that CSV, and prints per-arm rates for BOTH judges
plus their agreement.

Usage:
    python scripts/rejudge_stage3.py --judge opus       # claude-opus-4-8  (default)
    python scripts/rejudge_stage3.py --judge sonnet     # claude-sonnet-5
    python scripts/rejudge_stage3.py --judge-model <id> --tag <suffix>
    python scripts/rejudge_stage3.py --judge opus --models gemma3_4B  # subset
"""
import argparse
import os
import sys

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

_ALIASES = {"opus": "claude-opus-4-8", "sonnet": "claude-sonnet-5",
            "haiku": "claude-haiku-4-5"}
_DEFAULT_MODELS = ["qwen3_4B", "deepseek_r1_distill_8B", "gemma3_4B"]


def _load_dotenv(path):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="opus", choices=list(_ALIASES),
                    help="judge alias (default: opus)")
    ap.add_argument("--judge-model", default=None,
                    help="raw model id, overrides --judge")
    ap.add_argument("--tag", default=None,
                    help="column suffix (default: derived from --judge)")
    ap.add_argument("--models", nargs="+", default=_DEFAULT_MODELS)
    ap.add_argument("--out-root", default="outputs")
    args = ap.parse_args()

    _load_dotenv(os.path.join(_REPO_ROOT, ".env"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set (put it in .env). Aborting -- no calls made.")

    judge_model = args.judge_model or _ALIASES[args.judge]
    tag = args.tag or (args.judge if not args.judge_model else judge_model.split("-")[1])
    vcol, rcol = f"judge_verbalized_{tag}", f"judge_reason_{tag}"

    from cot_faith.config import Stage3Config
    from cot_faith.faithfulness import judge_captured_rows

    cfg = Stage3Config()
    cfg.judge_model = judge_model

    print(f"Re-judging with {judge_model}  ->  columns {vcol!r}, {rcol!r}\n")

    for m in args.models:
        path = os.path.join(args.out_root, f"stage3_{m}", "stage3_results.csv")
        if not os.path.exists(path):
            print(f"[skip] {m}: {path} not found")
            continue
        df = pd.read_csv(path)
        captured = df[df["captured"] == True].copy()  # noqa: E712
        if captured.empty:
            print(f"[skip] {m}: no captured rows")
            continue

        print(f"=== {m}: judging {len(captured)} captured rows with {judge_model} ===")
        scored = judge_captured_rows(captured, cfg)
        scored = scored.set_index("id")

        # Write new columns aligned by id; leave non-captured rows blank.
        df[vcol] = df["id"].map(scored["judge_verbalized"])
        df[rcol] = df["id"].map(scored["judge_reason"])
        df.to_csv(path, index=False)

        # Per-arm comparison of the two judges (on captured rows only).
        cap = df[df["captured"] == True]  # noqa: E712
        print(f"\n  {'arm':<16}{'n':>5}{'Haiku%':>9}{f'{tag}%':>9}{'agree%':>9}")
        for arm, g in cap.groupby("dataset"):
            both = g.dropna(subset=["judge_verbalized", vcol])
            if both.empty:
                continue
            h = both["judge_verbalized"].astype(bool)
            n2 = both[vcol].astype(bool)
            agree = (h == n2).mean() * 100
            print(f"  {arm:<16}{len(both):>5}{h.mean()*100:>8.0f}%"
                  f"{n2.mean()*100:>8.0f}%{agree:>8.0f}%")
        print(f"  saved -> {path}\n")

    print("Done. New judge columns added; original judge_verbalized untouched.")
    print("For validation-sample kappa vs this judge, compare judge_verbalized_"
          f"{tag} against second_llm_judge_verbalized in the human_validation_sample.csv.")


if __name__ == "__main__":
    main()
