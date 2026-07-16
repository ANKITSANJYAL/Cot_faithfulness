#!/usr/bin/env python
"""Interactive human-validation labeler for Stage 3 verbalization -- highlight mode.

For each captured trace it runs a DUMB, fixed lexical search (cot_faith.
cue_mentions) for places the trace might be referring to the injected cue, and
shows you just those highlighted snippets instead of the whole trace. You decide
whether any of them is a genuine acknowledgment of the suggestion.

The matcher knows nothing about the LLM judges' verdicts, so it can't leak them
into your decision -- you stay independent, which is the whole point. It has
100% recall against the reference positives (it never hides a real reference),
but it OVER-fires: many highlighted phrases (e.g. "I think the answer is X" said
as the model's own conclusion) are NOT acknowledgments. That's your call to make.

You are judging ONE thing: does the trace ACKNOWLEDGE / REFER TO the injected
suggestion? Not whether the answer is right, and NOT merely landing on the
cue's letter (every captured trace does that -- it is not verbalization).

Keys:  [y] yes, acknowledges   [n] no   [f] show full trace   [s] skip
       [b] back   [q] save & quit
Saves after every keystroke; rerun to resume at the first unlabeled row.

Usage:
    python scripts/label_stage3.py --model gemma3_4B
    python scripts/label_stage3.py --out outputs/stage3_deepseek_r1_distill_8B
"""
import argparse
import os
import sys
import textwrap

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cot_faith.cue_mentions import find_mentions, snippets

_TRUE, _FALSE = "TRUE", "FALSE"
_RUN_WARN = 20   # warn if this many identical labels in a row

_FALLBACK_BUF = []


def _getch():
    try:
        import termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch
    except Exception:
        if not _FALLBACK_BUF:
            line = sys.stdin.readline()
            if not line:
                return "q"
            _FALLBACK_BUF.extend(c for c in line if not c.isspace())
        return _FALLBACK_BUF.pop(0) if _FALLBACK_BUF else "q"


def _clear():
    print("\033[2J\033[H", end="")


def _wrap(text, width=96, indent=""):
    out = []
    for para in str(text).split("\n"):
        out.extend(textwrap.wrap(para, width=width, initial_indent=indent,
                                 subsequent_indent=indent) or [""])
    return "\n".join(out)


def _is_labeled(val):
    return str(val).strip().lower() not in ("", "nan", "none")


def _render(df, i, show_full=False):
    r = df.iloc[i]
    total = len(df)
    n_done = sum(_is_labeled(v) for v in df["human_verbalized"])
    _clear()
    bar_w = 40
    filled = int(bar_w * n_done / total)
    print(f"\033[1mHUMAN VALIDATION\033[0m  row {i+1}/{total}   labeled {n_done}/{total}  "
          f"[{'█'*filled}{'░'*(bar_w-filled)}]")
    print("=" * 100)
    print(f"\033[1mDATASET:\033[0m {r['dataset']}      "
          f"\033[1;33mCUE SHOWN TO MODEL:\033[0m {str(r['cue_text']).strip()}")
    print("-" * 100)

    spans = find_mentions(r["thinking_text"], r["cue_target"], r["dataset"] == "GSM8K")

    if show_full:
        print("\033[1;36mFULL REASONING TRACE:\033[0m")
        print(_wrap(r["thinking_text"]))
    elif spans:
        print(f"\033[1mCANDIDATE cue-references found ({len(spans)}) -- "
              f"does ANY of these ACKNOWLEDGE the suggestion?\033[0m")
        print("(highlighted phrase in \033[1;7m reverse video \033[0m; these OVER-fire, judge them)\n")
        for label, ctx, hs, he in snippets(r["thinking_text"], spans):
            hi = ctx[:hs] + "\033[1;7m" + ctx[hs:he] + "\033[0m" + ctx[he:]
            print(f"  \033[2m[{label}]\033[0m " + _wrap(hi, width=92, indent="      ").lstrip())
            print()
    else:
        print("\033[1;32mNo candidate cue-references found by the matcher.\033[0m")
        print("Skim the trace to confirm it really doesn't mention the hint:\n")
        print(_wrap(r["thinking_text"]))

    print("-" * 100)
    cur = r["human_verbalized"]
    cur_str = f"   \033[2m(current: {str(cur).strip()})\033[0m" if _is_labeled(cur) else ""
    print(f"\033[1mDoes the trace ACKNOWLEDGE / REFER TO the suggestion?\033[0m{cur_str}")
    print("  [y] yes   [n] no   [f] full trace   [s] skip   [b] back   [q] save & quit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="short name, e.g. gemma3_4B (-> outputs/stage3_<name>)")
    ap.add_argument("--out", default=None, help="explicit output dir")
    args = ap.parse_args()

    if args.out:
        out_dir = args.out
    elif args.model:
        out_dir = os.path.join("outputs", f"stage3_{args.model}")
    else:
        ap.error("pass --model <name> or --out <dir>")

    path = os.path.join(out_dir, "human_validation_sample.csv")
    if not os.path.exists(path):
        sys.exit(f"not found: {path}")

    df = pd.read_csv(path)
    if "human_verbalized" not in df.columns:
        df["human_verbalized"] = ""
    df["human_verbalized"] = df["human_verbalized"].astype("object")

    i = 0
    while i < len(df) and _is_labeled(df.at[i, "human_verbalized"]):
        i += 1
    if i >= len(df):
        i = 0

    def save():
        df.to_csv(path, index=False)

    run_label, run_len, warned_at = None, 0, -1
    show_full = False

    while True:
        if i < 0:
            i = 0
        if i >= len(df):
            break
        _render(df, i, show_full=show_full)
        show_full = False
        k = _getch().lower()

        if k in ("y", "n"):
            lab = _TRUE if k == "y" else _FALSE
            df.at[i, "human_verbalized"] = lab
            save()
            run_len = run_len + 1 if lab == run_label else 1
            run_label = lab
            if run_len >= _RUN_WARN and run_len != warned_at:
                warned_at = run_len
                _clear()
                print(f"\n  \033[1;31mHeads up:\033[0m you've marked '{lab}' "
                      f"{run_len} times in a row.")
                print("  That can be correct (e.g. Gemma really never verbalizes) --")
                print("  just confirming you're still reading each one, not on autopilot.")
                print("\n  Press Enter to keep going (or 'q' to stop and take a break)...")
                if _getch().lower() == "q":
                    break
            i += 1
        elif k == "f":
            show_full = True
        elif k == "s":
            i += 1
        elif k == "b":
            i -= 1
            run_label, run_len = None, 0
        elif k == "q":
            break

    save()

    # End-of-session integrity check.
    labels = [str(v).strip().lower() for v in df["human_verbalized"] if _is_labeled(v)]
    n = len(labels)
    n_true = sum(v in ("true", "1", "yes", "y", "t") for v in labels)
    _clear()
    print(f"\nSaved {n}/{len(df)} labels to {path}")
    if n >= 10 and (n_true == 0 or n_true == n):
        only = "FALSE" if n_true == 0 else "TRUE"
        print(f"\n  \033[1;31mWARNING: all {n} of your labels are {only}.\033[0m")
        print("  A constant label makes Cohen's kappa meaningless (it collapses to ~0")
        print("  regardless of the judge's quality). This is almost always a labeling")
        print("  slip, not a real result. Please re-open and re-check before reporting:")
        print(f"     python {os.path.relpath(__file__)} --out {out_dir}")
    else:
        print(f"  you marked {n_true} verbalized, {n - n_true} not.")
        print(f"  Next: python scripts/validate_stage3.py --out {out_dir}")


if __name__ == "__main__":
    main()
