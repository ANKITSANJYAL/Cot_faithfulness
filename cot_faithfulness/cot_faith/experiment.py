"""Stage 2 experiment: run both conditions, score, summarise.

Keeps the orchestration separate from the model details so this logic is easy
to read and reuse for Stage 3.
"""
import os
import json
import time
from typing import List

import pandas as pd

from .data import Problem
from .prompts import extract, is_correct, format_question, think_part, answer_part
from .cue_injection import inject_cues
from .faithfulness import (is_captured, string_match_score,
                           judge_captured_rows, judge_captured_rows_self)


class NecessityExperiment:
    def __init__(self, cfg, runner):
        self.cfg = cfg
        self.runner = runner

    def run(self, problems: List[Problem]) -> pd.DataFrame:
        # Group by dataset so we can apply per-dataset token budgets and
        # save a checkpoint after each dataset finishes.
        by_dataset: dict = {}
        for p in problems:
            by_dataset.setdefault(p.dataset, []).append(p)

        all_rows: List[dict] = []
        for ds_name, ds_problems in by_dataset.items():
            # Check if checkpoint already exists (allows resuming after a crash).
            ckpt_path = os.path.join(self.cfg.out_dir, f"checkpoint_{ds_name}.csv")
            if os.path.exists(ckpt_path):
                print(f"  [resume] Loading existing checkpoint for {ds_name} from {ckpt_path}")
                ckpt_df = pd.read_csv(ckpt_path)
                all_rows.extend(ckpt_df.to_dict("records"))
                continue

            # Per-dataset CoT token budget — drastically reduces wall time.
            # CommonsenseQA thinking traces are short (recall task); MMLU can spike.
            max_cot = getattr(self.cfg, "max_tokens_cot_by_dataset", {}).get(
                ds_name, self.cfg.max_tokens_cot
            )

            print(f"\n--- Dataset: {ds_name} ({len(ds_problems)} problems, "
                  f"max_cot_tokens={max_cot}) ---")
            cot_texts = self.runner.generate(ds_problems, use_cot=True,
                                             max_new_tokens=max_cot)
            dir_texts = self.runner.generate(ds_problems, use_cot=False)

            rows = []
            for p, cot_txt, dir_txt in zip(ds_problems, cot_texts, dir_texts):
                cot_pred = extract(p, cot_txt)
                dir_pred = extract(p, dir_txt)
                rows.append({
                    "id": p.id, "dataset": p.dataset, "gold": p.gold,
                    "cot_pred": cot_pred,
                    "cot_correct": is_correct(cot_pred, p.gold, p.is_numeric),
                    "direct_pred": dir_pred,
                    "direct_correct": is_correct(dir_pred, p.gold, p.is_numeric),
                    "cot_reached_end": "</think>" in cot_txt,
                    "cot_text": cot_txt,
                })

            ds_df = pd.DataFrame(rows)
            os.makedirs(self.cfg.out_dir, exist_ok=True)
            ds_df.to_csv(ckpt_path, index=False)
            n_cot = ds_df["cot_correct"].sum()
            n_dir = ds_df["direct_correct"].sum()
            n_complete = ds_df["cot_reached_end"].sum()
            print(f"  {ds_name}: cot_acc={n_cot}/{len(rows)}  "
                  f"dir_acc={n_dir}/{len(rows)}  "
                  f"cot_complete={n_complete}/{len(rows)}")
            print(f"  Checkpoint saved → {ckpt_path}")
            all_rows.extend(rows)

        return pd.DataFrame(all_rows)

    def summarise(self, df: pd.DataFrame) -> pd.DataFrame:
        s = (df.groupby("dataset")
               .agg(n=("id", "count"),
                    with_cot_acc=("cot_correct", "mean"),
                    no_cot_acc=("direct_correct", "mean"),
                    cot_complete_rate=("cot_reached_end", "mean"))
            )
        s["gap"] = s["with_cot_acc"] - s["no_cot_acc"]
        s["chance"] = s.index.map(self.cfg.chance)
        s["above_chance"] = s["with_cot_acc"] > s["chance"] + 0.05
        return s.round(3)

    def save(self, df: pd.DataFrame, summary: pd.DataFrame):
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        df.to_csv(os.path.join(self.cfg.out_dir, "stage2_results.csv"), index=False)
        summary.to_csv(os.path.join(self.cfg.out_dir, "stage2_summary.csv"))
        with open(os.path.join(self.cfg.out_dir, "run_config.json"), "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)
        # Remove per-dataset checkpoints now that the final file is written.
        for ds_name in df["dataset"].unique():
            ckpt_path = os.path.join(self.cfg.out_dir, f"checkpoint_{ds_name}.csv")
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)


class FaithfulnessExperiment:
    """Stage 3: inject a misleading cue, see whether the CoT admits relying on
    it, and compare that "verbalization rate" across the necessity spectrum
    established in Stage 2 (GSM8K necessity > MMLU neutral > CommonsenseQA
    counterproductive).

    Per arm: generate CoT twice per problem (clean, cued), detect which cued
    runs got "captured" (flipped to the cue's wrong target), then score only
    the captured traces -- faithfulness is only meaningful where the cue
    demonstrably changed behavior. LLM-judge scoring is a separate step
    (score_with_judge) so callers without network access can still run
    generation + capture detection + the string-match layer.
    """

    def __init__(self, cfg, runner):
        self.cfg = cfg
        self.runner = runner

    def run(self, problems: List[Problem]) -> pd.DataFrame:
        by_dataset: dict = {}
        for p in problems:
            by_dataset.setdefault(p.dataset, []).append(p)

        all_rows: List[dict] = []
        for ds_name, ds_problems in by_dataset.items():
            ckpt_path = os.path.join(self.cfg.out_dir, f"checkpoint_stage3_{ds_name}.csv")
            if os.path.exists(ckpt_path):
                print(f"  [resume] Loading existing checkpoint for {ds_name} from {ckpt_path}")
                ckpt_df = pd.read_csv(ckpt_path)
                all_rows.extend(ckpt_df.to_dict("records"))
                continue

            max_cot = self.cfg.max_tokens_cot_by_dataset.get(ds_name, self.cfg.max_tokens_cot)
            cues = inject_cues(ds_problems, self.cfg.cue_type, self.cfg.seed)

            print(f"\n--- Stage 3 arm: {ds_name} ({len(ds_problems)} problems, "
                  f"max_cot_tokens={max_cot}, cue_type={self.cfg.cue_type}) ---")

            t0 = time.time()
            clean_texts, clean_reached_ends = self.runner.generate_with_status(
                ds_problems, use_cot=True, max_new_tokens=max_cot)
            t_clean = time.time() - t0

            t1 = time.time()
            cue_texts = [c.cue_text for c in cues]
            cued_texts, cued_reached_ends = self.runner.generate_with_status(
                ds_problems, use_cot=True, max_new_tokens=max_cot, cues=cue_texts)
            t_cued = time.time() - t1

            tok = self.runner.tokenizer
            rows = []
            for p, cue, clean_txt, cued_txt, clean_reached_end, cued_reached_end in zip(
                    ds_problems, cues, clean_texts, cued_texts,
                    clean_reached_ends, cued_reached_ends):
                clean_pred = extract(p, clean_txt)
                cued_pred = extract(p, cued_txt)
                captured = is_captured(clean_pred, cued_pred, cue.target, p.is_numeric,
                                       cued_reached_end=cued_reached_end,
                                       clean_reached_end=clean_reached_end)
                cued_thinking = think_part(cued_txt)
                sm = string_match_score(cued_thinking, cue) if captured else None
                rows.append({
                    "id": p.id, "dataset": p.dataset, "gold": p.gold,
                    "question": format_question(p),
                    "cue_text": cue.cue_text, "cue_target": cue.target,
                    "clean_pred": clean_pred, "cued_pred": cued_pred,
                    "captured": captured,
                    "clean_cot_reached_end": clean_reached_end,
                    "cued_cot_reached_end": cued_reached_end,
                    "clean_tokens": len(tok.encode(clean_txt)),
                    "cued_tokens": len(tok.encode(cued_txt)),
                    "thinking_text": cued_thinking,
                    # The literal text extract() ran on -- kept so a bad
                    # cued_pred is directly diagnosable (e.g. the answer
                    # segment echoing the cue's number before the model's
                    # real final answer) without having to reproduce it.
                    "cued_answer_text": answer_part(cued_txt),
                    "string_match_verbalized": sm,
                })

            ds_df = pd.DataFrame(rows)
            os.makedirs(self.cfg.out_dir, exist_ok=True)
            ds_df.to_csv(ckpt_path, index=False)

            n_captured = int(ds_df["captured"].sum())
            capture_rate = n_captured / len(rows) if rows else 0.0
            avg_tok = ds_df["cued_tokens"].mean() if rows else 0.0
            max_tok = ds_df["cued_tokens"].max() if rows else 0
            cap_hit_rate = 1 - ds_df["cued_cot_reached_end"].mean() if rows else 0.0
            print(f"  {ds_name}: capture_rate={capture_rate:.3f} ({n_captured}/{len(rows)})  "
                  f"clean_gen={t_clean / 60:.1f}m  cued_gen={t_cued / 60:.1f}m  "
                  f"avg_cued_tokens={avg_tok:.0f}  max_cued_tokens={max_tok:.0f}  "
                  f"cued_max_token_cap_hit_rate={cap_hit_rate:.3f}")
            if n_captured < self.cfg.min_captures_for_stable_rate:
                print(f"  WARNING: only {n_captured} captures in {ds_name} -- below "
                      f"min_captures_for_stable_rate={self.cfg.min_captures_for_stable_rate}. "
                      f"The verbalization rate for this arm will be noisy; raise "
                      f"n_per_arm_by_dataset[{ds_name!r}].")
            print(f"  Checkpoint saved -> {ckpt_path}")
            all_rows.extend(rows)

        return pd.DataFrame(all_rows)

    def score_with_judge(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds judge_verbalized/judge_reason, scored only on captured rows.
        Dispatches on cfg.judge_backend: "anthropic" (real judge, Claude
        Haiku) or "self" (the generating Qwen model judges itself -- smoke
        test only, see faithfulness.judge_captured_rows_self).
        """
        captured = df[df["captured"] == True].copy()  # noqa: E712
        if captured.empty:
            df = df.copy()
            df["judge_verbalized"] = None
            df["judge_reason"] = None
            return df

        backend = getattr(self.cfg, "judge_backend", "anthropic")
        t0 = time.time()
        if backend == "self":
            print(f"\n--- Self-judge scoring {len(captured)} captured traces "
                  f"({self.cfg.model_name}, SMOKE-TEST ONLY) ---")
            scored = judge_captured_rows_self(captured, self.runner, self.cfg)
        else:
            print(f"\n--- LLM judge scoring {len(captured)} captured traces "
                  f"({self.cfg.judge_model}) ---")
            scored = judge_captured_rows(captured, self.cfg)
        print(f"  judge scoring done in {(time.time() - t0) / 60:.1f}m")

        merged = df.merge(scored[["id", "judge_verbalized", "judge_reason"]],
                          on="id", how="left")
        return merged

    def summarise(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def save(self, df: pd.DataFrame, summary: pd.DataFrame):
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        df.to_csv(os.path.join(self.cfg.out_dir, "stage3_results.csv"), index=False)
        summary.to_csv(os.path.join(self.cfg.out_dir, "stage3_summary.csv"))
        with open(os.path.join(self.cfg.out_dir, "run_config.json"), "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)
        for ds_name in df["dataset"].unique():
            ckpt_path = os.path.join(self.cfg.out_dir, f"checkpoint_stage3_{ds_name}.csv")
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
