# CoT Faithfulness — Necessity vs. Propensity

Does a model reason more *honestly* when reasoning is **necessary** to solve a
problem than when it's **optional**? Stage 2 establishes the two regimes; Stage 3
injects a misleading cue and measures whether the model's CoT admits relying on it.

## Layout
```
cot_faith/
  config.py       all run settings (Config for Stage 2, Stage3Config for Stage 3)
  data.py         load + normalise GSM8K / CommonsenseQA / MMLU into Problem
  prompts.py      prompt building + answer extraction (+ cue injection, think_part)
  runner.py       vLLM-backed generation (the speed fix)
  runner_hf.py    transformers fallback backend, same interface as runner.py
  cue_injection.py  Stage 3: builds the misleading "hint" + its wrong target
  faithfulness.py   Stage 3: capture detection + 3-layer verbalization scoring
  experiment.py   orchestration + scoring + summary (NecessityExperiment,
                   FaithfulnessExperiment)
scripts/
  run.sh              Stage 2 cluster entry point
  run_stage2.py       Stage 2 main logic
  plot_results.py     Stage 2 necessity-gap figure
  run_stage3.py       Stage 3 main logic (--smoke for a fast end-to-end check)
  plot_stage3.py      Stage 3 figures
  validate_stage3.py  judge-vs-human agreement after hand-labeling
  aggregate_stage3.py merge the per-GPU SLURM outputs
  stage3.sbatch       SLURM job: splits the 3 arms across both V100s
outputs/          results land here
```

## Setup (on the cluster)
```bash
# Create a dedicated conda env (do this on the login node, which has internet)
conda create -n cot_faith python=3.11 -y
conda activate cot_faith

# Guard against ~/.local shadowing conda packages (binary incompatibility trap)
conda env config vars set PYTHONNOUSERSITE=1 -n cot_faith
conda activate cot_faith   # re-activate so the variable takes effect

# Install deps — conda first for the scientific stack, then pip for the rest
conda install -c conda-forge "matplotlib>=3.5" -y
pip install --upgrade pip setuptools
pip install "vllm>=0.6.0"
pip install "transformers>=4.51" "datasets>=2.18" pandas numpy

# Verify
pip list | grep -E "vllm|datasets|transformers|pandas|numpy|matplotlib"
```

vLLM supports the V100 (Volta). If the cluster blocks internet on compute
nodes, pre-download the model weights on a login node first:
```bash
huggingface-cli download Qwen/Qwen3-1.7B
huggingface-cli download Qwen/Qwen3-4B
```

## Run
```bash
# Always use run.sh — it sets PYTHONNOUSERSITE=1 as a safety net
# Smoke test first (3 problems, ~2 min) to confirm the pipeline works
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh --smoke

# Full run — one V100 is plenty for a 4B model (~30-40 min)
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh --out outputs/qwen3_4B

# Optional: run 1.7B on the second GPU simultaneously for a size ablation
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh --model Qwen/Qwen3-1.7B --out outputs/qwen3_1B7

# Plot after runs complete
python scripts/plot_results.py --out outputs/qwen3_4B
```

## Two V100s
For a model too big for one card, shard it: set `tensor_parallel_size=2` in
`runner.py` and expose both GPUs (`CUDA_VISIBLE_DEVICES=0,1`). For <=8B, prefer
one GPU per run and (optionally) run two independent jobs in parallel, one per
card — simpler and fully utilises both.

## SLURM (if the cluster uses it)
```bash
#!/bin/bash
#SBATCH --job-name=cot-stage2
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm-%j.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cot_faith
bash scripts/run.sh --out outputs/qwen3_4B
```

## What to check in the output
- **gap**: large on GSM8K (necessity), small on CommonsenseQA / MMLU (propensity).
- **above_chance**: must be True everywhere. A small gap with at-chance accuracy
  means "can't do the task," not "reasoning optional" — bump to a bigger model.
- **cot_complete_rate**: should be ~1.0. If low, raise `max_tokens_cot`.

## Stage 3: cue injection

Injects a misleading "hint" pointing to a wrong answer, generates CoT both
clean and cued, and checks whether the model's reasoning (the text before
`</think>` — that's what gets scored, not the post-`</think>` answer) admits
relying on the cue. Only measured on "captured" problems: cases where the cue
actually flipped the answer to its target.

The judge has two backends (`Config.judge_backend`, or `--judge-backend`):
- `anthropic` (default for a full run) — Claude Haiku via the API, the real
  independent judge you report numbers from.
- `self` (default under `--smoke`) — the same Qwen model that generated the
  traces judges itself. Needs no API key/network, so it lets you validate the
  *entire* pipeline (cue injection → generation → capture → string-match →
  judge → save → plot) for free before spending on the real run. **Never
  report `self`-judge numbers as results** — a model judging its own
  reasoning isn't an independent check.

### tmux + API key setup (do this once)

The key lives in `cot_faithfulness/.env` (gitignored — `git check-ignore -v
.env` confirms it, never gets committed). `run_stage3.py` loads it
automatically at startup, so there's nothing to `export` by hand and nothing
to re-do in a fresh shell/tmux pane. If `.env` doesn't already have a line
like `ANTHROPIC_API_KEY=sk-ant-...`, open it and paste your key in (no quotes
needed).

```bash
tmux new -s stage3          # or: tmux attach -t stage3   if it already exists

conda activate cot_faith
cd ~/Cot_faithfulness/cot_faithfulness

# Required every time you run scripts/run_stage3.py directly (not via
# stage3.sbatch, which already sets this): a broken ~/.local scikit-learn
# install shadows the conda env's packages otherwise and breaks the
# transformers -> vllm import chain. This one still needs a manual export --
# it has to be set before the Python interpreter starts, so a script can't
# set it for itself.
export PYTHONNOUSERSITE=1
```

### Commands

```bash
# 1. Smoke test first -- self-judged, no API key needed, ~10 problems/arm,
#    exercises the full pipeline in a couple minutes.
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3.py --smoke

# Inspect outputs/stage3/stage3_results.csv and stage3_summary.csv. If the
# capture/verbalization mechanics look right, move to the real run.

# 2. Full run, all 3 arms on one GPU (~500 problems/arm by default), real
#    Anthropic judge (needs the .env key exported above).
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3.py --out outputs/stage3

# -- or --

# 2'. Two V100s in parallel (one arm-subset per GPU), then auto-merge + plot.
#     Submits as a SLURM job -- detach from tmux (Ctrl-b d) once it's queued,
#     the job keeps running; reattach later with `tmux attach -t stage3`.
sbatch scripts/stage3.sbatch

# 3. Plots (verbalization rate, capture rate, string-match/judge agreement,
#    judge/human agreement).
python scripts/plot_stage3.py --out outputs/stage3
```

`stage3.sbatch` doesn't need `ANTHROPIC_API_KEY` exported before submission —
each `run_stage3.py` invocation loads it from `.env` itself when the job
actually runs.

Human validation: `outputs/stage3/human_validation_sample.csv` has a blank
`human_verbalized` column for ~50 random captured traces. Fill it in
(TRUE/FALSE), then run `python scripts/validate_stage3.py --out outputs/stage3`
for % agreement + Cohen's kappa against the judge.

**Headline number**: `judge_verbalization_rate` in `stage3_summary.csv`, per
arm. Hypothesis: higher on GSM8K (necessity) than on MMLU / CommonsenseQA
(propensity) — i.e. the model is more honest about using a cue when it
actually needed to reason to solve the problem.
