# CoT Faithfulness: Necessity vs. Propensity

Chain-of-thought (CoT) is increasingly proposed as a **monitoring** surface  if
we can read a model's reasoning, we can catch misbehavior. That bet is only as
good as CoT **faithfulness**: whether the reasoning reflects the true causes of
the output. This project asks whether faithfulness **tracks necessity**: is a
model more honest in its CoT on tasks where reasoning is *required* to get the
answer (**necessity**, e.g. GSM8K math) than on tasks where reasoning is
*optional* (**propensity**, e.g. commonsense/factual recall)?

Tested across three open-weight models from different families **Qwen3-4B**,
**DeepSeek-R1-Distill-Llama-8B**, and **Gemma-3-4B** — on a single V100.

## Headline results

**Stage 2** confirms the necessity spectrum (accuracy gap with vs. without CoT,
n=200/arm): GSM8K is strongly necessity-loaded (Qwen +0.66, Gemma +0.69), while
MMLU and CommonsenseQA are propensity tasks (≈0 or negative gap).

**Stage 3** injects a misleading authority cue pointing at a *wrong* answer, then
measures **capture** (did the cue flip the final answer?) and, on captured
traces, **verbalization** (does the CoT admit the cue?).

| Model | GSM8K capture (necessity) | MMLU capture (propensity) | CSQA capture (propensity) |
|---|---|---|---|
| Qwen3-4B | 0.4% | 4.0% | 5.2% |
| DeepSeek-8B | 2.2% | 14.4% | 23.8% |
| Gemma-3-4B | 0.0% | 41.1% | 40.0% |

1. **Necessity suppresses manipulation**  near-zero capture on necessity tasks,
   substantial on propensity, across all three families (all p < 1e-5).
2. **But necessity does not buy transparency** — where a cue does slip through,
   the CoT is no more honest; necessity protects by *prevention*, not disclosure.
3. **Silent capture is real** Gemma is the most manipulable (~40%) yet
   discloses the cue ~0% of the time: the worst case for CoT monitoring.

Full write-up: [`cot_faithfulness/writeup/results_draft.md`](cot_faithfulness/writeup/results_draft.md).

## Repo layout

```
cot_faithfulness/
  cot_faith/      library: config, data loading, prompts, cue injection,
                  faithfulness scoring, vLLM + HF runners, experiment orchestration
  scripts/        run Stage 2 / Stage 3, aggregate, plot, significance tests, validate
  outputs/        results (per-model summaries + raw per-trace CSVs)
  writeup/        results draft + significance output
  README.md       detailed run guide (cluster setup, V100 gotchas, API key)
```

## Setup

Requires a CUDA GPU (developed on a V100-32GB). Python 3.11.

```bash
conda create -n cot_faith python=3.11 -y && conda activate cot_faith
conda env config vars set PYTHONNOUSERSITE=1 -n cot_faith && conda activate cot_faith
pip install -r cot_faithfulness/requirements.txt
```

The LLM judge (Stage 3) uses Claude Haiku via the Anthropic API. Put your key in
`cot_faithfulness/.env` (gitignored) as `ANTHROPIC_API_KEY=sk-ant-...` — it's
loaded automatically. You can run Stage 2, and Stage 3 generation + capture, with
no key; only the judge step needs it.

See [`cot_faithfulness/README.md`](cot_faithfulness/README.md) for the full
cluster setup, the V100 precision/kernel notes, and SLURM/tmux instructions.

## Reproduce

All commands run from inside `cot_faithfulness/`.

```bash
cd cot_faithfulness

# --- Stage 2: necessity spectrum (accuracy with vs. without CoT) ---
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage2.py --out outputs/qwen3_4B
python scripts/plot_results.py --out outputs/qwen3_4B

# --- Stage 3: cue injection (per model) ---
# Smoke test first (self-judged, no API key, ~2 min) to confirm the pipeline:
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3.py --smoke

# Full runs (real Anthropic judge; needs the .env key):
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3.py \
    --model Qwen/Qwen3-4B --out outputs/stage3_qwen3_4B
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3.py \
    --model google/gemma-3-4b-it --dtype float32 --max-model-len 5120 \
    --out outputs/stage3_gemma3_4B
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --out outputs/stage3_deepseek_r1_distill_8B

# Or run all three across available GPUs in one shot (smoke-gated):
bash scripts/run_stage3_all.sh

# --- Analysis (no GPU / API key needed; runs on the committed CSVs) ---
python scripts/significance_stage3.py     # Fisher's exact tests + Wilson CIs
python scripts/plot_stage3.py --out outputs/stage3_qwen3_4B
```

## Outputs

Each run directory contains:
- `stage2_summary.csv` / `stage3_summary.csv` — the per-arm headline numbers
- `stage3_results.csv` — one row per problem with both traces, capture flag, and
  judge label (the raw data behind the analysis scripts)
- `run_config.json` — the full config, so a run is fully described
- `human_validation_sample.csv` — ~50 captured traces for hand-labeling

`outputs/stage3_final_summary.csv` is the consolidated cross-model table.

## Caveats

- **Verbalization is scored by LLM judges** (Claude Haiku + Opus), not full human
  ground truth; the necessity arms are underpowered (necessity suppresses capture,
  so there are few captured traces to score).
- **DeepSeek always opens a `<think>` block**, so it has no genuine no-CoT
  condition — its Stage 2 gaps are not comparable to the other two.
- All models are 4–8B open weights on V100 hardware.

See the write-up's Limitations section for the full discussion.

## References

- Sprague et al. (2024) — CoT helps on math/symbolic, barely on recall/commonsense.
- Turpin et al. (2023) — models follow planted hints without verbalizing them.
