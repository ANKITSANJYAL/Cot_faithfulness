#!/bin/bash
#SBATCH --job-name=cot-stage3
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/stage3/slurm-%j.out
#
# Stage 3 faithfulness run, split across both V100s. Since Stage 3 uses a
# single model (Qwen3-4B -- it's the one whose Stage 2 gaps we're extending),
# the split is by ARM, not by model: one process per GPU via
# CUDA_VISIBLE_DEVICES, each running independent problems, so there's no
# cross-GPU coordination beyond `wait`. Do NOT set tensor_parallel_size=2 in
# runner.py for this -- a 4B model fits on one V100, sharding it across both
# would just add overhead for nothing.
#
# GPU0 takes GSM8K alone (longest CoT budget, heaviest arm).
# GPU1 takes CommonsenseQA + MMLU sequentially (lighter CoT budgets each).
# Rebalance these --arms lists after a smoke test shows actual per-arm timing:
#   CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3.py --smoke --arms GSM8K

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cot_faith

export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=0
export TRANSFORMERS_VERBOSITY=error

# ANTHROPIC_API_KEY doesn't need to be exported here -- run_stage3.py loads
# it from cot_faithfulness/.env automatically (and warns at runtime if it's
# missing from both the environment and .env).

OUT=outputs/stage3
mkdir -p "$OUT"

CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3.py \
    --arms GSM8K \
    --out "$OUT/gpu0_gsm8k" \
    > "$OUT/gpu0.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 python scripts/run_stage3.py \
    --arms CommonsenseQA MMLU \
    --out "$OUT/gpu1_csqa_mmlu" \
    > "$OUT/gpu1.log" 2>&1 &
PID1=$!

wait $PID0
STATUS0=$?
wait $PID1
STATUS1=$?

if [ $STATUS0 -ne 0 ] || [ $STATUS1 -ne 0 ]; then
    echo "One or both GPU jobs failed (gpu0=$STATUS0 gpu1=$STATUS1) -- check $OUT/gpu*.log" >&2
    exit 1
fi

# Aggregation: merge the per-GPU results into one run, recompute the
# cross-arm summary, and re-sample the human validation set from the pooled data.
python scripts/aggregate_stage3.py \
    --inputs "$OUT/gpu0_gsm8k" "$OUT/gpu1_csqa_mmlu" \
    --out "$OUT"

python scripts/plot_stage3.py --out "$OUT"

echo "Stage 3 complete -> $OUT"
