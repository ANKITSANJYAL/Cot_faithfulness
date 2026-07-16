#!/bin/bash
# Run Stage 3 (cue-injection faithfulness) for all three model families,
# parallelized across however many GPUs are free on this node.
#
# For each model: smoke test (n=5/arm, self-judged so no API key burned
# before we know generation even works) -> only if that exits 0, the full
# run (config defaults: GSM8K=500/MMLU=800/CommonsenseQA=400, real Claude
# Haiku judge). A model whose smoke test fails is skipped and reported, it
# does not block the other two.
#
# Concurrency is capped at NUM_GPUS via flock on one lock file per GPU index
# -- not a fixed round-robin assignment, since 3 models don't divide evenly
# over 2 GPUs. Whichever GPU a job's flock acquires first is the one it uses
# for BOTH its smoke test and (if smoke passes) its full run, so a model
# never migrates GPUs mid-pipeline.
#
# Usage:
#   bash scripts/run_stage3_all.sh          # run inside tmux -- full runs
#                                             # can take hours per model
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

NUM_GPUS=2
LOCK_DIR=/tmp/cotfaith_gpu_locks
mkdir -p "$LOCK_DIR"
for ((i=0; i<NUM_GPUS; i++)); do : > "$LOCK_DIR/gpu$i.lock"; done

run_model() {
    local name="$1" model="$2" extra="$3"
    local gpu=-1 fd

    # Try each GPU's lock non-blocking first; if all are busy, block on GPU 0.
    for ((i=0; i<NUM_GPUS; i++)); do
        exec {fd}<>"$LOCK_DIR/gpu$i.lock"
        if flock -n "$fd"; then
            gpu=$i
            break
        fi
        exec {fd}<&-
    done
    if [ "$gpu" = -1 ]; then
        exec {fd}<>"$LOCK_DIR/gpu0.lock"
        flock "$fd"
        gpu=0
    fi

    echo "[$name] acquired GPU $gpu -- smoke test (n=5/arm, self-judged)..."
    if CUDA_VISIBLE_DEVICES=$gpu python scripts/run_stage3.py --model "$model" $extra \
            --n 5 --judge-backend self --out "outputs/stage3_smoke_${name}" \
            > "logs/stage3_smoke_${name}.log" 2>&1; then
        echo "[$name] smoke PASSED (GPU $gpu) -- starting full run..."
        if CUDA_VISIBLE_DEVICES=$gpu python scripts/run_stage3.py --model "$model" $extra \
                --out "outputs/stage3_${name}" \
                > "logs/stage3_full_${name}.log" 2>&1; then
            echo "[$name] FULL RUN DONE -- outputs/stage3_${name}/"
        else
            echo "[$name] FULL RUN FAILED -- see logs/stage3_full_${name}.log"
        fi
    else
        echo "[$name] SMOKE FAILED -- see logs/stage3_smoke_${name}.log -- skipping full run"
    fi
    exec {fd}<&-
}

pids=()
run_model "qwen3_4B" "Qwen/Qwen3-4B" "" &
pids+=($!)
run_model "gemma3_4B" "google/gemma-3-4b-it" "--dtype float32 --max-model-len 5120" &
pids+=($!)
run_model "deepseek_r1_distill_8B" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "" &
pids+=($!)

wait "${pids[@]}"
echo
echo "=== All Stage 3 pipelines finished. Check logs/stage3_*.log and outputs/stage3_*/  ==="
