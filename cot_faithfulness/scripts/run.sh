#!/bin/bash
# Wrapper for run_stage2.py that guards against common cluster environment issues.
#
# - PYTHONNOUSERSITE=1    prevents ~/.local packages from shadowing conda env
# - CUDA_DEVICE_ORDER     ensures GPU indices match nvidia-smi / CUDA_VISIBLE_DEVICES
# - VLLM_USE_V1=0         disables vLLM v1 engine (workaround for some glibc issues)
#
# If vLLM still fails (e.g. glibc too old for llguidance), pass --backend hf
# to use HuggingFace transformers instead. Slower but compatible everywhere.
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=0
# Suppress "generation flags not valid" warning — temperature/top_p are in the
# model's generation_config but ignored under greedy (do_sample=False).
export TRANSFORMERS_VERBOSITY=error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/run_stage2.py" "$@"
