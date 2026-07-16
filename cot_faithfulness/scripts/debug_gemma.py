#!/usr/bin/env python
"""One-off diagnostic for the Gemma empty-output issue -- not part of the
main pipeline. Renders one real prompt, generates a short response via vLLM
directly, and prints the raw token IDs plus decoded text with and without
skip_special_tokens, so we can see exactly what the model is emitting
instead of inferring it indirectly from an empty final string.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/debug_gemma.py
"""
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPTS_DIR))  # repo root, for cot_faith imports
sys.path.insert(0, _SCRIPTS_DIR)                    # scripts/ itself, for run_stage2 import

# Reuse the exact same CUDA-lib-path / re-exec fix as the real entry points.
from run_stage2 import _ensure_cuda_libs_on_path  # noqa: E402
_ensure_cuda_libs_on_path()

from cot_faith.config import Config  # noqa: E402
from cot_faith.data import load_problems  # noqa: E402
from cot_faith.prompts import build_messages  # noqa: E402
from cot_faith.runner import _patch_gemma3_config_compat  # noqa: E402

_patch_gemma3_config_compat()

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "google/gemma-3-4b-it"

print(f"Loading {MODEL} ...")
llm = LLM(
    model=MODEL,
    dtype="float16",
    seed=0,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
    enforce_eager=True,
    enable_chunked_prefill=False,
    max_model_len=8192,
)
tok = llm.get_tokenizer()

problems = load_problems(("GSM8K",), 1)
p = problems[0]
messages = build_messages(p, use_cot=True)
prompt_text = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
)

print("\n=== RENDERED PROMPT ===")
print(repr(prompt_text[:800]))

print("\n=== GENERATING (100 tokens, no stop tokens at all -- see everything) ===")
params = SamplingParams(temperature=0.0, max_tokens=100, skip_special_tokens=False)
outputs = llm.generate([prompt_text], params)
out = outputs[0].outputs[0]

print("\nfinish_reason:", out.finish_reason)
print("num raw token ids:", len(out.token_ids))
print("first 20 token ids:", out.token_ids[:20])
print("last 20 token ids:", out.token_ids[-20:])

print("\n=== decoded WITHOUT skip_special_tokens (raw, everything visible) ===")
print(repr(tok.decode(out.token_ids, skip_special_tokens=False)))

print("\n=== decoded WITH skip_special_tokens (what our pipeline actually uses) ===")
print(repr(tok.decode(out.token_ids, skip_special_tokens=True)))

print("\n=== vLLM's own .text field (what Runner.generate() actually returns) ===")
print(repr(out.text))
