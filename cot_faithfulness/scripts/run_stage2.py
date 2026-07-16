#!/usr/bin/env python
"""Run the Stage 2 necessity/propensity experiment.

Usage on the cluster:
    python scripts/run_stage2.py
    python scripts/run_stage2.py --model Qwen/Qwen3-8B --n 300

Pick the GPU with CUDA_VISIBLE_DEVICES, e.g.:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_stage2.py
"""
import argparse
import glob
import sys
import os


def _ensure_cuda_libs_on_path():
    """Point LD_LIBRARY_PATH at this env's own pip-installed nvidia-*-cuXX
    packages before torch/vllm ever get imported.

    vLLM's compiled _C extension dlopen()s libcudart.so.12 directly at
    import time. torch preloads its OWN bundled CUDA runtime at import (its
    _load_global_deps()), but that only covers the cu11 libs matching our
    cu118 torch build -- vLLM's cu12 dependency is a separate package
    (nvidia-cuda-runtime-cu12) that torch's preloading doesn't touch, so it's
    resolved via the plain dynamic-linker search path instead. That path is
    apparently node-dependent on this cluster: this exact environment
    imported vllm fine on one node and failed with
    "ImportError: libcudart.so.12: cannot open shared object file" on
    another.

    Setting os.environ["LD_LIBRARY_PATH"] alone is not enough -- confirmed by
    testing: some dynamic linkers only consult LD_LIBRARY_PATH once, at
    process start, and don't re-read it for a dlopen() happening later in an
    already-running process. So: set it, then re-exec this same script as a
    brand new process (guarded by a marker env var so this only happens
    once) -- that's the same effect as if LD_LIBRARY_PATH had been exported
    in the shell before python was ever invoked.
    """
    if os.environ.get("_COT_FAITH_CUDA_LIBS_FIXED") == "1":
        return  # already re-exec'd once with the fix in place; don't loop
    env_root = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
    nvidia_lib_dirs = glob.glob(
        os.path.join(env_root, "lib", "python3.*", "site-packages", "nvidia", "*", "lib")
    )
    if not nvidia_lib_dirs:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_lib_dirs) + (":" + current if current else "")
    os.environ["_COT_FAITH_CUDA_LIBS_FIXED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_cuda_libs_on_path()

# allow running from repo root without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cot_faith.config import Config
from cot_faith.data import load_problems
from cot_faith.experiment import NecessityExperiment


def _release_cuda_memory():
    """A vLLM Runner() that fails partway through construction can leave GPU
    memory allocated (weights already loaded, KV cache already reserved)
    even after the exception -- without this, the HF fallback tries to load
    its own copy of the model into the same still-encumbered GPU and OOMs
    immediately, on a model that would otherwise fit comfortably.
    """
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _load_runner(backend: str, cfg):
    if backend == "hf":
        from cot_faith.runner_hf import HFRunner
        return HFRunner(cfg)
    if backend == "vllm":
        from cot_faith.runner import Runner
        return Runner(cfg)
    # auto: try vllm, fall back to hf
    try:
        from cot_faith.runner import Runner
        return Runner(cfg)
    except Exception as e:
        print(f"  vLLM unavailable ({type(e).__name__}: {e})")
        print("  Falling back to HuggingFace transformers backend.")
        _release_cuda_memory()
        from cot_faith.runner_hf import HFRunner
        return HFRunner(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="override model_name")
    ap.add_argument("--n", type=int, default=None, help="problems per dataset")
    ap.add_argument("--out", default=None, help="output dir")
    ap.add_argument("--dataset", default=None, choices=["GSM8K", "CommonsenseQA", "MMLU"],
                    help="run a single dataset instead of all three")
    ap.add_argument("--smoke", action="store_true",
                    help="quick pipeline check: 3 problems per dataset, prints raw output")
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"],
                    help="inference backend: vllm (fast), hf (transformers, compatible everywhere), "
                         "auto (try vllm, fall back to hf)")
    ap.add_argument("--dtype", default=None, choices=["float16", "float32", "bfloat16"],
                    help="override cfg.dtype. Gemma3 needs float32 on V100: it overflows "
                         "float16's range (trained/shipped in bf16 with large activation "
                         "norms + logit soft-capping) and V100 lacks bf16 tensor cores, so "
                         "float32 is the only precision that's both correct and V100-compatible "
                         "for that model family.")
    ap.add_argument("--max-tokens-direct", type=int, default=None,
                    help="override cfg.max_tokens_direct (e.g. for DeepSeek-R1-Distill, which "
                         "has no prompt-level way to suppress its own reasoning and may need "
                         "more budget to reach a stated answer even in the 'direct' condition)")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="override cfg.max_model_len. vLLM's startup memory-profiling pass "
                         "runs one dummy forward at this length, and its activation memory "
                         "cost scales with it -- large in float32 (Gemma3 on V100), which can "
                         "starve the KV cache. Lower this toward the real worst case (prompt "
                         "+ max_tokens_cot) to free that memory back to the KV cache.")
    args = ap.parse_args()

    cfg = Config()
    if args.model:
        cfg.model_name = args.model
    if args.n:
        cfg.n_per_dataset = args.n
    if args.out:
        cfg.out_dir = args.out
    if args.dataset:
        cfg.datasets = (args.dataset,)
    if args.smoke:
        cfg.n_per_dataset = 3
    if args.dtype:
        cfg.dtype = args.dtype
    if args.max_tokens_direct:
        cfg.max_tokens_direct = args.max_tokens_direct
    if args.max_model_len:
        cfg.max_model_len = args.max_model_len

    print("=== Config ===")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    print(f"  backend: {args.backend}")
    print()

    print("Loading problems...")
    problems = load_problems(cfg.datasets, cfg.n_per_dataset)
    print(f"  {len(problems)} problems across {len(cfg.datasets)} datasets\n")

    print("Loading model …")
    runner = _load_runner(args.backend, cfg)

    if args.smoke:
        _run_smoke(runner, problems, cfg)
        return

    print("Running both conditions...")
    exp = NecessityExperiment(cfg, runner)
    df = exp.run(problems)
    summary = exp.summarise(df)

    print("\n=== Necessity / Propensity summary ===")
    print(summary.to_string())
    print("\nReading it: large gap = necessity, small gap = propensity.")
    print("above_chance must be True or a small gap just means 'cannot do task'.")

    exp.save(df, summary)
    print(f"\nSaved results to {cfg.out_dir}/")


def _run_smoke(runner, problems, cfg):
    """Print one CoT and one direct response per dataset — fast pipeline sanity check."""
    print("\n=== SMOKE TEST ===")
    print("Checks: CoT reaches </think>, direct has no </think>, answers parse.\n")
    from cot_faith.prompts import extract, is_correct

    seen = set()
    sample = [p for p in problems if p.dataset not in seen and not seen.add(p.dataset)]

    # Use per-dataset token budget so the smoke test reflects real run conditions.
    by_ds_tokens = getattr(cfg, "max_tokens_cot_by_dataset", {})
    cot_texts = []
    for p in sample:
        max_cot = by_ds_tokens.get(p.dataset, cfg.max_tokens_cot)
        cot_texts.extend(runner.generate([p], use_cot=True, max_new_tokens=max_cot))
    dir_texts = runner.generate(sample, use_cot=False)

    hard_fail = False
    for p, cot_txt, dir_txt in zip(sample, cot_texts, dir_texts):
        cot_finished = "</think>" in cot_txt
        dir_clean = "</think>" not in dir_txt
        cot_pred = extract(p, cot_txt)
        dir_pred = extract(p, dir_txt)

        # Hard failures: direct mode leaked thinking, or extraction returned nothing.
        # CoT truncation is a WARN (raise max_tokens_cot), not a pipeline break.
        pipeline_broken = (not dir_clean
                           or cot_pred is None
                           or dir_pred is None)
        if pipeline_broken:
            hard_fail = True

        if pipeline_broken:
            status = "FAIL"
        elif not cot_finished:
            status = "WARN"  # CoT truncated — raise max_tokens_cot
        else:
            status = "OK"

        print(f"[{status}] {p.dataset} | gold={p.gold} | cot_pred={cot_pred} | dir_pred={dir_pred}")
        print(f"       cot_reached_think_end={cot_finished}  direct_has_no_thinking={dir_clean}")
        if not cot_finished:
            ds_budget = getattr(cfg, "max_tokens_cot_by_dataset", {}).get(p.dataset, cfg.max_tokens_cot)
            print(f"       NOTE: CoT truncated — raise max_tokens_cot_by_dataset[{p.dataset!r}] (currently {ds_budget})")
        print(f"       cot_tail:  {cot_txt[-120:].strip()!r}")
        print(f"       dir_raw:   {dir_txt[:120].strip()!r}")
        print()

    if hard_fail:
        print("Smoke test FAILED — pipeline broken. Fix before the full run.")
    else:
        print("Smoke test PASSED. Run without --smoke for the full experiment.")


if __name__ == "__main__":
    main()
