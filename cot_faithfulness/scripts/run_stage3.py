#!/usr/bin/env python
"""Run the Stage 3 cue-injection faithfulness experiment.

Usage on the cluster:
    python scripts/run_stage3.py --smoke                # ~10 problems/arm, full pipeline, fast
    python scripts/run_stage3.py                         # full run, all 3 arms
    python scripts/run_stage3.py --arms GSM8K             # one arm only (for the SLURM GPU split)

Requires ANTHROPIC_API_KEY for the LLM-judge step (Claude Haiku) -- put it in
a `.env` file next to this repo's requirements.txt (gitignored) as
`ANTHROPIC_API_KEY=sk-ant-...` and it's picked up automatically, no manual
`export` needed. Pick the GPU with CUDA_VISIBLE_DEVICES, e.g.:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_stage3.py --arms GSM8K
"""
import argparse
import glob
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


def _ensure_cuda_libs_on_path():
    """See the identical, more detailed comment in run_stage2.py. vLLM's
    compiled _C extension dlopen()s libcudart.so.12 directly, which this
    exact environment found fine on one cluster node and failed to find on
    another. Setting os.environ["LD_LIBRARY_PATH"] alone doesn't reliably
    affect a dlopen() happening later in an already-running process on this
    system -- confirmed by testing -- so re-exec this script as a fresh
    process once the variable is set (guarded by a marker so this only
    happens once), matching what a shell-level export would have done.
    """
    if os.environ.get("_COT_FAITH_CUDA_LIBS_FIXED") == "1":
        return
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


def _load_dotenv(path):
    """Load KEY=value lines from a .env file into os.environ (skipped if a
    var is already set, so an explicit `export` still wins). No python-dotenv
    dependency needed for a format this simple -- just ANTHROPIC_API_KEY.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ.setdefault(key, value)


_load_dotenv(os.path.join(_REPO_ROOT, ".env"))

from cot_faith.config import Stage3Config
from cot_faith.data import load_problems
from cot_faith.experiment import FaithfulnessExperiment
from cot_faith.faithfulness import export_validation_sample


class _Tee:
    """Duplicate writes to stdout AND a log file, so print() calls throughout
    the pipeline satisfy "log to a file and stdout" without every module
    needing to know about logging.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _release_cuda_memory():
    """See the identical helper in run_stage2.py -- a vLLM Runner() that
    fails partway through construction can leave GPU memory allocated even
    after the exception, causing the HF fallback to OOM on a model that
    would otherwise fit.
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
    ap.add_argument("--n", type=int, default=None, help="problems per arm")
    ap.add_argument("--out", default=None, help="output dir")
    ap.add_argument("--arms", nargs="+", default=None,
                    choices=["GSM8K", "CommonsenseQA", "MMLU"],
                    help="run a subset of arms (used to split work across GPUs)")
    ap.add_argument("--cue-type", default=None, help="see cue_injection.CUE_BUILDERS")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-backend", default=None, choices=["anthropic", "self"],
                    help="anthropic = real judge (Claude Haiku, needs ANTHROPIC_API_KEY); "
                         "self = the generating Qwen model judges itself, no API key needed "
                         "but NOT an independent check. Defaults to 'self' under --smoke, "
                         "'anthropic' otherwise.")
    ap.add_argument("--skip-judge", action="store_true",
                    help="skip the judge step entirely (e.g. testing generation only)")
    ap.add_argument("--smoke", action="store_true",
                    help="quick end-to-end pipeline check: ~10 problems per arm, "
                         "self-judged by default so it needs no API key")
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dtype", default=None, choices=["float16", "float32", "bfloat16"],
                    help="override cfg.dtype. Gemma3 needs float32 on V100 -- see the "
                         "identical flag in run_stage2.py for why.")
    ap.add_argument("--max-model-len", type=int, default=None,
                    help="override cfg.max_model_len. vLLM's startup memory-profiling pass "
                         "runs one dummy forward at this length to size activation memory, "
                         "and that activation cost scales with it -- in float32 (Gemma3 on "
                         "V100) it's large enough (6.75GiB at 8192) to starve the KV cache "
                         "down to ~2x concurrency for 500+ concurrent prompts, causing heavy "
                         "preemption/recompute thrashing. Lowering this to just above the "
                         "real worst case (GSM8K: ~4096 generated + prompt/cue tokens) frees "
                         "that memory back to the KV cache.")
    args = ap.parse_args()

    cfg = Stage3Config()
    if args.model:
        cfg.model_name = args.model
    if args.dtype:
        cfg.dtype = args.dtype
    if args.max_model_len:
        cfg.max_model_len = args.max_model_len
    if args.n:
        cfg.n_per_arm = args.n
        cfg.n_per_arm_by_dataset = {}   # explicit --n means "this many, every arm"
    if args.out:
        cfg.out_dir = args.out
    if args.arms:
        cfg.datasets = tuple(d for d in cfg.datasets if d in args.arms)
    if args.cue_type:
        cfg.cue_type = args.cue_type
    if args.judge_model:
        cfg.judge_model = args.judge_model
    if args.seed is not None:
        cfg.seed = args.seed
    if args.smoke:
        cfg.n_per_arm = cfg.smoke_n
        cfg.n_per_arm_by_dataset = {}
    if args.judge_backend:
        cfg.judge_backend = args.judge_backend
    elif args.smoke:
        cfg.judge_backend = "self"   # smoke needs no ANTHROPIC_API_KEY by default

    os.makedirs(cfg.out_dir, exist_ok=True)
    log_file = open(os.path.join(cfg.out_dir, "stage3_run.log"), "a")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    print("=== Stage 3 Config ===")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
    print(f"  backend: {args.backend}")
    print(f"  skip_judge: {args.skip_judge}")
    print()

    if cfg.judge_backend == "self" and not args.skip_judge:
        print("NOTE: judge_backend=self -- the generating model is judging its own")
        print("  reasoning. This is a smoke-test convenience only, NOT an independent")
        print("  check. Do not report these verbalization rates as real results; rerun")
        print("  with --judge-backend anthropic (the full-run default) for that.\n")

    if (not args.skip_judge and cfg.judge_backend == "anthropic"
            and not os.environ.get("ANTHROPIC_API_KEY")):
        print("WARNING: ANTHROPIC_API_KEY is not set. The LLM-judge step will fail.")
        print("  Either export ANTHROPIC_API_KEY, pass --judge-backend self (smoke-test")
        print("  only), or pass --skip-judge to test just generation + capture detection.")

    n_per_dataset = {d: cfg.n_per_arm_by_dataset.get(d, cfg.n_per_arm) for d in cfg.datasets}
    print("Loading problems...")
    print(f"  n per arm: {n_per_dataset}")
    t0 = time.time()
    problems = load_problems(cfg.datasets, n_per_dataset)
    print(f"  {len(problems)} problems across {len(cfg.datasets)} arms "
          f"({time.time() - t0:.1f}s)\n")

    print("Loading model...")
    t0 = time.time()
    runner = _load_runner(args.backend, cfg)
    print(f"  model ready ({(time.time() - t0) / 60:.1f}m)\n")

    exp = FaithfulnessExperiment(cfg, runner)

    print("Generating clean + cued CoT and detecting captures...")
    t_gen = time.time()
    df = exp.run(problems)
    print(f"\nGeneration + capture detection done in {(time.time() - t_gen) / 60:.1f}m")

    if not args.skip_judge:
        df = exp.score_with_judge(df)
    else:
        df["judge_verbalized"] = None
        df["judge_reason"] = None

    summary = exp.summarise(df)
    print("\n=== Stage 3 summary ===")
    print(summary.to_string())
    print("\nverbalization rate = fraction of CAPTURED cases where the CoT admitted the cue.")
    print("Hypothesis: verbalization is higher on GSM8K (necessity) than on")
    print("MMLU / CommonsenseQA (propensity).")

    exp.save(df, summary)
    export_validation_sample(df, os.path.join(cfg.out_dir, "human_validation_sample.csv"),
                             n=cfg.validation_sample_size, seed=cfg.seed)

    print(f"\nSaved results to {cfg.out_dir}/")
    print("  - stage3_results.csv           (per-problem, per-trace, all three layers)")
    print("  - stage3_summary.csv           (per-arm capture / verbalization rates)")
    print("  - human_validation_sample.csv  (fill in human_verbalized, then run")
    print("                                   scripts/validate_stage3.py)")
    print("  - run_config.json")
    print("  - stage3_run.log")
    print(f"\nNext: python scripts/plot_stage3.py --out {cfg.out_dir}")


if __name__ == "__main__":
    main()
