"""Central configuration for the necessity-vs-propensity experiment.

Everything tunable lives here so the scripts stay clean and a run is fully
described by one object you can print into the output for reproducibility.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict


@dataclass
class Config:
    # --- model ---
    model_name: str = "Qwen/Qwen3-4B"      # V100-32GB handles 4B comfortably
    # float16, not bfloat16: V100 is compute capability 7.0, and vLLM's engine
    # hard-requires >=8.0 (Ampere+) for bfloat16 -- it raises rather than
    # silently downcasting. V100 does have fp16 tensor cores (it was the first
    # generation to get them), so this is the correct precision here, not a
    # workaround. The HF backend tolerates bf16 on V100 (just without
    # tensor-core accel, so it's slow) which is why this only surfaces once
    # vLLM actually loads instead of falling back silently.
    dtype: str = "float16"
    # Both off by default for V100: vLLM's CUDA-graph capture and chunked
    # prefill both use Triton-compiled kernel paths that assume Ampere+ and
    # can hit a hard LLVM codegen crash on Volta ("Failed to compute parent
    # layout for slice layout" -- not a catchable Python exception, the
    # process aborts). Continuous batching, the main source of vLLM's speedup
    # over the HF backend, doesn't depend on either, so this is a correctness
    # fix, not a real performance sacrifice. Safe to flip back on if this ever
    # runs on Ampere+ hardware.
    enforce_eager: bool = True
    enable_chunked_prefill: bool = False
    # Without an explicit cap, vLLM uses the model's own native context length
    # for its memory-profiling pass (a dummy forward pass sized to max_model_len,
    # run before finalizing KV cache size) -- fine for Qwen3-4B (40960 native)
    # but DeepSeek-R1-Distill-Llama-8B natively supports 131072, and profiling
    # a dummy batch that long on an 8B model OOMs immediately (single 7GiB
    # allocation on one MLP layer, confirmed on a V100). Our longest real
    # generation budget is 4096 tokens (GSM8K, Stage 3) plus a few hundred for
    # the question -- nothing here needs anywhere near either model's native
    # max, so cap it well above what we use and well below what causes trouble.
    max_model_len: int = 8192

    # --- sampling ---
    n_per_dataset: int = 200
    max_tokens_cot: int = 2048             # fallback; per-dataset overrides below
    max_tokens_direct: int = 32
    temperature: float = 0.0               # greedy = reproducible

    # Per-dataset CoT budget. CommonsenseQA is recall-based (thinking very short);
    # GSM8K math needs moderate depth; MMLU hard science can spike to 4096.
    max_tokens_cot_by_dataset: Dict[str, int] = field(default_factory=lambda: {
        "GSM8K": 2048,
        "CommonsenseQA": 512,
        "MMLU": 2048,
    })

    # --- datasets ---
    datasets: tuple = ("GSM8K", "CommonsenseQA", "MMLU")

    # --- bookkeeping ---
    seed: int = 0
    out_dir: str = "outputs"

    # chance accuracy per dataset, for the "above chance?" sanity check
    chance: Dict[str, float] = field(default_factory=lambda: {
        "GSM8K": 0.0,
        "CommonsenseQA": 1 / 5,
        "MMLU": 1 / 4,
    })

    def to_dict(self):
        return asdict(self)


@dataclass
class Stage3Config:
    """Cue-injection faithfulness run: does the model verbalize a misleading
    suggestion in its CoT more often on the necessity arm (GSM8K) than on the
    propensity arms (MMLU, CommonsenseQA)?

    Defaults to the Qwen3-4B config whose Stage 2 gaps this experiment cites
    (GSM8K +0.66, MMLU +0.07, CommonsenseQA -0.15) — see outputs/qwen3_4B/.
    """
    # --- model ---
    model_name: str = "Qwen/Qwen3-4B"
    # float16, not bfloat16 -- see the identical note on Config.dtype above.
    # vLLM's engine refuses bf16 outright on V100 (compute capability 7.0,
    # needs >=8.0); V100 has fp16 tensor cores so this isn't a downgrade.
    dtype: str = "float16"
    # See the identical note on Config.enforce_eager/enable_chunked_prefill
    # above -- both are Volta-safety toggles, not a real perf sacrifice, since
    # continuous batching (vLLM's main speedup over the HF backend) needs
    # neither.
    enforce_eager: bool = True
    enable_chunked_prefill: bool = False
    # See the identical note on Config.max_model_len above -- caps the
    # memory-profiling pass so a model with a huge native context (e.g.
    # DeepSeek-R1-Distill-Llama-8B's 131072) doesn't OOM before it even starts.
    max_model_len: int = 8192

    # --- sampling ---
    n_per_arm: int = 500               # capture rate is often <50%, so this is
                                        # sized to yield ~100+ captures/arm
    temperature: float = 0.0
    max_tokens_cot: int = 2048         # fallback; per-arm overrides below
    max_tokens_cot_by_dataset: Dict[str, int] = field(default_factory=lambda: {
        # Bumped twice now. At 2048, 70% of cued GSM8K generations hit the
        # cap (vs 20% clean). At 3072, still 70% -- every single truncated
        # trace ran to exactly the cap, meaning it wasn't close to finishing,
        # not just occasionally grazing the ceiling. Inspecting one showed
        # why: the model gets stuck in a genuine repetition loop re-litigating
        # its own answer against the cue's, restating the same paragraph
        # verbatim, rather than running long on legitimately more reasoning.
        # More budget increases the chance it breaks out and concludes, but
        # isn't guaranteed to -- is_captured() now requires </think> to have
        # been reached on both generations specifically because truncated
        # traces like this produce unreliable extracted "answers" (see its
        # docstring), so a still-truncated trace is safely excluded rather
        # than silently corrupting the capture rate either way.
        "GSM8K": 4096,
        # Bumped from Stage 2's 512 for the same reason as GSM8K above: the
        # smoke test's one CommonsenseQA capture turned out to be mid-loop
        # oscillation ("the answer is D... the answer is A... the answer is
        # D...") truncated at 1024, not a genuine conclusion.
        "CommonsenseQA": 1536,
        "MMLU": 2048,
    })

    # Per-arm sample size, overriding n_per_arm. First sized from a bug-fixed
    # n=200/arm diagnostic, then bumped again for the paper-scale run once the
    # n=300/500/300 pass confirmed those rates hold (GSM8K 0/300 -- 4th
    # independent confirmation it's genuinely ~0%, not a bug; MMLU 70/500 =
    # 14.0%; CommonsenseQA 94/300 = 31.3%). MMLU gets the biggest bump since
    # its verbalization rate (95.7%) is the only one of the three not already
    # pinned near a ceiling -- that's where more samples actually tighten the
    # number. GSM8K's bump mostly just gives a rounder, bigger denominator
    # behind an already-solid "essentially never"; CommonsenseQA is already
    # comfortably powered and gets a smaller bump for consistency.
    n_per_arm_by_dataset: Dict[str, int] = field(default_factory=lambda: {
        "GSM8K": 500,
        "MMLU": 800,
        "CommonsenseQA": 400,
    })

    # --- arms, ordered by necessity (large gap -> ~0 -> negative) ---
    datasets: tuple = ("GSM8K", "MMLU", "CommonsenseQA")

    # --- cue injection (see cue_injection.CUE_BUILDERS) ---
    cue_type: str = "authority"

    # --- faithfulness scoring ---
    # "anthropic" (default, real judge -- Claude Haiku via API) or "self"
    # (reuse the already-loaded Qwen model as its own judge). "self" needs no
    # API key and no network, so it's the smoke-test default -- it is NOT an
    # independent check (the model judging its own reasoning) and should never
    # be used to draw real faithfulness conclusions.
    judge_backend: str = "anthropic"
    judge_model: str = "claude-haiku-4-5"
    judge_max_retries: int = 5
    judge_backoff_base: float = 1.0    # seconds; doubles each retry + jitter
    min_captures_for_stable_rate: int = 30   # warn if an arm has fewer than this

    # --- human validation export ---
    validation_sample_size: int = 50

    # --- smoke test ---
    smoke_n: int = 10

    # --- bookkeeping ---
    seed: int = 0
    out_dir: str = "outputs/stage3"

    def to_dict(self):
        return asdict(self)
