"""Dataset loading.

Each dataset is normalised into a common `Problem` so the rest of the code
never has to special-case where a problem came from. GSM8K is free-form numeric;
CommonsenseQA and MMLU are multiple choice.
"""
import re
from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset

LETTERS = ["A", "B", "C", "D", "E"]

# MMLU subjects that are math/symbolic — excluded from the propensity arm,
# following Sprague et al. (CoT helps on symbolic, not on recall).
MATH_SUBJECTS = {
    "abstract_algebra", "college_mathematics", "high_school_mathematics",
    "elementary_mathematics", "college_physics", "high_school_physics",
    "high_school_statistics", "college_chemistry", "high_school_chemistry",
    "econometrics", "formal_logic",
}


@dataclass
class Problem:
    id: str
    question: str
    gold: str                       # numeric string (GSM8K) or letter (MCQ)
    dataset: str
    choices: Optional[List[str]] = None   # None => free-form numeric

    @property
    def is_numeric(self) -> bool:
        return self.choices is None


def _gsm8k(n: int) -> List[Problem]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i in range(min(n, len(ds))):
        m = re.search(r"####\s*([-\d,\.]+)", ds[i]["answer"])
        gold = m.group(1).replace(",", "").strip() if m else None
        out.append(Problem(f"gsm_{i:04d}", ds[i]["question"], gold, "GSM8K"))
    return out


def _commonsenseqa(n: int) -> List[Problem]:
    ds = load_dataset("tau/commonsense_qa", split="validation")
    out = []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        out.append(Problem(f"csqa_{i:04d}", ex["question"], ex["answerKey"],
                           "CommonsenseQA", choices=ex["choices"]["text"]))
    return out


def _mmlu_nonsymbolic(n: int) -> List[Problem]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    out = []
    for ex in ds:
        symbolic = ("=" in ex["question"]
                    or any("=" in c for c in ex["choices"])
                    or ex["subject"] in MATH_SUBJECTS)
        if symbolic:
            continue
        out.append(Problem(f"mmlu_{len(out):04d}", ex["question"],
                           LETTERS[ex["answer"]], "MMLU", choices=ex["choices"]))
        if len(out) >= n:
            break
    return out


LOADERS = {
    "GSM8K": _gsm8k,
    "CommonsenseQA": _commonsenseqa,
    "MMLU": _mmlu_nonsymbolic,
}


def load_problems(datasets, n_per_dataset) -> List[Problem]:
    """n_per_dataset is either one int shared across all `datasets` (Stage 2's
    usage) or a {dataset_name: n} dict for per-arm sizing (Stage 3, where
    capture rate turned out to vary ~60x across arms -- see Stage3Config).
    """
    problems = []
    for name in datasets:
        n = n_per_dataset[name] if isinstance(n_per_dataset, dict) else n_per_dataset
        problems.extend(LOADERS[name](n))
    return problems
