"""HuggingFace transformers backend — drop-in replacement for runner.py.

Use this when vLLM is unavailable (e.g., glibc version mismatch on the
compute node). Same public interface as Runner: __init__(cfg) + generate().

Speed: ~3-5x slower than vLLM on V100 due to the lack of continuous batching,
but fully correct and needs no native library beyond transformers + torch.
"""
import time
import torch
from typing import List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import Problem
from .prompts import build_messages

# Starting batch size. Automatically halved on OOM, so set this high — it will
# self-correct downward. Raise to 16 if you have headroom after a clean run.
_BATCH_SIZE = 8


class HFRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)

        print(f"  Loading tokenizer: {cfg.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=True
        )
        # Left-padding is required for correct batched decoding on decoder-only models.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Explicit device: don't rely on device_map="auto" which can silently fall
        # back to CPU when accelerate mis-detects GPU availability.
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._device.type == "cpu":
            print("  WARNING: CUDA not available — inference will be extremely slow.")
        else:
            print(f"  CUDA available: {torch.cuda.get_device_name(self._device)}"
                  f" ({torch.cuda.get_device_properties(self._device).total_memory // 1024**3} GB)")

        dtype = getattr(torch, cfg.dtype)
        print(f"  Loading model weights ({cfg.dtype}) to {self._device} …")
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            dtype=dtype,
            trust_remote_code=True,
        ).to(self._device)
        self.model.eval()
        print(f"  Model ready on {self._device}")

        # Stop at turn-end tokens so generation doesn't run to max_new_tokens.
        # See the identical, more detailed comment in runner.py's
        # _build_stop_ids -- convert_tokens_to_ids on a string that isn't in
        # THIS tokenizer's vocab returns its UNK id (not None), so checking
        # only ">= 0" silently added a foreign model's turn-end token,
        # resolved to this model's UNK token, into the stop set.
        unk_id = self.tokenizer.unk_token_id
        stop = {self.tokenizer.eos_token_id}
        for turn_end in ("<|im_end|>", "<end_of_turn>"):
            candidate = self.tokenizer.convert_tokens_to_ids(turn_end)
            if isinstance(candidate, int) and candidate >= 0 and candidate != unk_id:
                stop.add(candidate)
        self._stop_ids = list(stop)

    def _render(self, problems: List[Problem], use_cot: bool,
                cues: Optional[List[Optional[str]]] = None) -> List[str]:
        prompts = []
        for i, p in enumerate(problems):
            cue = cues[i] if cues is not None else None
            messages = build_messages(p, use_cot, cue=cue)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=use_cot,
            )
            prompts.append(text)
        return prompts

    @torch.no_grad()
    def _generate_batch(self, prompts: List[str], max_new_tokens: int):
        """Generate for one batch; auto-halves batch size on OOM.

        Returns (texts, reached_end) -- reached_end[i] is True iff sequence i
        actually emitted one of self._stop_ids (a real stop token), vs
        running the full max_new_tokens budget without ever stopping. With
        eos_token_id=pad_token_id, HF pads a finished sequence's tail with
        that same id, so "does a stop id appear anywhere in the generated
        tokens" is equivalent to "did this sequence stop naturally" -- the
        model-agnostic way to detect truncation (see the identical, more
        detailed rationale on Runner.generate_with_status in runner.py).
        """
        if not prompts:
            return [], []
        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,    # high enough for any of our question formats
            ).to(self._device)

            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self._stop_ids,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            stop_ids_set = set(self._stop_ids)
            texts, reached_end = [], []
            for i in range(len(prompts)):
                gen_ids = out[i][prompt_len:].tolist()
                texts.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
                reached_end.append(any(t in stop_ids_set for t in gen_ids))
            return texts, reached_end
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(prompts) == 1:
                raise RuntimeError(
                    "OOM even with batch_size=1 — model may be too large for this GPU."
                )
            mid = len(prompts) // 2
            print(f"\n  OOM: splitting batch {len(prompts)} → {mid} + {len(prompts) - mid}")
            texts_a, end_a = self._generate_batch(prompts[:mid], max_new_tokens)
            texts_b, end_b = self._generate_batch(prompts[mid:], max_new_tokens)
            return texts_a + texts_b, end_a + end_b

    def generate(self, problems: List[Problem], use_cot: bool,
                 max_new_tokens: Optional[int] = None,
                 cues: Optional[List[Optional[str]]] = None) -> List[str]:
        texts, _ = self.generate_with_status(problems, use_cot, max_new_tokens, cues)
        return texts

    def generate_with_status(self, problems: List[Problem], use_cot: bool,
                 max_new_tokens: Optional[int] = None,
                 cues: Optional[List[Optional[str]]] = None):
        """See Runner.generate_with_status in runner.py -- same contract."""
        label = "CoT" if use_cot else "direct"
        if max_new_tokens is None:
            max_new_tokens = self.cfg.max_tokens_cot if use_cot else self.cfg.max_tokens_direct
        print(f"  generating {len(problems)} prompts [{label}] "
              f"(batch_size={_BATCH_SIZE}, max_new_tokens={max_new_tokens}) …")

        prompts = self._render(problems, use_cot, cues=cues)
        results: List[str] = []
        reached_ends: List[bool] = []
        t_start = time.time()
        for i in range(0, len(prompts), _BATCH_SIZE):
            batch = prompts[i: i + _BATCH_SIZE]
            texts, reached_end = self._generate_batch(batch, max_new_tokens)
            results.extend(texts)
            reached_ends.extend(reached_end)
            done = min(i + _BATCH_SIZE, len(prompts))
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(prompts) - done) / rate if rate > 0 else 0
            print(f"    {done}/{len(prompts)}  "
                  f"elapsed={elapsed/60:.1f}m  "
                  f"ETA=~{eta/60:.1f}m  "
                  f"({rate:.1f} prompts/min)",
                  end="\r", flush=True)
        elapsed_total = time.time() - t_start
        print(f"    {len(prompts)}/{len(prompts)}  "
              f"done in {elapsed_total/60:.1f}m  "
              f"({len(prompts)/elapsed_total*60:.1f} prompts/min)          ")
        return results, reached_ends

    def generate_chat(self, message_lists: List[List[dict]], max_new_tokens: int) -> List[str]:
        """Generate from raw chat message lists, bypassing Problem/build_messages
        entirely. Used by the Stage 3 self-judge smoke-test path (see
        faithfulness.judge_captured_rows_self) so the pipeline can be exercised
        end-to-end without the Anthropic API.
        """
        prompts = [
            self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for messages in message_lists
        ]
        print(f"  generating {len(prompts)} prompts [self-judge] "
              f"(batch_size={_BATCH_SIZE}, max_new_tokens={max_new_tokens}) …")
        results: List[str] = []
        for i in range(0, len(prompts), _BATCH_SIZE):
            results.extend(self._generate_batch(prompts[i: i + _BATCH_SIZE], max_new_tokens))
        return results
