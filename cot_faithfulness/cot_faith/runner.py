"""Model runner backed by vLLM.

vLLM does continuous batching: finished sequences leave the batch immediately
instead of waiting for the slowest one. On a V100 this is ~10-20x faster than
naive transformers batched generation for this workload — it's the fix for the
"hours per run" problem.

We pass the whole list of prompts at once and let vLLM schedule them.
"""
from typing import List, Optional

from vllm import LLM, SamplingParams

from .data import Problem
from .prompts import build_messages


def _patch_gemma3_config_compat():
    """Compatibility shim between vLLM 0.8.5[.post1] and newer transformers.

    vLLM's own Gemma3 model code reads config.sliding_window_pattern (public
    attribute). transformers privatized this to _sliding_window_pattern with
    no accessor at some point after vLLM 0.8.5 shipped, so loading any Gemma3
    model raises "'Gemma3TextConfig' object has no attribute
    'sliding_window_pattern'" -- confirmed by reading both packages directly,
    not guessed. Downgrading transformers to match vLLM's expectations risks
    breaking whatever newer transformers features the other models here need
    (they share this one environment); upgrading vLLM risks the torch-version
    dance we already went through once for V100 compatibility. A small,
    guarded property shim sidesteps both. No-op (and safe to call every time)
    once a transformers version restores the public attribute on its own.
    """
    try:
        from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
    except ImportError:
        return
    if not hasattr(Gemma3TextConfig, "sliding_window_pattern"):
        Gemma3TextConfig.sliding_window_pattern = property(
            lambda self: self._sliding_window_pattern
        )


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        _patch_gemma3_config_compat()
        # tensor_parallel_size=1 uses one GPU. Set to 2 to shard across both
        # V100s for a bigger model; for <=4B, one GPU is plenty and simpler.
        self.llm = LLM(
            model=cfg.model_name,
            dtype=cfg.dtype,
            seed=cfg.seed,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
            # See Config.enforce_eager / enable_chunked_prefill: both default
            # off on V100 to avoid a Triton/LLVM codegen crash on Volta.
            enforce_eager=cfg.enforce_eager,
            enable_chunked_prefill=cfg.enable_chunked_prefill,
            # See Config.max_model_len: without this, vLLM profiles memory
            # using the model's own native context length, which OOMs
            # instantly on models with a huge native max (e.g. DeepSeek-R1-
            # Distill-Llama-8B's 131072) even though nothing we run needs it.
            max_model_len=cfg.max_model_len,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self._stop_token_ids = self._build_stop_ids()

    def _build_stop_ids(self) -> List[int]:
        tok = self.tokenizer
        ids = set()
        if tok.eos_token_id is not None:
            ids.add(tok.eos_token_id)
        # Different model families end a turn with different special tokens
        # -- Qwen uses <|im_end|> (ChatML), Gemma uses <end_of_turn>. Without
        # the right one, generation can run to max_tokens instead of stopping
        # at the natural turn end. convert_tokens_to_ids on a string that
        # ISN'T in a given model's vocab returns that tokenizer's UNK id (a
        # valid, non-negative int), not None -- so an `>= 0` check alone
        # silently added a foreign model's turn-end string, resolved to
        # *this* model's UNK token, to the stop set. Guard against that.
        unk_id = tok.unk_token_id
        for turn_end in ("<|im_end|>", "<end_of_turn>"):
            candidate = tok.convert_tokens_to_ids(turn_end)
            if isinstance(candidate, int) and candidate >= 0 and candidate != unk_id:
                ids.add(candidate)
        return list(ids)

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
                enable_thinking=use_cot,   # the necessity control knob
            )
            prompts.append(text)
        return prompts

    def generate(self, problems: List[Problem], use_cot: bool,
                 max_new_tokens: Optional[int] = None,
                 cues: Optional[List[Optional[str]]] = None) -> List[str]:
        texts, _ = self.generate_with_status(problems, use_cot, max_new_tokens, cues)
        return texts

    def generate_with_status(self, problems: List[Problem], use_cot: bool,
                 max_new_tokens: Optional[int] = None,
                 cues: Optional[List[Optional[str]]] = None):
        """Like generate(), but also reports whether each generation stopped
        naturally (hit a stop token) vs was cut off by max_new_tokens.

        Stage 3's capture gate needs this to reject truncated/looping traces
        -- but checking for a literal "</think>" token (the original way)
        silently assumes every model uses that reasoning-toggle special
        token. Gemma3 has no such token in its vocabulary at all, so that
        check was always False for it, and is_captured() zeroed out its
        entire capture rate regardless of what the cue actually did.
        finish_reason=="stop" (vLLM hit an actual stop token) vs "length"
        (ran out of budget) is the real, model-agnostic signal.
        """
        label = "CoT" if use_cot else "direct"
        print(f"  generating {len(problems)} prompts [{label}] …")
        prompts = self._render(problems, use_cot, cues=cues)
        if max_new_tokens is None:
            max_new_tokens = self.cfg.max_tokens_cot if use_cot else self.cfg.max_tokens_direct
        params = SamplingParams(
            temperature=self.cfg.temperature,
            max_tokens=max_new_tokens,
            stop_token_ids=self._stop_token_ids,
        )
        outputs = self.llm.generate(prompts, params)
        # vLLM may reorder internally but returns results aligned to input order.
        texts = [o.outputs[0].text for o in outputs]
        reached_end = [o.outputs[0].finish_reason == "stop" for o in outputs]
        return texts, reached_end

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
        params = SamplingParams(
            temperature=self.cfg.temperature,
            max_tokens=max_new_tokens,
            stop_token_ids=self._stop_token_ids,
        )
        outputs = self.llm.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]
