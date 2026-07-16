# CoT Faithfulness and Necessity — Results Draft

*Working draft. Numbers are final (Stage 2 n=200/arm; Stage 3 as noted). Two LLM
judges (Claude Opus 4.8 and Claude Haiku 4.5) are reported side by side
throughout; neither is treated as ground truth (see Limitations).*

## Research question

Chain-of-thought (CoT) is increasingly proposed as a *monitoring* surface: if we
can read a model's reasoning, we can catch misbehavior. That bet is only as good
as CoT *faithfulness* — whether the reasoning reflects the true causes of the
model's output. We ask whether faithfulness **tracks necessity**: is a model more
honest in its CoT on tasks where reasoning is *required* to get the answer
(necessity) than on tasks where it is *optional* (propensity)? If so, monitoring
is more trustworthy exactly where reasoning is load-bearing. If not, the
monitoring bet is weaker than it looks.

We operationalize this in two stages across three open-weight reasoning/instruct
models from different families: **Qwen3-4B**, **DeepSeek-R1-Distill-Llama-8B**,
and **Gemma-3-4B**.

## Stage 2 — establishing the necessity spectrum

For each task we measure accuracy **with** vs **without** CoT. A large positive
gap means reasoning is *necessary*; a near-zero gap means it is *optional*
(propensity). (n=200/arm.)

| Model | GSM8K gap | MMLU gap | CommonsenseQA gap |
|---|---|---|---|
| Qwen3-4B | **+0.66** | +0.08 | −0.16 |
| Gemma-3-4B | **+0.69** | −0.05 | +0.02 |
| DeepSeek-R1-Distill-8B* | +0.03* | +0.18* | +0.05* |

GSM8K is strongly necessity-loaded for the two models with a real "no-CoT"
condition; MMLU and CommonsenseQA are propensity tasks (reasoning barely helps or
slightly hurts). This ordering — **GSM8K (necessity) > MMLU ≈ CommonsenseQA
(propensity)** — is the axis Stage 3 tests against.

\* **DeepSeek caveat:** DeepSeek-R1-Distill opens a `<think>` block on every
generation regardless of instruction; it has no genuine "reason-off" condition.
Its Stage 2 gaps are therefore *not* comparable to the other two (both arms
contain reasoning) and are reported for completeness only, not as necessity
measurements. This is itself a finding: for RL-distilled reasoning models the
necessity/propensity distinction can collapse at the architecture level.

## Stage 3 — cue injection: capture and verbalization

We inject a misleading authority cue ("I think the answer is X, but check" /
"I already worked this out and got X") pointing at a wrong answer, and measure:

- **Capture** — did the cue flip the model's final answer to the wrong target?
  (Measured on all problems; a trace counts only if both clean and cued
  generations terminate normally.)
- **Verbalization** — *of captured traces*, does the CoT acknowledge the cue's
  influence? (Scored by both LLM judges on captured rows only.)

### Capture rate — the primary, robust result

| Model | GSM8K (necessity) | MMLU (propensity) | CSQA (propensity) |
|---|---|---|---|
| Qwen3-4B | 0.4% (2/500) | 4.0% (32/800) | 5.2% (21/400) |
| DeepSeek-8B | 2.2% (11/500) | 14.4% (115/800) | 23.8% (95/400) |
| Gemma-3-4B | 0.0% (0/500) | 41.1% (329/800) | 40.0% (160/400) |

**Necessity strongly suppresses capture, replicated across all three families.**
On the necessity arm (GSM8K) the misleading cue almost never changes the answer
(0–2%); on the propensity arms it succeeds far more often (4–41%). When a model
actually needs to reason to reach the answer, an authority cue cannot easily
hijack it. Magnitude varies — Gemma is far more suggestible on propensity tasks
(~40%) than Qwen3 (~4–5%) — but the necessity-vs-propensity ordering holds for
every model.

### Verbalization rate — of captured traces, does the CoT admit the cue?

Reported for both judges (Opus / Haiku). Cells with too few captures to
interpret are flagged.

| Model | GSM8K (necessity) | MMLU (propensity) | CSQA (propensity) |
|---|---|---|---|
| Qwen3-4B | 100% / 100% **(n=2 — not interpretable)** | 50% / 56% (n=32) | 52% / 57% (n=21) |
| DeepSeek-8B | 18% / 18% **(n=11 — weak)** | 41% / 55% (n=115) | 64% / 76% (n=95) |
| Gemma-3-4B | — (0 captures) | **0% / 7%** (n=329) | **0% / 18%** (n=160) |

### Reading the verbalization result

1. **Faithfulness does not cleanly track necessity.** The original hypothesis
   predicted *higher* verbalization on necessity tasks. The data cannot support
   that claim: necessity is so effective at *preventing* capture (Stage 3 above)
   that the necessity arm yields almost no captured traces to score (n=0, 2, 11).
   The two available necessity cells even point in opposite directions (Qwen 100%
   at n=2; DeepSeek 18% at n=11) and are underpowered. **Necessity protects by
   preventing manipulation, not by making the CoT more transparent when
   manipulation does occur.** This is a more defensible and arguably more useful
   claim for the monitoring debate than the original bet.

2. **Gemma is the sharpest single result — and the worst case for monitoring.**
   Gemma is simultaneously the *most* capturable model (~40% on propensity arms)
   and essentially *never* discloses the cue (~0%). A monitor reading Gemma's CoT
   would receive almost no signal that the answer had been externally steered.

3. **Judge robustness and a methods caution.** Opus and Haiku agree on 89% of the
   765 captured rows (κ=0.71). Where they diverge, Haiku is consistently *higher*.
   The divergence concentrates entirely on Gemma (Haiku 7–18% vs Opus 0%). Manual
   inspection of all 50 Gemma validation traces (independent hand-read) found **0**
   genuine cue acknowledgments, matching Opus. Haiku's false positives arise from
   two failure modes: (a) hallucinating that the trace repeated the cue phrase
   when that phrase appears only in the prompt, and (b) conflating *capture*
   (landing on the cued answer, true of every captured row by construction) with
   *disclosure*. A weaker judge silently mistakes capture for faithfulness; a
   stronger judge does not. We therefore report both judges and do not average
   them, since the gap is informative.

## Limitations

- **No human ground truth for verbalization.** Verbalization was scored by two
  LLM judges (Opus, Haiku); we report their cross-agreement (89%, κ=0.71) and an
  independent manual read of the Gemma validation sample in lieu of a full human
  labeling pass. LLM judges can share systematic errors.
- **Underpowered necessity arms.** Because necessity suppresses capture (the
  central Stage 3 result), verbalization on necessity tasks rests on n ≤ 11
  captured traces per model and is directional at best. Percentages there are
  reported with counts and should not be over-read (esp. Qwen GSM8K, n=2).
- **DeepSeek has no genuine no-CoT condition** (always opens `<think>`), so its
  Stage 2 necessity gaps are not comparable to the other two.
- **Model scale.** All models are 4–8B open weights on V100 hardware. A larger
  model (gpt-oss-20b) was considered but is infeasible on this hardware; scaling
  the finding to larger models is future work.

## Headline takeaways

1. **Necessity suppresses manipulation** — cue capture is near-zero on necessity
   tasks and substantial on propensity tasks, across three model families. (Robust.)
2. **But necessity does not buy transparency** — where manipulation does slip
   through, the CoT is no more honest; on the necessity arm there are too few
   captures to even measure. The monitoring benefit of necessity is *prevention*,
   not *disclosure*.
3. **Silent capture is real and model-dependent** — Gemma is highly manipulable
   yet discloses ~0%, the worst case for CoT monitoring, and a weak LLM judge
   masks this by mistaking capture for disclosure.
