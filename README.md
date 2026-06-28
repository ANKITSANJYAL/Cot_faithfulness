# Cot_faithfulness

As my first cheap test I will be testing whether my two task arms actually behave as different regimes on my model. I run a sample (~200 problems each) from GSM8K (necessity arm) and CommonsenseQA / non-symbolic MMLU (propensity arms), each twice; once with chain-of-thought, once with it suppressed and compare accuracy. I've already run this on GSM8K (Stage 2: ~71-point accuracy drop without CoT); this test extends it to the propensity datasets to confirm the contrast. The output is one number per dataset: the with/without-CoT accuracy gap.

What I expect, and why: A large gap on GSM8K (reasoning is computationally required as the model must carry arithmetic across steps it can't do in one forward pass) and a small gap on the commonsense/factual sets (answers come from recall, so suppressing reasoning shouldn't hurt much). This is what Sprague et al. (2024) found across frontier models; I'm checking it reproduces on a small open model. Success here means I have two genuinely different regimes to run the faithfulness comparison across but nothing about faithfulness is measured yet, only that the regimes exist.

What would make me change course:





Propensity gap is also large (CoT removal hurts commonsense accuracy too): my propensity dataset isn't propensity on this model. I'd switch to an easier/more recall-pure set rather than treat it as a reasoning puzzle.



Necessity gap is small (GSM8K survives CoT removal): unlikely given Stage 2, but would mean the model is shortcutting the math, and I'd need harder necessity problems.



Propensity accuracy is near chance in both conditions: the most difficult case i.e,a "small gap" that's really "the model can't do the task at all." I'd check that with-CoT accuracy is clearly above chance; if not, I move to a larger model (Qwen3-4B) before trusting the propensity arm.
