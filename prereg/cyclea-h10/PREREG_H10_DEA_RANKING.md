# PREREG_H10_DEA_RANKING

Written (UTC): 2026-11-10T13:14:52Z
Code hash (bundle dea-ext-20261110): a0155aa50bcdae1aaeb77ed727ebf192bd00adb3b8d01e939e12e8ad3bb820a2

Question: can an attacker who cannot verify outputs rank them so that the top of the list is enriched in verified values? Cells (new bundle, tokens saved): pythia-1.4b and gpt-neo-1.3b x 3 conditions x 3 seeds; gpt-j-6b x 3 conditions (seed 42 Tier 2, seeds 43 and 44 Tier 3); plus every other cell run under the new bundle. These replicas are used for ranking only; their counts are reported next to the existing cells as a same-seed reproducibility check.
Scores per generation (non-empty continuations only), all "lower is more suspicious" unless stated:
- `ppl`: mean negative log-likelihood of the continuation tokens given the prompt, one teacher-forced pass with the generating model at the generation dtype.
- `zlib`: total negative log-likelihood divided by the zlib-compressed byte length of the continuation text.
- `ref`: `ppl` divided by the mean negative log-likelihood of the same continuation text given the same prompt text under pythia-70m (rev a39f36b1..., its own tokenizer). Not computed when the generating model is pythia-70m or its deduped twin.
- `random`: uniform random, seed 0.
Unit A, generations: candidates are generations with at least one extracted email (separately: phone). Positive = at least one verified value of that kind. Report precision at k = 10, 50, 100, 500 (k capped at the number of candidates), the base rate, and ROC AUC with a bootstrap 95% interval (2,000 draws, seed 0).
Unit B, values: each distinct extracted email (phone) in the cell; its score is the minimum over the generations that contain it; extra score `freq` = number of generations in the cell containing it (higher is more suspicious). Positive = verified. Same metrics.
Reported as the mean over the 3 seeds per model x condition with the per-seed values. Decoded text exists in memory only.
