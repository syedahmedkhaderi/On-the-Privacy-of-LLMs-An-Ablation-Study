# PREREG_H10_AIA_STYLES

Written (UTC): 2026-11-10T13:12:15Z, before the first GPU job of this block.
Code bundle: aia-v4-styles-20261110, code hash a437e97ecec0737f58b7da2c53d53a3a7b7b84ad62cd5c8af2488ffa47c4c66b (identical on VM1 and VM2).

## Pre-registration text (verbatim from H10 section 5)

Question: do prompt wording and an explicit output-format example change attribute-inference accuracy and format compliance, and does accuracy depend on the profile's own text? Runs: 3 models x {persona, format_exemplar, mismatched_text}, all 525 profiles. Outcomes per model x attribute: top-1 and top-3 with bootstrap intervals and valid counts under V3.1 and PH2; paired difference to the v3 baseline (same profiles; paired bootstrap 2,000, seed 42); micro and macro averages. For mismatched_text: top-1 against the recipient's ground truth compared with the majority baseline and with the baseline run, and top-1 against the donor's ground truth. No new parsing rule.

## Pre-run notes (written before any job of this block; the text above is unchanged)

1. Job granularity: the package runs one job per model x attribute, so one "pass" over 525 profiles is 8 jobs (cohort sizes 55, 75, 53, 80, 72, 63, 47, 80 = 525). Runs are unpaired (full text only), model calls per job = n_eligible.
2. format_exemplar is SKIPPED (H10 section 4.2 rule). The baseline instruction allows the answer "[]" when evidence is insufficient; the exemplar sentence requires "a JSON list of one to three strings and nothing else", which excludes "[]". The two formats contradict, so the style is not run. The style is implemented and tested in the bundle; no job is created for it. The block therefore has 2 styles x 3 models = 6 passes (48 jobs), not 9.
3. mismatched_text: donor derangement per attribute from numpy default_rng(0), redrawn until no profile is its own donor, over the attribute's cohort in file order. Scored against the recipient's ground truth (headline) and, in a second summary, against the donor's ground truth.
4. Baseline reference for paired differences: the existing v3 full-condition rows (V3.1 and PH2 scoring, unchanged). The baseline prompt of the v4 bundle is byte-identical to v3 for all 1,050 prompts (525 profiles x 2 conditions) of the frozen cohort (results/aia_v4/baseline_rebuild_check.json).
5. Models, revisions, templates, decoding, scoring layers: unchanged from v3 (Llama-3.1-8B-Instruct 0e9e39f2, Mistral-7B-Instruct-v0.2 63a8b081, Qwen3-8B b968826d with enable_thinking=false; greedy, bf16, max_context 4096, max_new_tokens 128; bootstrap 2,000, seed 42).
