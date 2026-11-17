# PREREG_H10_DEA_DEDUP

Written (UTC): 2026-11-10T13:14:52Z
Code hash (bundle dea-ext-20261110): a0155aa50bcdae1aaeb77ed727ebf192bd00adb3b8d01e939e12e8ad3bb820a2

Question: is verified extraction lower for Pythia models trained on the deduplicated Pile than for the standard models of the same size? Cells: pythia-{70m,160m,410m,1.4b}-deduped x 3 conditions x seeds 42, 43, 44 (36 cells), compared with the existing standard cells (same seeds, prompts, index). Tier 3 adds 2.8b and 6.9b pairs. Outcomes per model x condition: mean distinct verified emails and phones with 95% t-interval (n=3); ratio deduped / standard of the means with a bootstrap interval over seeds (10,000 draws, seed 0); verified union per model by repetition band (bands 1, 2-4, 5-9, 10-24, 25-99, 100+, counts from the existing index, which describes the standard Pile). Stated limits: repetition counts come from the standard Pile; the deduplicated models are a different training run. No significance claim from n=3.

STATUS ADDENDUM (written before any job): NOT RUN. The DEA review gate cannot qualify model keys outside the submitted paper's cell map (results/GATE_BLOCKER.md). This pre-registration is kept so the block can run unchanged if the owner later authorizes the gate extension.
