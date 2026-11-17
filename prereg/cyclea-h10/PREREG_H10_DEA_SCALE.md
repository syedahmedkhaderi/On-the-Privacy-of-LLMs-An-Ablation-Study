# PREREG_H10_DEA_SCALE

Written (UTC): 2026-11-10T13:14:52Z
Code hash (bundle dea-ext-20261110): a0155aa50bcdae1aaeb77ed727ebf192bd00adb3b8d01e939e12e8ad3bb820a2

Question: does verified extraction keep growing within the Pythia family above 1.4B? Cells: pythia-2.8b and pythia-6.9b (standard, step143000, float16) x 3 conditions x 3 seeds. Control for precision: pythia-1.4b in float16, generic_prefix, 3 seeds, compared with the existing float32-path cells. Outcomes: the within-family series 70M, 160M, 410M, 1.4B, 2.8B, 6.9B of mean verified emails and phones per condition with t-intervals; growth ratios between consecutive sizes; the fp16 vs existing difference for 1.4B.

STATUS ADDENDUM (written before any job): NOT RUN. (1) The review gate cannot qualify pythia-2.8b / pythia-6.9b (results/GATE_BLOCKER.md). (2) The precision control is void: the existing runner loads every model as float16 (modeling.py load_model), so there is no float32 path to compare against; the 1.4B float16 replicas of PREREG_H10_DEA_RANKING serve as the same-seed reproducibility check instead.
