# PREREG_H10_DEA_BUDGET (CPU, existing cells)
Written: 2026-11-10T13:10:04Z (before any C1 computation). Code: work/cyclea-h10/tools/c1_budget.py (hash recorded in c1 output).

Question: how does the number of distinct verified values depend on the query budget? For each of the 63 `ca-deab2` cells and the 9 step3000 cells: 1,000 random orderings of the 5,000 queries (`default_rng(0)`); for k in {50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000} the number of distinct verified emails and phones after the first k queries: mean and 2.5/97.5 percentiles over orderings; then the mean over seeds per model x condition, also as a fraction of the 5,000-query value. Interpretation limit: this is the effect of a cap on the number of queries, not of any other rate-limiting policy.
