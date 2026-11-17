import sys, json, os
sys.path.insert(0, "work/cyclea-h10/tools")
from h10cpu import *
cells = cell_dirs("ca-deab2"); cells.update(step3k())
KS = [50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000]
rng = np.random.default_rng(0)
N = 5000
perms = np.stack([rng.permutation(N) for _ in range(1000)])      # perms[o, pos] = query ; rank = inverse
ranks = np.argsort(perms, axis=1).astype(np.int32)                # ranks[o, query] = position
rows = []; src = []
for (m, c, s), cd in sorted(cells.items()):
    g, d = load(cd)
    pi = dict(zip(g.generation_id, g.prompt_index))
    assert g.prompt_index.min() == 0 and g.prompt_index.max() == N - 1 and len(g) == N
    src.append({"path": cd + "/generations.parquet", "sha256": sha(cd + "/generations.parquet")}); src.append({"path": cd + "/detections.parquet", "sha256": sha(cd + "/detections.parquet")})
    for kind in ("email", "phone"):
        v = d[(d.kind == kind) & d.verified]
        curves = np.zeros((1000, len(KS)))
        if len(v):
            v = v.assign(q=v.generation_id.map(pi))
            first = np.stack([ranks[:, grp.q.values].min(axis=1) for _, grp in v.groupby("value_hash")], axis=1)  # (1000, nvalues)
            for j, k in enumerate(KS): curves[:, j] = (first < k).sum(axis=1)
        full = curves[:, -1]
        for j, k in enumerate(KS):
            rows.append(dict(model=m, condition=c, seed=s, kind=kind, k=k, mean=curves[:, j].mean(),
                             p2_5=np.percentile(curves[:, j], 2.5), p97_5=np.percentile(curves[:, j], 97.5),
                             total_distinct_verified_5000=int(full[0]), frac=float(curves[:, j].mean() / full[0]) if full[0] else float("nan")))
df = pd.DataFrame(rows)
df.to_csv(OUT + "/cpu/c1_budget_per_cell.csv", index=False)
agg = df.groupby(["model", "condition", "kind", "k"]).agg(mean=("mean", "mean"), p2_5=("p2_5", "mean"), p97_5=("p97_5", "mean"),
        frac=("frac", "mean"), n_seeds=("seed", "nunique")).reset_index()
agg.to_csv(OUT + "/cpu/c1_budget_per_model_condition.csv", index=False)
json.dump({"prereg": "PREREG_H10_DEA_BUDGET", "prereg_sha256": sha("work/cyclea-h10/prereg/PREREG_H10_DEA_BUDGET.md"),
           "code_sha256": sha(__file__), "n_cells": len(cells), "orderings": 1000, "ks": KS,
           "note": "percentiles over orderings are per cell, then averaged over seeds; cap-on-queries effect only",
           "sources": src + [{"path": INV, "sha256": sha(INV)}]}, open(OUT + "/cpu/c1_budget.json", "w"), indent=1)
print(len(cells), "cells;", len(df))
