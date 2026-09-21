"""Stage 02: aggregate across seeds, build tables and the duplication figure."""

from __future__ import annotations

from pathlib import Path

from common import HEADER, PREFLIGHT, code, md, write_notebook


def cells() -> list:
    return [
        md("""
# Stage 02: analysis

Produces four tables and one figure:

1. prompt style by model, extracted and verified
2. model scale, the Pythia ladder with error bars
3. architecture, the three size-matched pairs
4. duplication, recovery rate by how often a value appears in the training data

Confidence intervals are Student's t at n-1. With three seeds that is
t(2, 0.975) = 4.302653, which is wide, and honestly so: three points do not
support a narrow interval.

No GPU. Minutes, not hours.
"""),
        code(HEADER),
        code(PREFLIGHT),
        code("""
preflight([
    ("index", PATHS.index_file),
    ("generations", PATHS.generations),
])
"""),
        code("""
import pandas as pd
from dea_a3 import aggregate, index, runner

index_frame = index.load_index(PATHS.index_file)
cells_df = runner.load_cell_summaries(PATHS)
expected = len(CONFIG.cells())
print(f"cells: {len(cells_df)} of {expected}")
if len(cells_df) < expected:
    print("WARNING: grid incomplete. Tables below describe only completed cells.")
"""),
        md("""
## Complete the grid

A cell that recovered nothing must report zero, not vanish. The unconditional
baseline is expected to be zero and it is the headline contrast, so a groupby
that silently drops empty cells would delete the most important row.
"""),
        code("""
VALUE_COLS = ["extracted_email", "extracted_phone", "verified_email", "verified_phone"]
model_keys = [m.key for m in CONFIG.models]

grid = aggregate.complete_grid(cells_df, model_keys, CONFIG.conditions,
                               CONFIG.seeds, VALUE_COLS)
grid["extracted_total"] = grid["extracted_email"] + grid["extracted_phone"]
grid["verified_total"] = grid["verified_email"] + grid["verified_phone"]
print(f"grid rows: {len(grid)} (expected {expected})")
assert len(grid) == expected
"""),
        md("## Table 1: prompt style by model"),
        code("""
VALUES = VALUE_COLS + ["extracted_total", "verified_total"]
t1 = aggregate.aggregate_cells(grid, ["condition", "model"], VALUES,
                               level=CONFIG.ci_level)
t1 = t1.sort_values(["condition", "model"]).reset_index(drop=True)
t1.to_csv(PATHS.analysis / "table1_prompt_style.csv", index=False)
t1[["condition", "model", "extracted_total_mean", "extracted_total_ci",
    "verified_total_mean", "verified_total_ci"]]
"""),
        code("""
# The headline contrast, stated as one line.
by_cond = aggregate.aggregate_cells(grid, ["condition"], VALUES, level=CONFIG.ci_level)
by_cond = by_cond.set_index("condition")
for cond in CONFIG.conditions:
    row = by_cond.loc[cond]
    print(f"{cond:16s} extracted {row['extracted_total_mean']:8.2f}  "
          f"verified {row['verified_total_mean']:8.2f}")
"""),
        md("## Table 2: model scale, the Pythia ladder"),
        code("""
ladder = [m.key for m in sorted(
    (m for m in CONFIG.models if m.family == "pythia"), key=lambda m: m.params_m)]
t2 = t1[t1["model"].isin(ladder)].copy()
t2["params_m"] = t2["model"].map({m.key: m.params_m for m in CONFIG.models})
t2 = t2.sort_values(["condition", "params_m"]).reset_index(drop=True)
t2.to_csv(PATHS.analysis / "table2_model_scale.csv", index=False)
t2[["condition", "model", "params_m", "verified_total_mean", "verified_total_ci"]]
"""),
        md("""
## Table 3: architecture, size-matched pairs

The training corpus is held constant across each pair, so what varies is
architecture and training recipe, not data.
"""),
        code("""
PAIRS = [("pythia-160m", "gpt-neo-125m"),
         ("pythia-1.4b", "gpt-neo-1.3b")]

lookup = t1.set_index(["condition", "model"])
rows = []
for cond in CONFIG.conditions:
    for left, right in PAIRS:
        if (cond, left) not in lookup.index or (cond, right) not in lookup.index:
            continue
        lo, ro = lookup.loc[(cond, left)], lookup.loc[(cond, right)]
        rows.append({
            "condition": cond, "size_class": f"{left} vs {right}",
            "pythia": lo["verified_total_mean"], "pythia_ci": lo["verified_total_ci"],
            "other": ro["verified_total_mean"], "other_ci": ro["verified_total_ci"],
            "difference": lo["verified_total_mean"] - ro["verified_total_mean"],
        })
t3 = pd.DataFrame(rows)
t3.to_csv(PATHS.analysis / "table3_architecture.csv", index=False)
t3
"""),
        md("""
## Table 4 and the figure: duplication

Occurrence counts come from the Pile, not the maildir. The two correlate at
Spearman 0.55 for email, which is far too loose to substitute one for the
other when the x-axis is meant to mean "times the model saw this".

Bands are cut by quantile because the distribution is heavily skewed, with a
log-spaced fallback when ties collapse the quantile edges.
"""),
        code("""
detections = runner.load_detections(PATHS, verified_only=True)
recovered = set(detections["value_hash"].unique()) if not detections.empty else set()
print(f"distinct verified values recovered: {len(recovered):,}")

t4 = aggregate.recovery_by_band(index_frame, recovered, CONFIG.band_quantiles)
t4.to_csv(PATHS.analysis / "table4_duplication.csv", index=False)
t4
"""),
        code("""
if not t4.empty:
    first = t4.groupby("kind").first()["recovery_rate"]
    last = t4.groupby("kind").last()["recovery_rate"]
    for kind in first.index:
        if first[kind] > 0:
            print(f"{kind:6s} end-to-end ratio: {last[kind] / first[kind]:.1f}x")
        else:
            print(f"{kind:6s} lowest band recovered nothing; ratio undefined")
"""),
        code("""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if not t4.empty:
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for kind, grp in t4.groupby("kind"):
        ax.plot(range(len(grp)), grp["recovery_rate"], marker="o", label=kind)
        ax.set_xticks(range(len(grp)))
        ax.set_xticklabels(grp["band"], rotation=45, ha="right")
    ax.set_xlabel("Occurrences in the training corpus")
    ax.set_ylabel("Recovery rate")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PATHS.analysis / "fig_duplication.png", dpi=200)
    print("figure written")
"""),
        md("""
## Comparison with Zeng et al.

Their Table 3 row `None` reports 245 emails and 27 phones from 5,000 prompts on
GPT-Neo-1.3B, with retrieval switched off. Our matching cell uses the same model
and the same prompt count.

Zeng et al. never state how they count. The paper says only that they count the
number of extracted PIIs, and neither the text nor the appendix describes
checking an extracted value against the Enron corpus. So two things follow:

- Their 245 is a regex count. It is comparable to our **extracted** column and
  not to our **verified** column.
- Whether 245 is distinct values or total occurrences is unresolvable from the
  paper. Both are printed below. Do not choose the one that is closer to 245.

`extracted_*` sums per-generation distinct counts across all 5,000 generations,
so it is an occurrence count. `distinct_extracted_*` counts each value once per
cell, recomputed from the stored detection hashes.
"""),
        code("""
distinct = runner.distinct_by_cell(PATHS)
DISTINCT_VALUES = ["distinct_extracted_email", "distinct_extracted_phone",
                   "distinct_verified_email", "distinct_verified_phone"]

if not distinct.empty:
    distinct.to_csv(PATHS.analysis / "table6_distinct_by_cell.csv", index=False)
    print(f"distinct-per-cell table written, {len(distinct)} cells")

wide = grid.merge(distinct, on=["model", "condition", "seed"], how="left")
for col in DISTINCT_VALUES:
    if col not in wide.columns:
        wide[col] = 0
    wide[col] = wide[col].fillna(0)

zeng = wide[(wide["model"] == "gpt-neo-1.3b") & (wide["condition"] == "pii_leading")]
if zeng.empty:
    print("gpt-neo-1.3b pii_leading cells not present")
else:
    comp = aggregate.aggregate_cells(zeng, ["model", "condition"],
                                     VALUES + DISTINCT_VALUES,
                                     level=CONFIG.ci_level)
    row = comp.iloc[0]
    print(f"prompts per cell             : {CONFIG.prompts_per_cell:,}")
    print(f"seeds                        : {len(CONFIG.seeds)}")
    print("")
    for kind, reported in (("email", 245), ("phone", 27)):
        print(f"{kind}")
        print(f"  Zeng et al. reported       : {reported}")
        print(f"  ours, distinct extracted   : "
              f"{row[f'distinct_extracted_{kind}_mean']:.1f} "
              f"+/- {row[f'distinct_extracted_{kind}_ci']:.1f}")
        print(f"  ours, occurrences extracted: "
              f"{row[f'extracted_{kind}_mean']:.1f} "
              f"+/- {row[f'extracted_{kind}_ci']:.1f}")
        print(f"  ours, distinct verified    : "
              f"{row[f'distinct_verified_{kind}_mean']:.1f} "
              f"+/- {row[f'distinct_verified_{kind}_ci']:.1f}")
        print(f"  ours, occurrences verified : "
              f"{row[f'verified_{kind}_mean']:.1f} "
              f"+/- {row[f'verified_{kind}_ci']:.1f}")
        print("")
    print("Zeng's counting method is not stated in the paper. Report the")
    print("ambiguity rather than selecting whichever row is closest.")
    comp.to_csv(PATHS.analysis / "table5_zeng_comparison.csv", index=False)
"""),
        code("""
from dea_a3 import manifest

manifest.write_manifest(
    PATHS.analysis, "02_analyse", CONFIG.to_dict(), ROOT,
    extra={"cells_analysed": int(len(cells_df)),
           "distinct_recovered": int(len(recovered))},
)
print("stage 02 complete")
"""),
    ]


def build(out_dir: Path) -> Path:
    return write_notebook(cells(), Path(out_dir) / "02_analyse.ipynb")


if __name__ == "__main__":
    print(build(Path(__file__).resolve().parents[1] / "notebooks"))
