"""Paper figures for dea_a3, restricted to the three conditioned prompt styles.

The published paper does not report the "unconditional" baseline, so it must
not appear in any figure, legend or tick label. The 84 runs stay in results/;
only the plotting is narrowed.

Nothing here re-runs generation or detection. Per-run numbers are read from the
stage 02 tables and the stored detection hashes. The one quantity that has to
be recounted is the "recovered" bar of the duplication figure: stage 02 pooled
all four styles into a single hash set, so dropping unconditional changes it.
That recount uses aggregate.recovery_by_band (src/dea_a3/aggregate.py:132) with
the same band quantiles, so the only difference from the stored table is which
runs feed the hash set.

It also writes the two paper tables to results/02_analysis/:
paper_per_run_3styles.csv (mean and CI of distinct verified and extracted
values per run, per model and style, 63 runs) and
table_duplication_3styles.csv (the three-style duplication bands with exact,
lower-inclusive, upper-exclusive edges).

Hashes stay in memory. They are never printed, plotted or written out.

    python tools/make_a3_figs_3styles.py [--data-root DIR] [--out DIR] [--tables-out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd

# Small panels, so type has to be sized down or the axis labels eat the plot.
plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.7,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dea_a3 import aggregate  # noqa: E402
from dea_a3.config import CONFIG  # noqa: E402

# Order, markers and line styles are fixed by the paper. Greyscale-safe: the
# three series are told apart by marker and dash pattern, not by colour.
STYLES = [
    ("generic_prefix", "s", "-", "0.35"),
    ("pii_leading", "^", "--", "0.65"),
    ("true_prefix", "D", "-.", "0.05"),
]
STYLE_NAMES = [s[0] for s in STYLES]
ALL_STYLES = list(CONFIG.conditions)

T_CRIT_N3 = 4.302653  # Student t, 2 df, two-sided 95%
BAR_GREYS = {"in corpus": "0.75", "recovered": "0.25"}

# Stage 02 verification rates, percent verified of extracted, per condition.
# Dropping unconditional does not change a per-condition rate; these are the
# stored values and the script stops if the results no longer produce them.
EXPECTED_VERIFY_RATE = {
    "generic_prefix": (8.35, 19.50),
    "pii_leading": (1.97, 2.65),
    "true_prefix": (12.94, 25.76),
}


def fail(msg: str) -> None:
    raise SystemExit(f"STOP: {msg}")


def load_recovered_hashes(gen_dir: Path, conditions) -> set[str]:
    """Distinct value hashes verified at least once in any run of `conditions`.

    Same selection stage 02 makes (verified rows of every cell's
    detections.parquet, see src/dea_a3/runner.py:233), narrowed by condition.
    """
    wanted = set(conditions)
    recovered: set[str] = set()
    for cell_dir in sorted(gen_dir.glob("*__*__seed*")):
        if cell_dir.name.split("__")[1] not in wanted:
            continue
        det_file = cell_dir / "detections.parquet"
        if not det_file.exists():
            continue
        frame = pd.read_parquet(det_file)
        if frame.empty:
            continue
        recovered.update(frame.loc[frame["verified"], "value_hash"].unique())
    return recovered


def distinct_by_kind(gen_dir: Path, conditions) -> dict[str, int]:
    """Distinct verified values per kind, pooled over the given conditions."""
    wanted = set(conditions)
    seen: dict[str, set[str]] = {"email": set(), "phone": set()}
    for cell_dir in sorted(gen_dir.glob("*__*__seed*")):
        if cell_dir.name.split("__")[1] not in wanted:
            continue
        det_file = cell_dir / "detections.parquet"
        if not det_file.exists():
            continue
        frame = pd.read_parquet(det_file)
        if frame.empty:
            continue
        ok = frame[frame["verified"]]
        for kind, grp in ok.groupby("kind"):
            seen.setdefault(kind, set()).update(grp["value_hash"].unique())
    return {k: len(v) for k, v in seen.items()}


def self_check(index_frame: pd.DataFrame, gen_dir: Path, stored: pd.DataFrame) -> None:
    """Reproduce the stored four-style table before trusting the recount."""
    recovered = load_recovered_hashes(gen_dir, ALL_STYLES)
    rebuilt = aggregate.recovery_by_band(index_frame, recovered, CONFIG.band_quantiles)
    cols = ["kind", "band", "in_corpus", "recovered"]
    a = rebuilt[cols].reset_index(drop=True)
    b = stored[cols].reset_index(drop=True)
    rows_match = a.equals(b)

    totals = distinct_by_kind(gen_dir, ALL_STYLES)
    totals_match = totals.get("email") == 1846 and totals.get("phone") == 1092

    print(f"SELF-CHECK rows vs table4_duplication.csv: {'MATCH' if rows_match else 'MISMATCH'}")
    print(f"SELF-CHECK four-style distinct verified: email {totals.get('email'):,} "
          f"phone {totals.get('phone'):,} "
          f"({'MATCH' if totals_match else 'MISMATCH'}, expected 1,846 / 1,092)")
    if not (rows_match and totals_match):
        if not rows_match:
            print(a.compare(b) if a.shape == b.shape else f"shape {a.shape} vs {b.shape}")
        fail("self-check MISMATCH; figures not written")
    print("SELF-CHECK: MATCH")


def per_run_stats(table6: pd.DataFrame, model: str, style: str, col: str):
    """Mean and 95% t half-width over the three seeds of one model and style."""
    vals = table6[(table6["model"] == model) & (table6["condition"] == style)][col]
    vals = vals.to_numpy(dtype=float)
    if vals.size != 3:
        fail(f"{style}/{model} has {vals.size} seeds, expected 3")
    return vals.mean(), T_CRIT_N3 * vals.std(ddof=1) / np.sqrt(3)


def fig_scale(table6: pd.DataFrame, out: Path) -> list[Path]:
    """Distinct verified emails per run against model size.

    Only the Pythia models differ in size alone, so only they are joined by a
    line. GPT-Neo and GPT-J points are hollow and unconnected, because a line
    through them would also cross a change of family and training recipe.
    """
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    pythia = sorted((m for m in CONFIG.models if m.family == "pythia"),
                    key=lambda m: m.params_m)
    others = [m for m in CONFIG.models if m.family != "pythia"]
    col = "distinct_verified_email"
    handles = []
    for name, marker, ls, colour in STYLES:
        stats = [per_run_stats(table6, m.key, name, col) for m in pythia]
        ax.errorbar([m.params_m for m in pythia], [s[0] for s in stats],
                    yerr=[s[1] for s in stats], marker=marker, linestyle=ls,
                    color=colour, capsize=2, markersize=4, linewidth=1.1)
        for m in others:
            mean, ci = per_run_stats(table6, m.key, name, col)
            ax.errorbar([m.params_m], [mean], yerr=[ci], marker=marker,
                        linestyle="none", color=colour, markerfacecolor="white",
                        markeredgecolor=colour, markeredgewidth=1.0, capsize=2,
                        markersize=4, elinewidth=1.1)
        handles.append((Line2D([], [], marker=marker, linestyle=ls, color=colour,
                               markersize=4, linewidth=1.1), name))
    family_key = (
        Line2D([], [], marker="o", linestyle="-", color="0.3", markersize=4,
               linewidth=1.1),
        Line2D([], [], marker="o", linestyle="none", color="0.3",
               markerfacecolor="white", markersize=4),
    )
    handles.append((family_key, "filled, joined: Pythia\nhollow: GPT-Neo / GPT-J"))
    ax.set_xscale("log")
    ticks = [100, 300, 1000, 3000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{int(v):,}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(50, 9000)
    ax.set_xlabel("Model parameters (millions)")
    ax.set_ylabel("Distinct verified emails per run")
    ax.margins(y=0.08)
    ax.legend([h for h, _ in handles], [lab for _, lab in handles],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
              loc="upper left", frameon=False, handlelength=2.6)
    written = []
    for suffix in (".png", ".pdf"):
        path = out.with_suffix(suffix)
        # No creation date in the PDF, so a rerun gives the same bytes.
        meta = {"CreationDate": None} if suffix == ".pdf" else None
        fig.savefig(path, dpi=300, bbox_inches="tight", metadata=meta)
        written.append(path)
    plt.close(fig)
    return written


def fig_verify_rate(rates: dict[str, tuple[float, float]], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    kinds = ["email", "phone"]
    x = np.arange(len(kinds))
    width = 0.26
    for i, (name, _marker, _ls, colour) in enumerate(STYLES):
        vals = list(rates[name])
        ax.bar(x + (i - 1) * width, vals, width, label=name, color=colour,
               edgecolor="black", linewidth=0.5, hatch=["", "//", "xx"][i])
    ax.set_xticks(x)
    ax.set_xticklabels(kinds)
    ax.set_ylabel("Verified share of extracted (%)")
    ax.set_xlabel("PII kind")
    ax.margins(y=0.22)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def exact_band_labels(grp: pd.DataFrame) -> list[str]:
    """Band labels as the integer ranges they contain.

    Bands are lower-inclusive and upper-exclusive, except the last, which is
    closed. So [2, 4) is "2-3", [1, 2) is "1" and the last band is "50+".
    """
    labels = []
    rows = list(grp.itertuples(index=False))
    for i, r in enumerate(rows):
        lo, hi = int(round(r.band_lo)), int(round(r.band_hi))
        if i == len(rows) - 1:
            labels.append(f"{lo}+")
        elif hi - 1 == lo:
            labels.append(str(lo))
        else:
            labels.append(f"{lo}-{hi - 1}")
    return labels


def write_per_run_table(table6: pd.DataFrame, out: Path) -> Path:
    """One row per model and style over the three reported styles."""
    rows = []
    for m in sorted(CONFIG.models, key=lambda m: m.params_m):
        for name in STYLE_NAMES:
            row = {"model": m.key, "params": m.params_m, "family": m.family,
                   "style": name}
            for col in ("verified_email", "verified_phone",
                        "extracted_email", "extracted_phone"):
                mean, ci = per_run_stats(table6, m.key, name, f"distinct_{col}")
                row[f"{col}_mean"] = round(float(mean), 4)
                row[f"{col}_ci"] = round(float(ci), 4)
            row["n_seeds"] = int(((table6["model"] == m.key)
                                  & (table6["condition"] == name)).sum())
            rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_duplication_table(three: pd.DataFrame, index_frame: pd.DataFrame,
                            out: Path) -> Path:
    """Three-style duplication bands with explicit integer edges.

    upper_exclusive of the last band is the largest occurrence count plus one,
    because the last band is closed. The "all" row covers every indexed value.
    """
    rows = []
    for kind, grp in three.groupby("kind", sort=True):
        grp = grp.reset_index(drop=True)
        top = int(index_frame.loc[index_frame["kind"] == kind, "occurrences"].max())
        for label, r in zip(exact_band_labels(grp), grp.itertuples(index=False)):
            last = label.endswith("+")
            rows.append({"kind": kind, "band_label": label,
                         "lower_inclusive": int(round(r.band_lo)),
                         "upper_exclusive": top + 1 if last else int(round(r.band_hi)),
                         "in_data": int(r.in_corpus), "recovered": int(r.recovered),
                         "rate": round(r.recovered / r.in_corpus, 6)})
        in_data = int(grp["in_corpus"].sum())
        recovered = int(grp["recovered"].sum())
        rows.append({"kind": kind, "band_label": "all",
                     "lower_inclusive": int(round(grp["band_lo"].min())),
                     "upper_exclusive": top + 1, "in_data": in_data,
                     "recovered": recovered, "rate": round(recovered / in_data, 6)})
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def fig_duplication(table: pd.DataFrame, out: Path) -> Path:
    kinds = ["email", "phone"]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    for ax, kind in zip(axes, kinds):
        grp = table[table["kind"] == kind]
        x = np.arange(len(grp))
        width = 0.38
        ax.bar(x - width / 2, grp["in_corpus"], width, label="in corpus",
               color=BAR_GREYS["in corpus"], edgecolor="black", linewidth=0.5)
        bars = ax.bar(x + width / 2, grp["recovered"], width, label="recovered",
                      color=BAR_GREYS["recovered"], edgecolor="black",
                      linewidth=0.5, hatch="//")
        for rect, val in zip(bars, grp["recovered"]):
            ax.annotate(f"{int(val)}", (rect.get_x() + rect.get_width() / 2,
                                        max(int(val), 1)),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=6)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(exact_band_labels(grp), rotation=45, ha="right")
        ax.set_xlabel(f"Occurrences in corpus ({kind})")
        # log axes ignore margins, so widen the top by a decade-fraction instead
        ax.set_ylim(top=ax.get_ylim()[1] * 3.0)
    axes[0].set_ylabel("Distinct values")
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(REPO_ROOT),
                    help="directory holding results/ (default: this repo)")
    ap.add_argument("--out", default=str(REPO_ROOT / "figs"))
    ap.add_argument("--tables-out", default=str(REPO_ROOT / "results" / "02_analysis"),
                    help="where the two paper CSVs go")
    args = ap.parse_args()

    results = Path(args.data_root) / "results"
    gen_dir = results / "01_generations"
    analysis = results / "02_analysis"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_frame = pd.read_parquet(results / "00_index" / "index.parquet")
    stored_t4 = pd.read_csv(analysis / "table4_duplication.csv")
    table6 = pd.read_csv(analysis / "table6_distinct_by_cell.csv")
    summaries = pd.read_csv(gen_dir / "cell_summaries.csv")

    self_check(index_frame, gen_dir, stored_t4)

    # Verification rates, recomputed from the stored per-cell counts.
    rates = {}
    for name in STYLE_NAMES:
        sub = summaries[summaries["condition"] == name]
        pair = []
        for kind in ("email", "phone"):
            ex = float(sub[f"extracted_{kind}"].sum())
            ve = float(sub[f"verified_{kind}"].sum())
            pair.append(round(100.0 * ve / ex, 2) if ex else 0.0)
        rates[name] = tuple(pair)
        if pair != list(EXPECTED_VERIFY_RATE[name]):
            fail(f"verify rate for {name} is {pair}, stored {EXPECTED_VERIFY_RATE[name]}")
    print("verify rates match the stored stage 02 values")

    three = aggregate.recovery_by_band(
        index_frame, load_recovered_hashes(gen_dir, STYLE_NAMES), CONFIG.band_quantiles)
    # Band edges and in_corpus come from the corpus index, so they must be
    # identical to the stored table; only "recovered" may move.
    edge_cols = ["kind", "band", "band_lo", "band_hi", "in_corpus"]
    if not three[edge_cols].reset_index(drop=True).equals(
            stored_t4[edge_cols].reset_index(drop=True)):
        fail("band edges or in_corpus drifted from table4_duplication.csv")

    table6_3 = table6[table6["condition"].isin(STYLE_NAMES)]
    paths = [
        *fig_scale(table6_3, out_dir / "a3_scale.png"),
        fig_verify_rate(rates, out_dir / "a3_verify_rate.png"),
        fig_duplication(three, out_dir / "a3_duplication.png"),
    ]
    tables_out = Path(args.tables_out)
    tables_out.mkdir(parents=True, exist_ok=True)
    paths += [
        write_per_run_table(table6_3, tables_out / "paper_per_run_3styles.csv"),
        write_duplication_table(three, index_frame,
                                tables_out / "table_duplication_3styles.csv"),
    ]

    totals = distinct_by_kind(gen_dir, STYLE_NAMES)
    gpu_seconds = 0.0
    cells = 0
    for cell_dir in sorted(gen_dir.glob("*__*__seed*")):
        if cell_dir.name.split("__")[1] not in set(STYLE_NAMES):
            continue
        marker = cell_dir / "COMPLETE"
        if marker.exists():
            gpu_seconds += float(marker.read_text().strip())
            cells += 1

    print()
    print("NUMBERS FOR THE PAPER")
    print("three-style duplication table (kind, band, in_corpus, recovered):")
    for _, r in three.iterrows():
        print(f"  {r['kind']:6s} {r['band']:>6s} {int(r['in_corpus']):8d} {int(r['recovered']):8d}")
    print(f"three-style distinct verified: email {totals.get('email', 0):,} "
          f"phone {totals.get('phone', 0):,} "
          f"total {totals.get('email', 0) + totals.get('phone', 0):,}")
    print(f"GPU hours, {cells} three-style runs: {gpu_seconds / 3600.0:.2f}")
    for p in paths:
        print(f"sha256 {p.name}: {hashlib.sha256(p.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
