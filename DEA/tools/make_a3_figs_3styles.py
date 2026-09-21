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

Hashes stay in memory. They are never printed, plotted or written out.

    python tools/make_a3_figs_3styles.py [--data-root DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def fig_scale(table6: pd.DataFrame, out: Path) -> Path:
    params = {m.key: m.params_m for m in CONFIG.models}
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for name, marker, ls, colour in STYLES:
        sub = table6[table6["condition"] == name]
        xs, ys, es = [], [], []
        for model, grp in sub.groupby("model"):
            vals = grp["distinct_verified_email"].to_numpy(dtype=float)
            if vals.size != 3:
                fail(f"{name}/{model} has {vals.size} seeds, expected 3")
            xs.append(params[model])
            ys.append(vals.mean())
            es.append(T_CRIT_N3 * vals.std(ddof=1) / np.sqrt(3))
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        ys = np.asarray(ys)[order]
        es = np.asarray(es)[order]
        ax.errorbar(xs, ys, yerr=es, marker=marker, linestyle=ls, color=colour,
                    capsize=2, markersize=4, linewidth=1.1, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Model parameters (millions)")
    ax.set_ylabel("Distinct verified emails per run")
    ax.margins(y=0.18)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


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
        ax.set_xticklabels(grp["band"], rotation=45, ha="right")
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

    paths = [
        fig_scale(table6[table6["condition"].isin(STYLE_NAMES)], out_dir / "a3_scale.png"),
        fig_verify_rate(rates, out_dir / "a3_verify_rate.png"),
        fig_duplication(three, out_dir / "a3_duplication.png"),
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
