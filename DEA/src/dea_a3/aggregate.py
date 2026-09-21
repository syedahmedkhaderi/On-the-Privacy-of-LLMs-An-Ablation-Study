"""Aggregation across seeds, and the duplication analysis.

Confidence intervals use Student's t at n-1 degrees of freedom, matching the
convention settled in the rs-1 CI correction. With three seeds that is
t(2, 0.975) = 4.302653, which is wide. It is honest: three points do not
support a narrow interval, and reporting a z interval here would understate
the spread by a factor of more than two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(values, level: float = 0.95) -> dict[str, float]:
    """Mean and half-width of a t confidence interval.

    A single observation gets a NaN half-width rather than zero, because one
    point carries no information about spread.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    n = int(arr.size)
    if n == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0,
                "half_width": float("nan"), "lo": float("nan"), "hi": float("nan")}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "sd": float("nan"), "n": 1,
                "half_width": float("nan"), "lo": float("nan"), "hi": float("nan")}
    sd = float(arr.std(ddof=1))
    tcrit = float(stats.t.ppf(0.5 + level / 2.0, n - 1))
    half = tcrit * sd / np.sqrt(n)
    return {"mean": mean, "sd": sd, "n": n, "half_width": float(half),
            "lo": mean - half, "hi": mean + half}


def aggregate_cells(df: pd.DataFrame, group_cols, value_cols,
                    level: float = 0.95) -> pd.DataFrame:
    """Reduce per-seed cell totals to mean and CI per group.

    Zero-yield groups are preserved. A cell that recovered nothing reports
    zero; it must never vanish from the table, because the unconditional
    baseline is the headline contrast and it is expected to be zero.
    """
    group_cols = list(group_cols)
    value_cols = list(value_cols)
    rows = []
    for key, grp in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        for col in value_cols:
            stat = mean_ci(grp[col].to_numpy(), level=level)
            row[f"{col}_mean"] = stat["mean"]
            row[f"{col}_sd"] = stat["sd"]
            row[f"{col}_n"] = stat["n"]
            row[f"{col}_ci"] = stat["half_width"]
            row[f"{col}_lo"] = stat["lo"]
            row[f"{col}_hi"] = stat["hi"]
        rows.append(row)
    return pd.DataFrame(rows)


def complete_grid(df: pd.DataFrame, models, conditions, seeds,
                  value_cols) -> pd.DataFrame:
    """Insert explicit zero rows for cells that produced no detections.

    Without this, a groupby silently drops empty cells and the unconditional
    row disappears from the results table.
    """
    value_cols = list(value_cols)
    full = pd.MultiIndex.from_product(
        [list(models), list(conditions), list(seeds)],
        names=["model", "condition", "seed"],
    ).to_frame(index=False)
    merged = full.merge(df, on=["model", "condition", "seed"], how="left")
    for col in value_cols:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = merged[col].fillna(0)
    return merged


def quantile_bands(counts, quantiles, min_bands: int = 3) -> list[tuple[float, float]]:
    """Band edges over an occurrence-count distribution.

    Duplication counts are heavily skewed, so fixed boundaries put almost
    everything in one bucket and quantile edges adapt better. But occurrence
    counts are also heavily TIED: when most values share the same count,
    several quantiles land on the same number, np.unique collapses them, and
    the analysis silently degrades to one band.

    So quantile edges are tried first, and if ties collapsed them below
    min_bands the edges are augmented with log-spaced boundaries over the
    observed range. Counts span orders of magnitude, so log spacing is the
    right fallback shape.
    """
    arr = np.asarray(list(counts), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return []
    lo, hi = float(arr.min()), float(arr.max())
    if lo == hi:
        return [(lo, hi)]

    edges = np.unique(np.quantile(arr, list(quantiles)))
    if edges.size - 1 < min_bands:
        n = max(min_bands, 1)
        log_edges = np.geomspace(max(lo, 1.0), hi, n + 1)
        edges = np.unique(np.concatenate([edges, log_edges]))

    edges = np.unique(np.clip(edges, lo, hi))
    if edges.size < 2:
        return [(lo, hi)]
    edges[0], edges[-1] = lo, hi
    edges = np.unique(edges)
    if edges.size < 2:
        return [(lo, hi)]
    return [(float(edges[i]), float(edges[i + 1])) for i in range(edges.size - 1)]


def band_label(lo: float, hi: float, is_last: bool) -> str:
    lo_i, hi_i = int(round(lo)), int(round(hi))
    if lo_i == hi_i:
        return str(lo_i)
    return f"{lo_i}-{hi_i}" if not is_last else f"{lo_i}+"


def recovery_by_band(index_counts: pd.DataFrame, recovered_hashes: set[str],
                     quantiles) -> pd.DataFrame:
    """Recovery rate per duplication band.

    index_counts needs columns: kind, value_hash, occurrences. The occurrence
    count must come from the training corpus, not the source maildir; P8
    measured the two at Spearman 0.55 for email, which is too loose to treat
    one as the other.
    """
    out = []
    for kind, grp in index_counts.groupby("kind", sort=True):
        bands = quantile_bands(grp["occurrences"], quantiles)
        if not bands:
            continue
        for i, (lo, hi) in enumerate(bands):
            is_last = i == len(bands) - 1
            if is_last:
                sel = grp[(grp["occurrences"] >= lo) & (grp["occurrences"] <= hi)]
            else:
                sel = grp[(grp["occurrences"] >= lo) & (grp["occurrences"] < hi)]
            total = int(len(sel))
            if total == 0:
                continue
            hit = int(sel["value_hash"].isin(recovered_hashes).sum())
            out.append({
                "kind": kind,
                "band": band_label(lo, hi, is_last),
                "band_lo": lo,
                "band_hi": hi,
                "in_corpus": total,
                "recovered": hit,
                "recovery_rate": hit / total,
            })
    return pd.DataFrame(out)
