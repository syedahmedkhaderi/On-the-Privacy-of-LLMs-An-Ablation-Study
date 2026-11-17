"""The ground-truth index: V-B intersect V-P.

V-B is the maildir under DEA's filters with quote stripping disabled. V-P is
the Pile's Enron component. The intersection has two properties that neither
side has alone:

  every value is provably in the corpus the models trained on, so a verified
  recovery is a training-data extraction and not merely a real-looking string
  that happens to exist in a court archive;

  bot addresses, mailer-daemons and all-company traffic are removed, because
  V-P is body text with no headers to filter on and V-B's message-level
  filters do that work.

P8 measured the cost at 4.7% of V-P's coverage.

Only hashes are persisted. No plaintext value is written to disk at any point.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from . import pii


def build_index(
    maildir_values: dict[str, set[str]],
    pile_counts: dict[str, Counter],
    salt: str,
    kinds: tuple[str, ...] = ("email", "phone"),
) -> pd.DataFrame:
    """Intersect the two sides and attach Pile occurrence counts.

    Returns a frame of kind, value_hash, occurrences, value_length. The
    occurrence count comes from the Pile side because that is what the model
    saw; P8 measured Spearman 0.55 between the two sides' counts for email,
    which is too loose to substitute one for the other.
    """
    rows = []
    for kind in kinds:
        left = maildir_values.get(kind, set())
        right = pile_counts.get(kind, Counter())
        for value in sorted(left & set(right.keys())):
            rows.append({
                "kind": kind,
                "value_hash": pii.salted_hash(kind, value, salt),
                "occurrences": int(right[value]),
                "value_length": len(pii.normalize_value(kind, value)),
            })
    frame = pd.DataFrame(rows, columns=["kind", "value_hash", "occurrences", "value_length"])
    if frame.empty:
        return frame
    return frame.sort_values(["kind", "value_hash"]).reset_index(drop=True)


def index_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-kind counts, for the corpus table."""
    if frame.empty:
        return pd.DataFrame(columns=["kind", "distinct", "occ_median", "occ_max"])
    grp = frame.groupby("kind", sort=True)["occurrences"]
    return pd.DataFrame({
        "kind": list(grp.groups.keys()),
        "distinct": grp.size().to_numpy(),
        "occ_median": grp.median().to_numpy(),
        "occ_max": grp.max().to_numpy(),
    })


def variant_stats(
    maildir_values: dict[str, set[str]],
    pile_counts: dict[str, Counter],
    kinds: tuple[str, ...] = ("email", "phone"),
) -> pd.DataFrame:
    """Coverage and precision of the maildir side against the Pile side.

    This reproduces the P8 comparison inside the study, so the index-choice
    subsection is backed by an artifact this pipeline produced rather than by
    a number quoted from a report.
    """
    rows = []
    for kind in kinds:
        left = maildir_values.get(kind, set())
        right = set(pile_counts.get(kind, Counter()).keys())
        inter = left & right
        rows.append({
            "kind": kind,
            "maildir_distinct": len(left),
            "pile_distinct": len(right),
            "intersection": len(inter),
            "coverage": len(inter) / len(right) if right else float("nan"),
            "precision": len(inter) / len(left) if left else float("nan"),
        })
    return pd.DataFrame(rows)


def load_index(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def hashes_by_kind(frame: pd.DataFrame) -> dict[str, set[str]]:
    """The lookup structure detection uses."""
    if frame.empty:
        return {}
    return {
        str(kind): set(grp["value_hash"].tolist())
        for kind, grp in frame.groupby("kind", sort=True)
    }
