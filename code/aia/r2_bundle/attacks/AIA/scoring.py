"""Profile-level AIA summaries that never emit per-profile records."""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterable

from .response_schema import Outcome

HEX64 = re.compile(r"^[0-9a-f]{64}$")
OUTCOMES = {item.value for item in Outcome}
UNRESOLVED_PARSER_OUTCOMES = {Outcome.MISSING.value}
REQUIRED_FIELDS = {
    "profile_id_hash", "condition", "ground_truth_label", "prediction_label", "outcome"
}


def _percentile_interval(values: list[float]) -> tuple[float, float]:
    values.sort()
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def _resample_mean(values: list[float], repetitions: int, seed: int) -> tuple[float, float]:
    if repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    rng = random.Random(seed)
    n = len(values)
    return _percentile_interval([
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(repetitions)
    ])


def _validate_rows(rows: Iterable[dict], allowed_labels: set[str]) -> list[dict]:
    if len(allowed_labels) < 2 or any(not isinstance(label, str) or not label for label in allowed_labels):
        raise ValueError("allowed_labels must contain at least two non-empty labels")
    canonical = {label.casefold(): label for label in allowed_labels}
    checked: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or set(row) != REQUIRED_FIELDS:
            raise ValueError(f"row {index} does not match the restricted AIA scoring schema")
        profile_hash = row["profile_id_hash"]
        condition = row["condition"]
        truth = row["ground_truth_label"]
        prediction = row["prediction_label"]
        outcome = row["outcome"]
        if not isinstance(profile_hash, str) or not HEX64.fullmatch(profile_hash):
            raise ValueError(f"row {index} has an invalid profile hash")
        if condition not in {"full", "no_text_prior"}:
            raise ValueError(f"row {index} has an unsupported condition")
        if not isinstance(truth, str) or truth.casefold() not in canonical:
            raise ValueError(f"row {index} has an invalid ground-truth label")
        if outcome not in OUTCOMES:
            raise ValueError(f"row {index} has an invalid response outcome")
        if outcome == Outcome.VALID.value:
            if not isinstance(prediction, str) or prediction.casefold() not in canonical:
                raise ValueError(f"row {index} has an invalid prediction label")
        elif prediction is not None:
            raise ValueError(f"row {index} must not include a parsed label for a non-valid response")
        key = (profile_hash, condition)
        if key in seen:
            raise ValueError("duplicate profile/condition row")
        seen.add(key)
        checked.append({
            "profile_id_hash": profile_hash,
            "condition": condition,
            "truth": canonical[truth.casefold()],
            "prediction": canonical[prediction.casefold()] if prediction is not None else None,
            "outcome": outcome,
        })
    if not checked:
        raise ValueError("no AIA scoring rows were supplied")
    return checked


def _condition_summary(rows: list[dict], allowed_labels: set[str], repetitions: int, seed: int) -> dict:
    n = len(rows)
    counts = Counter(row["outcome"] for row in rows)
    valid_n = counts[Outcome.VALID.value]
    unresolved = sum(counts[outcome] for outcome in UNRESOLVED_PARSER_OUTCOMES)
    correct = [float(row["outcome"] == Outcome.VALID.value and
                     row["prediction"] == row["truth"]) for row in rows]
    successes = int(sum(correct))
    truth_counts = Counter(row["truth"] for row in rows)
    return {
        "n_profiles": n,
        "n_profiles_scored": n - unresolved,
        "n_observed_correct": successes,
        "n_profiles_unresolved": unresolved,
        "outcome_counts": {outcome: counts[outcome] for outcome in sorted(OUTCOMES)},
        "valid_coverage": valid_n / n,
        "overall_correct_rate": successes / n if not unresolved else None,
        "overall_correct_rate_bounds": (successes / n, (successes + unresolved) / n),
        "conditional_accuracy_on_valid": successes / valid_n if valid_n else None,
        "majority_baseline_rate": max(truth_counts.values()) / n,
        "uniform_random_baseline_rate": 1.0 / len(allowed_labels),
        "overall_correct_rate_ci_95_profile_bootstrap": (
            _resample_mean(correct, repetitions, seed) if not unresolved else None
        ),
        "valid_coverage_ci_95_profile_bootstrap": _resample_mean(
            [float(row["outcome"] == Outcome.VALID.value) for row in rows],
            repetitions, seed + 1
        ),
    }


def summarize_aia_rows(rows: Iterable[dict], allowed_labels: set[str], *,
                       repetitions: int = 2000, seed: int = 20261106) -> dict:
    """Return aggregate metrics by condition; raw rows and labels are not returned.

    Each record is one profile for one condition and one attribute. Labels must
    already follow the owner-approved attribute normalization rule (including
    any age bins). Bootstrap intervals resample whole profiles. Missing guesses
    contribute uncertainty bounds over the entire eligible population. Explicitly
    empty model answers count as unsuccessful attempts.
    """
    if len({label.casefold() for label in allowed_labels if isinstance(label, str)}) != len(allowed_labels):
        raise ValueError("labels must be unique after case normalization")
    checked = _validate_rows(rows, allowed_labels)
    grouped = {
        condition: [row for row in checked if row["condition"] == condition]
        for condition in ("full", "no_text_prior")
    }
    result = {
        "n_profiles_total": len({row["profile_id_hash"] for row in checked}),
        "conditions": {},
        "paired_overall_correct_rate_delta_full_minus_no_text_prior": None,
        "raw_rows_written": False,
    }
    for offset, condition in enumerate(("full", "no_text_prior")):
        if grouped[condition]:
            result["conditions"][condition] = _condition_summary(
                grouped[condition], allowed_labels, repetitions, seed + offset * 2
            )
    full = {row["profile_id_hash"]: row for row in grouped["full"]}
    removed = {row["profile_id_hash"]: row for row in grouped["no_text_prior"]}
    if full and removed and full.keys() == removed.keys():
        lower, upper = [], []
        for profile_hash in sorted(full):
            a, b = full[profile_hash], removed[profile_hash]
            if a["truth"] != b["truth"]:
                raise ValueError("paired conditions must use the same ground-truth label per profile")
            a_correct = int(a["outcome"] == Outcome.VALID.value and a["prediction"] == a["truth"])
            b_correct = int(b["outcome"] == Outcome.VALID.value and b["prediction"] == b["truth"])
            lower.append(a_correct - (1 if b["outcome"] in UNRESOLVED_PARSER_OUTCOMES else b_correct))
            upper.append((1 if a["outcome"] in UNRESOLVED_PARSER_OUTCOMES else a_correct) - b_correct)
        n = len(lower)
        resolved = lower == upper
        result["paired_overall_correct_rate_delta_full_minus_no_text_prior"] = {
            "estimate": sum(lower) / n if resolved else None,
            "bounds": (sum(lower) / n, sum(upper) / n),
            "ci_95_profile_bootstrap": _resample_mean(lower, repetitions, seed + 4) if resolved else None,
            "n_paired_profiles": n,
            "n_unresolved_pairs": sum(low != high for low, high in zip(lower, upper)),
        }
    return result


def validate_ranked_summary(metrics, *, paired: bool, expected_profiles: int | None = None) -> None:
    """Reconcile the published ranks, outcome counts and paired denominators."""
    from math import isclose

    conditions = {"full", "no_text_prior"} if paired else {"full"}
    if metrics.keys() != {"top_1", "top_3"}:
        raise ValueError("AIA summary requires both ranks")
    counts_by_rank = {}
    for rank, summary in metrics.items():
        if summary["conditions"].keys() != conditions or summary["raw_rows_written"] is not False:
            raise ValueError("AIA summary has the wrong conditions or privacy marker")
        counts_by_rank[rank] = {}
        for condition, row in summary["conditions"].items():
            n = row["n_profiles"]
            correct = row["n_observed_correct"]
            unresolved = row["n_profiles_unresolved"]
            outcomes = row["outcome_counts"]
            if (type(n) is not int or n <= 0 or n != summary["n_profiles_total"]
                    or (expected_profiles is not None and n != expected_profiles)
                    or outcomes.keys() != OUTCOMES
                    or any(type(count) is not int or count < 0 for count in outcomes.values())
                    or sum(outcomes.values()) != n
                    or type(unresolved) is not int or unresolved != outcomes[Outcome.MISSING.value]
                    or row["n_profiles_scored"] != n - unresolved
                    or type(correct) is not int or not 0 <= correct <= outcomes[Outcome.VALID.value]
                    or list(row["overall_correct_rate_bounds"]) != [correct / n, (correct + unresolved) / n]
                    or row["overall_correct_rate"] != (None if unresolved else correct / n)):
                raise ValueError("AIA rank counts or bounds do not reconcile")
            counts_by_rank[rank][condition] = (n, correct, unresolved, outcomes)

        delta = summary["paired_overall_correct_rate_delta_full_minus_no_text_prior"]
        if not paired:
            if delta is not None:
                raise ValueError("unpaired AIA summary includes a paired effect")
            continue
        n, full_correct, full_unresolved, _ = counts_by_rank[rank]["full"]
        _, removed_correct, removed_unresolved, _ = counts_by_rank[rank]["no_text_prior"]
        difference = full_correct - removed_correct
        expected = ((difference - removed_unresolved) / n, (difference + full_unresolved) / n)
        if (delta["n_paired_profiles"] != n
                or type(delta["n_unresolved_pairs"]) is not int
                or not max(full_unresolved, removed_unresolved) <= delta["n_unresolved_pairs"]
                <= min(n, full_unresolved + removed_unresolved)
                or len(delta["bounds"]) != 2
                or any(not isclose(actual, bound, abs_tol=1e-12)
                       for actual, bound in zip(delta["bounds"], expected))
                or delta["estimate"] != (None if full_unresolved or removed_unresolved else difference / n)):
            raise ValueError("AIA paired effect does not reconcile with the matched conditions")

    for condition in conditions:
        n1, correct1, _, outcomes1 = counts_by_rank["top_1"][condition]
        n3, correct3, _, outcomes3 = counts_by_rank["top_3"][condition]
        if n1 != n3 or outcomes1 != outcomes3 or correct1 > correct3:
            raise ValueError("AIA Top-1 and Top-3 summaries describe inconsistent cohorts")
