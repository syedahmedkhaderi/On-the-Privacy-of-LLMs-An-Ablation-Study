import numpy as np
import pandas as pd

from dea_a3 import aggregate


def test_mean_ci_uses_student_t_not_z():
    stat = aggregate.mean_ci([10.0, 12.0, 14.0])
    assert stat["n"] == 3
    assert stat["mean"] == 12.0
    # t(2, 0.975) = 4.302653, not 1.96
    expected = 4.302653 * 2.0 / np.sqrt(3)
    assert abs(stat["half_width"] - expected) < 1e-4


def test_single_observation_has_no_interval():
    stat = aggregate.mean_ci([7.0])
    assert stat["mean"] == 7.0
    assert np.isnan(stat["half_width"])


def test_empty_is_safe():
    assert aggregate.mean_ci([])["n"] == 0


def test_zero_cells_survive_aggregation():
    df = pd.DataFrame({
        "model": ["m"] * 3,
        "condition": ["unconditional"] * 3,
        "seed": [42, 43, 44],
        "verified_email": [0, 0, 0],
    })
    out = aggregate.aggregate_cells(df, ["model", "condition"], ["verified_email"])
    assert len(out) == 1
    assert out["verified_email_mean"].iloc[0] == 0.0
    assert out["verified_email_n"].iloc[0] == 3


def test_complete_grid_inserts_missing_cells_as_zero():
    df = pd.DataFrame({
        "model": ["m1"], "condition": ["pii_leading"], "seed": [42],
        "verified_email": [5],
    })
    out = aggregate.complete_grid(df, ["m1", "m2"], ["pii_leading", "unconditional"],
                                  [42, 43], ["verified_email"])
    assert len(out) == 8
    assert out["verified_email"].sum() == 5
    uncond = out[out["condition"] == "unconditional"]
    assert (uncond["verified_email"] == 0).all()


def test_quantile_bands_adapt_to_skew():
    counts = [1] * 100 + [50] * 10 + [5000]
    bands = aggregate.quantile_bands(counts, (0.0, 0.5, 0.9, 1.0))
    assert len(bands) >= 3, "ties must not collapse the analysis to one band"
    assert bands[0][0] == 1.0
    assert bands[-1][1] == 5000.0
    for lo, hi in bands:
        assert lo < hi


def test_bands_survive_extreme_ties():
    """Almost every value sharing one count is the realistic case."""
    counts = [2] * 5000 + [3] * 20 + [900]
    bands = aggregate.quantile_bands(counts, (0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0))
    assert len(bands) >= 3
    assert bands[0][0] == 2.0
    assert bands[-1][1] == 900.0


def test_all_identical_counts_gives_one_band_not_crash():
    bands = aggregate.quantile_bands([7] * 50, (0.0, 0.5, 1.0))
    assert bands == [(7.0, 7.0)]


def test_bands_cover_every_value():
    import numpy as _np
    counts = list(_np.random.default_rng(0).integers(1, 400, size=2000))
    bands = aggregate.quantile_bands(counts, (0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0))
    covered = 0
    for i, (lo, hi) in enumerate(bands):
        last = i == len(bands) - 1
        covered += sum(1 for c in counts if (lo <= c <= hi) if last or c < hi)
    assert covered == len(counts)


def test_recovery_by_band_rate_rises_with_duplication():
    rows = []
    for i in range(100):
        rows.append({"kind": "email", "value_hash": f"lo{i}", "occurrences": 1})
    for i in range(100):
        rows.append({"kind": "email", "value_hash": f"hi{i}", "occurrences": 500})
    idx = pd.DataFrame(rows)
    recovered = {f"lo{i}" for i in range(2)} | {f"hi{i}" for i in range(60)}
    out = aggregate.recovery_by_band(idx, recovered, (0.0, 0.5, 1.0))
    assert len(out) == 2
    assert out["recovery_rate"].iloc[0] < out["recovery_rate"].iloc[-1]
    assert out["in_corpus"].sum() == 200


def test_recovery_by_band_handles_no_recoveries():
    idx = pd.DataFrame({"kind": ["email"] * 10,
                        "value_hash": [f"h{i}" for i in range(10)],
                        "occurrences": list(range(1, 11))})
    out = aggregate.recovery_by_band(idx, set(), (0.0, 0.5, 1.0))
    assert (out["recovery_rate"] == 0).all()
    assert out["in_corpus"].sum() == 10
