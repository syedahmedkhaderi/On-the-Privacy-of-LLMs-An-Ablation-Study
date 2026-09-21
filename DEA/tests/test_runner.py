"""End-to-end test of the runner with a stub model.

No torch, no weights, no GPU. The stub returns deterministic text so the
whole path from prompt construction through detection to parquet can be
checked, including resume behaviour.
"""
import random
from dataclasses import replace

import pandas as pd
import pytest

from dea_a3 import modeling, pii, runner
from dea_a3.config import Config, ModelSpec
from dea_a3.paths import Paths

SALT = "stub-salt"
REAL_EMAIL = "vince.kaminski@enron.com"
FAKE_EMAIL = "not.in.corpus@example.org"


class StubTokenizer:
    bos_token = "<|endoftext|>"
    eos_token = "<|endoftext|>"
    pad_token = "<|endoftext|>"
    pad_token_id = 0


def _stub_generate(loaded, prompts, decoder, seed):
    """Emit a real value, a fake one, or nothing, drawn from the batch seed.

    The draw must come from the seeded RNG rather than from arithmetic on the
    index, otherwise every seed produces the same totals and the test cannot
    tell reproducibility from coincidence.
    """
    rng = random.Random(seed)
    out = []
    for prompt in prompts:
        roll = rng.random()
        if roll < 0.34:
            tail = f" reach him at {REAL_EMAIL} anytime"
        elif roll < 0.67:
            tail = f" try {FAKE_EMAIL} instead"
        else:
            tail = " no contact details here at all"
        out.append(prompt + tail + f" {rng.randint(0, 9)}")
    return out


@pytest.fixture
def wired(tmp_path, monkeypatch):
    monkeypatch.setattr(modeling, "generate_batch", _stub_generate)
    paths = Paths(tmp_path)
    paths.ensure()
    cfg = Config(
        seeds=(42, 43),
        prompts_per_cell=60,
        conditions=("unconditional", "pii_leading"),
        models=(ModelSpec("stub", "stub/stub", "stub", 1, batch_size=16),),
    )
    loaded = modeling.LoadedModel("stub", object(), StubTokenizer(), "cpu")
    index_hashes = {"email": {pii.salted_hash("email", REAL_EMAIL, SALT)},
                    "phone": set()}
    return cfg, paths, loaded, index_hashes


def test_cell_runs_and_writes_artifacts(wired):
    cfg, paths, loaded, idx = wired
    summary = runner.run_cell(loaded, "stub", "pii_leading", 42, cfg, paths,
                              idx, SALT, progress=False)
    cell_dir = paths.cell_dir("stub", "pii_leading", 42)
    assert (cell_dir / "generations.parquet").exists()
    assert (cell_dir / "detections.parquet").exists()
    assert (cell_dir / "COMPLETE").exists()
    assert summary["generations"] == 60


def test_extracted_exceeds_verified(wired):
    cfg, paths, loaded, idx = wired
    s = runner.run_cell(loaded, "stub", "pii_leading", 42, cfg, paths, idx,
                        SALT, progress=False)
    assert s["extracted_email"] > s["verified_email"] > 0


def test_no_plaintext_pii_in_written_artifacts(wired):
    cfg, paths, loaded, idx = wired
    runner.run_cell(loaded, "stub", "pii_leading", 42, cfg, paths, idx, SALT,
                    progress=False)
    cell_dir = paths.cell_dir("stub", "pii_leading", 42)
    for name in ("generations.parquet", "detections.parquet"):
        blob = (cell_dir / name).read_bytes()
        assert REAL_EMAIL.encode() not in blob
        assert b"vince.kaminski" not in blob


def test_rerun_is_skipped_and_identical(wired):
    cfg, paths, loaded, idx = wired
    first = runner.run_cell(loaded, "stub", "pii_leading", 42, cfg, paths, idx,
                            SALT, progress=False)
    second = runner.run_cell(loaded, "stub", "pii_leading", 42, cfg, paths, idx,
                             SALT, progress=False)
    for key in ("generations", "extracted_email", "verified_email"):
        assert first[key] == second[key]


def test_same_seed_reproduces_bit_for_bit(wired, tmp_path):
    cfg, paths, loaded, idx = wired
    runner.run_cell(loaded, "stub", "pii_leading", 42, cfg, paths, idx, SALT,
                    progress=False)
    a = pd.read_parquet(paths.cell_dir("stub", "pii_leading", 42) / "generations.parquet")

    other = Paths(tmp_path / "second")
    other.ensure()
    runner.run_cell(loaded, "stub", "pii_leading", 42, cfg, other, idx, SALT,
                    progress=False)
    b = pd.read_parquet(other.cell_dir("stub", "pii_leading", 42) / "generations.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_give_different_results(wired):
    cfg, paths, loaded, idx = wired
    a = runner.run_cell(loaded, "stub", "pii_leading", 42, cfg, paths, idx,
                        SALT, progress=False)
    b = runner.run_cell(loaded, "stub", "pii_leading", 43, cfg, paths, idx,
                        SALT, progress=False)
    assert a["extracted_email"] != b["extracted_email"] or \
           a["verified_email"] != b["verified_email"]


def test_unconditional_cell_is_recorded_even_when_empty(wired, monkeypatch):
    cfg, paths, loaded, idx = wired
    monkeypatch.setattr(modeling, "generate_batch",
                        lambda l, p, d, s: [x + " nothing here" for x in p])
    s = runner.run_cell(loaded, "stub", "unconditional", 42, cfg, paths, idx,
                        SALT, progress=False)
    assert s["generations"] == 60
    assert s["verified_email"] == 0
    assert (paths.cell_dir("stub", "unconditional", 42) / "COMPLETE").exists()


def test_summaries_collect_across_cells(wired):
    cfg, paths, loaded, idx = wired
    for cond in cfg.conditions:
        for seed in cfg.seeds:
            runner.run_cell(loaded, "stub", cond, seed, cfg, paths, idx, SALT,
                            progress=False)
    out = runner.load_cell_summaries(paths)
    assert len(out) == 4
    dets = runner.load_detections(paths, verified_only=True)
    assert not dets.empty
    assert dets["verified"].all()


def test_distinct_by_cell_separates_values_from_occurrences(wired):
    """The stub emits exactly one real and one fake value, many times over.

    So distinct must be 1 for each while occurrences are far higher. If the two
    ever come out equal the distinct computation has collapsed into the sum.
    """
    cfg, paths, loaded, idx = wired
    summary = runner.run_cell(loaded, "stub", "pii_leading", 42, cfg, paths,
                              idx, SALT, progress=False)
    distinct = runner.distinct_by_cell(paths)

    assert len(distinct) == 1
    row = distinct.iloc[0]
    assert row["model"] == "stub"
    assert row["condition"] == "pii_leading"
    assert row["seed"] == 42

    assert row["distinct_extracted_email"] == 2   # one real, one fake
    assert row["distinct_verified_email"] == 1    # only the real one is indexed
    assert row["distinct_extracted_phone"] == 0
    assert row["distinct_verified_phone"] == 0

    assert summary["extracted_email"] > row["distinct_extracted_email"]
    assert summary["verified_email"] > row["distinct_verified_email"]


def test_distinct_by_cell_is_empty_without_detections(tmp_path):
    paths = Paths(tmp_path)
    paths.ensure()
    frame = runner.distinct_by_cell(paths)
    assert frame.empty
    assert "distinct_verified_email" in frame.columns


def test_batch_size_is_recorded_with_each_cell(wired):
    import json
    cfg, paths, loaded, idx = wired
    runner.run_cell(loaded, "stub", "pii_leading", 42, cfg, paths, idx, SALT,
                    batch_size=12, progress=False)
    config = paths.cell_dir("stub", "pii_leading", 42) / "cell_config.json"
    assert config.exists()
    assert json.loads(config.read_text())["batch_size"] == 12


def test_a_mixed_batch_size_is_detectable_on_resume(wired):
    """Batch size decides which prompts share a seeded sampling pass.

    Two cells of one model run at different batch sizes are not comparable, so
    the mismatch has to be visible rather than silent.
    """
    cfg, paths, loaded, idx = wired
    runner.run_cell(loaded, "stub", "pii_leading", 42, cfg, paths, idx, SALT,
                    batch_size=12, progress=False)
    assert runner.completed_batch_sizes(paths, "stub") == {12}

    runner.run_cell(loaded, "stub", "pii_leading", 43, cfg, paths, idx, SALT,
                    batch_size=20, progress=False)
    assert runner.completed_batch_sizes(paths, "stub") == {12, 20}


def test_batch_size_changes_the_generations(wired):
    """Not a performance knob. Different batching, different sampling."""
    import pandas as pd
    cfg, paths, loaded, idx = wired

    runner.run_cell(loaded, "stub", "pii_leading", 42, cfg, paths, idx, SALT,
                    batch_size=10, progress=False)
    a = pd.read_parquet(paths.cell_dir("stub", "pii_leading", 42) / "generations.parquet")

    runner.run_cell(loaded, "stub", "pii_leading", 44, cfg, paths, idx, SALT,
                    batch_size=30, progress=False)
    b = pd.read_parquet(paths.cell_dir("stub", "pii_leading", 44) / "generations.parquet")

    assert len(a) == len(b) == cfg.prompts_per_cell
