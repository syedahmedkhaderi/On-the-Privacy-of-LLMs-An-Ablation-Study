import subprocess
import sys
from pathlib import Path

from dea_a3 import seeds

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_cell_seed_is_deterministic_within_process():
    a = seeds.cell_seed(42, "pythia-70m", "pii_leading")
    b = seeds.cell_seed(42, "pythia-70m", "pii_leading")
    assert a == b


def test_cell_seed_distinguishes_cells():
    vals = {
        seeds.cell_seed(42, "pythia-70m", "pii_leading"),
        seeds.cell_seed(42, "pythia-70m", "true_prefix"),
        seeds.cell_seed(42, "pythia-160m", "pii_leading"),
        seeds.cell_seed(43, "pythia-70m", "pii_leading"),
    }
    assert len(vals) == 4


def test_cell_seed_is_stable_across_processes():
    """The whole point: Python's hash() is salted per process, crc32 is not."""
    code = (
        f"import sys; sys.path.insert(0, {SRC!r});"
        "from dea_a3 import seeds;"
        "print(seeds.cell_seed(42, 'pythia-1.4b', 'generic_prefix'))"
    )
    outs = set()
    for hashseed in ("0", "1", "12345"):
        env = {"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, env=env)
        assert out.returncode == 0, out.stderr
        outs.add(out.stdout.strip())
    assert len(outs) == 1, f"seed varied across processes: {outs}"


def test_batch_seeds_are_distinct_and_stable():
    cell = seeds.cell_seed(42, "m", "c")
    a = [seeds.batch_seed(cell, i) for i in range(50)]
    b = [seeds.batch_seed(cell, i) for i in range(50)]
    assert a == b
    assert len(set(a)) == 50


def test_seeds_stay_in_range():
    for i in range(0, 5000, 97):
        s = seeds.cell_seed(i, "model", "cond")
        assert 0 <= s < 2**31 - 1
        assert 0 <= seeds.batch_seed(s, i) < 2**31 - 1


def test_set_all_seeds_makes_python_rng_reproducible():
    import random
    seeds.set_all_seeds(123)
    a = [random.random() for _ in range(5)]
    seeds.set_all_seeds(123)
    assert a == [random.random() for _ in range(5)]
