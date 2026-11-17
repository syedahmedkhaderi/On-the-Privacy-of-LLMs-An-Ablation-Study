"""Deterministic seeding.

Two mechanisms, taken from the two rs-3 notebooks that each solved half the
problem:

  cell_seed()  derives an integer from a cell's identity using zlib.crc32,
               which is stable across processes. Python's built-in hash() is
               salted per process and must never be used for a seed.

  batch_seed() derives a per-batch generator seed from the cell seed and the
               batch index alone, so batch b of a cell starts from an
               identical sampling state on every run and on every machine.

Together these make a run reproducible from its identifiers alone. Nothing in
this module reads the clock, the environment, or a random device.
"""

from __future__ import annotations

import os
import random
import zlib

import numpy as np

MODULUS = 2**31 - 1
_MULT = 1_000_003


def cell_seed(base_seed: int, *parts: str) -> int:
    """Stable seed for a named cell.

    >>> cell_seed(42, "pythia-70m", "pii_leading") == cell_seed(42, "pythia-70m", "pii_leading")
    True
    """
    key = "|".join(str(p) for p in parts).encode("utf-8")
    return (int(base_seed) * _MULT + zlib.crc32(key)) % MODULUS


def batch_seed(cell: int, batch_index: int) -> int:
    """Stable seed for one batch inside a cell."""
    return (int(cell) * _MULT + int(batch_index)) % MODULUS


def set_all_seeds(seed: int, deterministic_torch: bool = True) -> None:
    """Seed every generator this project touches.

    torch is imported lazily so the pure-python parts of the package, and the
    test suite, run without it installed.
    """
    seed = int(seed) % MODULUS
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_torch_for_batch(seed: int) -> int:
    """Reset torch's global RNG immediately before one generate() call.

    transformers samples with torch.multinomial on the *global* RNG and its
    generate() does not accept a `generator` argument: unknown keywords reach
    _validate_model_kwargs and raise ValueError. So reproducibility has to come
    from reseeding globally right before the call, which gives the same
    property an explicit generator would have had. Batch b of a cell starts
    from an identical sampling state however many draws preceded it.

    Returns the seed actually applied, for logging and testing.

    See tests/test_seeds.py::test_generate_is_never_passed_a_generator_kwarg,
    which fails if a `generator` keyword is reintroduced.
    """
    import torch

    seed = int(seed) % MODULUS
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
