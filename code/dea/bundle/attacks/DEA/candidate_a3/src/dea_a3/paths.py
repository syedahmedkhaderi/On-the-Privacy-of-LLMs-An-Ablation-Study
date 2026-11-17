"""Filesystem layout.

Every path derives from one root so the study can be relocated by moving one
directory. Nothing here reads an environment variable except HF_HOME, and the
resolved value is recorded in every manifest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def maildir(self) -> Path:
        return self.data / "enron-mail"

    @property
    def pile_cache(self) -> Path:
        return self.data / "pile-enron"

    @property
    def zeng_prompts(self) -> Path:
        return self.data / "zeng_prompts"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def index_dir(self) -> Path:
        return self.results / "00_index"

    @property
    def index_file(self) -> Path:
        return self.index_dir / "index.parquet"

    @property
    def true_prefix_file(self) -> Path:
        return self.index_dir / "true_prefixes.parquet"

    @property
    def generations(self) -> Path:
        return self.results / "01_generations"

    @property
    def analysis(self) -> Path:
        return self.results / "02_analysis"

    @property
    def salt_file(self) -> Path:
        return self.root / "secrets" / "pii_salt.txt"

    @property
    def hf_cache(self) -> str | None:
        return os.environ.get("HF_HOME") or None

    def cell_dir(self, model: str, condition: str, seed: int) -> Path:
        return self.generations / f"{model}__{condition}__seed{seed}"

    def ensure(self) -> None:
        for path in (self.index_dir, self.generations, self.analysis,
                     self.pile_cache, self.salt_file.parent):
            path.mkdir(parents=True, exist_ok=True)


def find_root(start: Path | None = None) -> Path:
    """Walk up for the directory containing src/dea_a3."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "src" / "dea_a3").is_dir():
            return candidate
    return here


PATHS = Paths(find_root())
