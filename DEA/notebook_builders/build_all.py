"""Regenerate every notebook from its builder."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_00_index
import build_01_generate
import build_02_analyse

BUILDERS = (build_00_index, build_01_generate, build_02_analyse)


def build_all(out_dir: Path | None = None) -> list[Path]:
    out_dir = Path(out_dir or Path(__file__).resolve().parents[1] / "notebooks")
    return [module.build(out_dir) for module in BUILDERS]


if __name__ == "__main__":
    for path in build_all():
        print("built", path)
