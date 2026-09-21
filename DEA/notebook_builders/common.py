"""Shared helpers for building notebooks.

Notebooks are generated, never hand-edited. rs-3 kept five independent copies
of the same pipeline code across its notebooks and they had already diverged;
generating from one source is what stops that.

tools/verify_notebooks.py checks that every notebook on disk is byte-identical
to its builder's output. Editing a notebook by hand is therefore a build
failure, not a silent divergence.
"""

from __future__ import annotations

from pathlib import Path

import nbformat


def md(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(text.strip("\n"))


HEADER = """
import os
import sys
from pathlib import Path

# PYTHONHASHSEED must be exported before the interpreter starts to have any
# effect. Setting it here would do nothing, so instead we assert that nothing
# in this project depends on it: every seed is derived with zlib.crc32.
ROOT = Path.cwd()
while not (ROOT / "src" / "dea_a3").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from dea_a3.config import CONFIG
from dea_a3.paths import Paths

PATHS = Paths(ROOT)
PATHS.ensure()
print("root      :", ROOT)
print("study     :", CONFIG.study)
print("index     :", CONFIG.index_variant)
"""

PREFLIGHT = """
# Preflight. Every notebook checks its inputs before doing any work; round 10
# of the earlier study produced two wrong conclusions by running against a
# half-populated tree.
def preflight(required_paths, required_packages=("numpy", "pandas", "pyarrow", "scipy")):
    ok = True
    for label, path in required_paths:
        exists = Path(path).exists()
        print(f"[{'PASS' if exists else 'FAIL'}] {label}: {path}")
        ok &= exists
    for pkg in required_packages:
        try:
            module = __import__(pkg)
            print(f"[PASS] {pkg} {getattr(module, '__version__', '?')}")
        except ImportError:
            print(f"[FAIL] {pkg} not importable")
            ok = False
    if not ok:
        raise SystemExit("preflight failed; fix the above before continuing")
    print("preflight OK")
"""


def _build(cells: list) -> nbformat.NotebookNode:
    """Assemble a notebook with deterministic cell identifiers.

    nbformat assigns every new cell a random UUID, so rebuilding the same
    notebook twice produces different bytes and no drift check is possible.
    Numbering the cells by position makes the build reproducible.
    """
    nb = nbformat.v4.new_notebook()
    for position, cell in enumerate(cells):
        cell["id"] = f"c{position:03d}"
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    return nb


def write_notebook(cells: list, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(nbformat.writes(_build(cells), version=4) + "\n",
                    encoding="utf-8")
    return path


def render(cells: list) -> str:
    return nbformat.writes(_build(cells), version=4)
