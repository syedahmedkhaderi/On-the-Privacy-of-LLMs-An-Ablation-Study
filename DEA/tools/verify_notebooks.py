"""Verify that notebooks match their builders and that every cell compiles.

Two checks:

  drift    each notebook on disk must be byte-identical to what its builder
           emits. Hand-editing a notebook is therefore a build failure rather
           than a silent divergence between five near-copies.

  syntax   every code cell must compile. A syntax error in a late cell would
           otherwise surface only after the preceding cells had already spent
           GPU hours.

Exit code 0 if everything passes, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebook_builders"))
sys.path.insert(0, str(ROOT / "src"))

from build_all import BUILDERS  # noqa: E402
from common import render  # noqa: E402


def check_drift() -> list[str]:
    problems = []
    for module in BUILDERS:
        name = module.build(ROOT / "notebooks").name
        path = ROOT / "notebooks" / name
        expected = render(module.cells())
        actual = path.read_text(encoding="utf-8")
        if expected.strip() != actual.strip():
            problems.append(f"{name}: on-disk notebook differs from its builder")
        else:
            print(f"[PASS] {name} matches its builder")
    return problems


def check_syntax() -> list[str]:
    problems = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(str(path), as_version=4)
        n_code = 0
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            n_code += 1
            try:
                compile(cell.source, f"{path.name}[cell {i}]", "exec")
            except SyntaxError as exc:
                problems.append(f"{path.name} cell {i}: {exc}")
        print(f"[{'PASS' if not problems else 'FAIL'}] {path.name}: "
              f"{n_code} code cells compile")
    return problems


def check_structure() -> list[str]:
    problems = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(str(path), as_version=4)
        try:
            nbformat.validate(nb)
        except nbformat.ValidationError as exc:
            problems.append(f"{path.name}: invalid notebook: {exc}")
            continue
        if not any(c.cell_type == "markdown" for c in nb.cells):
            problems.append(f"{path.name}: no markdown cells")
        for cell in nb.cells:
            if cell.cell_type == "code" and cell.get("outputs"):
                problems.append(f"{path.name}: committed with cell outputs")
                break
        print(f"[PASS] {path.name} is a valid, output-free notebook")
    return problems


def main() -> int:
    problems: list[str] = []
    problems += check_drift()
    problems += check_syntax()
    problems += check_structure()
    if problems:
        print("\nFAILURES")
        for p in problems:
            print("  -", p)
        return 1
    print("\nall notebook checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
