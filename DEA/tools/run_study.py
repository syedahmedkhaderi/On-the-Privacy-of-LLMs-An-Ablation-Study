"""Execute the three notebooks in order, with checkpoints between them.

Intended for a long tmux session. Each stage writes its executed notebook
next to the source so the run is auditable afterwards.

Usage:
    python tools/run_study.py            # all three stages
    python tools/run_study.py --stage 1  # one stage
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    0: ("00_index.ipynb", "index build, no GPU, ~40 min"),
    1: ("01_generate.ipynb", "generation, ~37 GPU h, resumable"),
    2: ("02_analyse.ipynb", "analysis, no GPU, minutes"),
}


def execute(name: str) -> int:
    src = ROOT / "notebooks" / name
    out = ROOT / "results" / "executed" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute",
        "--ExecutePreprocessor.timeout=-1",
        "--output", str(out), str(src),
    ]
    print(f"\n>>> {name}", flush=True)
    started = time.time()
    code = subprocess.call(cmd)
    print(f"<<< {name} finished in {(time.time() - started) / 60:.1f} min "
          f"with exit {code}", flush=True)
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=sorted(STAGES))
    args = parser.parse_args()

    stages = [args.stage] if args.stage is not None else sorted(STAGES)
    for stage in stages:
        name, note = STAGES[stage]
        print(f"stage {stage}: {note}")
        code = execute(name)
        if code != 0:
            print(f"stage {stage} failed. Nothing after it was run.")
            return code
    print("\nall requested stages complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
