"""Check the machine can run this study before any GPU time is spent.

Reports PASS or FAIL per check and exits non-zero if anything blocking fails.
Nothing is installed or changed; this only reports.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASS, FAIL, WARN = "[PASS]", "[FAIL]", "[WARN]"

# 96 cells of 5,000 generations, plus the Pile cache, with headroom.
MIN_FREE_GB = 40


def check(results, blocking, ok, label, detail=""):
    marker = PASS if ok else (FAIL if blocking else WARN)
    print(f"{marker} {label}{': ' + detail if detail else ''}")
    if not ok and blocking:
        results.append(label)


def main() -> int:
    failures: list[str] = []

    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    check(failures, True, sys.version_info[:2] == (3, 10),
          "Python is 3.10.x", version)

    versions = {}
    for name, blocking in (("numpy", True), ("pandas", True), ("pyarrow", True),
                           ("scipy", True), ("matplotlib", True),
                           ("torch", True), ("transformers", True),
                           ("accelerate", True), ("nbformat", True),
                           ("nbconvert", True), ("ipykernel", True)):
        try:
            module = __import__(name)
            got = getattr(module, "__version__", "?")
            versions[name] = got
            check(failures, blocking, True, f"{name} importable", got)
        except ImportError:
            versions[name] = None
            check(failures, blocking, False, f"{name} importable", "not installed")

    if versions.get("numpy"):
        major = int(versions["numpy"].split(".")[0])
        check(failures, True, major < 2, "numpy below 2.0",
              f"{versions['numpy']} (2.x breaks the pinned torch ABI)")

    if versions.get("torch"):
        import torch
        check(failures, True, torch.cuda.is_available(), "CUDA available",
              str(torch.version.cuda))
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        check(failures, True, n >= 2, "at least two GPUs", f"found {n}")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            gb = props.total_memory / 1024 ** 3
            check(failures, False, gb >= 20, f"cuda:{i} has >= 20 GB",
                  f"{props.name}, {gb:.1f} GB")
        if torch.cuda.is_available():
            try:
                x = torch.randn(64, 64, device="cuda:0", dtype=torch.float16)
                check(failures, True, bool((x @ x).isfinite().all()),
                      "fp16 matmul on cuda:0")
            except Exception as exc:
                check(failures, True, False, "fp16 matmul on cuda:0", str(exc)[:80])

    from dea_a3 import pile
    decomp = pile.have_zstandard() or pile.have_zstd_cli()
    how = ("zstandard package" if pile.have_zstandard()
           else "zstd and curl CLI" if pile.have_zstd_cli() else "none")
    check(failures, True, decomp, "zstd decompression available", how)

    usage = shutil.disk_usage(ROOT)
    free_gb = usage.free / 1024 ** 3
    check(failures, True, free_gb >= MIN_FREE_GB,
          f"free disk at least {MIN_FREE_GB} GB", f"{free_gb:.1f} GB free")

    # run_study.py shells out to `jupyter nbconvert`. If that entry point is
    # missing, stage 0 dies in seconds and the failure looks like a notebook
    # problem rather than a missing package. Catch it here instead.
    import subprocess
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--version"],
            capture_output=True, text=True, timeout=120)
        nbconvert_ok = completed.returncode == 0
        detail = (completed.stdout or completed.stderr).strip().splitlines()
        detail = detail[-1] if detail else ""
    except Exception as exc:
        nbconvert_ok, detail = False, str(exc)
    check(failures, True, nbconvert_ok,
          "jupyter nbconvert runnable", detail or "pip install nbconvert ipykernel")

    try:
        from jupyter_client.kernelspec import KernelSpecManager
        specs = KernelSpecManager().find_kernel_specs()
        kernel_ok = "python3" in specs
        kernel_detail = ", ".join(sorted(specs)) or "none found"
    except Exception as exc:
        kernel_ok, kernel_detail = False, str(exc)
    check(failures, True, kernel_ok,
          "python3 kernelspec registered", kernel_detail
          + ("" if kernel_ok else "; run: python -m ipykernel install --user"))

    maildir = ROOT / "data" / "enron-mail"
    check(failures, True, maildir.is_dir(), "maildir present", str(maildir))

    from dea_a3.config import CONFIG
    check(failures, True, len(CONFIG.cells()) == 84, "grid is 84 cells",
          f"{len(CONFIG.models)} models x {len(CONFIG.conditions)} conditions "
          f"x {len(CONFIG.seeds)} seeds")
    from dea_a3.paths import Paths as _Paths
    from dea_a3 import prompts as _prompts
    _pool, _source = _prompts.pii_leading_pool(_Paths(ROOT).zeng_prompts,
                                               CONFIG.pii_leading_variants)
    check(failures, False, _source == "zeng_repo",
          "Zeng's own attack prompts present",
          f"pool is {len(_pool)} prompts from '{_source}'; run "
          "python tools/fetch_zeng_prompts.py for the comparable set")

    check(failures, True, CONFIG.decoder.repetition_penalty == 1.0,
          "repetition penalty disabled",
          "a penalty above 1.0 suppresses verbatim recall")
    check(failures, True, CONFIG.decoder.top_p == 1.0, "nucleus sampling disabled",
          "top_p with top_k narrows the candidate set twice")

    print()
    if failures:
        print(f"{len(failures)} blocking check(s) failed:")
        for item in failures:
            print("  -", item)
        return 1
    print("environment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
