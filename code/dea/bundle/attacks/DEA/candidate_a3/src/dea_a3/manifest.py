"""Run manifests.

Every artifact this study writes gets a manifest recording the configuration,
the seed, the environment and the git state. The rs-1 corpus artifact was
built from a dirty tree and only its manifest recorded that fact, which is
the argument for manifests existing at all.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "torch", "transformers", "tokenizers", "numpy", "scipy", "pandas",
    "pyarrow", "accelerate", "safetensors", "huggingface_hub",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_state(repo_root: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args, cwd=str(repo_root), capture_output=True, text=True, timeout=15
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--porcelain"])
    if commit and status:
        commit = f"{commit}-dirty"
    return {"commit": commit, "dirty": bool(status)}


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", None)
        except ImportError:
            out[name] = None
    return out


def gpu_info() -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info.append({
            "index": i,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
        })
    return info


def write_manifest(run_dir: Path, stage: str, config: dict[str, Any],
                   repo_root: Path, extra: dict[str, Any] | None = None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "written_at": _now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "gpus": gpu_info(),
        "git": git_state(repo_root),
        "config": config,
        "extra": extra or {},
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def read_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
