"""The Pile's Enron Emails component.

This is the corpus the models actually read, so it is what a claim about
memorization has to be verified against. The original Pile host is dead
(expired certificate); monology/pile-uncopyrighted is an ungated
redistribution and its Enron records are intact, since the uncopyrighted
variant removes Books3, BookCorpus2, OpenSubtitles, YouTubeSubtitles and
OpenWebText2, none of which touch Enron.

Records are filtered inline while streaming so no full shard is ever written
to disk. Only the Enron-tagged text is cached.
"""

from __future__ import annotations

import io
import json
import shlex
import shutil
import subprocess
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import pii

SHARD_URL = (
    "https://huggingface.co/datasets/{repo}/resolve/main/train/{shard:02d}.jsonl.zst"
)


def shard_url(repo: str, shard: int) -> str:
    return SHARD_URL.format(repo=repo, shard=shard)


def have_zstandard() -> bool:
    try:
        import zstandard  # noqa: F401
        return True
    except ImportError:
        return False


def have_zstd_cli() -> bool:
    return shutil.which("zstd") is not None and shutil.which("curl") is not None


def _iter_lines_python(url: str, timeout: int):
    """Decompress in-process with the zstandard package."""
    import zstandard

    # The Pile uses a long window, so the default limit rejects these frames.
    dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        with dctx.stream_reader(resp) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8", errors="replace"):
                yield line


def _iter_lines_subprocess(url: str):
    """Fall back to `curl | zstd -dc`, which needs no Python package.

    P8 used exactly this route on the target machine, so it is known to work
    there. Nothing is written to disk; the shard is consumed as a stream.
    """
    proc = subprocess.Popen(
        f"curl -sSL --retry 3 --retry-delay 2 {shlex.quote(url)} | zstd -dc --long=31",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            yield line
    finally:
        proc.stdout.close() if proc.stdout else None
        code = proc.wait()
        if code != 0:
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"shard stream failed ({code}): {err[:400]}")


def iter_shard_records(url: str, timeout: int = 600):
    if have_zstandard():
        lines = _iter_lines_python(url, timeout)
    elif have_zstd_cli():
        lines = _iter_lines_subprocess(url)
    else:
        raise RuntimeError(
            "no way to decompress zstd: install the zstandard package "
            "(pip install zstandard) or the zstd and curl command-line tools"
        )
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def extract_shard(repo: str, shard: int, set_name: str, out_dir: Path,
                  timeout: int = 600) -> Path:
    """Stream one shard, keep only the records tagged with set_name.

    Writes to a temporary file and renames on success, so an interrupted
    download never leaves a truncated shard that a resume would trust. P8 saw
    two of thirty shards truncate mid-stream, so this matters.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"enron_{shard:02d}.jsonl"
    done_path = out_dir / f"enron_{shard:02d}.done"
    if done_path.exists() and out_path.exists():
        return out_path

    tmp_path = out_dir / f"enron_{shard:02d}.partial"
    url = shard_url(repo, shard)
    kept = 0
    with tmp_path.open("w", encoding="utf-8") as fh:
        for record in iter_shard_records(url, timeout):
            meta = record.get("meta") or {}
            if meta.get("pile_set_name") != set_name:
                continue
            fh.write(json.dumps({"text": record.get("text", "")}) + "\n")
            kept += 1
    tmp_path.replace(out_path)
    done_path.write_text(str(kept), encoding="utf-8")
    return out_path


def extract_all(repo: str, n_shards: int, set_name: str, out_dir: Path,
                workers: int = 6) -> list[Path]:
    """Extract every shard, resuming any that already completed."""
    out_dir = Path(out_dir)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(extract_shard, repo, shard, set_name, out_dir)
            for shard in range(n_shards)
        ]
        return [f.result() for f in futures]


def iter_documents(out_dir: Path):
    """Yield every cached Enron document text."""
    out_dir = Path(out_dir)
    for path in sorted(out_dir.glob("enron_*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue


def build_value_counts(out_dir: Path,
                       kinds: tuple[str, ...] = ("email", "phone")
                       ) -> tuple[dict[str, Counter], int]:
    """Distinct values and their occurrence counts in the training corpus.

    The occurrence count is the number of Pile documents containing the value,
    which is the quantity the duplication analysis needs. Counting per
    document rather than per mention avoids one document that repeats an
    address forty times looking like forty exposures.
    """
    counts: dict[str, Counter] = {k: Counter() for k in kinds}
    n_docs = 0
    for text in iter_documents(out_dir):
        n_docs += 1
        found = pii.extract_values(text, kinds=kinds)
        for kind in kinds:
            for value in found[kind]:
                counts[kind][value] += 1
    return counts, n_docs
