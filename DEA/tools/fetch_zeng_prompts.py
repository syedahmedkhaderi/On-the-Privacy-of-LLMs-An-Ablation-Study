"""Fetch the attack prompt pools published by Zeng et al. 2024.

Their targeted-attack {information} components are the prompts that produced
Table 3. Using them directly removes the largest remaining difference between
this study and the one it reproduces: our own pool was 100 short hand-written
stems, theirs is 813 longer sentences.

The files are downloaded rather than vendored, so provenance is a URL and a
checksum rather than a copy sitting in this tree with no licence attached.

    python tools/fetch_zeng_prompts.py

Writes data/zeng_prompts/email.json and data/zeng_prompts/phone.json. If the
network is unavailable the study still runs; prompts.py falls back to the
enumerated pool and records which source was used in the manifest.

Source: https://github.com/phycholosogy/RAG-privacy
Paper:  arXiv:2402.16893
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

BASE = ("https://raw.githubusercontent.com/phycholosogy/RAG-privacy/"
        "main/Information/")

FILES = {
    "email": (
        "Target_Email%20Address.json",
        "68459cd2f938f127cb80e1be707bf9af2c1492edf068f0b6ddfed9aa0bfe08f9",
        489,
    ),
    "phone": (
        "Target_Phone%20Numer.json",
        "dad1ea1e28c22499d4722b5f106ea8e0c2b0ce1d76d23b1c6eb96ecf3c067fdb",
        324,
    ),
}


def fetch(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    for kind, (remote, expected_sha, expected_n) in FILES.items():
        url = BASE + remote
        target = out_dir / f"{kind}.json"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
        except Exception as exc:
            print(f"[FAIL] {kind}: {exc}")
            failures += 1
            continue

        actual_sha = hashlib.sha256(payload).hexdigest()
        if actual_sha != expected_sha:
            print(f"[FAIL] {kind}: checksum mismatch")
            print(f"       expected {expected_sha}")
            print(f"       got      {actual_sha}")
            print("       the upstream file changed; do not use it silently")
            failures += 1
            continue

        prompts = json.loads(payload.decode("utf-8"))
        if not isinstance(prompts, list) or len(prompts) != expected_n:
            print(f"[FAIL] {kind}: expected {expected_n} prompts, "
                  f"got {len(prompts) if isinstance(prompts, list) else '?'}")
            failures += 1
            continue

        target.write_bytes(payload)
        print(f"[OK]   {kind}: {len(prompts)} prompts -> {target}")

    if failures:
        print("\nSome files were not fetched. The study will fall back to the")
        print("enumerated pool, which is recorded in the manifest.")
        return 1

    print("\nBoth pools fetched. prompts.py will use them automatically.")
    return 0


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    return fetch(root / "data" / "zeng_prompts")


if __name__ == "__main__":
    sys.exit(main())
