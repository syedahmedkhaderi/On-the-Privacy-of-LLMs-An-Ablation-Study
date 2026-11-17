#!/usr/bin/env python3
"""Cycle A safety and manifest checks. This script never launches experiments."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import json
import hashlib
import os
import re
import stat
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXISTING_WORK_HOLD = ROOT / "runs" / "gates" / "EXISTING_WORK_HOLD.json"
REQUIRED_COLUMNS = {
    "run_id", "status", "attack", "reviewer_issue_or_claim_id", "host",
    "project_path", "code_hash", "config_hash", "dataset_revision_or_hash",
    "prompt_version_or_hash", "model_id_revision", "tokenizer_revision", "seed",
    "n_member", "n_nonmember", "n_examples_or_queries", "query_budget",
    "decoder_settings", "primary_metric", "secondary_metrics", "metric_code_revision",
    "safe_aggregate_path", "artifact_sensitivity_review", "ci_method_and_level",
    "owner", "safe_for_release", "hypothesis", "access_mode", "sample_rule",
    "approval_status", "output_dir", "run_gate_evidence",
}
RUN_STATUSES = {"planned", "queued", "running", "completed", "blocked", "failed"}
APPROVAL_STATUSES = {"draft", "ready_for_review", "approved_for_run", "blocked", "not_approved"}
ACTIVE_MARKERS = (
    "logged_runner", "accelerate launch", "torchrun", "vllm", "text-generation",
    "run_generation", "model.generate", "analysis.reproduce",
)
EXTERNAL_QUEUE_FILES = {
    "machine_A": Path("work/rs-3-ext/state/B/queue.json"),
    "machine_B": Path("work/rs-3-ext/state/A/queue.json"),
}
TERMINAL_QUEUE_STATUSES = {
    "done", "completed", "complete", "cancelled", "canceled", "skipped",
}
IN_FLIGHT_QUEUE_STATUSES = {"queued", "running"}
BLOCKING_QUEUE_STATUSES = {"failed", "orphaned"}


def validate_manifest(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return validate_manifest_rows(handle)


def validate_manifest_rows(handle) -> list[str]:
    errors: list[str] = []
    reader = csv.DictReader(handle)
    if not reader.fieldnames:
        return ["manifest has no header"]
    missing = REQUIRED_COLUMNS - set(reader.fieldnames)
    if missing:
        return ["missing columns: " + ", ".join(sorted(missing))]
    seen: set[str] = set()
    for line_number, row in enumerate(reader, start=2):
        run_id = (row.get("run_id") or "").strip()
        if not run_id or run_id.upper() in {"TEMPLATE", "TEMPLATE_NOT_STARTED"} or run_id.upper().startswith("TODO"):
            continue  # The supplied template intentionally has a blank row.
        if run_id in seen:
            errors.append(f"line {line_number}: duplicate run_id")
        seen.add(run_id)
        for field in ("status", "attack", "reviewer_issue_or_claim_id", "hypothesis",
                      "host", "project_path", "code_hash", "config_hash",
                      "dataset_revision_or_hash", "model_id_revision", "tokenizer_revision",
                      "seed", "n_member", "n_nonmember", "n_examples_or_queries",
                      "query_budget", "decoder_settings", "metric_code_revision",
                      "access_mode", "sample_rule", "primary_metric", "ci_method_and_level",
                      "safe_aggregate_path", "restricted_raw_artifact_path", "artifact_sensitivity_review",
                      "table_figure_mapping", "owner",
                      "approval_status", "output_dir"):
            if not (row.get(field) or "").strip():
                errors.append(f"line {line_number}: {field} is empty")
        if (row.get("attack") or "").strip().casefold() == "mia":
            mia_evidence_fields = (
                "exposure_evidence_path", "exposure_evidence_sha256",
                "holdout_evidence_path", "holdout_evidence_sha256",
            )
            for field in mia_evidence_fields:
                if field not in reader.fieldnames:
                    errors.append(f"line {line_number}: missing MIA evidence column {field}")
                elif not (row.get(field) or "").strip():
                    errors.append(f"line {line_number}: {field} is empty for MIA")
            for field in ("exposure_evidence_sha256", "holdout_evidence_sha256"):
                value = (row.get(field) or "").strip().casefold()
                if value and (len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)):
                    errors.append(f"line {line_number}: {field} must be a SHA-256 digest")
        if (row.get("status") or "").strip().casefold() not in RUN_STATUSES:
            errors.append(f"line {line_number}: unknown status")
        if (row.get("approval_status") or "").strip().casefold() not in APPROVAL_STATUSES:
            errors.append(f"line {line_number}: unknown approval_status")
        if (row.get("approval_status") or "").strip().casefold() == "approved_for_run" and not (
            (row.get("run_gate_evidence") or "").strip()
        ):
            errors.append(f"line {line_number}: approved run is missing assigned-resource gate evidence")
        if (row.get("approval_status") or "").strip().casefold() == "approved_for_run" and (
            (row.get("status") or "").strip().casefold() != "queued"
        ):
            errors.append(f"line {line_number}: an approved run must be queued")
        out = (row.get("output_dir") or "").strip()
        out_path = Path(out) if out else None
        if out_path is not None and not out_path.is_absolute():
            out_path = ROOT / out_path
        if (out_path is not None and out_path.exists()
                and (row.get("status") or "").strip().casefold() in {"planned", "queued"}):
            errors.append(f"line {line_number}: output_dir already exists; use a new run ID")
        release = (row.get("safe_for_release") or "").strip().casefold()
        if release not in {"yes", "no", "pending", "not_applicable"}:
            errors.append(f"line {line_number}: safe_for_release must be yes/no/pending/not_applicable")
    return errors


def gpu_status() -> tuple[bool, str]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False, "nvidia-smi is unavailable"
    query = subprocess.run(
        [executable, "--query-gpu=index,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True, timeout=15,
    )
    if query.returncode:
        return False, "nvidia-smi query failed"
    busy: list[str] = []
    for line in query.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 4:
            return False, "unrecognized nvidia-smi output"
        index, utilization, used, total = map(int, fields)
        if utilization > 5 or used > max(1000, total // 20):
            busy.append(f"GPU {index}: {utilization}% utilization, {used}/{total} MiB")
    return not busy, "; ".join(busy) if busy else "no GPU load detected"


def process_status() -> tuple[bool, str]:
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        gpu_apps = subprocess.run(
            [nvidia, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=15,
        )
        if gpu_apps.returncode == 0:
            gpu_pids = sorted({line.strip() for line in gpu_apps.stdout.splitlines() if line.strip().isdigit()})
            if gpu_pids:
                return False, "GPU compute process detected: " + ", ".join(f"pid={pid}" for pid in gpu_pids)

    query = subprocess.run(
        ["ps", "-eo", "pid=,etime=,args="],
        check=False, capture_output=True, text=True, timeout=15,
    )
    if query.returncode:
        return False, "process listing failed"
    matches = []
    for line in query.stdout.splitlines():
        lower = line.lower()
        if any(marker in lower for marker in ACTIVE_MARKERS):
            # Keep names only; never print arguments because they may contain data.
            matches.append(line.strip().split(maxsplit=2)[:2])
    if matches:
        summary = ", ".join(f"pid={pid} elapsed={elapsed}" for pid, elapsed, *_ in matches)
        return False, summary
    return True, "no known experiment process detected"


def existing_work_status(path: Path = EXISTING_WORK_HOLD) -> tuple[bool, str]:
    """Require a fresh, explicit clearance for work already queued on either VM."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        status = str(record["status"]).strip().casefold()
        checked = datetime.fromisoformat(str(record["checked_utc"]).replace("Z", "+00:00"))
        evidence = str(record["evidence"]).strip()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False, "existing-work clearance is missing or unreadable"
    if checked.utcoffset() is None:
        return False, "existing-work clearance requires a timezone"
    age = (datetime.now(timezone.utc) - checked).total_seconds()
    if status != "clear":
        return False, "pre-existing VM work has not been confirmed complete"
    if age < -60 or (record.get("clear_until_revoked") is not True and age > 300):
        return False, "existing-work clearance is stale or has an invalid timestamp"
    if record.get("terminal_failures_reviewed") is not True:
        return False, "terminal failures from pre-existing work have not been reviewed"
    if not evidence:
        return False, "existing-work clearance has no verification evidence"
    return True, "pre-existing VM work explicitly cleared"


def external_queue_status(path: Path, acknowledged_queue_sha256: str | None = None) -> tuple[bool, str]:
    """Accept known in-flight or terminal rows; block failed, orphaned, or unknown rows.

    Do not infer that a queued/running row is orphaned from its age or a process
    scan. The external queue owner confirms row liveness; this preflight must
    never mutate the external queue or submit Cycle A work to it.
    """
    try:
        queue_bytes = path.read_bytes()
        items = json.loads(queue_bytes)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False, "external queue is missing or unreadable"
    if not isinstance(items, list) or not items:
        return False, "external queue has an unrecognized or empty record set"
    statuses: list[str] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("status"), str):
            return False, "external queue contains an unrecognized record"
        status = item["status"].strip().casefold()
        if not status:
            return False, "external queue contains an empty status"
        statuses.append(status)
    counts = Counter(statuses)
    summary = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    known_statuses = TERMINAL_QUEUE_STATUSES | IN_FLIGHT_QUEUE_STATUSES | BLOCKING_QUEUE_STATUSES
    unknown = set(statuses) - known_statuses
    if unknown:
        return False, "external queue contains unknown status: " + summary
    blocking = set(statuses) & BLOCKING_QUEUE_STATUSES
    if acknowledged_queue_sha256 == hashlib.sha256(queue_bytes).hexdigest():
        blocking.discard("failed")
    if blocking:
        return False, "external queue contains failed or orphaned rows: " + summary
    return True, "external queue has known in-flight or terminal rows: " + summary


def read_host_gate(path: Path, host: str) -> dict | None:
    """Read an optional owner-only declaration; invalid declarations fail closed."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("host gate must be a regular owner-owned mode 600 file")
        def unique_keys(pairs):
            record = {}
            for key, value in pairs:
                if key in record:
                    raise ValueError("duplicate host gate field")
                record[key] = value
            return record
        record = json.loads(stream.read(), object_pairs_hook=unique_keys)
    fields = {"host", "external_queue", "acknowledged_queue_sha256", "review_ref"}
    if not isinstance(record, dict) or record.keys() != fields or record["host"] != host:
        raise ValueError("host gate fields or host mismatch")
    queue = record["external_queue"]
    if queue is not None and (not isinstance(queue, str) or not Path(queue).is_absolute()):
        raise ValueError("host gate queue must be absolute or null")
    digest = record["acknowledged_queue_sha256"]
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("host gate acknowledgement must be a SHA-256 or null")
    if not isinstance(record["review_ref"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", record["review_ref"]):
        raise ValueError("host gate needs a short review reference")
    return record


def local_external_queue_status() -> tuple[bool, str]:
    host = socket.gethostname().split(".")[0]
    try:
        gate = read_host_gate(Path.home() / ".cyclea" / "host_gate.json", host)
    except (OSError, ValueError, TypeError, UnicodeDecodeError):
        return False, "host gate is malformed, unreadable, or not owner-only"
    path = EXTERNAL_QUEUE_FILES.get(host)
    if path is None:
        if gate is None:
            return False, "external queue path is not configured for this host"
        if gate["external_queue"] is None:
            return True, "owner declares no external queue on this host"
        path = Path(gate["external_queue"])
    # A declaration cannot replace or disable a built-in queue path.
    if gate is not None and gate["external_queue"] != str(path):
        return False, "host gate queue does not match configured queue"
    return external_queue_status(path, gate["acknowledged_queue_sha256"] if gate else None)


def check_idle() -> list[str]:
    errors: list[str] = []
    gpu_idle, gpu_message = gpu_status()
    if not gpu_idle:
        errors.append("GPU gate closed: " + gpu_message)
    proc_idle, proc_message = process_status()
    if not proc_idle:
        errors.append("process gate closed: " + proc_message)
    ext_queue_clear, ext_queue_message = local_external_queue_status()
    if not ext_queue_clear:
        errors.append("external-queue gate closed: " + ext_queue_message)
    queue_clear, queue_message = existing_work_status()
    if not queue_clear:
        errors.append("existing-work gate closed: " + queue_message)
    return errors


def capture_idle_status(path: Path) -> tuple[bool, str]:
    gpu_idle, gpu_message = gpu_status()
    proc_idle, proc_message = process_status()
    ext_queue_clear, ext_queue_message = local_external_queue_status()
    queue_clear, queue_message = existing_work_status()
    nvidia = shutil.which("nvidia-smi")
    gpu_count = 0
    if nvidia:
        query = subprocess.run([nvidia, "-L"], check=False, capture_output=True,
                               text=True, timeout=15)
        if query.returncode == 0:
            gpu_count = sum(line.startswith("GPU ") for line in query.stdout.splitlines())
    payload = {
        "host": socket.gethostname().split(".")[0],
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu_idle": gpu_idle,
        "gpu_count": gpu_count,
        "gpu_status": gpu_message,
        "process_idle": proc_idle,
        "process_status": proc_message,
        "external_queue_clear": ext_queue_clear,
        "external_queue_status": ext_queue_message,
        "existing_work_clear": queue_clear,
        "existing_work_status": queue_message,
        "idle": gpu_idle and proc_idle and ext_queue_clear and queue_clear and gpu_count == 2,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    detail = "all host gates passed" if payload["idle"] else "one or more host gates are blocked"
    return bool(payload["idle"]), detail


def verify_two_vm_status(paths: list[Path], max_age_seconds: int = 300) -> list[str]:
    errors: list[str] = []
    expected = {"machine_A", "machine_B"}
    observed: set[str] = set()
    now = datetime.now(timezone.utc)
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            host = str(record["host"])
            captured = datetime.fromisoformat(str(record["captured_utc"]).replace("Z", "+00:00"))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            errors.append("unreadable VM idle-status record")
            continue
        if host not in expected or host in observed:
            errors.append("VM idle-status records must identify each expected host exactly once")
        observed.add(host)
        age = (now - captured).total_seconds()
        if age < -60 or age > max_age_seconds:
            errors.append("VM idle-status record is stale or has an invalid timestamp")
        if (record.get("idle") is not True or record.get("gpu_idle") is not True
                or record.get("process_idle") is not True
                or record.get("external_queue_clear") is not True
                or record.get("existing_work_clear") is not True):
            errors.append("a VM idle-status record fails one or more host gates")
        if record.get("gpu_count") != 2:
            errors.append("a VM idle-status record does not confirm two GPUs")
    if observed != expected:
        errors.append("VM idle-status records do not cover both expected hosts")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "Revision_Run_Manifest_Template.csv")
    parser.add_argument("--check-idle", action="store_true",
                        help="Read GPU/process/queue status and fail unless prior VM work is terminal and cleared.")
    parser.add_argument("--run-id", help="Check that a specific run ID is in the manifest.")
    parser.add_argument("--require-approved", action="store_true",
                        help="Require the selected --run-id to be approved for execution.")
    parser.add_argument("--write-idle-status", type=Path,
                        help="Write a safe host/GPU/process status JSON; requires --check-idle.")
    parser.add_argument("--require-both-vm-status", nargs=2, type=Path, metavar=("VM1_JSON", "VM2_JSON"),
                        help="Require fresh idle records from both VMs.")
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}")
        return 2
    errors = validate_manifest(args.manifest)
    if errors:
        for error in errors:
            print("ERROR:", error)
        return 1
    if args.require_approved and not args.run_id:
        print("ERROR: --require-approved needs --run-id")
        return 2
    if args.write_idle_status and not args.check_idle:
        print("ERROR: --write-idle-status requires --check-idle")
        return 2
    if args.run_id:
        with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
            matching = [row for row in csv.DictReader(handle)
                        if (row.get("run_id") or "").strip() == args.run_id]
        if len(matching) != 1:
            print("ERROR: run_id must identify exactly one manifest row")
            return 1
        if args.require_approved and (matching[0].get("approval_status") or "").strip().casefold() != "approved_for_run":
            print("ERROR: selected run is not approved_for_run")
            return 1
    print("Manifest schema is valid. No experiment was launched.")
    if args.check_idle:
        if args.write_idle_status:
            idle, detail = capture_idle_status(args.write_idle_status)
            if not idle:
                print("BLOCKED: local status written; VM is not idle or GPU inventory is incomplete")
                return 3
            print("Local idle status written.")
        idle_errors = check_idle()
        if idle_errors:
            for error in idle_errors:
                print("BLOCKED:", error)
            return 3
        print("Idle gate passed.")
    if args.require_both_vm_status:
        gate_errors = verify_two_vm_status(args.require_both_vm_status)
        if gate_errors:
            for error in gate_errors:
                print("BLOCKED:", error)
            return 3
        print("Fresh idle status is present for both VMs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
