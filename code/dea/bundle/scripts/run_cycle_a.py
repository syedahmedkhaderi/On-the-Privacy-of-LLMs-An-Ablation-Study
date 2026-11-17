"""Validate and execute one frozen Cycle A job. Use --check before approval."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import sys
from datetime import datetime, timezone

from attacks.MIA.score_tokenized import reject_duplicate_json_keys
from scripts.cache_smoke import run_smoke, validate_smoke_result, validate_smoke_settings
from scripts.preflight_cycle_a import EXTERNAL_QUEUE_FILES
from scripts.run_safety import (GPU_UUID, private_dir, private_file, publish_json,
                                read_bound_bytes, read_cuda_runtime, reserve_device,
                                source_digest, validate_cuda_runtime)

ROOT = Path(__file__).resolve().parents[1]
DEA_SOURCE = ROOT / "attacks" / "DEA" / "candidate_a3" / "src"
GPU_KINDS = {"cache_smoke", "dea_cell", "aia_inference", "mia_pilot", "mia_primary"}
CPU_KINDS = {"dea_control", "dea_replay", "aia_rescore"}
KINDS = GPU_KINDS | CPU_KINDS
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}")
DIGEST = re.compile(r"[0-9a-f]{64}")
FIELDS = {"schema", "run_id", "kind", "host", "code_sha256", "packages", "resources",
          "depends_on", "settings"}
BASE_PACKAGES = {
    "cache_smoke": set(),
    "dea_control": {"numpy", "pandas", "pyarrow", "scipy"},
    "dea_cell": {"numpy", "pandas", "pyarrow", "scipy"},
    "dea_replay": {"numpy", "pandas", "pyarrow", "scipy", "transformers", "tokenizers", "huggingface-hub"},
    "aia_inference": set(), "aia_rescore": set(),
    "mia_pilot": set(), "mia_primary": set(),
}
MIA_FINAL_MODEL = "EleutherAI/pythia-1.4b@9cc5c8c8148a4e0115d9e29c6b4f21124cfe748a"
GPU_PACKAGES = {"torch", "transformers", "tokenizers", "safetensors", "huggingface-hub", "accelerate"}


def current_code_hash() -> str:
    files = [path for path in (ROOT / "scripts").glob("*.py") if not path.name.startswith("test_")]
    for directory in (ROOT / "attacks" / "AIA", ROOT / "attacks" / "MIA",
                      ROOT / "attacks" / "DEA", DEA_SOURCE / "dea_a3"):
        files.extend(path for path in directory.glob("*.py") if not path.name.startswith("test_"))
    files.append(ROOT / "attacks" / "__init__.py")
    return source_digest(ROOT, [str(path.relative_to(ROOT)) for path in files])


def read_job(path: Path):
    private_file(path)
    raw = path.read_bytes()
    job = json.loads(raw, object_pairs_hook=reject_duplicate_json_keys)
    if not isinstance(job, dict) or job.keys() != FIELDS or job["schema"] != "cycleA.job.v1":
        raise ValueError("unsupported run manifest")
    if not RUN_ID.fullmatch(job["run_id"]) or job["kind"] not in KINDS:
        raise ValueError("unsupported run identity or job kind")
    if not DIGEST.fullmatch(job["code_sha256"]) or job["code_sha256"] != current_code_hash():
        raise ValueError("job does not match the current code bundle")
    if not isinstance(job["settings"], dict) or not job["settings"]:
        raise ValueError("job settings must be a nonempty JSON object")
    packages = job["packages"]
    required = BASE_PACKAGES[job["kind"]] | {"python"}
    if job["kind"] in GPU_KINDS:
        required |= GPU_PACKAGES
    if (not isinstance(packages, dict) or not required <= packages.keys()
            or any(not isinstance(pin, str) or not re.fullmatch(r"[\w.+!-]+", pin)
                   for pin in packages.values())):
        raise ValueError("job must freeze exact Python and package versions")
    resources = job["resources"]
    resource_keys = {"min_disk_mib", "min_ram_mib"}
    if job["kind"] in GPU_KINDS:
        resource_keys |= {"gpu_uuid", "min_free_mib"}
        sharing_keys = {"allow_sharing", "peak_memory_mib"}
        if isinstance(resources, dict) and sharing_keys & resources.keys():
            resource_keys |= sharing_keys
    if (not isinstance(resources, dict) or resources.keys() != resource_keys
            or any(type(resources[key]) is not int or resources[key] <= 0
                   for key in resource_keys - {"gpu_uuid", "allow_sharing"})):
        raise ValueError("invalid resource settings")
    if "allow_sharing" in resources and (type(resources["allow_sharing"]) is not bool
            or resources["peak_memory_mib"] <= 1024
            or resources["min_free_mib"] < resources["peak_memory_mib"] + 5120):
        raise ValueError("shared resource budget must reserve at least 5120 MiB")
    if job["kind"] in GPU_KINDS and not GPU_UUID.fullmatch(resources["gpu_uuid"]):
        raise ValueError("GPU jobs require a full GPU UUID")
    if job["kind"] in GPU_KINDS and job["host"] not in EXTERNAL_QUEUE_FILES:
        raise ValueError("GPU jobs require a configured research host")
    if not isinstance(job["depends_on"], list):
        raise ValueError("dependencies must be a list")
    seen = set()
    for dep in job["depends_on"]:
        if (not isinstance(dep, dict) or dep.keys() != {"run_id", "kind", "summary_sha256"}
                or not RUN_ID.fullmatch(dep["run_id"]) or dep["run_id"] == job["run_id"]
                or dep["run_id"] in seen or dep["kind"] not in KINDS
                or not DIGEST.fullmatch(dep["summary_sha256"])):
            raise ValueError("invalid or duplicate dependency")
        seen.add(dep["run_id"])
    if job["kind"] == "cache_smoke":
        validate_smoke_settings(job["settings"])
    if job["kind"] in GPU_KINDS - {"cache_smoke"} and sum(
        dep["kind"] == "cache_smoke" for dep in job["depends_on"]
    ) != 1:
        raise ValueError("GPU research job requires one matching cache-only smoke result")
    if job["kind"] == "mia_primary" and sum(dep["kind"] == "mia_pilot" for dep in job["depends_on"]) != 1:
        raise ValueError("primary MIA requires a verified pilot")
    return job, hashlib.sha256(raw).hexdigest()


def smoke_identity(job):
    spec = job["settings"]
    kind = job["kind"]
    if kind == "dea_cell":
        model = spec["model"]
        return {"model_id": model["hf_id"], "revision": model["revision"],
                "tokenizer_revision": model.get("tokenizer_revision") or model["revision"],
                "dtype": "fp16", "chat_template_sha256": None, "template_options": {}}
    if kind == "aia_inference":
        return {"model_id": spec["model_id"], "revision": spec["revision"],
                "tokenizer_revision": spec["revision"], "dtype": spec["dtype"],
                "chat_template_sha256": spec["chat_template_sha256"],
                "template_options": spec["template_options"]}
    model_id, revision = MIA_FINAL_MODEL.split("@")
    config = json.loads(read_bound_bytes(spec["config"]), object_pairs_hook=reject_duplicate_json_keys)
    return {"model_id": model_id, "revision": revision, "tokenizer_revision": revision,
            "dtype": config["dtype"], "chat_template_sha256": None, "template_options": {}}


def verify_dependencies(job, root):
    smoke_runtime = None
    for dep in job["depends_on"]:
        path = root / dep["run_id"] / "summary.json"
        value = json.loads(read_bound_bytes({"path": str(path), "sha256": dep["summary_sha256"]}),
                           object_pairs_hook=reject_duplicate_json_keys)
        if value["run_id"] != dep["run_id"] or value["kind"] != dep["kind"] or value["status"] != "complete":
            raise ValueError("dependency has not completed successfully")
        if value["code_sha256"] != job["code_sha256"]:
            raise ValueError("dependency was produced with another code bundle")
        if dep["kind"] == "cache_smoke" and job["kind"] in GPU_KINDS - {"cache_smoke"}:
            actual = value["result"]
            validate_smoke_result(actual)
            if (value["host"] != job["host"]
                    or value["resources"]["gpu_uuid"] != job["resources"]["gpu_uuid"]
                    or value["packages"] != job["packages"]
                    or any(actual.get(key) != expected for key, expected in smoke_identity(job).items())):
                raise ValueError("smoke test does not qualify the assigned device, runtime and checkpoint")
            smoke_runtime = value.get("cuda_runtime")
            validate_cuda_runtime(smoke_runtime, job["resources"]["gpu_uuid"])
    return smoke_runtime


def verify_environment(job):
    if job["host"] != socket.gethostname().split(".")[0]:
        raise ValueError("job is assigned to a different host")
    for name, expected in job["packages"].items():
        actual = platform.python_version() if name == "python" else version(name)
        if actual != expected:
            raise ValueError("runtime does not match the pinned package environment")
    if sys.platform == "linux":
        with open("/proc/meminfo", encoding="ascii") as stream:
            entries = dict(line.split(":", 1) for line in stream)
        if int(entries["MemAvailable"].split()[0]) < job["resources"]["min_ram_mib"] * 1024:
            raise RuntimeError("host is below required available memory")
    elif job["kind"] in GPU_KINDS:
        raise RuntimeError("GPU launcher requires the verified Linux environment")


def dispatch(job, out):
    kind, spec = job["kind"], job["settings"]
    if kind == "cache_smoke":
        return run_smoke(spec)
    if kind.startswith("dea_"):
        sys.path.insert(0, str(DEA_SOURCE))
        from dea_a3.experiment import run_cell, run_control
        from dea_a3.replay import run_replay

        return {"dea_cell": run_cell, "dea_control": run_control, "dea_replay": run_replay}[kind](spec, out)
    if kind.startswith("aia_"):
        from attacks.AIA.experiment import run_inference, run_rescore

        return (run_inference if kind == "aia_inference" else run_rescore)(spec, out)
    scoring, _, audit = prepare_mia(job, out.parent)
    result = scoring()
    result["preparation_sha256"] = spec["preparation"]["sha256"]
    result["data_audit_identity"] = audit
    result["publication_eligible_stage"] = kind == "mia_primary"
    return result


def verify_mia_disjoint(records, companion):
    for field in ("document_id_hash", "example_id_hash", "token_ids"):
        def value(row):
            return tuple(row[field]) if field == "token_ids" else row[field]

        first = {value(row) for row in records}
        second = {value(row) for row in companion}
        if first & second or len(first) != len(records) or len(second) != len(companion):
            raise ValueError("MIA pilot and primary must contain unique, disjoint documents and windows")


def prepare_mia(job, output_root=None):
    from attacks.MIA import score_tokenized

    kind, spec = job["kind"], job["settings"]
    if spec.keys() != {"manifest", "input", "companion_input", "config", "preparation"}:
        raise ValueError("MIA scheduler expects its CSV, both inputs, config and preparation")
    row = score_tokenized.selected_run(read_bound_bytes(spec["manifest"]), job["run_id"])
    if row["host"] != job["host"]:
        raise ValueError("MIA CSV and job manifest designate different hosts")
    prepared = json.loads(read_bound_bytes(spec["preparation"]), object_pairs_hook=reject_duplicate_json_keys)
    split, other = ("pilot", "primary") if kind == "mia_pilot" else ("primary", "pilot")
    if (prepared.get("schema") != "cycleA.mia.projection_preparation.v1"
            or prepared["split_hashes"][split] != spec["input"]["sha256"]
            or prepared["split_hashes"][other] != spec["companion_input"]["sha256"]
            or prepared["seed"] != int(row["seed"]) or prepared["token_length"] != 256
            or prepared["class_counts"] != {"pilot": 100, "primary": 1000}):
        raise ValueError("MIA inputs must match the frozen disjoint preparation")
    size = prepared["class_counts"][split]
    if int(row["n_member"]) != size or int(row["n_nonmember"]) != size:
        raise ValueError("MIA class counts differ from the approved pilot or primary design")
    if row["model_id_revision"] != MIA_FINAL_MODEL:
        raise ValueError("MIA job must use the approved final Pythia-1.4B checkpoint")
    scoring, records, config, audit = score_tokenized.validate_run(
        row, read_bound_bytes(spec["config"]), spec["input"], prepared["source_hashes"],
    )
    if audit["checkpoint_token_cutoff"] != prepared["checkpoint_token_cutoff"]:
        raise ValueError("owner checkpoint cutoff differs from the prepared projection")
    if any(config.get(key) != job["resources"][key] for key in ("gpu_uuid", "min_free_mib", "min_disk_mib")):
        raise ValueError("MIA resources disagree with the scheduler manifest")
    if config["token_length"] != 256:
        raise ValueError("MIA job must use the frozen 256-token document window")
    companion, counts = score_tokenized.parse_records(
        StringIO(read_bound_bytes(spec["companion_input"]).decode("utf-8")),
    )
    size = prepared["class_counts"][other]
    if counts != {"member": size, "nonmember": size} or any(len(record["token_ids"]) != 256 for record in companion):
        raise ValueError("MIA companion split has the wrong class counts or token length")
    verify_mia_disjoint(records, companion)
    if kind == "mia_primary" and output_root is not None:
        dep = next(dep for dep in job["depends_on"] if dep["kind"] == "mia_pilot")
        path = output_root / dep["run_id"] / "summary.json"
        pilot = json.loads(read_bound_bytes({"path": str(path), "sha256": dep["summary_sha256"]}),
                           object_pairs_hook=reject_duplicate_json_keys)["result"]
        if (pilot["preparation_sha256"] != spec["preparation"]["sha256"]
                or f'{pilot["model_id"]}@{pilot["model_revision"]}' != row["model_id_revision"]
                or pilot["input_sha256"] != prepared["split_hashes"]["pilot"]
                or pilot["data_audit_identity"] != audit or pilot["precision"] != config["dtype"]):
            raise ValueError("pilot does not qualify the planned primary data, provenance and model")
        score_tokenized.verify_pilot_scores(pilot, dep["run_id"], companion, prepared["seed"])
    return scoring, config, audit


def validate_job_inputs(job, output_root=None):
    kind, spec = job["kind"], job["settings"]
    if output_root is not None:
        if not output_root.is_absolute():
            raise ValueError("dependency root must be absolute")
        verify_dependencies(job, output_root)
    if kind.startswith("mia_"):
        prepare_mia(job, output_root)
    elif kind.startswith("dea_"):
        sys.path.insert(0, str(DEA_SOURCE))
        from dea_a3.experiment import validate_cell, validate_control
        from dea_a3.replay import validate_replay

        {"dea_cell": validate_cell, "dea_control": validate_control, "dea_replay": validate_replay}[kind](spec)
    elif kind == "aia_inference":
        from attacks.AIA.experiment import validate_inference

        validate_inference(spec)
    elif kind == "aia_rescore":
        from attacks.AIA.experiment import validate_rescore

        validate_rescore(spec)
    else:
        validate_smoke_settings(spec)


def execute(job, job_hash, approval_hash, root):
    if not DIGEST.fullmatch(approval_hash) or approval_hash != job_hash:
        raise ValueError("provide the independently reviewed job SHA-256")
    verify_environment(job)
    private_dir(root)
    smoke_runtime = verify_dependencies(job, root)
    resources = job["resources"]
    if shutil.disk_usage(root).free < resources["min_disk_mib"] * 2**20:
        raise RuntimeError("run filesystem is below required free space")

    is_gpu_job = job["kind"] in GPU_KINDS
    if is_gpu_job:
        output_parent = ROOT / "runs" if job["kind"].startswith("mia_") else root
        reservation = reserve_device(resources["gpu_uuid"], resources["min_free_mib"],
                                     resources["min_disk_mib"], output_parent,
                                     allow_sharing=resources["allow_sharing"],
                                     peak_memory_mib=resources["peak_memory_mib"]) if "allow_sharing" in resources else reserve_device(
                                         resources["gpu_uuid"], resources["min_free_mib"], resources["min_disk_mib"], output_parent)
    else:
        reservation = nullcontext()
    out = root / job["run_id"]
    out.mkdir(mode=0o700, exist_ok=False)
    identity = {"run_id": job["run_id"], "kind": job["kind"], "code_sha256": job["code_sha256"],
                "job_sha256": job_hash, "host": job["host"], "resources": resources,
                "packages": job["packages"]}
    publish_json(out / "job.json", job)
    publish_json(out / "started.json", dict(identity, started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        with reservation as driver_version:
            if is_gpu_job:
                runtime = read_cuda_runtime(resources["gpu_uuid"], driver_version)
                identity["cuda_runtime"] = runtime
                if "peak_memory_mib" in resources:
                    import torch
                    budget = (resources["peak_memory_mib"] - 1024) * 2**20 / runtime["total_memory_bytes"]
                    if not 0 < budget <= 1:
                        raise ValueError("process GPU memory cap is outside device capacity")
                    torch.cuda.set_per_process_memory_fraction(budget, 0)
                if smoke_runtime is not None and runtime != smoke_runtime:
                    raise RuntimeError("active CUDA runtime differs from the qualifying smoke")
            result = dispatch(job, out)
        publish_json(out / "summary.json", dict(identity, status="complete", result=result,
                                               finished_utc=datetime.now(timezone.utc).isoformat()))
    except BaseException as error:
        publish_json(out / "failure.json", dict(identity, status="failed", error_type=type(error).__name__))
        raise
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path, nargs="?")
    parser.add_argument("--code-hash", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--approve-sha256")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.code_hash:
        print(current_code_hash())
        return 0
    if args.job is None:
        parser.error("supply a job manifest")
    try:
        job, digest = read_job(args.job)
        if args.check:
            validate_job_inputs(job, args.output_root)
            print(f'{job["run_id"]} manifest, code and protected inputs validated on CPU. Review SHA-256 {digest}.')
            if job["depends_on"] and args.output_root is None:
                print("Dependency summaries require --output-root. Target runtime and GPU require separate qualification.")
            return 0
        if not args.approve_sha256 or not args.output_root or not args.output_root.is_absolute():
            raise ValueError("execution needs an approved hash and an absolute private output root")
        def interrupt(_signal, _frame):
            raise InterruptedError("operator interrupted the run")

        signal.signal(signal.SIGTERM, interrupt)
        execute(job, digest, args.approve_sha256, args.output_root)
        print(f'{job["run_id"]} completed and wrote a private aggregate.')
    except Exception as error:
        print(f"Cycle A job blocked or failed ({type(error).__name__}).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
