"""Private output and single-device reservations for Cycle A workers."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import signal
import threading
import sys
import subprocess
import tempfile

from scripts.preflight_cycle_a import existing_work_status, local_external_queue_status

GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
DRIVER_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)+")
CUDA_RUNTIME_FIELDS = {"driver_version", "torch_cuda_build", "gpu_uuid", "compute_capability", "total_memory_bytes"}


def sha256_stream(stream) -> str:
    """Hash a binary stream in bounded chunks on Python 3.10 and later."""
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def source_digest(root: Path, relative_paths: list[str]) -> str:
    hashes = {name: sha256_file(root / name) for name in sorted(relative_paths)}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def private_file(path: Path) -> None:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("input must be a regular owner-only file")


def read_bound_bytes(spec: dict) -> bytes:
    """Return the same bytes whose identity was approved."""
    if not isinstance(spec, dict) or spec.keys() != {"path", "sha256"}:
        raise ValueError("expected a path and SHA-256")
    path = Path(spec["path"])
    if not path.is_absolute():
        raise ValueError("require an absolute path for protected inputs")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("input must be a regular owner-only file")
        data = stream.read()
    if hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError("protected input identity mismatch")
    return data


def canonical_json_digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("output or reservation directory must be owner-only")


def publish_json(path: Path, payload: dict) -> None:
    """Publish a fully written JSON file once, including under concurrent writers."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(directory: Path, name: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("invalid reservation name")
    private_dir(directory)
    fd = os.open(directory / name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("resource already reserved") from error
        yield
    finally:
        os.close(fd)


def _nvidia(*args: str) -> list[str]:
    return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                          check=True, capture_output=True, text=True, timeout=15).stdout.splitlines()


def check_device(gpu_uuid: str, min_free_mib: int, allow_sharing: bool = False) -> str:
    rows = _nvidia("--id=" + gpu_uuid, "--query-gpu=uuid,memory.free,utilization.gpu,driver_version")
    if len(rows) != 1:
        raise ValueError("assigned GPU UUID did not resolve uniquely")
    actual_uuid, free, utilization, driver = (value.strip() for value in rows[0].split(","))
    if actual_uuid != gpu_uuid or int(free) < min_free_mib or (not allow_sharing and int(utilization) > 5):
        raise RuntimeError("assigned GPU is unavailable or below memory requirement")
    processes = _nvidia("--query-compute-apps=gpu_uuid")
    if any(not GPU_UUID.fullmatch(line.strip()) for line in processes):
        raise ValueError("GPU process inventory could not be verified")
    if not allow_sharing and gpu_uuid in {line.strip() for line in processes}:
        raise RuntimeError("assigned GPU already has a compute process")
    if not DRIVER_VERSION.fullmatch(driver):
        raise ValueError("GPU driver version could not be verified")
    return driver


def validate_cuda_runtime(runtime, gpu_uuid: str) -> None:
    if (not isinstance(runtime, dict) or runtime.keys() != CUDA_RUNTIME_FIELDS
            or runtime["gpu_uuid"] != gpu_uuid
            or any(not isinstance(runtime[key], str) or not DRIVER_VERSION.fullmatch(runtime[key])
                   for key in ("driver_version", "torch_cuda_build", "compute_capability"))
            or type(runtime["total_memory_bytes"]) is not int or runtime["total_memory_bytes"] <= 0):
        raise ValueError("CUDA runtime evidence is missing or invalid")


def read_cuda_runtime(gpu_uuid: str, driver_version: str) -> dict:
    """Identify CUDA device 0 after setting visibility inside the reservation."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one usable reserved CUDA device")
    properties = torch.cuda.get_device_properties(0)
    actual_uuid = getattr(properties, "uuid", None)
    if actual_uuid is None:
        # PyTorch 2.4 does not expose UUIDs in device properties. Initialize only
        # visible device 0, then bind this process's real CUDA context through
        # NVIDIA's inventory. Never treat CUDA_VISIBLE_DEVICES as proof.
        probe = torch.empty(1, device="cuda:0")
        torch.cuda.synchronize(0)
        process_devices = set()
        for row in _nvidia("--query-compute-apps=pid,gpu_uuid"):
            fields = [field.strip() for field in row.split(",")]
            if len(fields) != 2 or not fields[0].isdigit() or not GPU_UUID.fullmatch(fields[1]):
                raise ValueError("CUDA process/device identity could not be verified")
            if int(fields[0]) == os.getpid():
                process_devices.add(fields[1])
        del probe
        if len(process_devices) != 1:
            raise RuntimeError("CUDA process does not identify exactly one assigned GPU UUID")
        actual_uuid = next(iter(process_devices))
    if str(actual_uuid).lower().removeprefix("gpu-") != gpu_uuid.lower().removeprefix("gpu-"):
        raise RuntimeError("CUDA device 0 does not match the assigned GPU UUID")
    runtime = {"driver_version": driver_version, "torch_cuda_build": torch.version.cuda,
               "gpu_uuid": gpu_uuid, "compute_capability": f"{properties.major}.{properties.minor}",
               "total_memory_bytes": properties.total_memory}
    validate_cuda_runtime(runtime, gpu_uuid)
    return runtime


@contextmanager
def reserve_device(gpu_uuid: str, min_free_mib: int, min_disk_mib: int, output_parent: Path,
                   allow_sharing: bool = False, peak_memory_mib: int | None = None):
    if not GPU_UUID.fullmatch(gpu_uuid):
        raise ValueError("require a full GPU UUID")
    if any(type(value) is not int or value <= 0 for value in (min_free_mib, min_disk_mib)):
        raise ValueError("memory and disk requirements must be positive integers")
    if type(allow_sharing) is not bool:
        raise ValueError("sharing permission must be boolean")
    if peak_memory_mib is not None and (type(peak_memory_mib) is not int
            or peak_memory_mib <= 1024 or min_free_mib < peak_memory_mib + 5120):
        raise ValueError("bound peak must leave at least 5120 MiB free")
    if allow_sharing and peak_memory_mib is None:
        raise ValueError("sharing requires a bound peak memory budget")
    lock_dir = Path.home() / ".cyclea-device-locks"
    with exclusive_lock(lock_dir, gpu_uuid.lower()):
        if not existing_work_status()[0] or not local_external_queue_status()[0]:
            raise RuntimeError("owner hold or this host's external queue gate is closed")
        if shutil.disk_usage(output_parent).free < min_disk_mib * 2**20:
            raise RuntimeError("output filesystem is below required free space")
        driver = check_device(gpu_uuid, min_free_mib, allow_sharing=True) if allow_sharing else check_device(gpu_uuid, min_free_mib)
        overrides = {"CUDA_VISIBLE_DEVICES": gpu_uuid, "HF_HUB_OFFLINE": "1",
                     "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
        previous = {key: os.environ.get(key) for key in overrides}
        os.environ.update(overrides)
        stopped = threading.Event()
        watcher = None
        if peak_memory_mib is not None:
            def watch():
                while not stopped.wait(1.0):
                    try:
                        safe = shared_headroom(gpu_uuid)
                    except Exception:
                        safe = False
                    if not safe and not stopped.is_set():
                        print("Cycle A GPU reserve below 5120 MiB or unverifiable; interrupting this job.", file=sys.stderr, flush=True)
                        # Only interrupt this process; never signal an existing GPU user.
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()
        try:
            yield driver
        finally:
            stopped.set()
            if watcher is not None:
                watcher.join(timeout=20)
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def shared_headroom(gpu_uuid: str) -> bool:
    """Require a verifiable live 5 GiB reserve during explicitly shared work."""
    rows = _nvidia("--id=" + gpu_uuid, "--query-gpu=memory.free")
    if len(rows) != 1 or not rows[0].strip().isdigit():
        raise ValueError("shared GPU memory inventory could not be verified")
    return int(rows[0].strip()) >= 5120
