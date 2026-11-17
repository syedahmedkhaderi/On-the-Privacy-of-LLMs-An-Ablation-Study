"""The experiment driver.

One cell is one (model, condition, seed). A cell generates prompts_per_cell
continuations, scans each for PII, and writes two files: per-generation counts
and per-detection hashes. Cells are independent and resumable; a completed
cell is skipped on a rerun, so an interrupted run costs at most one cell.

Models are loaded once and reused across all their cells, because loading a
6B model repeatedly is a large fraction of the run.
"""

from __future__ import annotations

import json
import os
import hashlib
import hmac
from io import BytesIO
import re
import random
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import detect, modeling, prompts as prompt_mod
from .config import Config
from .paths import Paths
from .seeds import batch_seed, cell_seed, set_all_seeds

GEN_COLUMNS = [
    "generation_id", "model", "condition", "seed", "prompt_index",
    "n_chars", "extracted_email", "extracted_phone",
    "verified_email", "verified_phone",
]
DET_COLUMNS = ["generation_id", "model", "condition", "seed", "kind",
               "value_hash", "verified"]


def _completed_bytes(cell_dir: Path):
    marker = cell_dir / "COMPLETE"
    if not marker.is_file():
        return None
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        files = {"cell_config.json", "generations.parquet", "detections.parquet"}
        hashes = metadata["sha256"]
        if metadata["version"] != 1 or set(hashes) != files:
            raise ValueError
        data = {name: (cell_dir / name).read_bytes() for name in files}
        if any(hashlib.sha256(content).hexdigest() != hashes[name] for name, content in data.items()):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError("completion evidence is corrupt or legacy; audit separately") from error
    return data


def cell_is_done(cell_dir: Path) -> bool:
    return _completed_bytes(cell_dir) is not None


def _validate_tables(generations, detections, model, condition, seed, queries, kinds):
    if (list(generations.columns) != GEN_COLUMNS or type(queries) is not int or queries <= 0
            or len(generations) != queries
            or not pd.api.types.is_integer_dtype(generations["prompt_index"])
            or sorted(generations["prompt_index"].tolist()) != list(range(queries))):
        raise ValueError("DEA generation table has an invalid schema or query count")
    if list(detections.columns) != DET_COLUMNS:
        raise ValueError("DEA detection table has an invalid schema")
    for frame in (generations, detections):
        if any(not frame[field].eq(expected).all() or frame[field].isna().any()
               for field, expected in (("model", model), ("condition", condition), ("seed", seed))):
            raise ValueError("DEA table identity differs from its completed cell")
    expected_ids = [f"{model}|{condition}|{seed}|{i}" for i in generations["prompt_index"]]
    if generations["generation_id"].tolist() != expected_ids:
        raise ValueError("DEA generation identities do not match prompt indices")
    if not detections.empty and (
        not pd.api.types.is_bool_dtype(detections["verified"])
        or detections["verified"].isna().any()
        or not detections["kind"].isin(kinds).all()
        or not detections["generation_id"].isin(expected_ids).all()
        or detections.duplicated(["generation_id", "kind", "value_hash"]).any()
        or not detections["value_hash"].map(
            lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
        ).all()
    ):
        raise ValueError("DEA detection table has invalid records or duplicate values")
    count_columns = ["n_chars", "extracted_email", "verified_email", "extracted_phone", "verified_phone"]
    if any(not pd.api.types.is_integer_dtype(generations[name])
           or generations[name].isna().any() or generations[name].lt(0).any() for name in count_columns):
        raise ValueError("DEA generation table contains invalid counts")
    for kind in ("email", "phone"):
        selected = detections[detections["kind"].eq(kind)]
        extracted = selected.groupby("generation_id").size()
        verified = selected[selected["verified"].eq(True)].groupby("generation_id").size()
        for prefix, counts in (("extracted", extracted), ("verified", verified)):
            actual = counts.reindex(expected_ids, fill_value=0).tolist()
            if actual != generations[f"{prefix}_{kind}"].tolist():
                raise ValueError("DEA detections do not reconcile with the per-generation counts")


def read_completed_cell(cell_dir: Path):
    """Read and reconcile the same bytes verified by the completion marker."""
    data = _completed_bytes(cell_dir)
    if data is None:
        return None
    config = json.loads(data["cell_config.json"])
    generations = pd.read_parquet(BytesIO(data["generations.parquet"]))
    detections = pd.read_parquet(BytesIO(data["detections.parquet"]))
    model, condition, seed_text = cell_dir.name.rsplit("__", 2)
    seed = int(seed_text.removeprefix("seed"))
    _validate_tables(generations, detections, model, condition, seed,
                     config["prompts_per_cell"], config["kinds"])
    return generations, detections


def _cell_config(size: int, cfg: Config, pii_pool: list[str] | None,
                 true_prefix_pool: list[str] | None, prompt_list: list[str],
                 salt: str, model_spec, index_hashes: dict[str, set[str]]) -> dict:
    prompt_bytes = json.dumps(prompt_list, ensure_ascii=False).encode("utf-8")
    index_bytes = json.dumps(
        {kind: sorted(values) for kind, values in index_hashes.items()}, sort_keys=True,
    ).encode("utf-8")
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode("utf-8") + b"\0")
        source.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "model_spec": asdict(model_spec),
        "source_sha256": source.hexdigest(),
        "index_hmac_sha256": hmac.new(salt.encode("utf-8"), index_bytes, hashlib.sha256).hexdigest(),
        "batch_size": int(size),
        "prompts_per_cell": int(cfg.prompts_per_cell),
        "decoder": asdict(cfg.decoder),
        "n_pii_pool": len(pii_pool) if pii_pool else None,
        "n_true_prefix_pool": len(true_prefix_pool) if true_prefix_pool else None,
        "kinds": list(cfg.kinds),
        "prompt_hmac_sha256": hmac.new(salt.encode("utf-8"), prompt_bytes,
                                        hashlib.sha256).hexdigest(),
    }


def _completed_cell_configs(paths: Paths, model_key: str) -> list[dict]:
    configs = []
    for cell_dir in sorted(paths.generations.glob(f"{model_key}__*__seed*")):
        if not cell_is_done(cell_dir):
            continue
        config_path = cell_dir / "cell_config.json"
        if not config_path.is_file():
            raise ValueError(f"completed cell {cell_dir.name} has no cell_config.json; use a new run root")
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or not isinstance(config.get("batch_size"), int):
                raise ValueError
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"completed cell {cell_dir.name} has an unreadable cell configuration") from error
        configs.append(config)
    return configs


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"inspect existing temporary artifact: {temporary.name}")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"inspect existing temporary artifact: {temporary.name}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def run_cell(
    loaded: modeling.LoadedModel,
    model_key: str,
    condition: str,
    seed: int,
    cfg: Config,
    paths: Paths,
    index_hashes: dict[str, set[str]],
    salt: str,
    true_prefix_pool: list[str] | None = None,
    pii_pool: list[str] | None = None,
    batch_size: int | None = None,
    progress: bool = True,
    token_dir=None,
) -> dict:
    """Run one cell and write its artifacts. Returns a summary dict."""
    size = cfg.model_map()[model_key].batch_size if batch_size is None else batch_size
    if size < 1:
        raise ValueError("batch_size must be positive")
    cell_dir = paths.cell_dir(model_key, condition, seed)
    base = cell_seed(seed, model_key, condition)
    set_all_seeds(base)
    rng = random.Random(base)

    bos = loaded.tokenizer.bos_token or loaded.tokenizer.eos_token or "<|endoftext|>"
    prompt_list = prompt_mod.build_prompts(
        condition=condition,
        n=cfg.prompts_per_cell,
        rng=rng,
        bos_token=bos,
        true_prefix_pool=true_prefix_pool,
        n_variants=cfg.pii_leading_variants,
        pii_pool=pii_pool,
    )
    cell_config = _cell_config(size, cfg, pii_pool, true_prefix_pool, prompt_list, salt,
                               cfg.model_map()[model_key], index_hashes)
    shared_keys = ("batch_size", "prompts_per_cell", "decoder", "kinds", "model_spec",
                   "source_sha256", "index_hmac_sha256")
    for prior in _completed_cell_configs(paths, model_key):
        if any(prior.get(key) != cell_config.get(key) for key in shared_keys):
            raise ValueError("completed cells use a different sampling configuration; use a new run root")
    if cell_is_done(cell_dir):
        previous = json.loads((cell_dir / "cell_config.json").read_text(encoding="utf-8"))
        if previous != cell_config:
            raise ValueError("completed cell configuration differs; use a new run root")
        if progress:
            print(f"  skip (done): {cell_dir.name}", flush=True)
        generations, _ = read_completed_cell(cell_dir)
        return _summarise(generations, model_key, condition, seed, 0.0)
    if cell_dir.exists():
        raise ValueError(f"incomplete cell output exists; preserve it and use a new run root: {cell_dir.name}")
    cell_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    modeling.reset_peak_vram()
    gen_rows: list[dict] = []
    det_rows: list[dict] = []
    tok_rows: list[dict] = []
    started = time.time()

    for batch_index, batch in modeling.iter_batches(prompt_list, size):
        bseed = batch_seed(base, batch_index)
        if token_dir is not None:
            continuations, batch_prompt_ids, batch_cont_ids = modeling.generate_batch_with_ids(
                loaded, batch, cfg.decoder, bseed)
        else:
            continuations = modeling.generate_batch(loaded, batch, cfg.decoder, bseed)
            batch_prompt_ids = batch_cont_ids = None
        if len(continuations) != len(batch):
            raise ValueError("generator returned a different number of continuations than prompts")
        for offset, continuation in enumerate(continuations):
            prompt_index = batch_index * size + offset
            append_counts(gen_rows, det_rows, model_key, condition, seed, prompt_index,
                          continuation, index_hashes, salt, cfg.kinds)
            if token_dir is not None:
                tok_rows.append({"generation_id": f"{model_key}|{condition}|{seed}|{prompt_index}",
                                 "prompt_index": prompt_index,
                                 "prompt_ids": batch_prompt_ids[offset],
                                 "continuation_ids": batch_cont_ids[offset]})
        if progress and batch_index % 20 == 0:
            done = min((batch_index + 1) * size, len(prompt_list))
            print(f"    {cell_dir.name}: {done}/{len(prompt_list)}", flush=True)

    elapsed = time.time() - started
    if token_dir is not None:
        write_tokens(token_dir, tok_rows)
    return commit_cell(cell_dir, cell_config, gen_rows, det_rows, model_key, condition, seed,
                       elapsed, peak=modeling.peak_vram_gb())


def write_tokens(token_dir, tok_rows):
    """H10: protected token IDs (mode 600). Not part of the cell's COMPLETE marker."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    token_dir = Path(token_dir)
    token_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    table = pa.table({
        "generation_id": pa.array([r["generation_id"] for r in tok_rows], pa.string()),
        "prompt_index": pa.array([r["prompt_index"] for r in tok_rows], pa.int64()),
        "prompt_ids": pa.array([r["prompt_ids"] for r in tok_rows], pa.list_(pa.int32())),
        "continuation_ids": pa.array([r["continuation_ids"] for r in tok_rows], pa.list_(pa.int32())),
    })
    tmp = token_dir / "tokens.parquet.tmp"
    pq.write_table(table, tmp, compression="zstd")
    os.chmod(tmp, 0o600)
    os.replace(tmp, token_dir / "tokens.parquet")


def append_counts(gen_rows, det_rows, model_key, condition, seed, prompt_index,
                  continuation, index_hashes, salt, kinds):
    """Scan a proven continuation and retain only safe counts and hashes."""
    gid = f"{model_key}|{condition}|{seed}|{prompt_index}"
    result = detect.scan_continuation(
        generation_id=gid, model=model_key, condition=condition,
        seed=seed, prompt_index=prompt_index, continuation=continuation,
        index_hashes=index_hashes, salt=salt,
        kinds=kinds,
    )
    gen_rows.append({
        "generation_id": gid, "model": model_key, "condition": condition,
        "seed": seed, "prompt_index": prompt_index,
        "n_chars": result.n_chars,
        "extracted_email": result.extracted.get("email", 0),
        "extracted_phone": result.extracted.get("phone", 0),
        "verified_email": result.verified.get("email", 0),
        "verified_phone": result.verified.get("phone", 0),
    })
    for det in result.detections:
        det_rows.append({
            "generation_id": gid, "model": model_key,
            "condition": condition, "seed": seed, "kind": det.kind,
            "value_hash": det.value_hash, "verified": det.verified,
        })


def commit_cell(cell_dir, cell_config, gen_rows, det_rows, model_key, condition, seed, elapsed, *, peak=None):
    """Publish the validated tables and bind them in the completion marker."""
    gen_df = pd.DataFrame(gen_rows, columns=GEN_COLUMNS)
    det_df = pd.DataFrame(det_rows, columns=DET_COLUMNS)
    _validate_tables(gen_df, det_df, model_key, condition, seed, cell_config["prompts_per_cell"], cell_config["kinds"])
    _write_parquet_atomic(gen_df, cell_dir / "generations.parquet")
    _write_parquet_atomic(det_df, cell_dir / "detections.parquet")
    _write_json_atomic(cell_dir / "cell_config.json", cell_config)

    # Peak VRAM for this cell. Written as its own small file so the parquet
    # schema is unchanged and an older result set stays readable.
    if peak is not None:
        _write_json_atomic(cell_dir / "vram.json", {
            "peak_allocated_gb": round(peak, 3), "batch_size": cell_config["batch_size"]
        })
    _write_json_atomic(cell_dir / "COMPLETE", {
        "version": 1,
        "seconds": elapsed,
        "sha256": {
            name: hashlib.sha256((cell_dir / name).read_bytes()).hexdigest()
            for name in ("cell_config.json", "generations.parquet", "detections.parquet")
        },
    })

    summary = _summarise(gen_df, model_key, condition, seed, elapsed)
    summary["peak_vram_gb"] = round(peak, 3) if peak is not None else None
    return summary


def _summarise(gen_df: pd.DataFrame, model: str, condition: str,
               seed: int, elapsed: float) -> dict:
    return {
        "model": model,
        "condition": condition,
        "seed": seed,
        "generations": int(len(gen_df)),
        "extracted_email": int(gen_df["extracted_email"].sum()) if len(gen_df) else 0,
        "extracted_phone": int(gen_df["extracted_phone"].sum()) if len(gen_df) else 0,
        "verified_email": int(gen_df["verified_email"].sum()) if len(gen_df) else 0,
        "verified_phone": int(gen_df["verified_phone"].sum()) if len(gen_df) else 0,
        "seconds": round(elapsed, 1),
    }


def completed_batch_sizes(paths: Paths, model_key: str) -> set[int]:
    """Batch sizes already used by completed cells of one model.

    More than one value means the model's cells were not run under a single
    sampling configuration and cannot be compared across seeds.
    """
    return {int(config["batch_size"])
            for config in _completed_cell_configs(paths, model_key)}


def load_cell_summaries(paths: Paths) -> pd.DataFrame:
    """Collect every completed cell into one frame."""
    rows = []
    for cell_dir in sorted(paths.generations.glob("*__*__seed*")):
        tables = read_completed_cell(cell_dir)
        if tables is None:
            continue
        frame, _ = tables
        rows.append(_summarise(frame, str(frame["model"].iloc[0]),
                               str(frame["condition"].iloc[0]),
                               int(frame["seed"].iloc[0]), 0.0))
    return pd.DataFrame(rows)


def distinct_by_cell(paths: Paths) -> pd.DataFrame:
    """Distinct values per cell, as opposed to occurrences.

    The `extracted_*` and `verified_*` columns on a cell summary are sums of
    per-generation distinct counts, so a value the model emits in 400 separate
    generations contributes 400. That is an occurrence count. Zeng et al. do
    not state which of the two their figures are, so both are reported.

    Returns one row per (model, condition, seed) with, per kind, the number of
    distinct values emitted anywhere in the cell and the distinct subset that
    is in the index.
    """
    det = load_detections(paths, verified_only=False)
    cols = ["model", "condition", "seed",
            "distinct_extracted_email", "distinct_extracted_phone",
            "distinct_verified_email", "distinct_verified_phone"]
    keys = ["model", "condition", "seed"]
    groups = dict(tuple(det.groupby(keys)))
    rows = []
    for cell in load_cell_summaries(paths).itertuples(index=False):
        model, condition, seed = cell.model, cell.condition, cell.seed
        grp = groups.get((model, condition, seed), det.iloc[:0])
        row = {"model": model, "condition": condition, "seed": int(seed)}
        row.update(distinct_counts(grp))
        rows.append(row)
    return pd.DataFrame(rows, columns=cols).sort_values(
        ["model", "condition", "seed"]
    ).reset_index(drop=True)


def distinct_counts(detections: pd.DataFrame) -> dict:
    result = {}
    for kind in ("email", "phone"):
        sub = detections[detections["kind"].eq(kind)]
        result[f"distinct_extracted_{kind}"] = int(sub["value_hash"].nunique())
        result[f"distinct_verified_{kind}"] = int(sub.loc[sub["verified"].eq(True), "value_hash"].nunique())
    return result


def load_detections(paths: Paths, verified_only: bool = True) -> pd.DataFrame:
    frames = []
    for cell_dir in sorted(paths.generations.glob("*__*__seed*")):
        tables = read_completed_cell(cell_dir)
        if tables is None:
            continue
        _, frame = tables
        if verified_only and not frame.empty:
            frame = frame[frame["verified"]]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=DET_COLUMNS)
    return pd.concat(frames, ignore_index=True)
