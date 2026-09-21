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
import random
import time
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


def cell_is_done(cell_dir: Path) -> bool:
    return (cell_dir / "COMPLETE").exists() and (cell_dir / "generations.parquet").exists()


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
) -> dict:
    """Run one cell and write its artifacts. Returns a summary dict."""
    cell_dir = paths.cell_dir(model_key, condition, seed)
    if cell_is_done(cell_dir):
        if progress:
            print(f"  skip (done): {cell_dir.name}", flush=True)
        summary = pd.read_parquet(cell_dir / "generations.parquet")
        return _summarise(summary, model_key, condition, seed, 0.0)

    cell_dir.mkdir(parents=True, exist_ok=True)
    modeling.reset_peak_vram()
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

    size = batch_size or cfg.model_map()[model_key].batch_size
    gen_rows: list[dict] = []
    det_rows: list[dict] = []
    started = time.time()

    for batch_index, batch in modeling.iter_batches(prompt_list, size):
        bseed = batch_seed(base, batch_index)
        texts = modeling.generate_batch(loaded, batch, cfg.decoder, bseed)
        for offset, (prompt, text) in enumerate(zip(batch, texts)):
            prompt_index = batch_index * size + offset
            gid = f"{model_key}|{condition}|{seed}|{prompt_index}"
            result = detect.scan_generation(
                generation_id=gid, model=model_key, condition=condition,
                seed=seed, prompt_index=prompt_index, prompt=prompt,
                full_text=text, index_hashes=index_hashes, salt=salt,
                kinds=cfg.kinds,
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
        if progress and batch_index % 20 == 0:
            done = min((batch_index + 1) * size, len(prompt_list))
            print(f"    {cell_dir.name}: {done}/{len(prompt_list)}", flush=True)

    elapsed = time.time() - started
    gen_df = pd.DataFrame(gen_rows, columns=GEN_COLUMNS)
    det_df = pd.DataFrame(det_rows, columns=DET_COLUMNS)
    gen_df.to_parquet(cell_dir / "generations.parquet", index=False)
    det_df.to_parquet(cell_dir / "detections.parquet", index=False)
    # Batch size is part of the result, not just a performance knob: it decides
    # which prompts share a batch, and each batch is a single seeded sampling
    # pass. Two cells of the same model run at different batch sizes are not
    # comparable. Record it so a resume can detect the mismatch.
    (cell_dir / "cell_config.json").write_text(
        json.dumps({"batch_size": int(size),
                    "prompts_per_cell": int(cfg.prompts_per_cell),
                    "decoder": {"max_new_tokens": cfg.decoder.max_new_tokens,
                                "top_k": cfg.decoder.top_k,
                                "temperature": cfg.decoder.temperature,
                                "top_p": cfg.decoder.top_p,
                                "repetition_penalty": cfg.decoder.repetition_penalty},
                    "n_pii_pool": len(pii_pool) if pii_pool else None},
                   sort_keys=True) + "\n",
        encoding="utf-8")
    (cell_dir / "COMPLETE").write_text(f"{elapsed:.1f}\n", encoding="utf-8")

    # Peak VRAM for this cell. Written as its own small file so the parquet
    # schema is unchanged and an older result set stays readable.
    peak = modeling.peak_vram_gb()
    if peak is not None:
        (cell_dir / "vram.json").write_text(
            json.dumps({"peak_allocated_gb": round(peak, 3),
                        "batch_size": int(size)}) + "\n",
            encoding="utf-8")

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
    sizes: set[int] = set()
    for cell_dir in sorted(paths.generations.glob(f"{model_key}__*__seed*")):
        config = cell_dir / "cell_config.json"
        if not (cell_dir / "COMPLETE").exists() or not config.exists():
            continue
        try:
            sizes.add(int(json.loads(config.read_text())["batch_size"]))
        except Exception:
            continue
    return sizes


def load_cell_summaries(paths: Paths) -> pd.DataFrame:
    """Collect every completed cell into one frame."""
    rows = []
    for cell_dir in sorted(paths.generations.glob("*__*__seed*")):
        gen_file = cell_dir / "generations.parquet"
        if not gen_file.exists():
            continue
        frame = pd.read_parquet(gen_file)
        if frame.empty:
            continue
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
    if det.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for (model, condition, seed), grp in det.groupby(["model", "condition", "seed"]):
        row = {"model": model, "condition": condition, "seed": int(seed)}
        for kind in ("email", "phone"):
            sub = grp[grp["kind"] == kind]
            row[f"distinct_extracted_{kind}"] = int(sub["value_hash"].nunique())
            row[f"distinct_verified_{kind}"] = int(
                sub[sub["verified"]]["value_hash"].nunique()
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=cols).sort_values(
        ["model", "condition", "seed"]
    ).reset_index(drop=True)


def load_detections(paths: Paths, verified_only: bool = True) -> pd.DataFrame:
    frames = []
    for cell_dir in sorted(paths.generations.glob("*__*__seed*")):
        det_file = cell_dir / "detections.parquet"
        if not det_file.exists():
            continue
        frame = pd.read_parquet(det_file)
        if verified_only and not frame.empty:
            frame = frame[frame["verified"]]
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=DET_COLUMNS)
    return pd.concat(frames, ignore_index=True)
