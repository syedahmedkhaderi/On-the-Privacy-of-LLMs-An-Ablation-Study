"""Stage 01: generate and detect, one cell per (model, condition, seed)."""

from __future__ import annotations

from pathlib import Path

from common import HEADER, PREFLIGHT, code, md, write_notebook


def cells() -> list:
    return [
        md("""
# Stage 01: generation and detection

96 cells: 8 models x 4 prompt styles x 3 seeds, 5,000 generations each.

Each cell writes `generations.parquet` (per-generation counts),
`detections.parquet` (salted hashes, never values) and a `COMPLETE` marker.
Cells are resumable, so an interrupted run costs at most one cell.

Decoder settings follow Carlini et al. 2021: 256 tokens, top-k 40, temperature
1.0. `repetition_penalty` and `top_p` are deliberately left at 1.0 because
both suppress the verbatim recall that memorization consists of.

Budget roughly 56 GPU hours. Models are loaded once and reused across all of
their cells.
"""),
        code(HEADER),
        code(PREFLIGHT),
        code("""
preflight([
    ("index", PATHS.index_file),
    ("true-prefix pool", PATHS.true_prefix_file),
    ("salt", PATHS.salt_file),
], required_packages=("numpy", "pandas", "pyarrow", "torch", "transformers"))
"""),
        code("""
import torch

print("cuda      :", torch.cuda.is_available())
print("devices   :", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {props.name} {props.total_memory / 1024**3:.1f} GB")
print("HF cache  :", PATHS.hf_cache or "(default)")
"""),
        md("## Load the index and the prefix pool"),
        code("""
import pandas as pd
from dea_a3 import index, pii

salt = pii.load_or_create_salt(PATHS.salt_file)
index_frame = index.load_index(PATHS.index_file)
index_hashes = index.hashes_by_kind(index_frame)
true_prefix_pool = pd.read_parquet(PATHS.true_prefix_file)["prefix"].tolist()

from dea_a3 import prompts as prompt_mod
pii_pool, pii_pool_source = prompt_mod.pii_leading_pool(
    PATHS.zeng_prompts, CONFIG.pii_leading_variants)

for kind, values in sorted(index_hashes.items()):
    print(f"{kind:6s} indexed: {len(values):,}")
print(f"true prefixes: {len(true_prefix_pool):,}")
print(f"pii_leading pool: {len(pii_pool):,} prompts, source {pii_pool_source}")
if pii_pool_source != "zeng_repo":
    print("  NOTE: Zeng's own prompts were not found. Run")
    print("        python tools/fetch_zeng_prompts.py")
    print("        Numbers from the enumerated pool are not directly")
    print("        comparable to his Table 3.")
"""),
        md("""
## Run the grid

Ordered smallest model first, so a configuration problem surfaces in minutes
rather than after a 6B model has been loaded.
"""),
        code("""
import time
from dea_a3 import modeling, runner

def modeling_vram():
    return modeling.vram_free_total_gb()

ORDER = [m.key for m in sorted(CONFIG.models, key=lambda m: m.params_m)]

# Optional subsetting, for running the Zeng anchor cell before the long tail.
# DEA_A3_MODELS=gpt-neo-1.3b runs only that model; unset runs everything.
# Order within the subset is still smallest first, and nothing already on disk
# is recomputed, so a subset run and a full run compose without conflict.
_only = [k.strip() for k in os.environ.get("DEA_A3_MODELS", "").split(",") if k.strip()]
if _only:
    unknown = [k for k in _only if k not in CONFIG.model_map()]
    if unknown:
        raise SystemExit(f"DEA_A3_MODELS names unknown models: {unknown}")
    ORDER = [k for k in ORDER if k in _only]
    print(f"DEA_A3_MODELS set, running {len(ORDER)} of "
          f"{len(CONFIG.models)} models: {ORDER}")

# Optional batch-size scaling, for running beside another job on a shared card.
# DEA_A3_BATCH_SCALE=0.5 halves every batch. Smaller batches use less VRAM and
# run slower; they do not change any result, because every batch is reseeded
# from the cell identity and the batch index rather than from position in a
# stream.
_scale = float(os.environ.get("DEA_A3_BATCH_SCALE", "1") or 1)
if _scale <= 0:
    raise SystemExit("DEA_A3_BATCH_SCALE must be positive")

# Headroom left free for whatever else is on the card. Raise it when sharing.
_reserve = float(os.environ.get("DEA_A3_RESERVE_GB", "1") or 1)

def batch_for(spec):
    return max(1, int(spec.batch_size * _scale))

if _scale != 1 or _reserve != 1:
    print(f"batch scale {_scale}, reserve {_reserve} GB")

for _free, _total in modeling_vram():
    print(f"visible GPU: {_free:.1f} GB free of {_total:.1f} GB")

summaries = []
skipped = []
started_all = time.time()

for model_key in ORDER:
    spec = CONFIG.model_map()[model_key]
    pending = [
        (cond, seed)
        for cond in CONFIG.conditions
        for seed in CONFIG.seeds
        if not runner.cell_is_done(PATHS.cell_dir(model_key, cond, seed))
    ]
    if not pending:
        print(f"{model_key}: all cells complete, skipping load")
        continue

    print(f"\\n=== {model_key} ({spec.params_m}M), {len(pending)} cells to run ===",
          flush=True)

    size = batch_for(spec)

    # A model's cells must all run at one batch size or its three seeds are not
    # comparable: batch size decides which prompts share a seeded sampling pass.
    prior = runner.completed_batch_sizes(PATHS, model_key)
    if prior and prior != {size}:
        raise SystemExit(
            f"{model_key}: completed cells used batch size(s) {sorted(prior)} "
            f"but this run would use {size}. Results would not be comparable "
            f"across seeds. Either rerun with the same DEA_A3_BATCH_SCALE, or "
            f"delete results/01_generations/{model_key}__* and start that "
            f"model over."
        )

    fits, detail = modeling.check_vram_headroom(
        spec, size, reserve_gb=_reserve,
        max_new_tokens=CONFIG.decoder.max_new_tokens)
    print(f"    vram: {detail}", flush=True)
    if not fits:
        print(f"    SKIPPING {model_key}: not enough free VRAM. Its cells stay "
              f"pending and a later run will pick them up.", flush=True)
        skipped.append(model_key)
        continue

    t0 = time.time()
    loaded = modeling.load_model(spec, cache_dir=PATHS.hf_cache)
    print(f"    loaded in {time.time() - t0:.0f}s on {loaded.device}", flush=True)

    for cond, seed in pending:
        summary = runner.run_cell(
            loaded, model_key, cond, seed, CONFIG, PATHS,
            index_hashes, salt, true_prefix_pool=true_prefix_pool,
            pii_pool=pii_pool, batch_size=size,
        )
        summaries.append(summary)
        peak = summary.get("peak_vram_gb")
        peak_txt = f", peak {peak:.1f} GB" if peak else ""
        print(f"    {model_key} {cond} seed{seed}: "
              f"extracted {summary['extracted_email'] + summary['extracted_phone']}, "
              f"verified {summary['verified_email'] + summary['verified_phone']}, "
              f"{summary['seconds']}s{peak_txt}", flush=True)

    modeling.free_model(loaded)
    for _free, _total in modeling_vram():
        print(f"    after unload: {_free:.1f} GB free of {_total:.1f} GB", flush=True)

print(f"\\ntotal {(time.time() - started_all) / 3600:.2f} h")
if skipped:
    print(f"skipped for VRAM: {skipped}")
    print("Rerun this stage when the card is free; completed cells are kept.")
"""),
        md("## Collect and checkpoint"),
        code("""
cell_summaries = runner.load_cell_summaries(PATHS)
cell_summaries.to_csv(PATHS.generations / "cell_summaries.csv", index=False)
print(f"cells complete: {len(cell_summaries)} of {len(CONFIG.cells())}")
cell_summaries.head(20)
"""),
        code("""
from dea_a3 import manifest

manifest.write_manifest(
    PATHS.generations, "01_generate", CONFIG.to_dict(), ROOT,
    extra={"cells_complete": int(len(cell_summaries)),
           "cells_expected": len(CONFIG.cells()),
           "pii_pool_source": pii_pool_source,
           "pii_pool_size": len(pii_pool)},
)
print("stage 01 complete")
"""),
    ]


def build(out_dir: Path) -> Path:
    return write_notebook(cells(), Path(out_dir) / "01_generate.ipynb")


if __name__ == "__main__":
    print(build(Path(__file__).resolve().parents[1] / "notebooks"))
