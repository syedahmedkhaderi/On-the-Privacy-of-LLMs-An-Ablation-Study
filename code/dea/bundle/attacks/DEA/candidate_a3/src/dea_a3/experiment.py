"""Strict entry points for targeted DEA cells and an independent CPU control."""

from __future__ import annotations

from collections import Counter
from io import BytesIO
import json
import os
import math
import random
import re
from pathlib import Path

from . import aggregate, modeling, runner
from .config import Config, DecoderSpec, ModelSpec
from .index import hashes_by_kind, load_index
from .paths import Paths
from .pii import salted_hash

from attacks.DEA.historical import comparison_digest, qualify_rerun
from scripts.run_safety import read_bound_bytes

RELEASED_PROMPTS = {
    "email": (489, "68459cd2f938f127cb80e1be707bf9af2c1492edf068f0b6ddfed9aa0bfe08f9"),
    "phone": (324, "dad1ea1e28c22499d4722b5f106ea8e0c2b0ce1d76d23b1c6eb96ecf3c067fdb"),
}
FORMATS = {
    "first.last": lambda first, last: f"{first}.{last}",
    "first_last": lambda first, last: f"{first}_{last}",
    "flast": lambda first, last: f"{first[0]}{last}",
    "firstlast": lambda first, last: f"{first}{last}",
}
TOKEN = re.compile(r"[a-z]{1,40}")


def protected_index(inputs):
    salt = read_bound_bytes(inputs["salt"]).decode("ascii").strip()
    if not re.fullmatch(r"[0-9a-f]{64}", salt):
        raise ValueError("invalid protected index salt")
    index = load_index(BytesIO(read_bound_bytes(inputs["index"])))
    if not {"kind", "value_hash"} <= set(index.columns):
        raise ValueError("invalid protected index schema")
    if (index.empty or not index["value_hash"].map(
            lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    ).all()):
        raise ValueError("invalid protected index values")
    return index, salt


def validate_model(values):
    model = ModelSpec(**values)
    paper_roster = Config().model_map()
    roster_match = paper_roster.get(model.key)
    if not roster_match or (model.hf_id.casefold(), model.family, model.params_m) != (
        roster_match.hf_id.casefold(), roster_match.family, roster_match.params_m
    ):
        raise ValueError("model does not match the submitted DEA roster")
    if any(not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
           for revision in (model.revision, model.tokenizer_revision or model.revision)):
        raise ValueError("DEA requires immutable model and tokenizer revisions")
    if model.dtype != roster_match.dtype:
        raise ValueError("model dtype differs from the roster")
    if model.device_map is not None:
        raise ValueError("this runner reserves one GPU and requires an unsharded config")
    if not 1 <= model.batch_size <= 256:
        raise ValueError("invalid DEA batch size")
    return model


def validate_reference_index(spec):
    index, salt = protected_index(spec)
    if (set(index["kind"]) != {"email", "phone"} or "occurrences" not in index
            or index.duplicated(["kind", "value_hash"]).any()
            or not index["occurrences"].map(
                lambda value: type(value) is int and value > 0
            ).all()):
        raise ValueError("DEA cell requires unique email and phone reference values with positive corpus counts")
    return index, salt


def validate_cell(spec):
    expected = {"model", "condition", "seed", "queries", "decoder", "index", "salt", "prompt_pool",
                "history_audit"}
    if spec.keys() != expected:
        raise ValueError("unexpected DEA cell settings")
    model = validate_model(spec["model"])
    condition, seed = spec["condition"], spec["seed"]
    if (condition not in ("generic_prefix", "pii_leading", "true_prefix")
            or type(seed) is not int or seed not in (42, 43, 44)):
        raise ValueError("cell is outside the paper's comparison grid")
    if type(spec["queries"]) is not int or not 1 <= spec["queries"] <= 5000:
        raise ValueError("invalid query budget")
    historical = qualify_rerun(spec)
    index, salt = validate_reference_index(spec)
    pii_pool = true_pool = None
    if condition == "pii_leading":
        pool = spec["prompt_pool"]
        if not isinstance(pool, dict) or set(pool) != {"email", "phone"}:
            raise ValueError("PII-leading cells require both pinned released prompt lists")
        pii_pool = []
        for kind, (count, checksum) in RELEASED_PROMPTS.items():
            if pool[kind].get("sha256") != checksum:
                raise ValueError("PII-leading pool differs from the released source")
            data = json.loads(read_bound_bytes(pool[kind]))
            if (not isinstance(data, list) or len(data) != count
                    or any(not isinstance(item, str) or not item.strip() for item in data)):
                raise ValueError("released prompt list failed the expected schema or count")
            pii_pool.extend(data)
    elif condition == "true_prefix":
        true_pool = json.loads(read_bound_bytes(spec["prompt_pool"]))
        if (not isinstance(true_pool, list) or not true_pool
                or any(not isinstance(item, str) or not item.strip() for item in true_pool)):
            raise ValueError("invalid true-prefix pool")
    elif spec["prompt_pool"] is not None:
        raise ValueError("generic prefix uses the pinned code-defined pool")
    decoder = DecoderSpec(**spec["decoder"])
    if (type(decoder.max_new_tokens) is not int or not 1 <= decoder.max_new_tokens <= 1024
            or type(decoder.do_sample) is not bool or type(decoder.top_k) is not int or decoder.top_k < 0
            or any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0
                   for value in (decoder.top_p, decoder.temperature, decoder.repetition_penalty))
            or decoder.top_p > 1):
        raise ValueError("invalid frozen DEA decoder")
    cfg = Config(models=(model,), seeds=(seed,), conditions=(condition,),
                 prompts_per_cell=spec["queries"], decoder=decoder)
    return cfg, index, salt, pii_pool, true_pool, historical


def run_cell(spec, out: Path):
    cfg, index, salt, pii_pool, true_pool, historical = validate_cell(spec)
    model, = cfg.models
    condition, seed = spec["condition"], spec["seed"]
    index_hashes = hashes_by_kind(index)

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one usable reserved CUDA device")
    if not modeling.check_vram_headroom(
        model, model.batch_size, max_new_tokens=cfg.decoder.max_new_tokens
    )[0]:
        raise RuntimeError("DEA memory estimate exceeds the reserved device")
    loaded = modeling.load_model(model)
    paths = Paths(out)
    # H10: token saving is controlled outside the job settings (settings schema unchanged).
    tokens_root = os.environ.get("H10_TOKENS_ROOT")
    token_dir = Path(tokens_root) / out.name if tokens_root else None
    summary = runner.run_cell(loaded, model.key, condition, seed, cfg, paths,
                              index_hashes, salt, true_prefix_pool=true_pool, pii_pool=pii_pool,
                              token_dir=token_dir)
    summary["comparison_sha256"] = comparison_digest(spec)
    summary["historical_evidence"] = historical
    summary["evidence_mode"] = "targeted_rerun"
    return finalize_cell_summary(summary, paths.cell_dir(model.key, condition, seed), index, index_hashes)


def finalize_cell_summary(summary, cell_dir, index, index_hashes):
    _, detections = runner.read_completed_cell(cell_dir)
    summary["distinct"] = runner.distinct_counts(detections)
    expected_verified = [
        row.value_hash in index_hashes.get(row.kind, ()) for row in detections.itertuples()
    ]
    if detections["verified"].tolist() != expected_verified:
        raise ValueError("DEA verification flags differ from the approved reference index")
    verified = detections[detections["verified"].eq(True)]
    summary["repetition"] = aggregate.fixed_repetition_summary(
        index, set(zip(verified["kind"], verified["value_hash"])),
    )
    return summary


def validate_guessing_protocol(first_names, last_names, weights, domain, budget, batch_size, seeds):
    for names in (first_names, last_names):
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not TOKEN.fullmatch(name) for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("name pools must contain unique lowercase ASCII tokens")
    if (not isinstance(weights, dict) or not weights or set(weights) - FORMATS.keys()
            or any(type(weight) is not int or weight < 1 for weight in weights.values())):
        raise ValueError("format weights must be frozen positive integers")
    if (type(budget) is not int or type(batch_size) is not int
            or not 1 <= batch_size <= budget <= 1_000_000 or budget % batch_size
            or not isinstance(domain, str) or not re.fullmatch(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+", domain)
            or not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int for seed in seeds)):
        raise ValueError("invalid guessing protocol")


def guessing_control(first_names, last_names, weights, domain, budget, batch_size,
                     seeds, index_hashes, salt):
    """Draw names independently from fixed lists, then compare typed hashes."""
    validate_guessing_protocol(first_names, last_names, weights, domain, budget, batch_size, seeds)
    ordered = sorted(weights)
    results = []
    for seed in seeds:
        rng = random.Random(seed)
        drawn = Counter()
        match_hashes = set()
        distinct = set()
        matches = 0
        for _ in range(budget):
            first, last = rng.choice(first_names), rng.choice(last_names)
            fmt = rng.choices(ordered, weights=[weights[name] for name in ordered])[0]
            drawn[fmt] += 1
            value_hash = salted_hash("email", FORMATS[fmt](first, last) + "@" + domain, salt)
            distinct.add(value_hash)
            if value_hash in index_hashes.get("email", ()):
                matches += 1
                match_hashes.add(value_hash)
        results.append({
            "seed": seed, "candidate_draws": budget, "candidates_per_batch": batch_size,
            "query_equivalent_batches": budget // batch_size, "draws_by_format": dict(drawn),
            "distinct_format_valid": len(distinct), "matching_draws": matches,
            "distinct_matches": len(match_hashes), "match_fraction_of_draws": matches / budget,
            "distinct_match_fraction": len(match_hashes) / len(distinct),
        })
    return {"endpoint": "email only", "results": results}


def validate_control(spec):
    if spec.keys() != {"first_names", "last_names", "weights", "domain", "budget",
                       "candidates_per_batch", "seeds", "index", "salt"}:
        raise ValueError("unexpected control settings")
    first = json.loads(read_bound_bytes(spec["first_names"]))
    last = json.loads(read_bound_bytes(spec["last_names"]))
    index, salt = protected_index(spec)
    protocol = (first, last, spec["weights"], spec["domain"], spec["budget"],
                spec["candidates_per_batch"], spec["seeds"])
    validate_guessing_protocol(*protocol)
    hashes = hashes_by_kind(index)
    if not hashes.get("email"):
        raise ValueError("guessing control requires a nonempty email reference index")
    return protocol, hashes, salt


def run_control(spec, _out: Path):
    protocol, hashes, salt = validate_control(spec)
    return guessing_control(*protocol, hashes, salt)
