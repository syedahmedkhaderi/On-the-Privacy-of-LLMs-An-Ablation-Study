"""Rescore owner-qualified historical DEA token traces without loading weights."""

from __future__ import annotations

import json
from pathlib import Path
import time

from attacks.DEA.historical import DIGEST, qualify_rescore, replay_comparison_digest
from attacks.MIA.score_tokenized import reject_duplicate_json_keys
from scripts.run_safety import canonical_json_digest, read_bound_bytes
from . import modeling, runner
from .experiment import finalize_cell_summary, validate_model, validate_reference_index
from .index import hashes_by_kind
from .paths import Paths

SETTINGS = {"model", "condition", "seed", "queries", "max_prompt_tokens", "max_new_tokens",
            "eos_token_ids", "trace", "index", "salt", "history_audit", "source_protocol"}
TRACE_FIELDS = {"prompt_index", "padded_input_ids", "sequence_ids"}


def read_trace(spec):
    rows, observed_ids = [], set()
    data = read_bound_bytes(spec["trace"]).decode("utf-8")
    for offset, line in enumerate(data.removesuffix("\n").split("\n")):
        row = json.loads(line, object_pairs_hook=reject_duplicate_json_keys)
        if (not isinstance(row, dict) or row.keys() != TRACE_FIELDS
                or type(row["prompt_index"]) is not int or row["prompt_index"] != offset
                or offset >= spec["queries"]):
            raise ValueError("historical trace has invalid or unordered prompt rows")
        prompt, sequence = row["padded_input_ids"], row["sequence_ids"]
        if (not isinstance(prompt, list) or not 1 <= len(prompt) <= spec["max_prompt_tokens"]
                or not isinstance(sequence, list)
                or not 1 <= len(sequence) - len(prompt) <= spec["max_new_tokens"]
                or any(type(token) is not int or token < 0 for token in (*prompt, *sequence))
                or sequence[:len(prompt)] != prompt):
            raise ValueError("historical trace fails token boundary or budget validation")
        observed_ids.update(sequence)
        rows.append(sequence[len(prompt):])
    if len(rows) != spec["queries"]:
        raise ValueError("historical trace count differs from the reviewed query budget")
    return rows, observed_ids


def validate_source_protocol(spec, model):
    source = json.loads(read_bound_bytes(spec["source_protocol"]), object_pairs_hook=reject_duplicate_json_keys)
    expected = {field: spec[field] for field in ("condition", "queries", "max_prompt_tokens",
                                                "max_new_tokens", "eos_token_ids")}
    expected.update(schema="cycleA.dea.replay_protocol.v1", model_id=model.hf_id, model_revision=model.revision,
                    tokenizer_revision=model.tokenizer_revision or model.revision)
    source_digests = {"prompt_recipe_sha256", "generation_config_sha256"}
    if (not isinstance(source, dict) or source.keys() != expected.keys() | source_digests
            or canonical_json_digest({key: source[key] for key in expected}) != canonical_json_digest(expected)
            or any(not isinstance(source[key], str) or not DIGEST.fullmatch(source[key]) for key in source_digests)):
        raise ValueError("historical protocol does not match the reviewed replay settings")


def validate_replay(spec):
    if not isinstance(spec, dict) or spec.keys() != SETTINGS:
        raise ValueError("unexpected DEA replay settings")
    model = validate_model(spec["model"])
    if (spec["condition"] not in ("generic_prefix", "pii_leading", "true_prefix")
            or type(spec["seed"]) is not int or spec["seed"] not in (42, 43, 44)
            or type(spec["queries"]) is not int or not 1 <= spec["queries"] <= 5000
            or type(spec["max_prompt_tokens"]) is not int or not 1 <= spec["max_prompt_tokens"] <= 4096
            or type(spec["max_new_tokens"]) is not int or not 1 <= spec["max_new_tokens"] <= 1024):
        raise ValueError("historical replay needs a published cell and a bounded source protocol")
    stops = spec["eos_token_ids"]
    if (not isinstance(stops, list)
            or any(type(token) is not int or token < 0 for token in stops)
            or stops != sorted(set(stops))):
        raise ValueError("historical replay requires the reviewed sorted, unique EOS token IDs")
    history = qualify_rescore(spec)
    validate_source_protocol(spec, model)
    index, salt = validate_reference_index(spec)
    rows, observed_ids = read_trace(spec)
    return model, index, salt, history, rows, observed_ids


def run_replay(spec, out: Path):
    started = time.monotonic()
    model, index, salt, history, rows, observed_ids = validate_replay(spec)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model.hf_id, revision=model.tokenizer_revision or model.revision,
        local_files_only=True, trust_remote_code=False,
    )
    stops = set(spec["eos_token_ids"])
    if (observed_ids | stops) - set(tokenizer.get_vocab().values()):
        raise ValueError("historical trace uses tokens outside the reviewed tokenizer vocabulary")
    model_key, condition, seed = model.key, spec["condition"], spec["seed"]
    kinds = ["email", "phone"]
    index_hashes = hashes_by_kind(index)
    generations, detections = [], []
    for prompt_index, suffix in enumerate(rows):
        continuation = tokenizer.decode(modeling.trim_at_eos(suffix, stops), skip_special_tokens=True)
        runner.append_counts(generations, detections, model_key, condition, seed, prompt_index,
                             continuation, index_hashes, salt, kinds)
    comparison = replay_comparison_digest(spec)
    config = {"prompts_per_cell": spec["queries"], "kinds": kinds,
              "comparison_sha256": comparison, "trace_sha256": spec["trace"]["sha256"],
              "audit_sha256": history["audit_sha256"], "evidence_mode": "historical_replay"}
    cell_dir = Paths(out).cell_dir(model_key, condition, seed)
    cell_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
    summary = runner.commit_cell(cell_dir, config, generations, detections, model_key, condition,
                                 seed, time.monotonic() - started)
    summary.update(comparison_sha256=comparison, historical_evidence=history,
                   trace_sha256=spec["trace"]["sha256"], evidence_mode="historical_replay")
    return finalize_cell_summary(summary, cell_dir, index, index_hashes)
