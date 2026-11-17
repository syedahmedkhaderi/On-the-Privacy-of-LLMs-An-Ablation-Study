#!/usr/bin/env python3
"""Metadata-only audit of protected AIA JSONL files; never prints row values."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_jsonl(path: Path, dataset: bool = False) -> dict:
    line_count = invalid_count = 0
    field_types: dict[str, Counter] = {}
    nested_keys: dict[str, set[str]] = {}
    category_counts: Counter = Counter()
    user_sequence: list[str] = []
    user_seen: set[str] = set()
    duplicate_users = 0
    prediction_models: set[str] = set()
    prediction_fields: set[str] = set()
    raw_text_field_present = False
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_count += 1
                continue
            if not isinstance(row, dict):
                invalid_count += 1
                continue
            for key, value in row.items():
                field_types.setdefault(key, Counter())[type(value).__name__] += 1
                if isinstance(value, dict):
                    nested_keys.setdefault(key, set()).update(map(str, value.keys()))
                if key in {"text", "response", "comments"}:
                    raw_text_field_present = True
            if dataset:
                feature = row.get("feature")
                if isinstance(feature, str):
                    category_counts[feature] += 1
            else:
                username = row.get("username")
                if isinstance(username, str):
                    user_sequence.append(username)
                    if username in user_seen:
                        duplicate_users += 1
                    user_seen.add(username)
                predictions = row.get("predictions")
                if isinstance(predictions, dict):
                    for model_id, payload in predictions.items():
                        prediction_models.add(str(model_id))
                        if isinstance(payload, dict):
                            prediction_fields.update(map(str, payload.keys()))
    return {
        "file_name": path.name,
        "file_sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": line_count,
        "invalid_rows": invalid_count,
        "top_level_field_names": sorted(field_types),
        "field_value_types": {key: dict(sorted(counter.items()))
                              for key, counter in sorted(field_types.items())},
        "nested_key_names": {key: sorted(values) for key, values in sorted(nested_keys.items())},
        "feature_counts": dict(sorted(category_counts.items())) if dataset else {},
        "unique_user_count": len(user_seen) if not dataset else None,
        "duplicate_user_count": duplicate_users if not dataset else None,
        "prediction_model_ids": sorted(prediction_models),
        "prediction_field_names": sorted(prediction_fields),
        "raw_text_field_present": raw_text_field_present,
        "_user_sequence": user_sequence,
    }


def audit(dataset_path: Path, prediction_files: list[Path]) -> dict:
    dataset = inspect_jsonl(dataset_path, dataset=True)
    dataset.pop("_user_sequence", None)
    predictions = [inspect_jsonl(path) for path in prediction_files]
    sequences = [item.pop("_user_sequence") for item in predictions]
    nonempty_sequences = [sequence for sequence in sequences if sequence]
    order_matches = bool(nonempty_sequences) and all(
        sequence == nonempty_sequences[0] for sequence in nonempty_sequences[1:]
    )
    dataset_keys = set(dataset["top_level_field_names"])
    candidate_id_fields = sorted(dataset_keys & {"username", "user", "id", "row_id", "profile_id"})
    return {
        "dataset": dataset,
        "prediction_files": predictions,
        "prediction_user_order_matches_across_models": order_matches,
        "dataset_has_common_row_id_field": bool(candidate_id_fields),
        "common_row_id_field_names": candidate_id_fields,
        "prediction_to_dataset_row_order_verified": False,
        "note": "Schema/count/order metadata only. This tool does not score, copy, or display raw values.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.predictions_dir.glob("*.jsonl"))
    if not paths:
        raise SystemExit("ERROR: no prediction JSONL files found")
    report = audit(args.dataset, paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("AIA metadata audit written; raw row values were not included.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
