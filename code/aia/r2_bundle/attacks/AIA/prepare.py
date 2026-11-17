"""Project the pinned synthetic AIA release into private inference/replay inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from attacks.AIA.source_evidence import load_selection, read_jsonl, source_projection
from attacks.AIA.ranked import parse_ranked_answer, remove_evidence
from attacks.AIA.verify_upstream_mapping import mismatch_fields
from attacks.MIA.hash_ids import hmac_id
from attacks.MIA.score_tokenized import reject_duplicate_json_keys
from scripts.run_safety import canonical_json_digest, private_file, publish_json, read_bound_bytes, sha256_file


def prepare(spec, out: Path):
    expected = {"dataset", "key", "expected_counts", "rubrics", "removals", "saved_outputs", "source_audit", "selection"}
    if not isinstance(spec, dict) or spec.keys() != expected:
        raise ValueError("unexpected protected AIA preparation settings")
    key = read_bound_bytes(spec["key"])
    if len(key) < 32:
        raise ValueError("profile HMAC key requires at least 32 bytes")
    audit, selection = load_selection(spec["selection"], spec["source_audit"])
    if spec["dataset"]["sha256"] != audit["dataset_sha256"]:
        raise ValueError("preparation source does not match the saved source audit")
    source = source_projection(spec["dataset"])
    counts = audit["attribute_counts"]
    if counts != spec["expected_counts"] or counts.keys() != spec["rubrics"].keys():
        raise ValueError("projected source population differs from the approved counts")
    removals = {}
    if spec["removals"] is not None:
        for row in read_jsonl(spec["removals"]):
            if row.keys() != {"source_row", "remove_indices"}:
                raise ValueError("unsupported evidence-removal annotation")
            index = row["source_row"]
            if type(index) is not int or not 0 <= index < len(source) or index in removals:
                raise ValueError("invalid source row in removal annotations")
            remove_evidence(source[index][1], row["remove_indices"])
            removals[index] = row["remove_indices"]
        if len(removals) != len(source):
            raise ValueError("paired control must annotate the entire eligible population")
    saved = spec["saved_outputs"]
    if (not isinstance(saved, list) or len(saved) > 3
            or len({item["model_id"] for item in saved}) != len(saved)):
        raise ValueError("freeze up to three unique saved model sources before scoring")
    rescore_models = {model_id for model_id, model in selection["models"].items() if model["mode"] == "rescore"}
    if {entry["model_id"] for entry in saved} != rescore_models:
        raise ValueError("prepared saved files must match the shortlisted rescore models")
    exports = {}
    for number, entry in enumerate(saved):
        if entry.keys() != {"model_id", "predictions"}:
            raise ValueError("invalid model source entry")
        model = entry["model_id"]
        if entry["predictions"]["sha256"] != audit["saved_outputs"][model]["sha256"]:
            raise ValueError("prepared saved file does not match the historical source audit")
        outputs = list(read_jsonl(entry["predictions"]))
        if len(outputs) != len(source):
            raise ValueError("saved row count differs from the source population")
        if not any(isinstance(row.get("predictions"), dict) and model in row["predictions"]
                   for row in outputs):
            raise ValueError("frozen model is absent from the saved source")
        for index, (projection, output) in enumerate(zip(source, outputs)):
            if mismatch_fields(output, projection, model) - {"prediction_record", "prediction_attribute"}:
                raise ValueError("saved prediction cannot be joined to the exact source row")
            _, _, attr, truth, _ = projection
            rubric = spec["rubrics"][attr]
            guesses = None
            predictions = output.get("predictions")
            if isinstance(predictions, dict) and isinstance(predictions.get(model), dict):
                attribute_prediction = predictions[model].get(attr)
                if isinstance(attribute_prediction, dict):
                    guesses = attribute_prediction.get("guess")
            # A blank legacy parser field cannot establish what the model answered.
            answer = None if guesses in (None, [], "") else json.dumps(
                [guesses] if isinstance(guesses, str) else guesses, ensure_ascii=False
            )
            outcome, ranked = parse_ranked_answer(answer, labels=rubric["labels"], aliases=rubric["aliases"])
            # Unsupported saved parser output stays unresolved pending adjudication.
            if outcome == "malformed":
                outcome, ranked = "missing", []
            identity = hmac_id(f'{spec["dataset"]["sha256"]}:{index}', key)
            exports.setdefault(f"saved-{number}-{attr}.jsonl", []).append({
                "profile_id_hash": identity, "ground_truth": truth, "condition": "full",
                "outcome": outcome, "guesses": ranked,
            })
    for index, (_, comments, attr, truth, _) in enumerate(source):
        exports.setdefault(f"profiles-{attr}.jsonl", []).append({
            "source_row": index, "attribute": attr, "comments": comments,
            "ground_truth": truth, "remove_indices": removals.get(index, []),
        })
    out.mkdir(mode=0o700, exist_ok=False)
    hashes = {}
    for name, rows in exports.items():
        path = out / name
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w",
                       encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        hashes[name] = sha256_file(path)
    publish_json(out / "preparation.json", {
        "schema": "cycleA.aia.preparation.v3", "dataset_sha256": spec["dataset"]["sha256"],
        "source_audit_sha256": spec["source_audit"]["sha256"], "selection_sha256": spec["selection"]["sha256"],
        "key_sha256": spec["key"]["sha256"],
        "rubric_sha256": {attribute: canonical_json_digest(rubric)
                          for attribute, rubric in spec["rubrics"].items()},
        "counts": counts, "has_complete_removal_annotations": bool(removals),
        "saved_model_ids": [entry["model_id"] for entry in saved], "output_sha256": hashes,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        private_file(args.spec)
        prepare(json.loads(args.spec.read_bytes(), object_pairs_hook=reject_duplicate_json_keys), args.output)
    except Exception as error:
        parser.exit(1, f"AIA preparation blocked or failed ({type(error).__name__}).\n")


if __name__ == "__main__":
    main()
