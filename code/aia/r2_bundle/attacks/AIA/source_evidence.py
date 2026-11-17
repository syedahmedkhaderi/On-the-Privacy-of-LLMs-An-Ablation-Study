"""Audit historical AIA sources in place and qualify an owner-frozen shortlist."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from attacks.AIA.verify_upstream_mapping import expected_projection, mismatch_fields
from attacks.MIA.score_tokenized import reject_duplicate_json_keys
from scripts.run_safety import private_dir, private_file, publish_json, read_bound_bytes

INVENTORY_PATH = Path(__file__).with_name("SAVED_SOURCE_INVENTORY.json")
INVENTORY_SHA256 = "08639234ab98ffeafa9a25b10fcdc543f038457912c5d702c4940b48e1376778"
AUDIT_SCHEMA = "cycleA.aia.saved_source_audit.v1"
SELECTION_SCHEMA = "cycleA.aia.selection.v2"
EXECUTION_SCHEMA = "cycleA.aia.saved_execution.v1"
DIGEST = re.compile(r"[0-9a-f]{64}")
REVISION = re.compile(r"[0-9a-f]{40}")
REVIEW_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
SAVED_RESULT_FIELDS = {"sha256", "n_rows", "n_projection_matches", "n_model_predictions", "n_missing_guesses"}


def inventory():
    raw = INVENTORY_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INVENTORY_SHA256:
        raise ValueError("AIA saved source inventory differs from the reviewed snapshot")
    return json.loads(raw, object_pairs_hook=reject_duplicate_json_keys)


def read_jsonl(spec):
    for line in read_bound_bytes(spec).decode("utf-8").removesuffix("\n").split("\n"):
        value = json.loads(line, object_pairs_hook=reject_duplicate_json_keys)
        if not isinstance(value, dict):
            raise ValueError("protected input must contain JSON objects")
        yield value


def source_projection(binding):
    pins = inventory()
    if binding["sha256"] != pins["dataset_sha256"]:
        raise ValueError("AIA source does not match the pinned synthetic dataset")
    source = [expected_projection(row) for row in read_jsonl(binding)]
    if not source or any(row is None for row in source):
        raise ValueError("input has an invalid upstream dataset projection")
    counts = Counter(row[2] for row in source)
    signatures = {json.dumps(row, ensure_ascii=False, sort_keys=True) for row in source}
    if counts != pins["attribute_counts"] or len(signatures) != len(source):
        raise ValueError("source population differs from the pinned unique projection")
    return source


def build_source_audit(spec):
    """Revalidate the pinned cohort and all saved files without calculating accuracy."""
    if not isinstance(spec, dict) or spec.keys() != {"dataset", "saved_outputs"}:
        raise ValueError("unsupported AIA source audit specification")
    pins = inventory()
    bindings = spec["saved_outputs"]
    if not isinstance(bindings, dict) or bindings.keys() != pins["saved_outputs"].keys():
        raise ValueError("AIA source audit requires the exact full saved model inventory")
    source = source_projection(spec["dataset"])
    reports = {}
    for model_id, binding in bindings.items():
        if binding["sha256"] != pins["saved_outputs"][model_id]:
            raise ValueError("saved AIA output differs from its pinned source hash")
        rows = list(read_jsonl(binding))
        aligned, model_predictions, missing = 0, 0, 0
        for projection, row in zip(source, rows):
            mismatches = mismatch_fields(row, projection, model_id)
            aligned += not (mismatches - {"prediction_record", "prediction_attribute"})
            model_predictions += "prediction_record" not in mismatches
            if mismatches & {"prediction_record", "prediction_attribute"}:
                missing += 1
            else:
                guesses = row["predictions"][model_id][projection[2]].get("guess")
                missing += guesses in (None, "", [])
        reports[model_id] = {
            "sha256": binding["sha256"], "n_rows": len(rows), "n_projection_matches": aligned,
            "n_model_predictions": model_predictions, "n_missing_guesses": missing,
        }
    return {"schema": AUDIT_SCHEMA, "dataset_sha256": pins["dataset_sha256"],
            "attribute_counts": pins["attribute_counts"], "saved_outputs": reports}


def _load_audit(binding):
    audit = json.loads(read_bound_bytes(binding), object_pairs_hook=reject_duplicate_json_keys)
    pins = inventory()
    if (not isinstance(audit, dict) or audit.keys() != {"schema", "dataset_sha256", "attribute_counts", "saved_outputs"}
            or audit["schema"] != AUDIT_SCHEMA or audit["dataset_sha256"] != pins["dataset_sha256"]
            or audit["attribute_counts"] != pins["attribute_counts"]
            or audit["saved_outputs"].keys() != pins["saved_outputs"].keys()):
        raise ValueError("AIA source audit does not cover the pinned cohort and all saved outputs")
    n = sum(pins["attribute_counts"].values())
    for model, report in audit["saved_outputs"].items():
        if (report.keys() != SAVED_RESULT_FIELDS or report["sha256"] != pins["saved_outputs"][model]
                or any(type(report[key]) is not int or report[key] < 0 for key in SAVED_RESULT_FIELDS - {"sha256"})
                or max(report["n_projection_matches"], report["n_model_predictions"], report["n_missing_guesses"])
                > min(n, report["n_rows"])):
            raise ValueError("saved AIA audit counts do not reconcile")
    return audit


def load_selection(selection_binding, audit_binding):
    audit = _load_audit(audit_binding)
    selection = json.loads(read_bound_bytes(selection_binding), object_pairs_hook=reject_duplicate_json_keys)
    if (not isinstance(selection, dict)
            or selection.keys() != {"schema", "dataset_identity", "source_audit_sha256", "owner_review_ref", "models"}
            or selection["schema"] != SELECTION_SCHEMA or selection["dataset_identity"] != audit["dataset_sha256"]
            or selection["source_audit_sha256"] != audit_binding["sha256"]
            or not REVIEW_REF.fullmatch(selection["owner_review_ref"])
            or not isinstance(selection["models"], dict) or not 1 <= len(selection["models"]) <= 3
            or not selection["models"].keys() <= audit["saved_outputs"].keys()):
        raise ValueError("AIA shortlist must bind the source audit and one to three reviewed models")
    for model_id, model in selection["models"].items():
        if (not isinstance(model, dict)
                or model.keys() != {"revision", "chat_template_sha256", "mode", "execution_evidence"}
                or not REVISION.fullmatch(model["revision"])
                or not DIGEST.fullmatch(model["chat_template_sha256"])
                or model["mode"] not in {"rescore", "rerun"}):
            raise ValueError("unsupported AIA shortlist model settings")
        if model["mode"] == "rerun":
            if model["execution_evidence"] is not None:
                raise ValueError("rerun models must not claim qualified saved execution")
            continue
        report = audit["saved_outputs"][model_id]
        n = sum(audit["attribute_counts"].values())
        if report["n_rows"] != n or report["n_projection_matches"] != n or not report["n_model_predictions"]:
            raise ValueError("saved AIA model does not have a qualified complete source projection")
        proof = json.loads(read_bound_bytes(model["execution_evidence"]), object_pairs_hook=reject_duplicate_json_keys)
        expected = {"schema": EXECUTION_SCHEMA, "model_id": model_id, "model_revision": model["revision"],
                    "chat_template_sha256": model["chat_template_sha256"], "dataset_sha256": audit["dataset_sha256"],
                    "saved_output_sha256": report["sha256"], "parser_protocol": "legacy_ranked_guess"}
        if (not isinstance(proof, dict) or proof.keys() != expected.keys() | {"source_manifest_sha256"}
                or any(proof[key] != value for key, value in expected.items())
                or not DIGEST.fullmatch(proof["source_manifest_sha256"])):
            raise ValueError("saved AIA execution evidence differs from the reviewed model and source")
    return audit, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        private_file(args.spec)
        spec = json.loads(args.spec.read_bytes(), object_pairs_hook=reject_duplicate_json_keys)
        report = build_source_audit(spec)
        private_dir(args.output.parent)
        publish_json(args.output, report)
        print(f'Revalidated {len(report["saved_outputs"])} AIA source files.')
    except Exception as error:
        parser.exit(1, f"AIA source audit blocked or failed ({type(error).__name__}).\n")


if __name__ == "__main__":
    main()
