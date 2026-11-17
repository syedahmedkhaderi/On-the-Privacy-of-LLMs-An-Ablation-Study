"""Verify saved AIA rows against the upstream synthetic-profile transformation.

Reads protected files in place and emits only hashes, counts, and mismatch field
names. It never prints, copies, or writes profile text, labels, usernames, or answers.
This is a data-provenance audit, not a model experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


FEATURE_TO_ATTRIBUTE = {
    "income_level": "income",
    "age": "age",
    "sex": "gender",
    "city_country": "location",
    "birth_city_country": "pobp",
    "education": "education",
    "occupation": "occupation",
    "relationship_status": "married",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    invalid = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    invalid += 1
                else:
                    rows.append(value)
            except (json.JSONDecodeError, UnicodeDecodeError):
                invalid += 1
    return rows, invalid


def expected_projection(row: dict[str, Any]) -> tuple[str, list[str], str, Any, Any] | None:
    feature = row.get("feature")
    attribute = FEATURE_TO_ATTRIBUTE.get(feature)
    personality = row.get("personality")
    if attribute is None or not isinstance(personality, dict):
        return None
    if feature not in personality:
        return None

    age = personality.get("age")
    sex = personality.get("sex")
    if age is None or sex is None:
        return None
    username = f"{age}{sex}"
    response = row.get("response")
    if not isinstance(response, str):
        return None
    comments = [line for line in response.split("\n") if line not in ("", " ")]
    return username, comments, attribute, personality[feature], row.get("hardness")


def mismatch_fields(
    output: dict[str, Any],
    projection: tuple[str, list[str], str, Any, Any],
    expected_model: str,
) -> set[str]:
    username, expected_comments, attribute, estimate, hardness = projection
    mismatches: set[str] = set()
    if output.get("username") != username:
        mismatches.add("username")

    comments = output.get("comments")
    if not isinstance(comments, list):
        mismatches.add("comments")
    else:
        texts = [item.get("text") if isinstance(item, dict) else None for item in comments]
        subreddits = [item.get("subreddit") if isinstance(item, dict) else None for item in comments]
        users = [item.get("user") if isinstance(item, dict) else None for item in comments]
        if texts != expected_comments or any(value != "synth" for value in subreddits):
            mismatches.add("comments")
        if any(value != username for value in users):
            mismatches.add("comment_user")

    reviews = output.get("reviews")
    source_review = reviews.get("synth") if isinstance(reviews, dict) else None
    target_review = source_review.get(attribute) if isinstance(source_review, dict) else None
    if not isinstance(target_review, dict):
        mismatches.add("review_attribute")
    else:
        if target_review.get("estimate") != estimate:
            mismatches.add("review_estimate")
        if target_review.get("hardness") != hardness:
            mismatches.add("review_hardness")
        if target_review.get("certainty") != 5:
            mismatches.add("review_certainty")
        if target_review.get("detect_from_subreddit") is not False:
            mismatches.add("review_source_flag")

    predictions = output.get("predictions")
    prediction = predictions.get(expected_model) if isinstance(predictions, dict) else None
    if not isinstance(prediction, dict):
        mismatches.add("prediction_record")
    elif not isinstance(prediction.get(attribute), dict):
        mismatches.add("prediction_attribute")
    if output.get("num_comments") != len(expected_comments):
        mismatches.add("num_comments")
    return mismatches


def audit(root: Path, audit_path: Path) -> int:
    metadata = json.loads(audit_path.read_text(encoding="utf-8"))
    dataset_meta = metadata["dataset"]
    dataset_path = root / "datasets" / dataset_meta["file_name"]
    if not dataset_path.is_file() or sha256(dataset_path) != dataset_meta["file_sha256"]:
        print("dataset hash does not match the recorded protected-source audit")
        return 2

    source_rows, source_invalid = read_jsonl(dataset_path)
    projections = [expected_projection(row) for row in source_rows]
    invalid_projection = sum(item is None for item in projections)
    hardness_counts = Counter(
        str(row.get("hardness")) for row in source_rows if row.get("hardness") is not None
    )
    source_signatures = Counter(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        for item in projections
        if item is not None
    )
    duplicate_projection_groups = sum(count > 1 for count in source_signatures.values())
    duplicate_projection_rows = sum(count for count in source_signatures.values() if count > 1)

    print(
        f"dataset rows={len(source_rows)} invalid_json={source_invalid} "
        f"unmappable_rows={invalid_projection}"
    )
    print(
        f"identical source-projection groups={duplicate_projection_groups} "
        f"rows_in_groups={duplicate_projection_rows}"
    )
    print("hardness row counts=" + ", ".join(f"{key}:{value}" for key, value in sorted(hardness_counts.items())))

    failed = bool(source_invalid or invalid_projection or len(source_rows) != dataset_meta["rows"])
    for prediction_meta in metadata["prediction_files"]:
        prediction_path = root / "results" / prediction_meta["file_name"]
        if (
            not prediction_path.is_file()
            or sha256(prediction_path) != prediction_meta["file_sha256"]
        ):
            print(f"{prediction_meta['file_name']}: source hash mismatch")
            failed = True
            continue

        output_rows, output_invalid = read_jsonl(prediction_path)
        field_counts: Counter[str] = Counter()
        guess_types: Counter[str] = Counter()
        inference_types: Counter[str] = Counter()
        full_answer_types: Counter[str] = Counter()
        aligned = 0
        for source, output in zip(projections, output_rows):
            if source is None:
                field_counts["source_projection"] += 1
                continue
            model_id = prediction_meta["prediction_model_ids"][0]
            predictions = output.get("predictions")
            prediction = predictions.get(model_id) if isinstance(predictions, dict) else None
            attribute_prediction = prediction.get(source[2]) if isinstance(prediction, dict) else None
            if not isinstance(attribute_prediction, dict):
                guess_types["missing_attribute"] += 1
                inference_types["missing_attribute"] += 1
            elif "guess" not in attribute_prediction:
                guess_types["missing_guess"] += 1
            else:
                guess = attribute_prediction["guess"]
                if isinstance(guess, list):
                    kind = "empty_list" if not guess else "nonempty_list"
                else:
                    kind = "string" if isinstance(guess, str) else type(guess).__name__
                guess_types[kind] += 1
            if isinstance(attribute_prediction, dict):
                inference = attribute_prediction.get("inference")
                inference_types[
                    "missing" if "inference" not in attribute_prediction else "string" if isinstance(inference, str) else type(inference).__name__
                ] += 1
            if isinstance(prediction, dict):
                answer = prediction.get("full_answer")
                full_answer_types[
                    "missing" if "full_answer" not in prediction else "string" if isinstance(answer, str) else type(answer).__name__
                ] += 1
            else:
                full_answer_types["missing_prediction"] += 1
            mismatch = mismatch_fields(output, source, model_id)
            if not mismatch:
                aligned += 1
            field_counts.update(mismatch)
            if not isinstance(predictions, dict) or model_id not in predictions:
                field_counts["prediction_model"] += 1

        row_count_matches = len(output_rows) == len(source_rows) == prediction_meta["rows"]
        all_rows_match = row_count_matches and output_invalid == 0 and aligned == len(source_rows)
        if field_counts["prediction_model"]:
            all_rows_match = False
        failed = failed or not all_rows_match
        mismatch_summary = ", ".join(
            f"{key}:{value}" for key, value in sorted(field_counts.items())
        ) or "none"
        guess_summary = ", ".join(
            f"{key}:{value}" for key, value in sorted(guess_types.items())
        ) or "none"
        inference_summary = ", ".join(
            f"{key}:{value}" for key, value in sorted(inference_types.items())
        ) or "none"
        full_answer_summary = ", ".join(
            f"{key}:{value}" for key, value in sorted(full_answer_types.items())
        ) or "none"
        print(
            f"{prediction_meta['file_name']}: rows={len(output_rows)} "
            f"invalid_json={output_invalid} row_order_matches={aligned}/{len(source_rows)} "
            f"field_mismatches={mismatch_summary} guess_field_types={guess_summary} "
            f"inference_field_types={inference_summary} full_answer_types={full_answer_summary}"
        )

    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Protected AIA source directory")
    parser.add_argument("--audit", required=True, type=Path, help="Schema audit JSON with expected hashes")
    args = parser.parse_args()
    return audit(args.root, args.audit)


if __name__ == "__main__":
    raise SystemExit(main())
