"""Prepare blinded AIA judge review and validate it from protected ratings."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from attacks.AIA.experiment import ATTRIBUTES
from attacks.AIA.scoring import OUTCOMES, _resample_mean
from attacks.AIA.source_evidence import read_jsonl
from attacks.MIA.hash_ids import hmac_id
from attacks.MIA.score_tokenized import reject_duplicate_json_keys
from scripts.run_safety import canonical_json_digest, private_dir, private_file, publish_json, read_bound_bytes

HEX = re.compile(r"[0-9a-f]{64}")
REVISION = re.compile(r"[0-9a-f]{40}")
REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}")
PROTOCOL_FIELDS = {"schema", "endpoint_sha256", "judge_model", "judge_revision", "judge_prompt_sha256",
                   "rubric_sha256", "frame_sha256", "key_sha256", "seed", "repetitions", "strata", "thresholds"}
SAMPLE_INPUTS = {"protocol", "frame", "key"}
RATING_ROLES = {"rater_a", "rater_b", "adjudication", "judge"}
THRESHOLD_FIELDS = {"min_human_exact", "min_judge_exact", "max_judge_mae"}


def read_json(binding):
    return json.loads(read_bound_bytes(binding), object_pairs_hook=reject_duplicate_json_keys)


def load_protocol(binding):
    protocol = read_json(binding)
    if (not isinstance(protocol, dict) or protocol.keys() != PROTOCOL_FIELDS
            or protocol["schema"] != "cycleA.aia.judge_protocol.v2"
            or not isinstance(protocol["judge_model"], str) or not protocol["judge_model"].strip()
            or not REVISION.fullmatch(protocol["judge_revision"])
            or any(not HEX.fullmatch(protocol[key]) for key in
                   ("endpoint_sha256", "judge_prompt_sha256", "rubric_sha256", "frame_sha256", "key_sha256"))
            or type(protocol["seed"]) is not int or protocol["seed"] < 0
            or type(protocol["repetitions"]) is not int or not 100 <= protocol["repetitions"] <= 20000):
        raise ValueError("unsupported frozen AIA judge protocol")
    thresholds = protocol["thresholds"]
    if (not isinstance(thresholds, dict) or thresholds.keys() != THRESHOLD_FIELDS
            or any(type(value) not in (int, float) or not 0 <= value <= 1
                   for value in thresholds.values())):
        raise ValueError("judge thresholds must be prespecified rates in [0, 1]")
    allocation = {}
    if not isinstance(protocol["strata"], list) or not protocol["strata"]:
        raise ValueError("judge protocol requires a nonempty stratum allocation")
    for row in protocol["strata"]:
        if (not isinstance(row, dict) or row.keys() != {"attribute", "outcome", "n"}
                or row["attribute"] not in ATTRIBUTES or row["outcome"] not in OUTCOMES
                or type(row["n"]) is not int or row["n"] < 2):
            raise ValueError("invalid judge stratum allocation")
        key = row["attribute"], row["outcome"]
        if key in allocation:
            raise ValueError("duplicate judge validation stratum")
        allocation[key] = row["n"]
    return protocol, allocation


def group_frame(binding, allocation):
    groups = defaultdict(list)
    seen = set()
    for row in read_jsonl(binding):
        if (row.keys() != {"case_id_hash", "attribute", "outcome"}
                or not HEX.fullmatch(row["case_id_hash"]) or row["case_id_hash"] in seen
                or (row["attribute"], row["outcome"]) not in allocation):
            raise ValueError("review frame contains invalid, duplicate or undeclared cases")
        case = row["case_id_hash"]
        seen.add(case)
        group = row["attribute"], row["outcome"]
        groups[group].append(case)
    return groups


def frozen_sample(spec, protocol, allocation):
    if any(spec[field]["sha256"] != protocol[f"{field}_sha256"] for field in ("frame", "key")):
        raise ValueError("review frame and blinding key must match the frozen protocol")
    key = read_bound_bytes(spec["key"])
    if len(key) < 32:
        raise ValueError("judge blinding key requires at least 32 bytes")
    groups = group_frame(spec["frame"], allocation)
    if any(len(groups[group]) < n for group, n in allocation.items()):
        raise ValueError("review frame cannot fill the prespecified stratum counts")
    namespace = f'{spec["protocol"]["sha256"]}:{spec["frame"]["sha256"]}:{protocol["seed"]}'
    sampled, join = [], {}
    for group, n in sorted(allocation.items()):
        cases = sorted(groups[group], key=lambda case: hmac_id(f"sample:{namespace}:{case}", key))[:n]
        for case in cases:
            blind = hmac_id(f"blind:{namespace}:{case}", key)
            sampled.append({"blind_id": blind, "attribute": group[0], "outcome": group[1]})
            join[blind] = case
    sampled.sort(key=lambda row: row["blind_id"])
    return {
        "schema": "cycleA.aia.judge_sample.v1", "protocol_sha256": spec["protocol"]["sha256"],
        "frame_sha256": spec["frame"]["sha256"], "items": sampled,
    }, join, groups


def prepare_sample(spec, out: Path):
    if not isinstance(spec, dict) or spec.keys() != SAMPLE_INPUTS:
        raise ValueError("judge sampling needs a protocol, review frame and private key")
    protocol, allocation = load_protocol(spec["protocol"])
    sample, join, _ = frozen_sample(spec, protocol, allocation)
    out.mkdir(mode=0o700, exist_ok=False)
    publish_json(out / "review_ids.json", [
        {"blind_id": row["blind_id"], "attribute": row["attribute"]} for row in sample["items"]
    ])
    publish_json(out / "private_join.json", join)
    publish_json(out / "sample.json", sample)


def read_ratings(binding, expected, *, id_field="blind_id", allow_unresolved=False):
    rows = read_json(binding)
    if not isinstance(rows, list):
        raise ValueError("judge ratings must be a list of case IDs and ordinal scores")
    ratings = {}
    for row in rows:
        if (not isinstance(row, dict) or row.keys() != {id_field, "score"}
                or row[id_field] in ratings or row[id_field] not in expected):
            raise ValueError("rating has an invalid or repeated case ID")
        score = row["score"]
        if not (allow_unresolved and score is None) and (type(score) is not int or score not in (0, 1, 2)):
            raise ValueError("rating has an invalid ordinal score")
        ratings[row[id_field]] = score
    if ratings.keys() != expected:
        raise ValueError("rating rows do not cover the exact required cases")
    return ratings


def evaluate_judge(spec):
    if not isinstance(spec, dict) or spec.keys() != RATING_ROLES | SAMPLE_INPUTS | {"sample", "review_receipt"}:
        raise ValueError("judge validation requires a frozen sample and independently reviewed ratings")
    protocol, allocation = load_protocol(spec["protocol"])
    sample, join, frame_groups = frozen_sample(spec, protocol, allocation)
    if read_json(spec["sample"]) != sample:
        raise ValueError("reviewed cases do not match the regenerated frozen sample")
    groups = defaultdict(list)
    for row in sample["items"]:
        groups[row["attribute"], row["outcome"]].append(row["blind_id"])

    hashes = {role: spec[role]["sha256"] for role in sorted(RATING_ROLES)}
    if Path(spec["rater_a"]["path"]).samefile(spec["rater_b"]["path"]):
        raise ValueError("independent rating artifacts must be separate records")
    receipt = read_json(spec["review_receipt"])
    expected = {"schema": "cycleA.aia.judge_review_receipt.v1", "sample_sha256": spec["sample"]["sha256"],
                "protocol_sha256": spec["protocol"]["sha256"], "rating_sha256": hashes}
    if (not isinstance(receipt, dict) or receipt.keys() != expected.keys() | {"review_ref", "review_checks"}
            or any(receipt[key] != value for key, value in expected.items())
            or not REF.fullmatch(receipt["review_ref"])
            or not isinstance(receipt["review_checks"], dict)
            or receipt["review_checks"].keys() != {"frozen_before_ratings", "independent_blinded_review",
                                                    "judge_execution_matches_protocol"}
            or any(value is not True for value in receipt["review_checks"].values())):
        raise ValueError("owner receipt must bind blinded human review and exact judge execution")
    ids = join.keys()
    a, b = [read_ratings(spec[role], ids) for role in ("rater_a", "rater_b")]
    disagreement = {case for case in ids if a[case] != b[case]}
    adjudication = read_ratings(spec["adjudication"], disagreement)
    judge = read_ratings(spec["judge"], ids)

    by_stratum = []
    for position, (group, items) in enumerate(sorted(groups.items())):
        truth = [adjudication.get(case, a[case]) for case in items]
        human = [float(a[case] == b[case]) for case in items]
        judge_exact = [float(judge[case] == label) for case, label in zip(items, truth)]
        error = [(judge[case] - label) / 2 for case, label in zip(items, truth)]
        absolute = [abs(value) for value in error]
        n = len(items)
        estimates = {"human_exact": sum(human) / n, "judge_exact": sum(judge_exact) / n,
                     "judge_mae": sum(absolute) / n, "judge_bias": sum(error) / n}
        limits = protocol["thresholds"]
        passed = (estimates["human_exact"] >= limits["min_human_exact"]
                  and estimates["judge_exact"] >= limits["min_judge_exact"]
                  and estimates["judge_mae"] <= limits["max_judge_mae"])
        intervals = {name: _resample_mean(values, protocol["repetitions"], protocol["seed"] + 4 * position + offset)
                     for offset, (name, values) in enumerate(zip(estimates, (human, judge_exact, absolute, error)))}
        by_stratum.append({"attribute": group[0], "outcome": group[1], "n": n,
                           "n_adjudicated": sum(case in disagreement for case in items),
                           "estimates": estimates, "bootstrap_intervals_95": intervals,
                           "meets_thresholds": passed})
    report = {"schema": "cycleA.aia.judge_validation.v2", "endpoint_sha256": protocol["endpoint_sha256"],
              "protocol_sha256": spec["protocol"]["sha256"], "sample_sha256": spec["sample"]["sha256"],
              "frame_sha256": sample["frame_sha256"],
              "review_receipt_sha256": spec["review_receipt"]["sha256"], "rating_sha256": hashes,
              "n_reviewed": len(ids), "strata": by_stratum, "thresholds": protocol["thresholds"],
              "meets_prespecified_thresholds": all(row["meets_thresholds"] for row in by_stratum)}
    return report, {join[blind]: score for blind, score in judge.items()}, frame_groups


def validate_judge(spec):
    report, _, _ = evaluate_judge(spec)
    return report


def partial_credit_summary(values):
    n = len(values)
    observed = sum(value for value in values if value is not None) / 2
    missing = values.count(None)
    return {"n_eligible": n, "n_unresolved": missing, "mean_partial_credit": None if missing else observed / n,
            "mean_partial_credit_bounds": [observed / n, (observed + missing) / n]}


def report_partial_credit(spec):
    if not isinstance(spec, dict) or spec.keys() != {"review_spec", "validation", "ratings", "application_receipt"}:
        raise ValueError("partial credit requires qualified validation and a full frozen frame")
    review = read_json(spec["review_spec"])
    reproduced, sampled_scores, groups = evaluate_judge(review)
    validation = read_json(spec["validation"])
    if (not reproduced["meets_prespecified_thresholds"]
            or canonical_json_digest(validation) != canonical_json_digest(reproduced)):
        raise ValueError("judge validation does not qualify this frozen endpoint and frame")
    proof = read_json(spec["application_receipt"])
    expected = {"schema": "cycleA.aia.judge_application_receipt.v2",
                **{f"{field}_sha256": spec[field]["sha256"]
                   for field in ("review_spec", "validation", "ratings")},
                "judge_execution_matches_protocol": True}
    if (proof.keys() != expected.keys() | {"review_ref"}
            or any(proof[key] != value for key, value in expected.items())
            or proof["judge_execution_matches_protocol"] is not True
            or not REF.fullmatch(proof["review_ref"])):
        raise ValueError("partial-credit scores require a bound application review receipt")
    cases = {case for items in groups.values() for case in items}
    scores = read_ratings(spec["ratings"], cases, id_field="case_id_hash", allow_unresolved=True)
    if any(scores[case] != score for case, score in sampled_scores.items()):
        raise ValueError("application scores differ from the validated judge ratings on reviewed cases")
    return {"schema": "cycleA.aia.partial_credit.v2", "endpoint_sha256": reproduced["endpoint_sha256"],
            "validation_sha256": spec["validation"]["sha256"], "ratings_sha256": spec["ratings"]["sha256"],
            "application_receipt_sha256": spec["application_receipt"]["sha256"],
            "overall": partial_credit_summary(list(scores.values())),
            "strata": [{"attribute": group[0], "outcome": group[1],
                         **partial_credit_summary([scores[case] for case in items])}
                        for group, items in sorted(groups.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sample", "validate", "partial-credit"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        private_file(args.spec)
        spec = json.loads(args.spec.read_bytes(), object_pairs_hook=reject_duplicate_json_keys)
        if args.command == "sample":
            prepare_sample(spec, args.output)
        else:
            report = (validate_judge if args.command == "validate" else report_partial_credit)(spec)
            private_dir(args.output.parent)
            publish_json(args.output, report)
    except Exception as error:
        parser.exit(1, f"AIA judge review blocked or failed ({type(error).__name__}).\n")


if __name__ == "__main__":
    main()
