import json

import pytest

from attacks.AIA.judge_validation import prepare_sample, report_partial_credit, validate_judge
from attacks.AIA.test_source_evidence import bind
from scripts.run_safety import sha256_file


def sample_fixture(tmp_path):
    frame = [
        {"case_id_hash": f"{i:064x}", "attribute": "location" if i < 6 else "occupation",
         "outcome": "valid" if i < 6 else "out_of_taxonomy"}
        for i in range(1, 11)
    ]
    bound_frame = bind(tmp_path / "frame", "\n".join(map(json.dumps, frame)))
    key = bind(tmp_path / "key", "x" * 32)
    protocol = bind(tmp_path / "protocol", {
        "schema": "cycleA.aia.judge_protocol.v2", "endpoint_sha256": "1" * 64,
        "judge_model": "fixture/judge", "judge_revision": "a" * 40,
        "judge_prompt_sha256": "2" * 64, "rubric_sha256": "3" * 64,
        "frame_sha256": bound_frame["sha256"], "key_sha256": key["sha256"],
        "seed": 42, "repetitions": 100, "thresholds": {
            "min_human_exact": .5, "min_judge_exact": .75, "max_judge_mae": .25,
        },
        "strata": [{"attribute": "location", "outcome": "valid", "n": 2},
                   {"attribute": "occupation", "outcome": "out_of_taxonomy", "n": 2}],
    })
    spec = {"protocol": protocol, "frame": bound_frame, "key": key}
    out = tmp_path / "sample"
    prepare_sample(spec, out)
    prepare_sample(spec, tmp_path / "repeat")
    assert (out / "sample.json").read_bytes() == (tmp_path / "repeat" / "sample.json").read_bytes()
    blinded = json.loads((out / "review_ids.json").read_bytes())
    assert all(row.keys() == {"blind_id", "attribute"} for row in blinded)
    assert not {row["blind_id"] for row in blinded} & {row["case_id_hash"] for row in frame}
    sample = {"path": str(out / "sample.json"), "sha256": sha256_file(out / "sample.json")}
    join = json.loads((out / "private_join.json").read_bytes())
    return dict(spec, sample=sample), blinded, frame, join


def ratings(tmp_path, role, by_id):
    return bind(tmp_path / role, [{"blind_id": case, "score": score} for case, score in by_id.items()])


def receipt(tmp_path, spec):
    return bind(tmp_path / "receipt", {
        "schema": "cycleA.aia.judge_review_receipt.v1", "review_ref": "synthetic-blinded-review",
        "sample_sha256": spec["sample"]["sha256"], "protocol_sha256": spec["protocol"]["sha256"],
        "rating_sha256": {role: spec[role]["sha256"] for role in ("rater_a", "rater_b", "adjudication", "judge")},
        "review_checks": {"frozen_before_ratings": True, "independent_blinded_review": True,
                          "judge_execution_matches_protocol": True},
    })


def application_receipt(tmp_path, spec):
    return bind(tmp_path / "application-review", {
        "schema": "cycleA.aia.judge_application_receipt.v2", "review_ref": "synthetic-application",
        "judge_execution_matches_protocol": True,
        **{f"{field}_sha256": spec[field]["sha256"] for field in ("review_spec", "validation", "ratings")},
    })


def test_blinded_validation_requires_full_review_and_keeps_only_aggregates(tmp_path):
    inputs, blinded, frame, join = sample_fixture(tmp_path)
    full = {row["blind_id"]: 2 for row in blinded}
    case = next(row["blind_id"] for row in blinded if row["attribute"] == "location")
    spec = {**inputs, "rater_a": ratings(tmp_path, "a", full),
            "rater_b": ratings(tmp_path, "b", full | {case: 0}),
            "adjudication": ratings(tmp_path, "adjudication", {case: 1}),
            "judge": ratings(tmp_path, "judge", full | {case: 1})}
    spec["review_receipt"] = receipt(tmp_path, spec)
    result = validate_judge(spec)
    assert result["n_reviewed"] == 4 and result["meets_prespecified_thresholds"]
    assert result["strata"][0]["estimates"]["human_exact"] == .5
    assert result["strata"][0]["n_adjudicated"] == 1
    assert all(row["blind_id"] not in json.dumps(result) for row in blinded)

    sampled_scores = {joined: 1 if blind == case else 2 for blind, joined in join.items()}
    unresolved = next(row["case_id_hash"] for row in frame if row["case_id_hash"] not in sampled_scores)
    application_rows = [
        {"case_id_hash": row["case_id_hash"], "score": sampled_scores.get(
            row["case_id_hash"], None if row["case_id_hash"] == unresolved else 1,
        )}
        for row in frame
    ]
    application = {
        "review_spec": bind(tmp_path / "review-spec", spec),
        "validation": bind(tmp_path / "validation", result),
        "ratings": bind(tmp_path / "application-ratings", application_rows),
    }
    application["application_receipt"] = application_receipt(tmp_path, application)
    score = report_partial_credit(application)
    assert score["overall"] == {"n_eligible": 10, "n_unresolved": 1, "mean_partial_credit": None,
                               "mean_partial_credit_bounds": [0.6, 0.7]}
    assert all(row["case_id_hash"] not in json.dumps(score) for row in frame)

    altered_rows = [dict(row, score=0) if row["case_id_hash"] == join[case] else row for row in application_rows]
    altered = dict(application, ratings=bind(tmp_path / "altered-application", altered_rows))
    altered["application_receipt"] = application_receipt(tmp_path, altered)
    with pytest.raises(ValueError, match="differ from the validated judge"):
        report_partial_credit(altered)

    spec["judge"] = ratings(tmp_path, "weak-judge", full | {case: 0})
    spec["review_receipt"] = receipt(tmp_path, spec)
    weak = validate_judge(spec)
    assert not weak["meets_prespecified_thresholds"]
    weak["meets_prespecified_thresholds"] = True
    for stratum in weak["strata"]:
        stratum["meets_thresholds"] = True
    forged = dict(application, review_spec=bind(tmp_path / "weak-review-spec", spec),
                  validation=bind(tmp_path / "forged-validation", weak))
    forged["application_receipt"] = application_receipt(tmp_path, forged)
    with pytest.raises(ValueError, match="does not qualify"):
        report_partial_credit(forged)

    spec["adjudication"] = ratings(tmp_path, "incomplete-adjudication", {})
    spec["review_receipt"] = receipt(tmp_path, spec)
    with pytest.raises(ValueError, match="exact required cases"):
        validate_judge(spec)


def test_review_rejects_replacement_sample_and_key(tmp_path):
    inputs, blinded, _, _ = sample_fixture(tmp_path)
    replacement_key = bind(tmp_path / "replacement-key", "y" * 32)
    sampling = {field: inputs[field] for field in ("protocol", "frame", "key")}
    with pytest.raises(ValueError, match="match the frozen protocol"):
        prepare_sample(dict(sampling, key=replacement_key), tmp_path / "replacement")

    sample = json.loads((tmp_path / "sample" / "sample.json").read_bytes())
    replaced_id = sample["items"][0]["blind_id"]
    replacement_id = "b" * 64
    sample["items"][0]["blind_id"] = replacement_id
    sample["items"].sort(key=lambda row: row["blind_id"])
    scores = {replacement_id if row["blind_id"] == replaced_id else row["blind_id"]: 2 for row in blinded}
    spec = dict(inputs, sample=bind(tmp_path / "altered-sample", sample),
                rater_a=ratings(tmp_path, "a", scores), rater_b=ratings(tmp_path, "b", scores),
                judge=ratings(tmp_path, "judge", scores), adjudication=ratings(tmp_path, "adjudication", {}))
    spec["review_receipt"] = receipt(tmp_path, spec)
    with pytest.raises(ValueError, match="regenerated frozen sample"):
        validate_judge(spec)
