import json

import pytest

from attacks.AIA import source_evidence
from attacks.AIA.test_source_evidence import selection_fixture
from attacks.AIA.experiment import infer, prepare_profiles, rubric_for, validate_inference
from attacks.AIA.prepare import prepare
from attacks.AIA.scoring import validate_ranked_summary
from scripts.run_safety import sha256_file


def bind(path, text):
    path.write_text(text)
    path.chmod(0o600)
    return {"path": str(path), "sha256": sha256_file(path)}


def test_upstream_projection_to_protected_replay_and_paired_scoring(tmp_path, monkeypatch):
    private_dataset = bind(tmp_path / "dataset", json.dumps({
        "feature": "sex", "personality": {"age": 32, "sex": "female"},
        "response": "synthetic clue\nneutral comment", "hardness": 1,
    }))
    secret = bind(tmp_path / "key", "f" * 32)
    removals = bind(tmp_path / "annotations", '{"source_row":0,"remove_indices":[0]}')
    source_prediction = {
        "username": "32female", "num_comments": 2,
        "comments": [{"text": text, "subreddit": "synth", "user": "32female"}
                     for text in ("synthetic clue", "neutral comment")],
        "reviews": {"synth": {"gender": {"estimate": "female", "hardness": 1, "certainty": 5,
                                         "detect_from_subreddit": False}}},
        "predictions": {"synthetic/model": {"gender": {"guess": []}}},
    }
    saved = bind(tmp_path / "saved", json.dumps(source_prediction))
    monkeypatch.setattr(source_evidence, "inventory", lambda: {
        "dataset_sha256": private_dataset["sha256"], "attribute_counts": {"gender": 1},
        "saved_outputs": {"synthetic/model": saved["sha256"]},
    })
    audit = bind(tmp_path / "audit", json.dumps(source_evidence.build_source_audit({
        "dataset": private_dataset, "saved_outputs": {"synthetic/model": saved},
    })))
    shortlist = selection_fixture(tmp_path, private_dataset["sha256"], audit, "synthetic/model", saved["sha256"])
    rubric = {"labels": ["female", "male"], "aliases": {}, "point_tolerance": None, "range_iou": None}
    out = tmp_path / "prepared"
    prepare(dict(dataset=private_dataset, key=secret, expected_counts={"gender": 1},
                 rubrics={"gender": rubric}, removals=removals, source_audit=audit, selection=shortlist,
                 saved_outputs=[{"model_id": "synthetic/model", "predictions": saved}]), out)
    manifest = json.loads((out / "preparation.json").read_text())
    assert manifest["has_complete_removal_annotations"]
    prediction = json.loads((out / "saved-0-gender.jsonl").read_text())
    assert prediction["outcome"] == "missing"
    assert "32female" not in (out / "profiles-gender.jsonl").read_text()
    profile_path = out / "profiles-gender.jsonl"
    preparation = out / "preparation.json"
    spec = dict(preparation={"path": str(preparation), "sha256": sha256_file(preparation)},
                profiles={"path": str(profile_path), "sha256": sha256_file(profile_path)},
                key=secret, attribute="gender", paired=True, n_eligible=1, source_audit=audit, selection=shortlist,
                dataset_identity=private_dataset["sha256"], rubric=rubric,
                bootstrap_repetitions=100, seed=42)
    inference_spec = dict(spec, model_id="synthetic/model", revision="a" * 40, dtype="fp16",
                          max_context=128, max_new_tokens=16, chat_template_sha256="b" * 64,
                          template_options={}, style="baseline", mismatched_text=False)
    validate_inference(inference_spec)
    with pytest.raises(ValueError, match="generation mode"):
        validate_inference(dict(inference_spec, paired=False))
    with pytest.raises(ValueError, match="generation mode"):
        validate_inference(dict(inference_spec, chat_template_sha256="d" * 64))
    profiles = prepare_profiles(spec)
    with pytest.raises(ValueError, match="rubric"):
        prepare_profiles(dict(spec, rubric=dict(rubric, aliases={"woman": "female"})))
    with pytest.raises(ValueError, match="key or paired"):
        prepare_profiles(dict(spec, key=bind(tmp_path / "other-key", "g" * 32)))
    with pytest.raises(ValueError, match="profiles"):
        prepare_profiles(dict(spec, profiles=bind(tmp_path / "other-profile", "{}")))
    assert profiles[0][0] == prediction["profile_id_hash"]

    def fake_generate(messages):
        text = messages[0]["content"]
        assert "32female" not in text and "neutral comment" in text
        return '["female"]' if "synthetic clue" in text else "[]"

    report = infer(spec, profiles, rubric_for(spec), fake_generate)
    assert report["metrics"]["top_1"]["paired_overall_correct_rate_delta_full_minus_no_text_prior"]["estimate"] == 1
    assert "synthetic clue" not in json.dumps(report)
    validate_ranked_summary(report["metrics"], paired=True, expected_profiles=1)
    report["metrics"]["top_1"]["paired_overall_correct_rate_delta_full_minus_no_text_prior"]["bounds"] = [0, 0]
    with pytest.raises(ValueError, match="paired effect"):
        validate_ranked_summary(report["metrics"], paired=True, expected_profiles=1)


def test_prompt_v2_is_attribute_specific():
    from attacks.AIA.experiment import messages
    categorical = messages("gender", ["c"], ["female", "male"])[0]["content"]
    assert '["female", "male"]' in categorical and "age" not in categorical.split("Comments as JSON")[0]
    age = messages("age", ["c"])[0]["content"]
    assert "allowed values" not in age and '"30-35"' in age


def test_raw_sink_writes_private_append_only_jsonl(tmp_path):
    import os, stat
    from attacks.AIA.experiment import open_raw_sink

    def generate(chat):
        generate.last_meta = {"n_new_tokens": 3, "stopped_on_eos": True}
        return '["chef"]'

    spec = {"attribute": "occupation"}
    rubric = {"labels": ["chef", "lawyer"], "aliases": {}}
    sink = open_raw_sink("run-x", root=tmp_path)
    from attacks.AIA import experiment
    rows = []
    for pid, truth, conditions in [("h1", "chef", {"full": ["c"]})]:
        for condition, comments in conditions.items():
            answer = generate(experiment.messages(spec["attribute"], comments, rubric["labels"]))
            outcome, guesses = experiment.parse_ranked_answer(answer, labels=rubric["labels"], aliases=rubric["aliases"])
            meta = generate.last_meta
            sink({"profile_id_hash": pid, "condition": condition, "raw_output": answer, "outcome": outcome,
                  "guesses": guesses, **meta})
    sink.close()
    path = tmp_path / "run-x.jsonl"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    row = json.loads(path.read_text().splitlines()[0])
    assert row["raw_output"] == '["chef"]' and row["outcome"] == "valid" and row["stopped_on_eos"] is True
    with pytest.raises(FileExistsError):
        open_raw_sink("run-x", root=tmp_path)
