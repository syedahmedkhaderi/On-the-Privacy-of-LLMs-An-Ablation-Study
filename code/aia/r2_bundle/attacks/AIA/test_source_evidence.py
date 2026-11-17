import json

import pytest

from attacks.AIA import source_evidence
from scripts.run_safety import sha256_file


def bind(path, data):
    if not isinstance(data, str):
        data = json.dumps(data)
    path.write_text(data)
    path.chmod(0o600)
    return {"path": str(path), "sha256": sha256_file(path)}


def selection_fixture(tmp_path, dataset, audit_binding, model_id, raw_output_hash):
    proof = bind(tmp_path / "execution", {
        "schema": source_evidence.EXECUTION_SCHEMA, "model_id": model_id,
        "model_revision": "a" * 40, "chat_template_sha256": "b" * 64,
        "dataset_sha256": dataset, "saved_output_sha256": raw_output_hash,
        "parser_protocol": "legacy_ranked_guess", "source_manifest_sha256": "c" * 64,
    })
    return bind(tmp_path / "selection", {
        "schema": source_evidence.SELECTION_SCHEMA, "dataset_identity": dataset,
        "source_audit_sha256": audit_binding["sha256"], "owner_review_ref": "synthetic-review",
        "models": {model_id: {"revision": "a" * 40, "chat_template_sha256": "b" * 64,
                              "mode": "rescore", "execution_evidence": proof}},
    })


def test_full_inventory_audit_and_shortlist_reject_weak_evidence(tmp_path, monkeypatch):
    dataset = bind(tmp_path / "dataset", {
        "feature": "sex", "personality": {"age": 32, "sex": "female"},
        "response": "synthetic private marker", "hardness": 1,
    })
    outputs = {}
    for i in range(10):
        model_id = f"fixture/model-{i}"
        outputs[model_id] = bind(tmp_path / f"saved-{i}", {
            "username": "32female", "num_comments": 1,
            "comments": [{"text": "synthetic private marker", "subreddit": "synth", "user": "32female"}],
            "reviews": {"synth": {"gender": {"estimate": "female", "hardness": 1, "certainty": 5,
                                             "detect_from_subreddit": False}}},
            "predictions": {model_id: {"gender": {"guess": []}, "full_answer": "synthetic answer"}},
        })
    pins = {"dataset_sha256": dataset["sha256"], "attribute_counts": {"gender": 1},
            "saved_outputs": {model: binding["sha256"] for model, binding in outputs.items()}}
    monkeypatch.setattr(source_evidence, "inventory", lambda: pins)
    with pytest.raises(ValueError, match="full saved model inventory"):
        source_evidence.build_source_audit({"dataset": dataset, "saved_outputs": dict(list(outputs.items())[:9])})
    report = source_evidence.build_source_audit({"dataset": dataset, "saved_outputs": outputs})
    assert len(report["saved_outputs"]) == 10
    assert all(row["n_missing_guesses"] == row["n_projection_matches"] == 1
               for row in report["saved_outputs"].values())
    assert all(secret not in json.dumps(report)
               for secret in ("synthetic private marker", "synthetic answer", "32female", str(tmp_path)))

    audit = bind(tmp_path / "audit", report)
    selected = "fixture/model-0"
    shortlist = selection_fixture(tmp_path, dataset["sha256"], audit, selected, outputs[selected]["sha256"])
    _, selection = source_evidence.load_selection(shortlist, audit)
    assert set(selection["models"]) == {selected}
    oversized = json.loads((tmp_path / "selection").read_text())
    for model in list(outputs)[1:4]:
        oversized["models"][model] = {
            "mode": "rerun", "revision": "a" * 40, "chat_template_sha256": "b" * 64, "execution_evidence": None,
        }
    with pytest.raises(ValueError, match="one to three"):
        source_evidence.load_selection(bind(tmp_path / "oversized", oversized), audit)

    selection["models"][selected]["chat_template_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="execution evidence differs"):
        source_evidence.load_selection(bind(tmp_path / "changed", selection), audit)
