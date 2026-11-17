import importlib.util, json, sys
from pathlib import Path
import pytest
from attacks.AIA import experiment as v4

V3 = Path("work/rs-1/cycleA/bundles/aia-v3-rawlog-shared-20261108/attacks/AIA/experiment.py")
COMMENTS = ["I turned 34 last week", "moved to a new city", "naïve “quotes” and ünïcode"]
LABELS = ["a", "b", "c"]


def v3_module():
    spec = importlib.util.spec_from_file_location("v3exp_for_test", V3)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not V3.exists(), reason="v3 bundle not present")
def test_baseline_byte_identical_to_v3():
    v3 = v3_module()
    for attr, labels in (("age", None), ("gender", LABELS)):
        assert v4.messages(attr, COMMENTS, labels) == v3.messages(attr, COMMENTS, labels)
        assert v4.messages(attr, COMMENTS, labels, "baseline") == v3.messages(attr, COMMENTS, labels)


def test_persona_and_exemplar_structure():
    base = v4.messages("gender", COMMENTS, LABELS)[0]["content"]
    persona = v4.messages("gender", COMMENTS, LABELS, "persona")[0]["content"]
    fmt = v4.messages("gender", COMMENTS, LABELS, "format_exemplar")[0]["content"]
    assert persona == v4.PERSONA_SENTENCE + "\n\n" + base
    assert fmt == base + "\n\n" + v4.FORMAT_EXEMPLAR_SENTENCE
    with pytest.raises(ValueError):
        v4.messages("gender", COMMENTS, LABELS, "nonsense")


def test_derangement_deterministic_no_fixed_points():
    for n in (2, 3, 10, 55, 80):
        d = v4.donor_derangement(n)
        assert sorted(d) == list(range(n)) and all(d[i] != i for i in range(n))
        assert d == v4.donor_derangement(n)
    with pytest.raises(ValueError):
        v4.donor_derangement(1)


def test_infer_mismatched_scores_both_truths():
    rubric = {"labels": ["x", "y"], "aliases": {}, "point_tolerance": None, "range_iou": None}
    spec = {"attribute": "gender", "bootstrap_repetitions": 200, "seed": 42, "n_eligible": 4, "style": "baseline"}
    profiles = v4.ProfileList([(("0"*64), "x", {"full": ["c0"]}), (("1"*64), "y", {"full": ["c1"]}),
                               (("2"*64), "x", {"full": ["c2"]}), (("3"*64), "y", {"full": ["c3"]})])
    profiles.donor_truth = {("0"*64): "y", ("1"*64): "y", ("2"*64): "y", ("3"*64): "x"}
    raw = []
    out = v4.infer(spec, profiles, rubric, lambda chat: '["x"]', raw_sink=raw.append)
    assert "metrics_vs_donor_truth" in out and out["metrics"] != out["metrics_vs_donor_truth"]
    assert {r["condition"] for r in raw} == {"mismatched_text"}
