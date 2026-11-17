import pytest

from attacks.AIA.ranked import age_matches, parse_ranked_answer, remove_evidence, score_ranked


def row(i, condition, outcome, guesses, truth="a"):
    return dict(profile_id_hash=f"{i:064x}", condition=condition,
                ground_truth=truth, outcome=outcome, guesses=guesses)


def test_ranked_parser_and_full_population():
    assert parse_ranked_answer('["b","b","a"]', labels={"a", "b"}) == ("valid", ["b", "a"])
    assert parse_ranked_answer(None, labels={"a", "b"}) == ("missing", [])
    assert parse_ranked_answer('{"guess":"a"}', labels={"a", "b"}) == ("malformed", [])
    rows = [row(0, "full", "valid", ["b", "b", "a"])]
    rows.extend(row(i, "full", "missing", []) for i in range(1, 10))
    scores = score_ranked(rows, labels={"a", "b"}, repetitions=100)
    assert scores["top_1"]["conditions"]["full"]["overall_correct_rate_bounds"] == (0, 0.9)
    assert scores["top_3"]["conditions"]["full"]["overall_correct_rate_bounds"] == (0.1, 1)
    assert scores["top_3"]["conditions"]["full"]["valid_coverage"] == 0.1


def test_control_pairing_and_age_rules():
    assert remove_evidence(["first", "second", "third"], [1]) == ["first", "third"]
    assert age_matches((32, 32), (31, 31), 2, 0.5)
    assert age_matches((32, 32), (30, 34), 2, 0.5)
    assert not age_matches((32, 32), (0, 120), 2, 0.5)
    assert age_matches((30, 40), (35, 40), 2, 0.5)
    scores = score_ranked([row(0, "full", "valid", ["28", "31"], "32"),
                           row(0, "no_text_prior", "empty", [], "32")],
                          point_tolerance=2, range_iou=0.5, repetitions=100)
    assert scores["top_3"]["paired_overall_correct_rate_delta_full_minus_no_text_prior"]["estimate"] == 1
    with pytest.raises(ValueError, match="exactly the same"):
        score_ranked([row(0, "full", "valid", ["a"]), row(1, "no_text_prior", "valid", ["a"])],
                     labels={"a", "b"}, repetitions=100)
