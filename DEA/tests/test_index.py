from collections import Counter

import pandas as pd

from dea_a3 import index, pii

SALT = "s"


def test_intersection_keeps_only_values_in_both_sides():
    maildir = {"email": {"a@x.com", "b@x.com", "c@x.com"}}
    pile = {"email": Counter({"b@x.com": 4, "c@x.com": 9, "d@x.com": 2})}
    frame = index.build_index(maildir, pile, SALT, kinds=("email",))
    assert len(frame) == 2
    got = set(frame["value_hash"])
    assert pii.salted_hash("email", "b@x.com", SALT) in got
    assert pii.salted_hash("email", "a@x.com", SALT) not in got


def test_occurrences_come_from_the_pile_side():
    maildir = {"email": {"a@x.com"}}
    pile = {"email": Counter({"a@x.com": 37})}
    frame = index.build_index(maildir, pile, SALT, kinds=("email",))
    assert frame["occurrences"].iloc[0] == 37


def test_index_holds_no_plaintext():
    maildir = {"email": {"jane.doe@enron.com"}}
    pile = {"email": Counter({"jane.doe@enron.com": 2})}
    frame = index.build_index(maildir, pile, SALT, kinds=("email",))
    blob = frame.to_csv(index=False)
    assert "jane.doe" not in blob and "enron.com" not in blob


def test_empty_intersection_is_an_empty_frame_not_an_error():
    frame = index.build_index({"email": {"a@x.com"}},
                              {"email": Counter({"z@x.com": 1})},
                              SALT, kinds=("email",))
    assert frame.empty
    assert list(frame.columns) == ["kind", "value_hash", "occurrences", "value_length"]


def test_variant_stats_matches_hand_computation():
    maildir = {"email": {"a", "b", "c", "d"}}
    pile = {"email": Counter({"a": 1, "b": 1, "e": 1})}
    stats = index.variant_stats(maildir, pile, kinds=("email",))
    row = stats.iloc[0]
    assert row["intersection"] == 2
    assert abs(row["coverage"] - 2 / 3) < 1e-9
    assert abs(row["precision"] - 2 / 4) < 1e-9


def test_hashes_by_kind_round_trips():
    frame = pd.DataFrame({
        "kind": ["email", "email", "phone"],
        "value_hash": ["h1", "h2", "h3"],
        "occurrences": [1, 2, 3],
        "value_length": [5, 5, 10],
    })
    out = index.hashes_by_kind(frame)
    assert out == {"email": {"h1", "h2"}, "phone": {"h3"}}


def test_summary_counts_per_kind():
    frame = pd.DataFrame({
        "kind": ["email"] * 3 + ["phone"] * 2,
        "value_hash": list("abcde"),
        "occurrences": [1, 5, 9, 2, 4],
        "value_length": [5] * 5,
    })
    s = index.index_summary(frame).set_index("kind")
    assert s.loc["email", "distinct"] == 3
    assert s.loc["phone", "distinct"] == 2
    assert s.loc["email", "occ_max"] == 9
