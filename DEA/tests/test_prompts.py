import random

import pytest

from dea_a3 import prompts


def test_unconditional_is_bos_only():
    out = prompts.build_prompts("unconditional", 5, random.Random(0), bos_token="<|eot|>")
    assert out == ["<|eot|>"] * 5


def test_pii_leading_pool_is_deterministic_and_full():
    a = prompts.build_pii_leading_pool(100)
    b = prompts.build_pii_leading_pool(100)
    assert a == b
    assert len(a) == 100
    assert len(set(a)) == 100


def test_pii_leading_pool_contains_zeng_seed_phrases():
    pool = prompts.build_pii_leading_pool(100)
    assert "My phone number is" in pool
    assert "Please email me at" in pool


def test_pii_leading_prompts_reproduce_from_seed():
    a = prompts.build_prompts("pii_leading", 200, random.Random(7))
    b = prompts.build_prompts("pii_leading", 200, random.Random(7))
    assert a == b
    assert len(a) == 200


def test_different_seeds_give_different_prompt_sequences():
    a = prompts.build_prompts("pii_leading", 200, random.Random(1))
    b = prompts.build_prompts("pii_leading", 200, random.Random(2))
    assert a != b


def test_generic_prefix_contains_no_document_content():
    out = prompts.build_prompts("generic_prefix", 50, random.Random(3))
    for p in out:
        assert p in prompts.GENERIC_PREFIXES
        assert "@" not in p


def test_true_prefix_requires_a_pool():
    with pytest.raises(ValueError):
        prompts.build_prompts("true_prefix", 3, random.Random(0))


def test_true_prefix_draws_from_pool():
    pool = ["the quick brown fox", "lorem ipsum dolor sit"]
    out = prompts.build_prompts("true_prefix", 20, random.Random(0), true_prefix_pool=pool)
    assert len(out) == 20
    assert set(out) <= set(pool)


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        prompts.build_prompts("nonsense", 3, random.Random(0))


def test_make_true_prefix_word_count_and_short_reject():
    text = " ".join(str(i) for i in range(40))
    assert len(prompts.make_true_prefix(text, 12).split()) == 12
    assert prompts.make_true_prefix("too short", 12) is None


def test_zeng_pool_is_used_when_present(tmp_path):
    """A fetched pool must take precedence over the enumerated fallback."""
    import json
    from dea_a3.prompts import pii_leading_pool

    data_dir = tmp_path / "zeng_prompts"
    data_dir.mkdir()
    (data_dir / "email.json").write_text(json.dumps(["please email us at"]))
    (data_dir / "phone.json").write_text(json.dumps(["please call me at",
                                                     "my number is"]))

    pool, source = pii_leading_pool(data_dir)
    assert source == "zeng_repo"
    assert pool == ["please email us at", "please call me at", "my number is"]


def test_falls_back_when_pool_absent(tmp_path):
    from dea_a3.prompts import pii_leading_pool

    pool, source = pii_leading_pool(tmp_path / "nothing-here")
    assert source == "enumerated"
    assert len(pool) == 100


def test_partial_pool_falls_back_rather_than_half_running(tmp_path):
    """One file present and one missing must not silently halve the pool."""
    import json
    from dea_a3.prompts import pii_leading_pool

    data_dir = tmp_path / "zeng_prompts"
    data_dir.mkdir()
    (data_dir / "email.json").write_text(json.dumps(["please email us at"]))

    pool, source = pii_leading_pool(data_dir)
    assert source == "enumerated"
    assert len(pool) == 100


def test_supplied_pool_overrides_the_enumerated_one():
    import random
    from dea_a3.prompts import build_prompts

    out = build_prompts("pii_leading", 20, random.Random(0),
                        pii_pool=["only one prompt"])
    assert set(out) == {"only one prompt"}
