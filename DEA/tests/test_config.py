from dea_a3.config import CONFIG


def test_grid_size_matches_the_plan():
    assert len(CONFIG.models) == 7
    assert len(CONFIG.conditions) == 4
    assert len(CONFIG.seeds) == 3
    assert len(CONFIG.cells()) == 84
    assert CONFIG.prompts_per_cell == 5000


def test_decoder_follows_carlini_2021():
    d = CONFIG.decoder
    assert d.max_new_tokens == 256
    assert d.top_k == 40
    assert d.temperature == 1.0
    assert d.do_sample is True


def test_yield_suppressing_settings_are_disabled():
    """Each of these punishes verbatim recall of memorized text."""
    d = CONFIG.decoder
    assert d.repetition_penalty == 1.0, "a penalty > 1 punishes reciting memorized text"
    assert d.top_p == 1.0, "nucleus on top of top-k narrows the candidate set twice"


def test_index_is_the_variant_p8_selected():
    assert CONFIG.index_variant == "V-B_intersect_V-P"
    assert CONFIG.strip_quoted is False, "the Pile keeps quoted text; stripping loses targets"
    assert CONFIG.extra_headers == (), "X- headers appear in under 1% of Pile records"


def test_model_keys_are_unique_and_cover_both_ladders():
    keys = [m.key for m in CONFIG.models]
    assert len(keys) == len(set(keys))
    families = {m.family for m in CONFIG.models}
    assert {"pythia", "gpt-neo", "gpt-j"} <= families


def test_size_matched_pairs_exist_for_the_architecture_axis():
    by_key = CONFIG.model_map()
    for a, b in (("pythia-160m", "gpt-neo-125m"),
                 ("pythia-1.4b", "gpt-neo-1.3b")):
        ratio = by_key[a].params_m / by_key[b].params_m
        assert 0.7 < ratio < 1.4, f"{a} and {b} are not size-matched"


def test_six_billion_models_are_sharded():
    by_key = CONFIG.model_map()
    assert by_key["gpt-j-6b"].device_map == "auto"
    assert by_key["gpt-j-6b"].revision == "float16"


def test_cells_are_stable_in_order():
    assert CONFIG.cells() == CONFIG.cells()


def test_config_serialises_for_the_manifest():
    d = CONFIG.to_dict()
    assert d["decoder"]["top_k"] == 40
    assert len(d["models"]) == 7
