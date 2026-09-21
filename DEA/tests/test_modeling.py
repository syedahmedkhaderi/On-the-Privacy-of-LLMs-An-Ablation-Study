"""Checks on the GPU path that can run without a GPU or torch installed.

The generation call itself cannot be exercised on CPU here, but the keyword
arguments it builds can, and a bad keyword is the failure mode that kills the
first batch of a 37-hour run rather than showing up in a dry run.
"""
import pytest

from dea_a3 import modeling
from dea_a3.config import DecoderSpec


def test_generation_kwargs_are_all_accepted_by_generate():
    kwargs = modeling.build_generation_kwargs(DecoderSpec(), pad_token_id=0)
    assert set(kwargs) <= modeling.GENERATE_ALLOWED_KEYS


def test_generator_is_not_among_them():
    """transformers 4.44.2 generate() has no `generator` parameter.

    Passing one reaches _validate_model_kwargs and raises ValueError on the
    first batch. This test exists because that bug shipped once.
    """
    kwargs = modeling.build_generation_kwargs(DecoderSpec(), pad_token_id=0)
    assert "generator" not in kwargs


def test_decoder_settings_are_carried_through_unchanged():
    kwargs = modeling.build_generation_kwargs(DecoderSpec(), pad_token_id=50256)
    assert kwargs["max_new_tokens"] == 256
    assert kwargs["top_k"] == 40
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 1.0
    assert kwargs["repetition_penalty"] == 1.0
    assert kwargs["pad_token_id"] == 50256
    assert kwargs["eos_token_id"] is None


def test_greedy_omits_the_sampling_arguments():
    kwargs = modeling.build_generation_kwargs(
        DecoderSpec(do_sample=False), pad_token_id=0)
    for key in ("temperature", "top_k", "top_p", "repetition_penalty"):
        assert key not in kwargs


def test_an_unknown_key_is_caught_rather_than_sent_to_generate(monkeypatch):
    monkeypatch.setattr(modeling, "GENERATE_ALLOWED_KEYS", frozenset({"top_k"}))
    with pytest.raises(ValueError, match="would reject"):
        modeling.build_generation_kwargs(DecoderSpec(), pad_token_id=0)


def test_weight_estimate_matches_fp16_arithmetic():
    from dea_a3.config import ModelSpec

    spec = ModelSpec("x", "x/x", "x", 6000)
    assert modeling.estimate_weights_gb(spec) == 12.0


def test_peak_estimate_exceeds_weights_and_grows_with_batch():
    from dea_a3.config import ModelSpec

    spec = ModelSpec("x", "x/x", "x", 1400)
    small = modeling.estimate_peak_gb(spec, batch_size=8)
    large = modeling.estimate_peak_gb(spec, batch_size=64)
    assert small > modeling.estimate_weights_gb(spec)
    assert large > small


def test_headroom_check_passes_without_cuda(monkeypatch):
    from dea_a3.config import ModelSpec

    monkeypatch.setattr(modeling, "vram_free_total_gb", lambda: [])
    ok, detail = modeling.check_vram_headroom(ModelSpec("x", "x/x", "x", 6000), 8)
    assert ok
    assert "nothing to check" in detail


def test_unsharded_model_is_checked_against_one_card_not_the_sum(monkeypatch):
    """Two cards with 8 GB each cannot hold a 12 GB unsharded model."""
    from dea_a3.config import ModelSpec

    monkeypatch.setattr(modeling, "vram_free_total_gb",
                        lambda: [(8.0, 24.0), (8.0, 24.0)])
    unsharded = ModelSpec("j", "x/x", "x", 6000)
    ok, _ = modeling.check_vram_headroom(unsharded, 8)
    assert not ok

    sharded = ModelSpec("j", "x/x", "x", 6000, device_map="auto")
    ok_sharded, _ = modeling.check_vram_headroom(sharded, 8)
    assert ok_sharded


def test_reserve_is_subtracted_from_available(monkeypatch):
    from dea_a3.config import ModelSpec

    monkeypatch.setattr(modeling, "vram_free_total_gb", lambda: [(5.0, 24.0)])
    spec = ModelSpec("small", "x/x", "x", 1400)
    assert modeling.check_vram_headroom(spec, 32, reserve_gb=0.0)[0]
    assert not modeling.check_vram_headroom(spec, 32, reserve_gb=4.0)[0]
