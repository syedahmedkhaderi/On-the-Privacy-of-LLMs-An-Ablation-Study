"""Study configuration.

Plain Python, no YAML. Import CONFIG and read it; do not mutate it at run
time. Every value that affects a result is recorded into each run's manifest
by manifest.py, so a result can be traced back to the configuration that
produced it.

Decoder settings follow Carlini et al. 2021 exactly: 256 tokens sampled with
top-k = 40 at temperature 1.0. Three settings are deliberately NOT used,
because each suppresses verbatim recall of memorized text and would depress
the yield for reasons unrelated to memorization:

  repetition_penalty  left at 1.0. A penalty above 1.0 punishes the model for
                      repeating tokens, which is exactly what reciting a
                      memorized string requires.
  top_p               left at 1.0. Combining nucleus with top-k narrows the
                      candidate set twice.
  no_repeat_ngram     unset. It forbids the literal repetition memorization
                      consists of.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    family: str
    params_m: int
    revision: str | None = None
    device_map: str | None = None
    batch_size: int = 32


@dataclass(frozen=True)
class DecoderSpec:
    name: str = "top_k_40"
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 1.0
    repetition_penalty: float = 1.0


@dataclass(frozen=True)
class Config:
    study: str = "dea_a3"

    # Grid
    seeds: tuple[int, ...] = (42, 43, 44)
    prompts_per_cell: int = 5_000
    conditions: tuple[str, ...] = (
        "unconditional",
        "generic_prefix",
        "pii_leading",
        "true_prefix",
    )

    models: tuple[ModelSpec, ...] = (
        ModelSpec("pythia-70m", "EleutherAI/pythia-70m", "pythia", 70, batch_size=64),
        ModelSpec("pythia-160m", "EleutherAI/pythia-160m", "pythia", 160, batch_size=64),
        ModelSpec("pythia-410m", "EleutherAI/pythia-410m", "pythia", 410, batch_size=48),
        ModelSpec("pythia-1.4b", "EleutherAI/pythia-1.4b", "pythia", 1400, batch_size=32),
        ModelSpec("gpt-neo-125m", "EleutherAI/gpt-neo-125M", "gpt-neo", 125, batch_size=64),
        ModelSpec("gpt-neo-1.3b", "EleutherAI/gpt-neo-1.3B", "gpt-neo", 1300, batch_size=32),
        ModelSpec("gpt-j-6b", "EleutherAI/gpt-j-6B", "gpt-j", 6000,
                  revision="float16", device_map="auto", batch_size=8),
    )

    decoder: DecoderSpec = field(default_factory=DecoderSpec)

    # Index
    index_variant: str = "V-B_intersect_V-P"
    strip_quoted: bool = False        # V-B: quote stripping OFF
    extra_headers: tuple[str, ...] = ()   # V-C's X- headers excluded, per D18
    kinds: tuple[str, ...] = ("email", "phone")
    broadcast_recipient_limit: int = 25

    # Pile extraction
    pile_repo: str = "monology/pile-uncopyrighted"
    pile_set_name: str = "Enron Emails"
    pile_shards: int = 30
    pile_workers: int = 6

    # Prompt construction
    generic_prefix_tokens: int = 21
    true_prefix_words: int = 12
    pii_leading_variants: int = 100

    # Duplication analysis
    band_quantiles: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0)

    # Statistics
    ci_level: float = 0.95

    def model_map(self) -> dict[str, ModelSpec]:
        return {m.key: m for m in self.models}

    def cells(self) -> list[tuple[str, str, int]]:
        """Every (model, condition, seed) triple, in a stable order."""
        out = []
        for m in self.models:
            for cond in self.conditions:
                for seed in self.seeds:
                    out.append((m.key, cond, seed))
        return out

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["models"] = [asdict(m) for m in self.models]
        d["decoder"] = asdict(self.decoder)
        return d


CONFIG = Config()
