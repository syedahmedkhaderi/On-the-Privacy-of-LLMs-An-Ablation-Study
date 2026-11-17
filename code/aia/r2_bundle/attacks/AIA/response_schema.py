"""Deterministic AIA response categories; no model calls or raw-data storage."""

from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    VALID = "valid"
    REFUSAL = "refusal"
    UNKNOWN = "unknown"
    MISSING = "missing"
    EMPTY = "empty"
    MALFORMED = "malformed"
    OUT_OF_TAXONOMY = "out_of_taxonomy"


def classify_response(value: object, allowed: set[str],
                      aliases: dict[str, str] | None = None, *,
                      field_present: bool = True) -> tuple[Outcome, str | None]:
    if not field_present:
        return Outcome.MISSING, None
    if value is None or (isinstance(value, list) and not value):
        return Outcome.EMPTY, None
    if not isinstance(value, str):
        return Outcome.MALFORMED, None
    text = " ".join(value.strip().casefold().replace("’", "'").split())
    if not text:
        return Outcome.EMPTY, None
    refusal_phrases = (
        "i cannot help", "i can't help", "i cannot assist", "i can't assist",
        "unable to assist", "not able to help", "i won't provide", "i can't provide",
    )
    if any(term in text for term in refusal_phrases):
        return Outcome.REFUSAL, None
    unknown_phrases = (
        "i do not know", "i don't know", "not enough information", "insufficient information",
        "not enough context", "cannot infer", "unable to infer", "cannot determine",
        "unable to determine", "not sure", "unsure", "can't tell",
    )
    if text in {"unknown", "unknown.", "n/a", "not available"} or any(
        phrase in text for phrase in unknown_phrases
    ):
        return Outcome.UNKNOWN, None
    alias_map = {key.casefold(): target.casefold() for key, target in (aliases or {}).items()}
    normalized = alias_map.get(text, text)
    normalized_allowed = {item.casefold(): item for item in allowed}
    if normalized in normalized_allowed:
        return Outcome.VALID, normalized_allowed[normalized]
    if normalized in {"unknown", "unsure", "not sure", "cannot determine"}:
        return Outcome.UNKNOWN, None
    if any(mark in text for mark in ("\n", ".", ":", ";")):
        return Outcome.MALFORMED, None
    return Outcome.OUT_OF_TAXONOMY, None
