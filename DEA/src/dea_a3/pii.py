"""PII patterns, normalisation, and salted hashing.

Carried forward from rs-1/src/dea/pii.py with the same validity conditions,
the same locked extraction order, and the same salted-hash storage. The
patterns are unchanged because P8 showed they contribute nothing to the
index gap: substituting CRELT's regex moved the count by 18 in 14,000.

No plaintext PII value is ever written to disk by this module. Values live
in memory during extraction and leave as salted SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

KINDS = ("email", "phone", "ssn", "card")

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
CARD_RE = re.compile(r"(?<![\d\-])(?:\d[ \-]?){14,15}\d(?![\d\-])")
PHONE_RE = re.compile(
    r"(?<![\d\-])(?:\+?1[ .\-]?)?(?:\(([2-9]\d{2})\)|([2-9]\d{2}))[ .\-]?"
    r"(\d{3})[ .\-]?(\d{4})(?![\d\-])"
)

_DIGITS_RE = re.compile(r"\D")


@dataclass(frozen=True)
class Span:
    """One extracted PII occurrence."""

    kind: str
    raw: str
    normalized: str
    start: int
    end: int


def luhn_valid(digits: str) -> bool:
    """Standard Luhn checksum over a digit string."""
    if not digits.isdigit():
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def normalize_value(kind: str, value: str) -> str:
    """Canonical form used for both indexing and matching."""
    if kind == "email":
        return value.strip().lower()
    return _DIGITS_RE.sub("", value)


def _phone_digits(match: re.Match) -> str | None:
    area = match.group(1) or match.group(2)
    if area is None:
        return None
    digits = f"{area}{match.group(3)}{match.group(4)}"
    if len(digits) != 10:
        return None
    return digits


def extract_spans(text: str) -> list[Span]:
    """Extract PII spans in a locked order: email, ssn, card, then phone.

    Earlier spans block later ones over the same characters, so a card number
    is never re-counted as a telephone number. This ordering is load-bearing
    and must not be changed without rebuilding every index.
    """
    if not text:
        return []

    spans: list[Span] = []
    claimed: list[tuple[int, int]] = []

    def _overlaps(a: int, b: int) -> bool:
        return any(not (b <= s or a >= e) for s, e in claimed)

    for m in EMAIL_RE.finditer(text):
        raw = m.group(0)
        spans.append(Span("email", raw, normalize_value("email", raw), m.start(), m.end()))
        claimed.append((m.start(), m.end()))

    for m in SSN_RE.finditer(text):
        if _overlaps(m.start(), m.end()):
            continue
        raw = m.group(0)
        spans.append(Span("ssn", raw, normalize_value("ssn", raw), m.start(), m.end()))
        claimed.append((m.start(), m.end()))

    for m in CARD_RE.finditer(text):
        if _overlaps(m.start(), m.end()):
            continue
        digits = _DIGITS_RE.sub("", m.group(0))
        if len(digits) not in (15, 16) or not luhn_valid(digits):
            continue
        spans.append(Span("card", m.group(0), digits, m.start(), m.end()))
        claimed.append((m.start(), m.end()))

    for m in PHONE_RE.finditer(text):
        if _overlaps(m.start(), m.end()):
            continue
        digits = _phone_digits(m)
        if digits is None:
            continue
        spans.append(Span("phone", m.group(0), digits, m.start(), m.end()))
        claimed.append((m.start(), m.end()))

    spans.sort(key=lambda s: (s.start, s.kind))
    return spans


def extract_values(text: str, kinds: tuple[str, ...] = KINDS) -> dict[str, set[str]]:
    """Normalised distinct values per kind found in one string."""
    out: dict[str, set[str]] = {k: set() for k in kinds}
    for span in extract_spans(text):
        if span.kind in out:
            out[span.kind].add(span.normalized)
    return out


def load_or_create_salt(path: Path) -> str:
    """Read the persisted salt, creating one on first use.

    The salt file must never be committed. Every index and every detection
    result in a study must share one salt or hashes will not compare.
    """
    path = Path(path)
    if path.exists():
        salt = path.read_text(encoding="utf-8").strip()
        if salt:
            return salt
    salt = secrets.token_hex(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(salt + "\n", encoding="utf-8")
    path.chmod(0o600)
    return salt


def salted_hash(kind: str, value: str, salt: str) -> str:
    """Typed, salted SHA-256 of a normalised value."""
    payload = f"{salt}|{kind}|{normalize_value(kind, value)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mask(kind: str, value: str) -> str:
    """Redact a value for logging. Never log an unmasked value."""
    if kind == "email":
        local, _, domain = value.partition("@")
        keep = local[:1] if local else ""
        return f"{keep}{'*' * max(len(local) - 1, 0)}@{domain}"
    digits = normalize_value(kind, value)
    if len(digits) <= 3:
        return "X" * len(digits)
    return digits[:3] + "X" * (len(digits) - 3)
