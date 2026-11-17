"""Maildir loading and the V-B index side.

V-B is DEA's pipeline with quote and forward stripping DISABLED. P8 measured
why: the Pile's Enron component keeps quoted and forwarded text (roughly a
fifth of its records carry explicit markers), so stripping removes addresses
a model trained on the Pile could genuinely have memorized. V-A, which
strips, has 59.5% precision against the Pile; V-B has 75.6%.

The message-level filters are kept. Automated senders and broadcast traffic
are heavily duplicated and trivially extractable, and counting them as
personal contact leakage would overstate the finding.
"""

from __future__ import annotations

import email
import email.policy
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from . import pii

BASE_HEADERS = ("From", "To", "Cc", "Bcc", "Subject")
X_HEADERS = ("X-From", "X-To", "X-cc", "X-bcc")

AUTOMATED_SENDERS = (
    "mailer-daemon", "postmaster", "no-reply", "noreply", "donotreply",
    "outlook.team", "announcements", "notification",
)
AUTOMATED_SUBJECTS = (
    "undeliverable", "delivery status notification", "out of office",
    "automatic reply", "returned mail",
)
BROADCAST_TOKENS = (
    "all.worldwide", "all.houston", "all_ena", "all.enron", "everyone",
    "allenron", "dl-ga-all", "enron.announcements",
)

QUOTE_MARKERS = (
    "-----original message-----",
    "----- forwarded by",
    "---------------------- forwarded",
    "-----forwarded by",
    "> -----original message-----",
)
WROTE_RE = re.compile(r"^\s*On .{0,80}\bwrote:\s*$", re.I)
UNDERSCORE_RULE_RE = re.compile(r"^_{10,}$")
ATTACHMENT_RE = re.compile(
    r"^\s*(?:<<.+?>>|\[IMAGE\]|- .+\.(?:doc|xls|pdf|ppt|zip|jpg|gif)\s*)$", re.I
)
_WS_RE = re.compile(r"[ \t]+")


@dataclass
class Record:
    path: str
    sender: str
    recipients: tuple[str, ...]
    subject: str
    body: str

    def indexed_text(self, extra_headers_text: str = "") -> str:
        parts = [self.sender, " ".join(self.recipients), self.subject]
        if extra_headers_text:
            parts.append(extra_headers_text)
        parts.append(self.body)
        return "\n".join(p for p in parts if p)


def iter_message_paths(root: Path) -> list[Path]:
    root = Path(root)
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() not in (".md", ".zip")
    )


def strip_quoted(body: str) -> str:
    """V-A's cleaning. Retained so the variant can be reproduced, not used."""
    lines: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        low = line.strip().lower()
        if any(low.startswith(marker) for marker in QUOTE_MARKERS):
            break
        if WROTE_RE.match(line) or UNDERSCORE_RULE_RE.match(line.strip()):
            break
        if line.lstrip().startswith(">"):
            continue
        if ATTACHMENT_RE.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def normalize_body(body: str) -> str:
    return "\n".join(_WS_RE.sub(" ", ln).rstrip() for ln in body.splitlines()).strip()


def _header(msg, name: str) -> str:
    try:
        value = msg.get(name, "")
    except Exception:
        return ""
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _split_addresses(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_message(path: Path) -> tuple[Record | None, str]:
    """Parse one maildir file. Returns (record, drop_reason)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None, "unreadable"
    msg = None
    for policy in (email.policy.default, email.policy.compat32):
        try:
            msg = email.message_from_bytes(raw, policy=policy)
            break
        except Exception:
            continue
    if msg is None:
        return None, "parse_error"

    body_parts: list[str] = []
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() != "text/plain":
                    continue
                if part.get_filename():
                    continue
                try:
                    payload = part.get_payload(decode=True)
                except Exception:
                    continue
                if payload:
                    body_parts.append(payload.decode("utf-8", errors="replace"))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body_parts.append(payload.decode("utf-8", errors="replace"))
    except Exception:
        return None, "content_error"

    return Record(
        path=str(path),
        sender=_header(msg, "From"),
        recipients=_split_addresses(_header(msg, "To")) + _split_addresses(_header(msg, "Cc")),
        subject=_header(msg, "Subject"),
        body="\n".join(body_parts),
    ), ""


def is_automated(sender: str, subject: str) -> bool:
    s, subj = sender.lower(), subject.lower()
    return (any(t in s for t in AUTOMATED_SENDERS)
            or any(t in subj for t in AUTOMATED_SUBJECTS))


def is_broadcast(recipients: Iterable[str], limit: int = 25) -> bool:
    recipients = list(recipients)
    if len(recipients) > limit:
        return True
    joined = " ".join(recipients).lower()
    return any(t in joined for t in BROADCAST_TOKENS)


def load_records(root: Path, strip: bool = False, limit: int | None = None,
                 broadcast_limit: int = 25) -> tuple[list[Record], Counter]:
    """Walk the maildir and return kept records plus drop statistics."""
    stats: Counter = Counter()
    kept: list[Record] = []
    paths = iter_message_paths(root)
    stats["files_seen"] = len(paths)
    if limit is not None:
        paths = paths[:limit]
        stats["files_considered"] = len(paths)
    for path in paths:
        rec, reason = parse_message(path)
        if rec is None:
            stats[f"dropped_{reason}"] += 1
            continue
        body = strip_quoted(rec.body) if strip else rec.body
        rec.body = normalize_body(body)
        if not rec.body.strip():
            stats["dropped_empty_after_clean"] += 1
            continue
        if is_automated(rec.sender, rec.subject):
            stats["dropped_automated"] += 1
            continue
        if is_broadcast(rec.recipients, broadcast_limit):
            stats["dropped_broadcast"] += 1
            continue
        kept.append(rec)
    stats["records_after_filtering"] = len(kept)
    return kept, stats


def iter_indexed_text(records: Iterable[Record]) -> Iterator[str]:
    for rec in records:
        yield rec.indexed_text()


def collect_values(records: Iterable[Record],
                   kinds: tuple[str, ...] = ("email", "phone")) -> dict[str, set[str]]:
    """Distinct normalised values across the corpus."""
    out: dict[str, set[str]] = {k: set() for k in kinds}
    for text in iter_indexed_text(records):
        found = pii.extract_values(text, kinds=kinds)
        for kind in kinds:
            out[kind].update(found[kind])
    return out
