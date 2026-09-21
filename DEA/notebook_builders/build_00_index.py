"""Stage 00: build the ground-truth index and the true-prefix pool."""

from __future__ import annotations

from pathlib import Path

from common import HEADER, PREFLIGHT, code, md, write_notebook


def cells() -> list:
    return [
        md("""
# Stage 00: ground-truth index

Builds `V-B intersect V-P`, the index every later stage verifies against.

- **V-B** is the Enron maildir under the message-level filters, with quote and
  forward stripping **disabled**. The Pile's Enron component keeps quoted text,
  so stripping would discard addresses a model could genuinely have memorized.
- **V-P** is the Pile's Enron Emails component, which is the corpus the models
  actually trained on.

Intersecting gives an index where every value is provably in the training data
and bot traffic has still been filtered out.

Nothing here needs a GPU. Expect roughly 40 minutes: about 15 for the Pile
download and 10 for the maildir pass.
"""),
        code(HEADER),
        code(PREFLIGHT),
        code("""
preflight([
    ("maildir", PATHS.maildir),
    ("data dir", PATHS.data),
])
"""),
        md("## 1. The Pile's Enron component\n\nStreams every shard, keeps only "
           "records tagged `Enron Emails`, and caches them. Resumable: a shard "
           "that already finished is skipped."),
        code("""
from dea_a3 import pile

paths_out = pile.extract_all(
    repo=CONFIG.pile_repo,
    n_shards=CONFIG.pile_shards,
    set_name=CONFIG.pile_set_name,
    out_dir=PATHS.pile_cache,
    workers=CONFIG.pile_workers,
)
print(f"shards cached: {len(paths_out)}")
"""),
        code("""
pile_counts, n_pile_docs = pile.build_value_counts(PATHS.pile_cache, kinds=CONFIG.kinds)
for kind in CONFIG.kinds:
    print(f"{kind:6s} distinct in Pile: {len(pile_counts[kind]):,}")
print(f"Pile Enron documents: {n_pile_docs:,}")
"""),
        md("## 2. The maildir side (V-B)\n\nQuote stripping is off. The automated-sender "
           "and broadcast filters stay on."),
        code("""
from dea_a3 import corpus

records, corpus_stats = corpus.load_records(
    PATHS.maildir,
    strip=CONFIG.strip_quoted,
    broadcast_limit=CONFIG.broadcast_recipient_limit,
)
for key in sorted(corpus_stats):
    print(f"{key:32s} {corpus_stats[key]:>8,}")
"""),
        code("""
maildir_values = corpus.collect_values(records, kinds=CONFIG.kinds)
for kind in CONFIG.kinds:
    print(f"{kind:6s} distinct in maildir (V-B): {len(maildir_values[kind]):,}")
"""),
        md("## 3. Coverage and precision\n\nThis reproduces the variant comparison "
           "inside the study, so the index-choice claim rests on an artifact this "
           "pipeline produced rather than on a number quoted from a report."),
        code("""
from dea_a3 import index

variant_table = index.variant_stats(maildir_values, pile_counts, kinds=CONFIG.kinds)
variant_table.to_csv(PATHS.index_dir / "variant_stats.csv", index=False)
variant_table
"""),
        md("## 4. Build and persist the index\n\nOnly salted hashes are written. "
           "No plaintext value reaches disk at any point."),
        code("""
from dea_a3 import pii

salt = pii.load_or_create_salt(PATHS.salt_file)
index_frame = index.build_index(maildir_values, pile_counts, salt, kinds=CONFIG.kinds)
index_frame.to_parquet(PATHS.index_file, index=False)

summary = index.index_summary(index_frame)
summary.to_csv(PATHS.index_dir / "index_summary.csv", index=False)
print(f"index rows: {len(index_frame):,}")
summary
"""),
        code("""
# Guard: assert no plaintext leaked into the artifact.
blob = index_frame.to_csv(index=False)
assert "@" not in blob, "plaintext email reached the index file"
print("no-plaintext check passed")
"""),
        md("## 5. True-prefix pool\n\nThe first twelve words of documents that "
           "contain at least one indexed value. Only prefixes are stored, and a "
           "prefix that itself contains PII is discarded so the attacker's own "
           "input can never be scored as a leak."),
        code("""
import pandas as pd
from dea_a3 import prompts as prompt_mod

known = index.hashes_by_kind(index_frame)
pool = []
for rec in records:
    text = rec.indexed_text()
    found = pii.extract_values(text, kinds=CONFIG.kinds)
    if not any(
        pii.salted_hash(kind, value, salt) in known.get(kind, set())
        for kind in CONFIG.kinds for value in found[kind]
    ):
        continue
    prefix = prompt_mod.make_true_prefix(rec.body, CONFIG.true_prefix_words)
    if prefix is None:
        continue
    if any(pii.extract_values(prefix, kinds=CONFIG.kinds)[k] for k in CONFIG.kinds):
        continue
    pool.append(prefix)

pool = sorted(set(pool))
pd.DataFrame({"prefix": pool}).to_parquet(PATHS.true_prefix_file, index=False)
print(f"true-prefix pool: {len(pool):,}")
"""),
        code("""
assert len(pool) >= 1000, "true-prefix pool is too small; check the maildir path"
for prefix in pool[:3]:
    print(repr(prefix[:70]))
"""),
        md("## 6. Manifest"),
        code("""
from dea_a3 import manifest

manifest.write_manifest(
    PATHS.index_dir, "00_index", CONFIG.to_dict(), ROOT,
    extra={
        "corpus_stats": dict(corpus_stats),
        "pile_documents": n_pile_docs,
        "index_rows": int(len(index_frame)),
        "true_prefix_pool": int(len(pool)),
    },
)
print("stage 00 complete")
"""),
    ]


def build(out_dir: Path) -> Path:
    return write_notebook(cells(), Path(out_dir) / "00_index.ipynb")


if __name__ == "__main__":
    print(build(Path(__file__).resolve().parents[1] / "notebooks"))
