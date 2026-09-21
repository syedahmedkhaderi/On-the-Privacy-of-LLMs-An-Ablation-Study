"""End-to-end smoke test on synthetic data. No GPU, no network, no downloads.

Builds a miniature maildir and a miniature Pile cache, runs all three stages,
and checks the artifacts. Run this on the VM before starting the real study:
if it passes, the wiring is sound and only scale and weights remain.

Usage:  python tools/smoke_test.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from dea_a3 import aggregate, corpus, detect, index, modeling, pii, runner  # noqa: E402
from dea_a3.config import Config, ModelSpec  # noqa: E402
from dea_a3.paths import Paths  # noqa: E402

OPENINGS = (
    "Following up on the meeting yesterday about the schedule for",
    "Please find below the revised numbers we discussed regarding",
    "As agreed on the call this morning I am sending over",
    "Quick update before the deadline tomorrow concerning the open",
    "Thanks for your patience while we worked through the detail of",
)

REAL = [f"user{i:03d}@enron.com" for i in range(60)]
MAILDIR_ONLY = [f"ghost{i:03d}@enron.com" for i in range(20)]
PILE_ONLY = [f"pileonly{i:03d}@enron.com" for i in range(15)]
PHONES = [f"713853{i:04d}" for i in range(30)]


def make_maildir(root: Path) -> None:
    """A synthetic maildir including traffic the filters should remove."""
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    n = 0
    for i, addr in enumerate(REAL + MAILDIR_ONLY):
        box = root / f"user{i % 5}" / "sent"
        box.mkdir(parents=True, exist_ok=True)
        phone = PHONES[i % len(PHONES)]
        pretty = f"{phone[:3]}-{phone[3:6]}-{phone[6:]}"
        # The opening words carry no PII, matching real messages: a prefix
        # that already contains the answer would make the true-prefix
        # condition trivial, so stage 00 discards those.
        body = (f"Message-ID: <{i}>\nFrom: sender{i}@enron.com\n"
                f"To: recipient{i}@enron.com\nSubject: Notes for {i}\n\n"
                f"{OPENINGS[i % len(OPENINGS)]} item {i} and the remaining "
                f"points we agreed to circulate before the end of the week.\n"
                f"Please contact {addr} or call {pretty}.\n"
                f"-----Original Message-----\n"
                f"From: quoted{i}@enron.com\nQuoted body {rng.random()}\n")
        (box / f"{i}").write_text(body, encoding="utf-8")
        n += 1
    # Traffic the message-level filters must drop.
    drop = root / "user9" / "inbox"
    drop.mkdir(parents=True, exist_ok=True)
    (drop / "1").write_text(
        "From: mailer-daemon@enron.com\nTo: a@enron.com\nSubject: Undeliverable\n\n"
        "bounced dropme@enron.com\n", encoding="utf-8")
    (drop / "2").write_text(
        "From: x@enron.com\nTo: all.worldwide@enron.com\nSubject: Notice\n\n"
        "broadcast blast@enron.com\n", encoding="utf-8")
    print(f"  synthetic maildir: {n} messages plus 2 that must be filtered")


def make_pile_cache(cache: Path) -> None:
    """A synthetic Pile cache with its own duplication structure."""
    cache.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, addr in enumerate(REAL):
        repeats = 1 + (i % 12) ** 2  # skewed, and heavily tied at the low end
        for _ in range(repeats):
            phone = PHONES[i % len(PHONES)]
            lines.append(json.dumps({
                "text": f"reach {addr} on {phone[:3]}.{phone[3:6]}.{phone[6:]} thanks"
            }))
    for addr in PILE_ONLY:
        lines.append(json.dumps({"text": f"only in the pile {addr} here"}))
    (cache / "enron_00.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  synthetic pile cache: {len(lines)} documents")


# Weight recovery toward heavily duplicated values, which is the effect the
# duplication analysis exists to measure. A stub that picks uniformly would
# make every band recover at the same rate and the gradient would go untested.
_WEIGHTS = [1 + (i % 12) ** 2 for i in range(len(REAL))]


def stub_generate(loaded, prompts, decoder, seed):
    rng = random.Random(seed)
    out = []
    for prompt in prompts:
        roll = rng.random()
        if roll < 0.40:
            picked = rng.choices(REAL, weights=_WEIGHTS, k=1)[0]
            tail = f" you can reach them at {picked} directly"
        elif roll < 0.55:
            tail = f" try {rng.choice(MAILDIR_ONLY)} instead"
        elif roll < 0.70:
            tail = f" or invented{rng.randint(0, 999)}@example.org perhaps"
        else:
            tail = " no contact information appears in this continuation"
        out.append(prompt + tail)
    return out


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="dea_a3_smoke_"))
    try:
        print("stage 00: index")
        paths = Paths(tmp)
        paths.ensure()
        make_maildir(paths.maildir)
        make_pile_cache(paths.pile_cache)

        cfg = Config(
            seeds=(42, 43, 44),
            prompts_per_cell=120,
            conditions=("unconditional", "generic_prefix", "pii_leading", "true_prefix"),
            models=(ModelSpec("stub-a", "s/a", "pythia", 70, batch_size=32),
                    ModelSpec("stub-b", "s/b", "gpt-neo", 125, batch_size=32)),
        )

        from dea_a3 import pile
        pile_counts, n_docs = pile.build_value_counts(paths.pile_cache, kinds=cfg.kinds)
        records, stats = corpus.load_records(paths.maildir, strip=cfg.strip_quoted)
        assert stats["dropped_automated"] == 1, stats
        assert stats["dropped_broadcast"] == 1, stats
        maildir_values = corpus.collect_values(records, kinds=cfg.kinds)

        salt = pii.load_or_create_salt(paths.salt_file)
        idx = index.build_index(maildir_values, pile_counts, salt, kinds=cfg.kinds)
        idx.to_parquet(paths.index_file, index=False)
        blob = idx.to_csv(index=False)
        assert "@" not in blob, "plaintext reached the index"

        vstats = index.variant_stats(maildir_values, pile_counts, kinds=cfg.kinds)
        email_row = vstats[vstats["kind"] == "email"].iloc[0]
        assert 0 < email_row["precision"] < 1, "precision should be strictly between 0 and 1"
        assert 0 < email_row["coverage"] < 1, "coverage should be strictly between 0 and 1"
        print(f"  index rows {len(idx):,}, pile docs {n_docs:,}, "
              f"email precision {email_row['precision']:.3f}, "
              f"coverage {email_row['coverage']:.3f}")

        from dea_a3 import prompts as prompt_mod
        known = index.hashes_by_kind(idx)
        pool = []
        for rec in records:
            prefix = prompt_mod.make_true_prefix(rec.body, cfg.true_prefix_words)
            if prefix and not any(
                pii.extract_values(prefix, kinds=cfg.kinds)[k] for k in cfg.kinds
            ):
                pool.append(prefix)
        pool = sorted(set(pool))
        pd.DataFrame({"prefix": pool}).to_parquet(paths.true_prefix_file, index=False)
        assert len(pool) >= 5, f"true-prefix pool too small: {len(pool)}"
        print(f"  true-prefix pool {len(pool):,}")

        print("stage 01: generation")
        modeling.generate_batch = stub_generate
        loaded = modeling.LoadedModel("stub", object(), _Tok(), "cpu")
        for spec in cfg.models:
            for cond in cfg.conditions:
                for seed in cfg.seeds:
                    runner.run_cell(loaded, spec.key, cond, seed, cfg, paths,
                                    known, salt, true_prefix_pool=pool,
                                    progress=False)
        summaries = runner.load_cell_summaries(paths)
        expected = len(cfg.cells())
        assert len(summaries) == expected, f"{len(summaries)} != {expected}"
        print(f"  cells complete {len(summaries)}/{expected}")

        print("stage 02: analysis")
        value_cols = ["extracted_email", "extracted_phone",
                      "verified_email", "verified_phone"]
        grid = aggregate.complete_grid(summaries, [m.key for m in cfg.models],
                                       cfg.conditions, cfg.seeds, value_cols)
        grid["extracted_total"] = grid["extracted_email"] + grid["extracted_phone"]
        grid["verified_total"] = grid["verified_email"] + grid["verified_phone"]
        assert len(grid) == expected

        t1 = aggregate.aggregate_cells(grid, ["condition", "model"],
                                       value_cols + ["extracted_total", "verified_total"])
        assert len(t1) == len(cfg.conditions) * len(cfg.models)
        assert t1["verified_total_n"].eq(3).all(), "every group needs three seeds"
        assert t1["verified_total_ci"].notna().all(), "CI must be computable"

        assert (grid["extracted_total"] >= grid["verified_total"]).all(), \
            "verified can never exceed extracted"

        detections = runner.load_detections(paths, verified_only=True)
        recovered = set(detections["value_hash"].unique())
        t4 = aggregate.recovery_by_band(idx, recovered, cfg.band_quantiles)
        assert not t4.empty, "duplication table is empty"
        assert len(t4[t4["kind"] == "email"]) >= 3, "too few duplication bands"
        assert t4["in_corpus"].sum() == len(idx), "bands must cover the whole index"
        email_bands = t4[t4["kind"] == "email"].reset_index(drop=True)
        assert email_bands["recovery_rate"].iloc[0] < \
               email_bands["recovery_rate"].iloc[-1], \
               "recovery must rise with duplication in the fixture"
        print(f"  tables built, {len(t4)} duplication bands, "
              f"{len(recovered):,} distinct values recovered")

        for _, row in t4[t4["kind"] == "email"].iterrows():
            print(f"    band {row['band']:>10s}  n={int(row['in_corpus']):4d}  "
                  f"rate={row['recovery_rate']:.3f}")

        print("\nSMOKE TEST PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class _Tok:
    bos_token = "<|endoftext|>"
    eos_token = "<|endoftext|>"
    pad_token = "<|endoftext|>"
    pad_token_id = 0


if __name__ == "__main__":
    raise SystemExit(main())
