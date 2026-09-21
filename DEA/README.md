# DEA

## dea_a3: corpus-verified reproduction study

### What the study is

`dea_a3` measures how much Enron personal data seven open language models
reproduce verbatim when prompted, and checks every extracted value against the
training corpus rather than trusting a regex. The grid is 7 models
(pythia-70m, pythia-160m, pythia-410m, pythia-1.4b, gpt-neo-125m, gpt-neo-1.3b,
gpt-j-6b) x 4 prompt styles (unconditional, generic_prefix, pii_leading,
true_prefix) x 3 seeds (42, 43, 44), with 5,000 prompts per run: 84 runs in
total. The last run completed 2026-09-17 22:13 UTC. Decoding was top-k 40,
temperature 1.0, 256 new tokens.

A value counts as *extracted* when the regex detector finds it in a generation,
and as *verified* only when its salted hash also appears in the corpus index
built from the Pile's Enron Emails subset. The gap between those two columns is
the point of the study: across the conditioned styles only 2 to 26 percent of
what looks like PII is PII the model actually saw.

### Which styles the paper reports

The paper reports three styles: `generic_prefix`, `pii_leading` and
`true_prefix`. The `unconditional` runs are kept in `results/` for
completeness, but they are not reported and they do not appear in any figure.
Every figure in `figs/` shows three styles only.

### Layout

```
src/                dea_a3 package: config, corpus index, prompts, runner,
                    detection, aggregation
tests/              pytest suite (89 tests, no GPU or network needed)
notebook_builders/  scripts that generate the three stage notebooks
notebooks/          the generated notebooks
tools/              study driver, environment checks, figure generation
results/00_index/   corpus index (hashes and occurrence counts)
results/01_generations/  one directory per run: generations.parquet,
                    detections.parquet, vram.json, COMPLETE (wall clock, s)
results/02_analysis/     tables 1-6 and the stage 02 duplication figure
results/executed/   the three notebooks as executed
figs/               the paper figures, three styles
```

No stored file contains a plaintext PII value. `index.parquet`,
`detections.parquet` and every table hold salted SHA-256 hashes, occurrence
counts and lengths only.

### Running the tests

```
python -m pytest tests/ -q          # 89 passed
```

### Regenerating the figures

```
python tools/make_a3_figs_3styles.py                 # reads ./results, writes ./figs
python tools/make_a3_figs_3styles.py --data-root /path/to/dea_a3
```

The script recounts one quantity and reads everything else. Band edges,
`in corpus` counts and the per-style verification rates come from
`results/02_analysis/`. The `recovered` bars have to be recounted, because
stage 02 pooled all four styles into one hash set; the recount uses the same
`aggregate.recovery_by_band` (`src/dea_a3/aggregate.py:132`) and the same band
quantiles, narrowed to the three reported styles. Before writing anything the
script re-runs itself over all four styles and stops unless it reproduces
`table4_duplication.csv` row for row and the stored totals of 1,846 distinct
verified emails and 1,092 phones.

### Library versions

python 3.10.17, torch 2.4.1+cu121, transformers 4.44.2, numpy 1.26.4,
scipy 1.13.1, pandas 2.2.2, pyarrow 17.0.0, accelerate 0.33.0,
matplotlib 3.8.4. The grid ran on NVIDIA L4 cards.

### Provenance

The study directory was not under version control while the grid ran, so every
stage manifest records `git.commit = None` and `git.dirty = false`. There is no
commit pinning the code that produced these results. This commit is the first
versioned copy of that code; it is byte-identical to what ran, but the link is
asserted here rather than proved by the manifests.

The grid also ran as three concurrent processes across two cards, so the stage
01 manifest's `cells_complete` reflects whichever process wrote last. The count
of 84 is correct, and is independently confirmed by the 84 `COMPLETE` markers.

### What is not published, and why

| Path | Size | Why |
| --- | --- | --- |
| `secrets/pii_salt.txt` | 65 B | The salt. Publishing it would make every stored hash reversible by dictionary attack. |
| `results/00_index/true_prefixes.parquet` | 6.7 MB | Plaintext corpus text, used to build the `true_prefix` prompts. |
| `data/` | 1.7 GB | The Enron maildir and the Pile shards. Third-party corpora, redistributed by their own sources. |

No published file exceeds 95 MB; the largest is
`results/00_index/index.parquet` at 7.3 MB.

`tools/fetch_zeng_prompts.py` and the stage 00 notebook rebuild the index and
the prefixes from the corpora, given a salt of your own. A different salt
produces different hashes, so the published hashes cannot be joined against a
rebuilt index.
