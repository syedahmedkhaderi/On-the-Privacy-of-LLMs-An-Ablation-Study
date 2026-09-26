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

### Paper to file map

The paper uses the 63 three-style runs: 7 models x 3 styles x 3 seeds. The 21
`unconditional` runs are in `results/01_generations/` but no paper number is
computed from them. Items are listed by caption title, because table numbers
change between drafts.

| Paper item (caption title) | File |
| --- | --- |
| Study design | `src/dea_a3/config.py` (models, styles, seeds, decoder); each run's `results/01_generations/<run>/cell_config.json` |
| Reference setting (GPT-Neo-1.3B, `pii_leading`, against Zeng et al.) | `results/02_analysis/table5_zeng_comparison.csv`, `distinct_*` columns; the same numbers are the `gpt-neo-1.3b`, `pii_leading` row of `paper_per_run_3styles.csv` |
| Verified per run | `results/02_analysis/paper_per_run_3styles.csv`, `verified_email_*` and `verified_phone_*` columns |
| Repetition (duplication bands) | `results/02_analysis/table_duplication_3styles.csv` |
| Scale figure | `figs/a3_scale.pdf` and `figs/a3_scale.png` |
| Extracted per-run counts | `results/02_analysis/paper_per_run_3styles.csv`, `extracted_email_*` and `extracted_phone_*` columns |

Counts are distinct values per run. Each mean is over the three seeds and each
`_ci` is the half-width of the 95% t interval with 2 degrees of freedom. The
per-seed values behind every mean are in
`results/02_analysis/table6_distinct_by_cell.csv`.

Duplication bands are lower-inclusive and upper-exclusive: the band with
`lower_inclusive` 2 and `upper_exclusive` 4 holds values seen 2 or 3 times. The
last band of each kind is closed, so its `upper_exclusive` is the largest
occurrence count plus one. A value counts as recovered if it is verified in at
least one of the 63 runs, and it is counted once.

`results/02_analysis/table4_duplication.csv` is the stage 02 duplication table.
It pools all four styles, including `unconditional`, and is not used in the
paper; `table_duplication_3styles.csv` replaces it. Tables 1 to 3 are not used
in the paper either: they cover all four styles and are built from occurrence
counts (per-generation counts summed over a run, as in
`results/01_generations/cell_summaries.csv`), not from distinct values per run.

All of the files above except `table5_zeng_comparison.csv` and `table6_distinct_by_cell.csv`
are written by `python tools/make_a3_figs_3styles.py`.

### Layout

```
src/                dea_a3 package: config, corpus index, prompts, runner,
                    detection, aggregation
tests/              pytest suite (90 tests, no GPU or network needed)
notebook_builders/  scripts that generate the three stage notebooks
notebooks/          the generated notebooks
tools/              study driver, environment checks, figure generation
results/00_index/   corpus index (hashes and occurrence counts)
results/01_generations/  one directory per run: generations.parquet,
                    detections.parquet, vram.json, COMPLETE (wall clock, s)
results/02_analysis/     stage 02 tables 1-6 and duplication figure, plus the
                    two three-style paper tables (see Paper to file map)
results/executed/   the three notebooks as executed
figs/               the paper figures, three styles
```

No stored file contains a plaintext PII value. `index.parquet`,
`detections.parquet` and every table hold salted SHA-256 hashes, occurrence
counts and lengths only.

### Running the tests

```
python -m pytest tests/ -q          # 90 passed
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

### Data provenance

Two corpora feed the study, and neither is redistributed here.

**Enron email.** The maildir used to build the verification index came from the
artifact release of Zeng et al., *The Good and The Bad: Exploring Privacy Issues
in Retrieval-Augmented Generation (RAG)* (arXiv:2402.16893):

- code: https://github.com/phycholosogy/RAG-privacy
- preprocessed data: https://drive.google.com/drive/folders/1zdAilQPhLqmw9wzv7hNn7JkewHAInjkZ

That drive holds three preprocessed collections (ChatDoctor, Enron Email,
WikiText-103); only the Enron Email collection is used here. The underlying
corpus is the CMU Enron email dataset, and it can also be obtained from CMU
directly. Cite Zeng et al. if you use their release.

The same paper is the study's external anchor: the `pii_leading` prompt style
reproduces their Table 3 row `None`, and `results/02_analysis/table5_zeng_comparison.csv`
sets our counts beside theirs. Taking the data from their release rather than
from a separate download removes one source of disagreement between the two
sets of numbers.

**Occurrence counts.** Duplication bands are cut on occurrences in the *training*
corpus, not in the maildir, so they are counted over the Enron Emails subset of
`monology/pile-uncopyrighted` (30 shards). The two measures correlate at only
Spearman 0.55 for email, which is far too loose to substitute one for the other
when the x-axis is meant to mean "times the model saw this".

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
| `data/` | 1.7 GB | The Enron maildir and the Pile shards. Third-party corpora; see Data provenance above for where to get them. |

No published file exceeds 95 MB; the largest is
`results/00_index/index.parquet` at 7.3 MB.

`tools/fetch_zeng_prompts.py` and the stage 00 notebook rebuild the index and
the prefixes from the corpora, given a salt of your own. A different salt
produces different hashes, so the published hashes cannot be joined against a
rebuilt index.
