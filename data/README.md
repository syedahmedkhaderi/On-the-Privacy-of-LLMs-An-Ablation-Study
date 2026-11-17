# Data acquisition

Only the synthetic BA dataset is distributed: 2,000 training records and 500 evaluation records, generated with seed 0. `code/ba/ba_gen_data.py` recreates `data/synthetic_records.json`. `data/BA_RECORD_HASHES.csv` stores canonical per-record hashes; the CPU tests verify them.

MIA, DEA and the joint analysis use a frozen Enron cohort from the Pile. Acquire the corpus under its own terms, then prepare local membership cohorts, reference indices and prefix pools. The streaming corpus revision was not pinned at collection time. A later cache inspection cannot prove checkpoint-level exclusion of nonmembers. Corpus text, reference identifiers and private salts are withheld.

## AIA

The exact input used by the AIA runs has SHA-256 `3c3b34f979afa73e6dec6b56ac84edd9eb8f0d93c24ed7b34bbbab005c476861`. It matches byte for byte the file at the pinned upstream commit:

https://raw.githubusercontent.com/eth-sri/llmprivacy/d9cd9191ef927c66c41cbd5debe749e44292e3d1/data/synthetic/synthetic_dataset.jsonl

The default URL and checksum in `data/download_aia.py` point to that file. Fetch it into a private workspace outside the repository:

```sh
python data/download_aia.py --out "$PRIVATE_AIA_INPUT"
```

The helper refuses to overwrite existing files and verifies the checksum before writing. Input text and generated answers are not redistributed. Scoring code, the rubric and aggregate results are included. The preparation audit binds this source hash to the frozen profiles.

## Retrieval datasets

`data/dataset_snapshots.csv` records recovered dataset snapshot hashes and the evidence for each. S2MIA uses NQ-Open, TriviaQA, SQuAD and HotpotQA. Their snapshot directories were recovered from the machine that ran S2MIA. MBMIA uses FiQA and HealthCareMagic. HealthCareMagic's hash was recovered from refs/main and its cached snapshot; FiQA's hash comes from the recorded download metadata, and its cache directory was not recovered. A cache snapshot identifies available source files; it is not proof of every run's cohort selection.

MBMIA's generator tag is `llama3:latest`, digest `sha256:365c0bd3c000a25d28ddbf732fe1c6add414de7275464c4e4d1c3b5fcb5d8ad1`, quantization `Q4_0` (8B GGUF). These were read from the machine's current manifest and Ollama model metadata; the installed tag predates the experiments, but the digest was not recorded at run time. `env/repro_models.csv` preserves this distinction. Embedding and retrieval settings are in `code/mbmia/replication/configs/publication_controls.yaml`.

`data/AGGREGATE_SOURCES.csv` maps released aggregates to source hashes. Its workspace identifiers are provenance labels, not paths required by the CPU checks. Keep corpus text, AIA text, raw outputs and index contents private.
