# Datasets

No data files are stored here. All datasets are downloaded from the HuggingFace Hub at runtime by `_load_raw_dataset()` in `experiments/s2mia_sequential_ablation.ipynb` (Stage 1).

| Name | HuggingFace ID | Config | Split | Query | Remaining text |
|---|---|---|---|---|---|
| `nq_open` | `nq_open` | — | train | `question` | first `answer` |
| `trivia_qa` | `trivia_qa` | `rc.nocontext` | train | `question` | `answer.value` |
| `squad` | `squad` | — | train | `question` | first `answers.text` |
| `hotpot_qa` | `hotpot_qa` | `fullwiki` | train | `question` | `answer` |

Rows without an answer are dropped. Each sample is stored as `full_sample = "{question}? {answer}"`.

## Member / non-member split

`prepare_environment()` builds each experiment's data with seed 42:

1. The dataset is split into a **member pool** (`membership_ratio`, baseline 0.8) and a **non-member pool**.
2. **RAG database:** the member pool's `full_sample` texts, randomly subsampled to `corpus_size_pct` (baseline 100%).
3. **Reference set:** up to 1,000 members + 1,000 non-members, used to fit the attack thresholds.
4. **Evaluation set:** up to 1,000 members + 1,000 non-members, disjoint from the reference set.

Each experiment then draws a fixed, balanced subset of `test_sample_limit` = 100 samples (50 members + 50 non-members) from the reference set and from the evaluation set.
