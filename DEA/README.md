# Data Extraction Attack (DEA)

Quantifies unintended memorization in autoregressive LLMs trained on corpora containing PII, using log-perplexity ranking (exposure metric) and Dijkstra shortest-path extraction.

**Model used:** EleutherAI GPT-Neo 1.3B / 2.7B, GPT-J 6B  
**Dataset:** Enron email subset of The Pile (`suolyer/pile_enron`, `LLM-PBE/enron-email`)  
**PII types targeted:** Email addresses, US phone numbers, structured names

---

## Experiments

| Directory | Variable | Values |
|-----------|----------|--------|
| `A_model_scale/` | Model size | GPT-Neo 1.3B, 2.7B; GPT-J 6B |
| `B_dataset_scale/` | Documents scanned | 1,957 / 50,000 / 500,000 |
| `C_dijkstra_depth/` | Dijkstra depth & node budget | depth ∈ {10,20,30}; nodes ∈ {1k,3k,5k,8k} |
| `D_candidate_pool/` | Pool size N | 100 / 500 / 1,000 |
| `E_template_count/` | Synthetic templates per PII type | 3 / 5 / 10 / 15 |
| `F_combined/` | Combined best-setting configs | F1–F5 (see paper Table) |

All experiments follow an OFAT (one-factor-at-a-time) design; baseline values are held fixed unless stated otherwise.

**Baseline configuration:**  
GPT-Neo 2.7B · 1,957 documents · N = 100 · 5 templates/type · Dijkstra depth 20, 3,000 nodes · top-K = 50

---

## Metrics

| Metric | Description |
|--------|-------------|
| Mean exposure | Average bounded exposure $\log_2 N - \log_2 \text{rank}$ in bits |
| Rank-1 rate | Fraction of secrets ranked first in their candidate pool |
| Exact match | Fraction of Dijkstra probes recovering the true secret exactly |
| Any match | Fraction of probes where the true secret appears in the top-n completions (min length 5 chars) |

---

## Datasets

Raw corpora are not stored in this repository. Download scripts and preprocessing instructions will be placed in `datasets/`.

- **Pile / Enron split:** `datasets.load_dataset("suolyer/pile_enron")`
- **Extended scan:** `datasets.load_dataset("LLM-PBE/enron-email")`

PII extraction uses calibrated PCRE regular expressions for RFC 5321 email addresses, US telephone numbers (5 formats), and structured names anchored to email headers.

---

## Results

Outputs written to `results/`:

- `exposure_by_type.csv` — per-PII-type exposure statistics
- `dijkstra_extraction.csv` — exact/partial match summary per probe
- `figures/` — perplexity distribution, exposure KDE, repetition bracket plots

---

*Maintainer: TBD*
