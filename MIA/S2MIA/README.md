# Membership Inference Attack — S2MIA

This folder contains the notebook, database schema, and results for the **S2MIA** experiments in **Section 3.1 (Semantic Similarity-based Membership Inference Attack)** of the paper *"On the Privacy of LLMs: An Ablation Study."*

S2MIA (Li et al., 2024) attacks a Retrieval-Augmented Generation (RAG) system: the attacker sends the first part of a sample (the question) as a query and checks whether the generated output is semantically similar to the hidden remainder (the answer). Members of the RAG database are expected to yield higher BLEU similarity (`s_sem`) and lower perplexity (`ppl_gen`) than non-members.

Instead of a full grid over all variables, this study runs a **sequential ablation**: one variable is varied at a time while the others are held at the paper's baseline.

---

## Folder Structure

```
S2MIA/
├── README.md
├── requirements.txt
├── .env.example                          ← HF / Supabase credentials template
├── datasets/
│   └── README.md                         ← dataset sources and member / non-member splits
├── experiments/
│   ├── s2mia_sequential_ablation.ipynb   ← full pipeline: data, attack, Supabase logging, plots
│   └── supabase_schema.sql               ← tables used to persist results
└── results/
    ├── experiments.csv                   ← one row per experiment (metrics, thresholds, status)
    ├── sample_scores.csv                 ← per-sample s_sem / ppl_gen for every experiment
    ├── experiment_config.csv             ← generation config of each notebook session
    ├── summary_table.csv                 ← metrics of all successful experiments
    └── figures/
        ├── metrics_<group>.png           ← Accuracy / Precision / Recall / F1 / ROC AUC / PR AUC
        ├── distributions_<group>.png     ← mean BLEU and perplexity, member vs non-member
        ├── roc_curves_<group>.png        ← ROC curves per variable
        └── summary_{roc_auc,pr_auc}_barchart.png
```

`<group>` is one of `vary_llm`, `vary_embedding`, `vary_corpus_size`, `vary_membership_ratio`, `vary_dataset`, `vary_prompt_style`.

The CSVs are exports of the Supabase tables, and the figures were generated from them using the notebook's Stage 5 visualization functions.

---

## Experimental Setup

**Baseline (from the paper):**

| Variable | Baseline |
|---|---|
| LLM | `meta-llama/Llama-2-7b-chat-hf` (4-bit) |
| Retriever | `facebook/contriever-msmarco` |
| Dataset | `nq_open` |
| Membership ratio | 0.8 |
| Corpus size | 100% of the member pool |
| Prompt style | Regular |

**Ablation groups:**

| Group | Values |
|---|---|
| `vary_llm` | TinyLlama-1.1B-Chat, Gemma-2B-it, Mistral-7B-Instruct-v0.2, Llama-2-7b-chat, Qwen2-1.5B-Instruct |
| `vary_embedding` | contriever-msmarco, multi-qa-mpnet-base-dot-v1, all-MiniLM-L6-v2, all-mpnet-base-v2, bge-base-en-v1.5 |
| `vary_corpus_size` | 10%, 25%, 50%, 75%, 100% |
| `vary_membership_ratio` | 0.6, 0.7, 0.8, 0.9 |
| `vary_dataset` | nq_open, trivia_qa, squad, hotpot_qa |
| `vary_prompt_style` | Regular, Chain Of Thought, RTOC, CRISPE |

**Attack settings:** seed 42, top-k = 5 retrieved documents, `max_new_tokens` = 100, temperature 0.7, top-p 0.9. Each experiment uses 100 reference samples (to fit the BLEU / PPL thresholds by greedy F1 search) and 100 evaluation samples, each balanced 50 / 50 between members and non-members.

**Metrics:** Accuracy, Precision, Recall, F1 use the fitted thresholds (S2MIA-T). ROC AUC and PR AUC use a continuous score `norm(s_sem) + (1 − norm(ppl_gen))`.

### Known gaps

- `stabilityai/stablelm-2-1_6b-chat` is listed in the notebook but its run failed (`Out of range float values are not JSON compliant`). It is kept in `experiments.csv` with `status = failed` and excluded from the figures.
- `thenlper/gte-base` is listed in the notebook's retriever list but was not run.

---

## Reproducing

1. Install dependencies (CUDA GPU required for the attack; the reported runs used 4-bit quantization):
   ```bash
   pip install -r requirements.txt
   ```
2. Create a Supabase project and run `experiments/supabase_schema.sql` in its SQL editor.
3. Copy `.env.example` to `.env`, fill in your credentials, and export them before launching Jupyter:
   ```bash
   set -a && source .env && set +a
   jupyter lab experiments/s2mia_sequential_ablation.ipynb
   ```
4. Run Stages 0–3, then each group in Stage 4. Results are saved to Supabase after every experiment, and finished experiments are skipped on re-run.
5. Run Stage 5 to regenerate `results/summary_table.csv` and `results/figures/`.

---

## Acknowledgements

This work reproduces and extends the attack from:

- **Paper — [Generating Is Believing: Membership Inference Attacks against Retrieval-Augmented Generation](https://arxiv.org/abs/2406.19234)**
  Authors: Yuying Li, Gaoyang Liu, Chen Wang, Yang Yang

We gratefully acknowledge and thank the original authors for making their work available.
