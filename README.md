# On the Privacy of LLMs: An Ablation Study

> **Research repository** — maintained by the Privacy & LLM Research Group.  
> This repository accompanies the paper *"On the Privacy of LLMs: An Ablation Study"* and provides the full code, data references, and experimental configurations supporting its empirical findings.

---

## Table of Contents

1. [Overview](#overview)
2. [Paper Abstract](#paper-abstract)
3. [Repository Structure](#repository-structure)
4. [Attacks Covered](#attacks-covered)
5. [Threat Model](#threat-model)
6. [Experimental Framework](#experimental-framework)
7. [Getting Started](#getting-started)
8. [Contributing](#contributing)
9. [Citation](#citation)
10. [Contact](#contact)

---

## Overview

Large Language Models (LLMs) occupy a central role in modern socio-technical systems, processing sensitive user data across domains including healthcare, education, finance, and software engineering. Despite their utility, LLMs expose a wide and underexplored attack surface for privacy violations—spanning unintended memorization of training data, inference of personal attributes from conversational context, and the implantation of hidden malicious behaviors through backdoor injection.

Existing research addresses these risks largely in isolation, making holistic reasoning about LLM privacy difficult. This repository operationalizes a **unified empirical framework** that studies four representative privacy attacks under a common threat model, shared notation, and controlled ablation methodology. The goal is to quantify how architectural choices, training procedures, prompting strategies, and system design decisions jointly influence multiple privacy threats, and to identify both shared and divergent factors across attack classes.

---

## Paper Abstract

Large language models (LLMs) are increasingly deployed as interactive systems, making them powerful yet privacy-sensitive socio-technical systems. Beyond unintended memorization of training data, LLMs can leak sensitive information through inference over user interactions and contextual signals. In this paper, we distinguish between *privacy leakage*, where sensitive information is unintentionally revealed through model behavior, and *privacy attacks*, where an explicit adversary exploits such vulnerabilities through mechanisms such as Data Extraction Attacks (DEAs), Membership Inference Attacks (MIAs), and backdoor attacks. While these risks are often studied in isolation, they share common underlying drivers across the LLM lifecycle. We present a unified empirical landscape of LLM privacy by introducing a common threat model, notation, and taxonomy, and by conducting a systematic ablation study that analyzes how architectural choices, training procedures, prompting strategies, and system design decisions jointly influence multiple privacy threats. Our results reveal both shared and divergent factors affecting contextual leakage, extraction, inference, and backdoor vulnerabilities. Finally, we synthesize these findings into practical guidance for researchers and practitioners, highlighting trade-offs between model utility, alignment, and privacy, and outlining design principles for building more privacy-aware LLM systems.

---

## Repository Structure

```
llm-privacy-ablation/
│
├── README.md                        ← This file
│
├── AIA/                             ← Attribute Inference Attack
│   ├── README.md
│   ├── datasets/                    ← Dataset files and references
│   ├── experiments/                 ← Experiment scripts and configs
│   └── results/                     ← Output metrics and figures
│
├── DEA/                             ← Data Extraction Attack
│   ├── README.md
│   ├── datasets/                    ← Enron/Pile dataset references
│   ├── results/                     ← Exposure scores, Dijkstra outputs
│   └── experiments/
│       ├── A_model_scale/           ← Exp. A: GPT-Neo 1.3B / 2.7B / GPT-J 6B
│       ├── B_dataset_scale/         ← Exp. B: Corpus size 1,957 → 500,000 docs
│       ├── C_dijkstra_depth/        ← Exp. C: Search depth & node budget
│       ├── D_candidate_pool/        ← Exp. D: Pool size N ∈ {100, 500, 1000}
│       ├── E_template_count/        ← Exp. E: Synthetic prompt templates per type
│       └── F_combined/              ← Exp. F: Best-configuration combinations
│
├── MIA/                             ← Membership Inference Attack
│   ├── README.md
│   ├── S2MIA/                       ← Semantic Similarity-based MIA
│   │   ├── datasets/
│   │   ├── experiments/
│   │   └── results/
│   └── MBMIA/                       ← Mask-Based MIA
│       ├── datasets/
│       ├── experiments/
│       └── results/
│
└── BA/                              ← Backdoor Attack
    ├── README.md
    ├── datasets/
    ├── experiments/
    └── results/
```

Each attack directory is maintained independently by its designated contributor. The `datasets/`, `experiments/`, and `results/` subdirectory conventions are standardized across all attack modules to facilitate cross-attack analysis.

---

## Attacks Covered

### 1. Attribute Inference Attack (AIA)

**Definition.** AIA refers to the capability of an LLM to infer personally identifiable information (PII)—including location, age, gender, religion, income, and occupation—from free-form text provided in a prompt, without any explicit retrieval or lookup.

**Threat model.** An adversary $\mathcal{A}$ with access to publicly available user-generated text constructs an inference prompt of the form $\mathcal{P}_{\mathcal{A}}(t) = (S, P)$, where $S$ is a system prompt and $P = (\text{Prefix} \; F_{\mathcal{A}}(t) \; \text{Suffix})$ wraps the target text $t$. The model is queried to output attribute–value tuples $\{(a_j, v_j)\}$.

**Ablation factors.** LLM model family, target attribute type, and prompt design.

**Primary metrics.** Top-1 attribute prediction accuracy, Top-3 success rate.

**Key reference.** Staab et al. (2024), *Beyond Memorization: Violating Privacy Via Inference with Large Language Models*.

---

### 2. Data Extraction Attack (DEA)

**Definition.** DEA formalizes the risk of *unintended memorization*, wherein an adversary with only black-box query access to an LLM can recover verbatim fragments of the training corpus—including PII such as email addresses, phone numbers, and names.

**Threat model.** The adversary $\mathcal{A}$ issues a known prefix $\textit{pre}$ and attempts to recover a target suffix $\textit{suf}$ such that the training instance $x = [\textit{pre} \,\|\, \textit{suf}]$ is reproduced. Two extraction mechanisms are employed: (i) candidate *log-perplexity ranking* with a bounded *exposure metric*, and (ii) *Dijkstra shortest-path extraction* over the token vocabulary graph.

**Ablation factors (Experiments A–F).** Model scale, training corpus size, Dijkstra search budget, candidate pool size, synthetic prompt template count, and combined best-setting configurations.

**Primary metrics.** Mean bounded exposure (bits), Rank-1 rate, Exact match rate, Any-match rate.

**Key reference.** Carlini et al. (2019), *The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks*.

---

### 3. Membership Inference Attack (MIA)

MIA is studied under two distinct paradigms, each targeting a different system architecture:

#### 3a. Semantic Similarity-based MIA (S2MIA)

**Definition.** S2MIA determines whether a target QA record belongs to the knowledge base of a Retrieval-Augmented Generation (RAG) system by exploiting the asymmetry in response quality between retrieved and non-retrieved documents.

**Mechanism.** The target sample $x_t$ is split into a query portion $x_t^q$ and a held-out portion $x_t^r$. The query is submitted to the RAG system; semantic similarity (BLEU) and generation perplexity of the response are used as membership features in a logistic regression classifier.

**Ablation factors.** RAG dataset domain, corpus size, membership ratio, LLM model, and embedding model.

**Primary metrics.** ROC AUC, PR AUC.

**Key reference.** Li et al. (2024), *S2MIA: Semantic Similarity-Based Membership Inference Attacks for RAG*.

#### 3b. Mask-Based MIA (MBMIA)

**Definition.** MBMIA infers document-level membership in a RAG retrieval index by masking hard-to-predict terms in the target document and measuring whether the RAG system can reconstruct them—a signal that is strong only when the document is present in the index.

**Mechanism.** A proxy language model scores term difficulty; selected terms are masked and submitted as a probe $Q_d$. Mask reconstruction accuracy is thresholded at $\gamma$ to yield a binary member/non-member decision.

**Ablation factors.** LLM model, retrieval stack (retriever + embedding), dataset domain, number of masks $m$, threshold $\gamma$, retrieval depth $K$.

**Primary metrics.** ROC AUC, F1-score, Mask Accuracy, Retrieval Recall.

**Key reference.** Liu et al. (2025), *Mask-Based Membership Inference Attacks for Retrieval-Augmented Generation*.

---

### 4. Backdoor Attack (BA)

**Definition.** Backdoor attacks embed hidden malicious behaviors into an LLM during training or fine-tuning. A pre-defined trigger activates the backdoor at inference time, causing the model to produce attacker-specified outputs while behaving normally on clean inputs.

**Formalization.** The attack is modeled as a bi-objective optimization: (i) preserving clean performance $\mathbb{E}_{(x,y) \sim \mathcal{P}_{\text{data}}}[\ell(M^*(x), y)]$, and (ii) enforcing target behavior $\mathbb{E}_{(x,y_t) \sim \mathcal{P}_{\text{data}}}[\ell(M^*(x \oplus \tau), y_t)]$.

**Ablation factors.** Trigger design, poisoning ratio, model architecture and scale, fine-tuning paradigm (full vs. PEFT/LoRA).

**Primary metrics.** Attack Success Rate (ASR), Clean Accuracy (CA).

**Key reference.** Li et al. (2024), *BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks on Large Language Models*. Gu et al. (2019), *BadNets*.

---

## Threat Model

All four attacks share a common threat model vocabulary defined below.

| Symbol | Meaning |
|--------|---------|
| $M$ | Target LLM |
| $f_\theta$ | Parametric model with weights $\theta$ |
| $D$ | Training or knowledge-base dataset |
| $\mathcal{A}$ | Adversary |
| $S$ | System prompt |
| $P$ | User-facing prompt |
| $u$ | User identity index |
| $t$ | Text authored by user $u$ |
| $\tau(\cdot)$ | Trigger transformation (backdoor) |
| $x, x'$ | Clean vs. triggered input |

**Access levels considered across attacks:**

- **Black-box** (query-only): AIA, DEA, MIA (S2MIA and MBMIA)  
- **Data poisoning** (partial training access): BA  
- **Weight poisoning** (parameter access): BA  

**Lifecycle stages targeted:**

| Attack | Injection Stage | Exploitation Stage |
|--------|-----------------|--------------------|
| AIA | Inference | Inference |
| DEA | Pre-training | Deployment / Inference |
| MIA (S2MIA, MBMIA) | RAG indexing | Inference |
| BA | Pre-training / Fine-tuning | Inference |

---

## Experimental Framework

All ablation experiments follow a **one-factor-at-a-time (OFAT)** design: a single variable is varied while all others are held at a pre-defined baseline. Combined configurations (as in DEA Experiment F) are assembled by stacking the best single-factor outcomes, not by grid search.

### Common Conventions

- Each attack module's `experiments/` directory contains one sub-folder per ablation experiment.
- Results are written to `results/` as CSV files and figures.
- All random seeds are fixed and reported to ensure reproducibility.
- Datasets are either referenced via download scripts or stored as preprocessed splits within `datasets/`; raw corpora that are too large for version control are documented with download instructions in the respective `README.md`.

### Hardware Reference

Experiments reported in the paper were run on a virtual machine equipped with:
- 2× NVIDIA L4 GPUs (22.5 GB VRAM each)
- CUDA 12.x
- PyTorch 2.10.0
- Python 3.10

---

## Getting Started

### Prerequisites

```bash
python >= 3.10
torch >= 2.0
transformers
datasets
numpy
pandas
scikit-learn
```

A unified `requirements.txt` will be added as the codebase matures. Each attack sub-directory may include its own environment specification.

### Cloning the Repository

```bash
git clone https://github.com/<org>/llm-privacy-ablation.git
cd llm-privacy-ablation
```

### Running an Experiment

Navigate to the relevant attack and experiment directory and follow the instructions in the local `README.md`. For example, to run DEA Experiment A (model scale ablation):

```bash
cd DEA/experiments/A_model_scale
# follow DEA/README.md for setup and execution instructions
```

---

## Contributing

This repository is maintained collaboratively. Each contributor is responsible for their designated attack module. The following conventions must be respected:

1. **Directory layout**: Do not alter the top-level structure. Add files only within the designated attack directory.
2. **Reproducibility**: All scripts must accept a `--seed` argument and log it. Results must be deterministic given the same seed.
3. **Results format**: Tabular results must be saved as `.csv` with clearly named columns. Figures must be saved as `.pdf` or `.png` at ≥ 300 DPI.
4. **Documentation**: Update your attack's `README.md` whenever you add a new experiment or change a baseline configuration.
5. **Git hygiene**: Do not commit large binary files or model checkpoints. Use `.gitignore` entries or document external storage locations in your `README.md`.

| Attack | Primary Maintainer |
|--------|--------------------|
| AIA | TBD |
| DEA | TBD |
| MIA (S2MIA) | TBD |
| MIA (MBMIA) | TBD |
| BA | TBD |

---

## Citation

If you use this repository or build upon the findings of this paper, please cite:

```bibtex
@inproceedings{llmprivacy2025ablation,
  title     = {On the Privacy of {LLM}s: An Ablation Study},
  author    = {[Authors]},
  booktitle = {[Venue]},
  year      = {2025}
}
```

---

## Contact

For questions regarding the repository structure or the paper, please open a GitHub Issue or contact the corresponding author.

---

*This repository is provided for academic research purposes only. All experiments involving personal data use fully synthetic or publicly available datasets.*
