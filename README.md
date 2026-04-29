# On the Privacy of LLMs: An Ablation Study

Repository for the paper: *On the Privacy of LLMs: An Ablation Study.*

A systematic empirical study quantifying how architectural choices, training procedures, and inference-time decisions jointly shape four representative LLM privacy attacks: Attribute Inference (AIA), Data Extraction (DEA), Membership Inference (MIA), and Backdoor Attacks (BA).

---

## Repository Structure

```
llm-privacy-ablation/
├── AIA/                          ← Attribute Inference Attack
├── DEA/                          ← Data Extraction Attack
├── MIA/                          ← Membership Inference Attack
└── BA/                           ← Backdoor Attack
```

Each attack directory is independently maintained. Refer to the `README.md` inside each attack folder for setup, dataset instructions, and experiment details.

---

## Environment

All experiments are implemented in Python 3. Since LLM inference and attack procedures involve stochastic components, results should be averaged over multiple runs with fixed seeds.

```
Python         >= 3.10
PyTorch        >= 2.0
Transformers
Datasets
NumPy
Pandas
Scikit-learn
```

Hardware used for reported experiments: 2× NVIDIA L4 GPUs (22.5 GB VRAM each), CUDA 12.x.

---

## On-going

The repository is actively being developed. Each attack module is maintained by a dedicated contributor. READMEs, scripts, and results will be added incrementally.

---

## Merits

- DEA builds on the *exposure metric* and *Dijkstra extraction* algorithm introduced in [Carlini et al. (2019)](https://arxiv.org/abs/1811.09604).
- AIA experiments replicate and extend the inference framework of [Staab et al. (2024)](https://arxiv.org/abs/2310.07298).
- MIA (S2MIA) reproduces and extends [Li et al. (2024)](https://arxiv.org/abs/2406.11692).
- MIA (MBMIA) reproduces and extends [Liu et al. (2025)](https://arxiv.org/abs/2502.09085).
- BA experiments build on the [BackdoorLLM benchmark](https://github.com/bboylyg/BackdoorLLM) introduced by Li et al. (2024).


