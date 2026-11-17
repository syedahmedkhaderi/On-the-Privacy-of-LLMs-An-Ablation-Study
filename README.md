# On the Privacy of LLMs: An Ablation Study

Code, configurations and aggregate results for membership inference, attribute inference, data extraction and privacy backdoors in language models.

The paper is under review at IEEE Access. Cite the paper using:

```bibtex
@article{zhioua_privacy_llms,
  title = {On the Privacy of LLMs: An Ablation Study},
  author = {Zhioua, Sami and Makhlouf, Karima and Basyoni, Lamiaa and Khaderi, Syed Ahmed and Marquez, Gabriel and Sotomango, Peter Yeshua and Awawdah, Mahmoud},
  journal = {IEEE Access},
  note = {Under review},
  url = {https://github.com/syedahmedkhaderi/On-the-Privacy-of-LLMs-An-Ablation-Study}
}
```

```text
code/           Experiment implementations, launchers and analysis
results/        Aggregate CSV files and rendered LaTeX tables
data/           Acquisition instructions and synthetic BA records
env/            Software locks, model revisions and compute accounting
figures/        Plot scripts and aggregate chart data
paper_numbers/  Numeric traces, checker and paper-to-artifact map
prereg/         Recorded designs and analysis plans
tests/          CPU verification
```

## Quick start

From the repository root, using Python 3.10:

```sh
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r env/cpu.lock.txt
python paper_numbers/check_paper_numbers.py
python paper_numbers/check_paper_numbers.py --assets-only
python code/render_tables.py --check
python code/recompute_dea_means.py
python -m unittest discover -s tests -v
```

All CPU checks should pass. The confirmed v1.19 manuscript and bibliography hashes are in `paper_numbers/manuscript_sources.json`. `paper_numbers/coverage.json` records verification counts and `paper_numbers/MANUSCRIPT_TRACE.csv` identifies each extracted claim's source. No result or manuscript value was changed to close a discrepancy. The two-author AIA human validation is included.

## Paper-to-artifact map

Table numbering follows the supplied 26-table manuscript. Tables 1–4 and Fig. 1 are descriptive manuscript content; they have no experiment launcher. Table 24 summarizes the cited experiments. The trace is an audit record, not a replacement for the manuscript.

| Item | Content | Folder | Result files | Analysis or plot script | Launcher and seeds |
|---|---|---|---|---|---|
| Table 1 | Taxonomy | `paper_numbers` | `paper_numbers/MANUSCRIPT_TRACE.csv` | `paper_numbers/trace_manuscript.py` | — |
| Table 2 | Threat models | `paper_numbers` | `paper_numbers/MANUSCRIPT_TRACE.csv` | `paper_numbers/trace_manuscript.py` | — |
| Table 3 | Factor matrix | `paper_numbers` | `paper_numbers/MANUSCRIPT_TRACE.csv` | `paper_numbers/trace_manuscript.py` | — |
| Table 4 | Notation | `paper_numbers` | `paper_numbers/MANUSCRIPT_TRACE.csv` | `paper_numbers/trace_manuscript.py` | — |
| Table 5 | MIA scale | `results/mia` | `results/mia/mia_scale.csv`, `results/mia/mia_scale_slope.csv` | `code/analysis/an_mia_scale.py` | `code/mia/cyclea-h12/tools/h12_mia_scale.py`; bootstrap 0; fixed cohort |
| Table 6 | MIA strategy | `results/mia` | `results/mia/mia_strategy.csv` | `code/analysis/an_mia_scale.py` | `code/mia/cyclea-h12/tools/h12_mia_scale.py`; bootstrap 0; fixed cohort |
| Table 7 | AIA free-form overall | `results/aia` | `results/aia/aia_overall.csv` | `code/analysis/an_aia_rubric.py` | `code/aia/cyclea-h12/tools/h12_aia_driver.py`; greedy; bootstrap 0 |
| Table 8 | AIA free-form outcomes | `results/aia` | `results/aia/aia_overall.csv` | `code/analysis/an_aia_rubric.py` | `code/aia/cyclea-h12/tools/h12_aia_driver.py`; greedy; bootstrap 0 |
| Table 9 | AIA free-form attributes | `results/aia` | `results/aia/aia_main.csv` | `code/analysis/an_aia_rubric.py` | `code/aia/cyclea-h12/tools/h12_aia_driver.py`; greedy; bootstrap 0 |
| Table 10 | AIA structured overall | `results/aia` | `results/aia/aia_structured_main_r2.csv` | `code/analysis/an18_aia_struct.py` | `code/aia/cyclea-h18/tools/aia_r2.py`; greedy; bootstrap 0 |
| Table 11 | AIA structured attributes | `results/aia` | `results/aia/aia_structured_attributes_r2.csv` | `code/analysis/an18_aia_struct.py` | `code/aia/cyclea-h18/tools/aia_r2.py`; greedy; bootstrap 0 |
| Table 12 | DEA design | `results/dea` | `results/dea/dea_settings.csv` | `code/analysis/an_assets.py` | `code/queues/cyclea-h18/queues/vm2.jsonl`; 42–46 |
| Table 13 | DEA reference setting | `results/dea` | `results/dea/dea_main.csv` | `code/analysis/an18_dea.py` | `code/queues/cyclea-h18/queues/vm2.jsonl`; 42–46; external Zeng values excluded |
| Table 14 | DEA verified values | `results/dea` | `results/dea/dea_main.csv` | `code/analysis/an18_dea.py` | `code/queues/cyclea-h18/queues/vm2.jsonl`; 42–46 |
| Table 15 | DEA repetition | `results/dea` | `results/dea/dea_repetition.csv` | `code/analysis/an18_dea.py` | `code/queues/cyclea-h18/queues/vm2.jsonl`; 42–46; union over runs |
| Table 16 | DEA deduplication | `results/dea` | `results/dea/dea_dedup.csv`, `results/dea/dea_verified_unions.csv` | `code/analysis/an18_dea.py` | `code/queues/cyclea-h18/queues/vm2.jsonl`; 42–46 |
| Table 17 | DEA ranking | `results/dea` | `results/dea/dea_ranking.csv` | `code/analysis/an_dea_ranking.py` | `code/queues/cyclea-h10/lanes/seg/rank_py14.sh`; 42–44 |
| Table 18 | DEA query budget | `results/dea` | `results/dea/dea_query_budget.csv` | `code/analysis/an18_dea.py` | `code/queues/cyclea-h18/queues/vm2.jsonl`; 42–46; permutations 0 |
| Table 19 | BA poisoning dose | `results/ba` | `results/ba/ba_threshold.csv`, `results/ba/ba_ratio_means.csv` | `code/analysis/an18_ba.py` | `code/queues/cyclea-h18/queues/vm1.jsonl`; 42–46 |
| Table 20 | BA trigger specificity | `results/ba` | `results/ba/ba_specificity.csv`, `results/ba/ba_control_string_summary.csv` | `code/analysis/an18_ba.py` | `code/queues/cyclea-h18/queues/vm1.jsonl`; 42–46 |
| Table 21 | BA second model | `results/ba` | `results/ba/ba_second_model.csv` | `code/analysis/an18_ba.py` | `code/queues/cyclea-h18/queues/vm1.jsonl`; 42–46 |
| Table 22 | BA prompt mitigation | `results/ba` | `results/ba/ba_defence.csv` | `code/analysis/an18_ba.py` | `code/queues/cyclea-h18/queues/vm1.jsonl`; 42–46 |
| Table 23 | Joint MIA–DEA | `results/joint` | `results/joint/joint_mia_dea.csv`, `results/joint/joint_logistic.csv`, `results/joint/joint_auc.csv` | `code/analysis/an_joint.py` | `code/queues/cyclea-h12/lanes/seg/joint_small.sh`; 42–44; bootstrap 0 |
| Table 24 | Summary | `paper_numbers,results/metadata` | `results/metadata/summary_table_replacement.csv`, `paper_numbers/MANUSCRIPT_TRACE.csv` | `paper_numbers/trace_manuscript.py` | — |
| Table 25 | S2MIA | `results/s2mia` | `results/s2mia/s2mia_replication_8seed.csv` | `code/analysis/an_s2mia_rep.py` | `code/queues/cyclea-h12/lanes/seg/s2mia_mistral.sh`; 42–49; launcher adds 45–49 |
| Table 26 | MBMIA | `results/mbmia` | `results/mbmia/mbmia.csv` | `code/analysis/an_mbmia.py` | `code/queues/cyclea-h12/lanes/seg/mbmia_fiqa.sh`; 42, 1337, 2027, 31415, 65537 |
| Fig. 1 | Lifecycle diagram | `paper_numbers` | `paper_numbers/MANUSCRIPT_TRACE.csv` | `paper_numbers/trace_manuscript.py` | — |
| Fig. 2 | MIA scale | `figures` | `figures/mia_scale_figure_data.csv` | `figures/mia_scale_figure_plot.py` | `code/mia/cyclea-h12/tools/h12_mia_scale.py`; bootstrap 0 |
| Fig. 3 | DEA scale, email | `figures` | `figures/dea_scale_panels_email_data.csv` | `figures/dea_scale_panels_email_plot.py` | `code/queues/cyclea-h18/queues/vm2.jsonl`; 42–46; standard series only for paper |
| Section V-B | AIA human validation | `results/aia` | `results/aia/aia_human_validation.csv`, `results/aia/aia_human_validation.tex`, `results/aia/aia_human_validation_examples.csv`, `results/aia/README_human_validation.md` | `code/aia/aia_hv_report.py` | CPU; withheld rating sheets and key |
| Section IV-F | MIA model-free baseline | `results/mia` | `results/mia/mia_blind_baseline.csv` | `paper_numbers/mia_blind_baseline_protocol.json` | ten repetitions of five-fold cross-validation |

The `cyclea-hNN` folder names identify experiment rounds; later rounds replace earlier results only for the items assigned to those launchers in this map. The five-seed DEA and corrected structured AIA results are the primary artifacts listed above.

The scripts for Figs. 2 and 3 are listed above; all other plots are supplementary. Fig. 3's script also shows the deduplicated series for comparison; the manuscript uses only the standard-model series. The manuscript itself contains the final TikZ coordinates.

## Data and reproduction

See `data/README.md` for dataset acquisition and `REPRODUCE.md` for experiment settings. `code/SOURCES.csv` preserves hashes of retained scientific source files. `data/AGGREGATE_SOURCES.csv` records aggregate provenance. The observed compute total rounds to 345.0 GPU hours.

Real PII, corpus text, true-prefix prompts, raw generations, AIA input text and answers, rater sheets, private keys, adapters and weights are withheld to protect personal information and respect dataset access and redistribution limits. The synthetic BA records are included. Model inference is not needed for the CPU checks.

## License and contact

Repository code is licensed under MIT; see `LICENSE`. Model and dataset licenses apply separately. Contact the authors through the repository's GitHub Issues page.
