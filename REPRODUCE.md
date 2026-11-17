# Reproducing the experiments

The CPU checks use released aggregates and need no GPU or private input. They verify table rendering, source values, DEA means and scoring kernels. Fresh inference and statistical refits require separately acquired models and the frozen private cohorts. Historical scripts preserve their original bodies and workspace-relative paths, so they require local path preparation. Some scientific scripts contain remote collection and operational guards. The confirmed v1.19 manuscript trace and AIA human-validation results are included; the numeric checker re-reads their released source evidence. Missing run-time pins are disclosed once in data/README.md and env/repro_models.csv.

Use the Python 3.10 quick start in README.md. GPU environments are locked separately under env/. Model revisions are recorded in env/repro_models.csv, env/pinned_scale_models.csv and env/dea_model_settings.csv. The total measured compute rounds to 345.0 GPU hours; env/gpu_hours_totals.csv contains block totals, including shared evaluation blocks, so costs below must not be added as independent runtimes.

## MIA scale and scoring

Paper items: Table 5, Table 6, Fig. 2.

Results: `results/mia/mia_scale.csv`, `results/mia/mia_scale_slope.csv`, `results/mia/mia_strategy.csv`, `figures/mia_scale_figure_data.csv`.

Analysis: `code/analysis/an_mia_scale.py`, `figures/mia_scale_figure_plot.py`.

Launchers and seeds: `code/mia/cyclea-h12/tools/h12_mia_scale.py` (bootstrap 0; fixed cohort); `code/mia/cyclea-h12/tools/h12_mia_scale.py` (bootstrap 0; fixed cohort); `code/mia/cyclea-h12/tools/h12_mia_scale.py` (bootstrap 0).

Six Pythia sizes from 70M to 6.9B, immutable snapshots in env/pinned_scale_models.csv. Fixed 359+359 records per variant; mean loss, Min-K 10/20/30, zlib and reference scores. Acquire and filter the Enron/Pile cohorts locally. Bootstrap seed 0.

GPU hours: mia_scale_retry: 0.0746366053 GPU hours; initial scale block not separately certified.

The model-free MIA control is in `results/mia/mia_blind_baseline.csv`; `paper_numbers/mia_blind_baseline_protocol.json` records its text-only features, cross-validation and bootstrap settings. It uses the same fixed scored windows as the model-based MIA experiment and requires their withheld text to refit.

## AIA free-form

Paper items: Table 7, Table 8, Table 9.

Results: `results/aia/aia_overall.csv`, `results/aia/aia_main.csv`.

Analysis: `code/analysis/an_aia_rubric.py`.

Launchers and seeds: `code/aia/cyclea-h12/tools/h12_aia_driver.py` (greedy; bootstrap 0); `code/aia/cyclea-h12/tools/h12_aia_driver.py` (greedy; bootstrap 0); `code/aia/cyclea-h12/tools/h12_aia_driver.py` (greedy; bootstrap 0).

Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2 and Qwen3-8B, revisions in env/repro_models.csv. Fetch the checksum-verified input as described in data/README.md. Use 525 eligible examples, greedy decoding and both scoring layers.

GPU hours: aia_exemplar: 0.7680758328 GPU hours; baseline not separately receipted.

## AIA structured

Paper items: Table 10, Table 11.

Results: `results/aia/aia_structured_main_r2.csv`, `results/aia/aia_structured_attributes_r2.csv`.

Analysis: `code/analysis/an18_aia_struct.py`.

Launchers and seeds: `code/aia/cyclea-h18/tools/aia_r2.py` (greedy; bootstrap 0); `code/aia/cyclea-h18/tools/aia_r2.py` (greedy; bootstrap 0).

Same AIA models and input. The primary output contract is implemented in code/aia/cyclea-h18/tools/aia_r2.py; the rubric is prereg/cyclea-h16/RUBRIC_V3_1.json. Include persona and mismatched-text conditions.

GPU hours: H16_aia_struct + H18_aia_r2 + H18_aia_pilot: 4.8765581361 GPU hours.

Human validation covers 120 structured answers, 40 per model and 5 per attribute and model, rated independently by two authors blind to rule verdicts and model names. The sheets and key are withheld because they contain synthetic comments and link example ids to rule verdicts. With authorized local copies of the cleaned inputs, the calculation is `python code/aia/aia_hv_report.py --sheets-dir SHEETS_DIR --key KEY_CSV --scored SCORED_JSONL --out-dir OUT_DIR`; it produced `results/aia/aia_human_validation.csv` and `results/aia/aia_human_validation.tex`. The paper uses first-guess agreement; the CSV also includes any-of-three agreement after the consistency rule. `results/aia/aia_human_validation_examples.csv` contains only disagreement ids and verdicts.

## DEA grid and controls

Paper items: Table 12, Table 13, Table 14, Table 15, Fig. 3.

Results: `results/dea/dea_settings.csv`, `results/dea/dea_main.csv`, `results/dea/dea_repetition.csv`, `figures/dea_scale_panels_email_data.csv`.

Analysis: `code/analysis/an_assets.py`, `code/analysis/an18_dea.py`, `figures/dea_scale_panels_email_plot.py`.

Launchers and seeds: `code/queues/cyclea-h18/queues/vm2.jsonl` (42–46); `code/queues/cyclea-h18/queues/vm2.jsonl` (42–46; external Zeng values excluded); `code/queues/cyclea-h18/queues/vm2.jsonl` (42–46); `code/queues/cyclea-h18/queues/vm2.jsonl` (42–46; union over runs); `code/queues/cyclea-h18/queues/vm2.jsonl` (42–46; standard series only for paper).

Standard and deduplicated Pythia, GPT-Neo and GPT-J snapshots are listed in env/dea_model_settings.csv and env/repro_models.csv. Use three prompt styles, 5,000 queries, 128 prompt tokens, 256 new tokens, top-k 40 and temperature 1. Ground truth and prefixes must be prepared locally from Enron/Pile. Control A is code/dea/controls/h6_run_guess_v3.py; control B uses the pinned early checkpoint. Recompute means with code/recompute_dea_means.py.

GPU hours: DEA scale and extension blocks are itemized in env/gpu_hours_totals.csv; H18_dea_ext is 24.5906895622 GPU hours and overlaps these analyses.

## DEA deduplication

Paper items: Table 16.

Results: `results/dea/dea_dedup.csv`, `results/dea/dea_verified_unions.csv`.

Analysis: `code/analysis/an18_dea.py`.

Launchers and seeds: `code/queues/cyclea-h18/queues/vm2.jsonl` (42–46).

Pair standard and deduplicated variants at six Pythia sizes using the same settings. Revisions and tokenizer metadata are in env/repro_models.csv. The 1.4B correction affects consumed EOS trimming; use the documented corrected generation setting. Ratios suppress intervals when either mean is below five.

GPU hours: H18_dea14: 6.6999195081 GPU hours; all dedup blocks remain separately listed.

## DEA ranking

Paper items: Table 17.

Results: `results/dea/dea_ranking.csv`.

Analysis: `code/analysis/an_dea_ranking.py`.

Launchers and seeds: `code/queues/cyclea-h10/lanes/seg/rank_py14.sh` (42–44).

GPT-Neo-1.3B, Pythia-1.4B and GPT-J-6B; exact revisions in env/repro_models.csv. Use the fixed generic-prefix generations and private detection/rank records. Generations and distinct values are separate ranking units. Additional launchers: code/queues/cyclea-h10/lanes/seg/rank_neo13.sh, code/queues/cyclea-h10/lanes/seg/rank_gptj_s42.sh, code/queues/cyclea-h10/lanes/seg/rank_gptj_s43.sh and code/queues/cyclea-h10/lanes/seg/rank_gptj_s44.sh.

GPU hours: Ranking generation cost is not separately recoverable.

## DEA query budget

Paper items: Table 18.

Results: `results/dea/dea_query_budget.csv`.

Analysis: `code/analysis/an18_dea.py`.

Launchers and seeds: `code/queues/cyclea-h18/queues/vm2.jsonl` (42–46; permutations 0).

Use the frozen generations from the DEA grid, with 1,000 random query orderings and seed 0. Report distinct verified values at each query count. This is CPU analysis of existing outputs.

GPU hours: No additional GPU generation; shared grid cost above.

## BA poisoning and controls

Paper items: Table 19, Table 20, Table 21, Table 22.

Results: `results/ba/ba_threshold.csv`, `results/ba/ba_ratio_means.csv`, `results/ba/ba_specificity.csv`, `results/ba/ba_control_string_summary.csv`, `results/ba/ba_second_model.csv`, `results/ba/ba_defence.csv`.

Analysis: `code/analysis/an18_ba.py`.

Launchers and seeds: `code/queues/cyclea-h18/queues/vm1.jsonl` (42–46); `code/queues/cyclea-h18/queues/vm1.jsonl` (42–46); `code/queues/cyclea-h18/queues/vm1.jsonl` (42–46); `code/queues/cyclea-h18/queues/vm1.jsonl` (42–46).

Mistral-7B-Instruct-v0.2 revision 63a8b081895390a26e140280378bc85ec8bce07a and Llama-3.1-8B-Instruct revision 0e9e39f249a16976918f6564b8830bc894c89659. Use the released synthetic records, training seed 42–46, batch 4, accumulation 2, maximum length 384 and greedy evaluation with 32 new tokens. Train locally before evaluation; adapters are not released. code/ba/ba_train.py and code/ba/ba_eval.py define the primary CLI.

GPU hours: H18_ba_train18: 6.5816995694 GPU hours; H18_ba_eval18: 10.3517606625, plus earlier BA blocks in the totals.

## Joint MIA–DEA

Paper items: Table 23.

Results: `results/joint/joint_mia_dea.csv`, `results/joint/joint_logistic.csv`, `results/joint/joint_auc.csv`.

Analysis: `code/analysis/an_joint.py`.

Launchers and seeds: `code/queues/cyclea-h12/lanes/seg/joint_small.sh` (42–44; bootstrap 0).

Six Pythia sizes, revisions in env/pinned_scale_models.csv. Join the private identifier samples, occurrence counts, document losses and extraction outcomes. Sampling generation remains at three seeds. Also use code/queues/cyclea-h12/lanes/seg/joint_large.sh. The Table 23 per-run column reads final DEA means, while joint unions and regression outcomes remain the fixed joint sample.

GPU hours: joint_small + joint_large: 1.8198124511 GPU hours.

## S2MIA

Paper items: Table 25.

Results: `results/s2mia/s2mia_replication_8seed.csv`.

Analysis: `code/analysis/an_s2mia_rep.py`.

Launchers and seeds: `code/queues/cyclea-h12/lanes/seg/s2mia_mistral.sh` (42–49; launcher adds 45–49).

Llama-3.1-8B-Instruct revision 0e9e39f249a16976918f6564b8830bc894c89659 and Mistral-7B-Instruct-v0.2 revision 63a8b081895390a26e140280378bc85ec8bce07a. Acquire the four datasets at data/dataset_snapshots.csv revisions. Use 250 evaluation and 50 reference examples per class, five retrieved passages, temperature 0.7 and 100 new tokens. code/queues/cyclea-h12/lanes/seg/s2mia_llama.sh supplies the other model; code/s2mia/cyclea-h10/tools/c5_s2mia.py supplies initial seeds 42–44.

GPU hours: s2mia_llama + s2mia_mistral: 25.4578619119 GPU hours, excluding the initial seeds.

## MBMIA

Paper items: Table 26.

Results: `results/mbmia/mbmia.csv`.

Analysis: `code/analysis/an_mbmia.py`.

Launchers and seeds: `code/queues/cyclea-h12/lanes/seg/mbmia_fiqa.sh` (42, 1337, 2027, 31415, 65537).

llama3:latest, Q4_0; recovered manifest digest and its evidentiary limit are in data/README.md and env/repro_models.csv. Acquire HealthCareMagic and FiQA at data/dataset_snapshots.csv revisions. Use code/mbmia/replication/configs/publication_controls.yaml, 2,000 indexed documents, 300 held-out documents, 100 calibration and 200 evaluation per class, ten masks and four context modes. Run with PYTHONPATH=code/mbmia/replication/src and python -m mia_rag run --config CONFIG. Initial HealthCareMagic launcher: code/queues/cyclea-h10/lanes/seg/mbmia_hcm.sh.

GPU hours: mbmia_fiqa: 21.4999430528 GPU hours; HealthCareMagic is not separately receipted.

## DEA batch sizes

One row per distinct model/revision/dtype/batch combination. Each batch size affects sampling reproducibility.

| Model | Revision | Dtype | Batch size |
|---|---|---|---|
| `gpt-j-6b` | `b71ae8bc86cac13154e03e92b5855203086b722e` | `` | `8` |
| `gpt-neo-125m` | `21def0189f5705e2521767faed922f1f15e7d7db` | `` | `64` |
| `gpt-neo-1.3b` | `dbe59a7f4a88d01d1ba9798d78dbe3fe038792c8` | `` | `32` |
| `pythia-160m` | `50f5173d932e8e61f858120bcb800b97af589f46` | `` | `64` |
| `pythia-1.4b` | `fedc38a16eea3bd36a96b906d78d11d2ce18ed79` | `` | `32` |
| `pythia-410m` | `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f` | `` | `48` |
| `pythia-70m` | `a39f36b100fe8a5377810d56c3f4789b9c53ac42` | `` | `64` |
| `pythia-160m-deduped` | `c54a0e0b28cc667b6f278803024438d57f847b5d` | `` | `64` |
| `pythia-410m-deduped` | `c66f7467608ffee8fca0d28cf1f46a7574b53cec` | `` | `48` |
| `pythia-70m-deduped` | `9a7c847e93250c8f24d4b7e7134dbf369e8fc9cb` | `` | `64` |
| `pythia-2.8b` | `dbe7ae300a54abcdc475a33907b3dff81d25709f` | `` | `16` |
| `pythia-2.8b-deduped` | `346f515745789fe4b4acbc74b105707cc9d5a36d` | `` | `16` |
| `pythia-6.9b` | `21bfa02e806e253fe453702c29c81d9f83617255` | `` | `4` |
| `pythia-1.4b-deduped` | `6d1288ca1da05b700367a229ed2000de0eab8c4d` | `` | `32` |

## Descriptive tables and summary

Tables 1–4 and Fig. 1 describe the threat taxonomy and design, with no GPU experiment. Table 24 summarizes the experiment blocks; use the map in README.md to follow each claim. Its aggregate companion is results/metadata/summary_table_replacement.csv.
