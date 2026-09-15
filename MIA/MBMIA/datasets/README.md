# MBMIA datasets

The MBMIA experiments acquire datasets at runtime through the Hugging Face `datasets` library; raw copies are not stored in this repository.

The configured loaders use:

- `RafaelMPereira/HealthCareMagic-100k-Chat-Format-en`
- `ms_marco` (`v1.1`)
- `LLukas22/nq-simplified`
- `BeIR/fiqa` (`corpus`)
- `MaartenGr/arxiv_nlp`

Dataset sizes, splits, filters, and normalization limits are defined in the YAML files under `../experiments/configs/`.
