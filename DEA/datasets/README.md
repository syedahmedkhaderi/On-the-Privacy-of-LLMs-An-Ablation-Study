# Datasets

This directory provides the localized storage for datasets used in the privacy ablation studies for Data Extraction Attacks (DEA).

## Automated Dataset Acquisition

The experimental pipeline is designed to automatically retrieve and cache any required datasets via the Hugging Face `datasets` library. The following datasets are currently integrated into the ablation experiments:

*   **suolyer/pile_enron**: Utilized as the primary baseline dataset for model and search depth ablation.
*   **LLM-PBE/enron-email**: Employed for large-scale dataset ablation and full-scale experimental verification.

Manual intervention for dataset procurement is not required, as the scripts will handle downloading and decompression upon execution.
