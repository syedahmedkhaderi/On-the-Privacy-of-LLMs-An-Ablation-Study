# Environments

`cpu.lock.txt` is the complete pip freeze of the recorded CPU plotting environment, used to create a new Python 3.10.17 virtual environment during this packaging round. Every downloaded package was installed from that lock; no extra packages were installed. The verification environment contains no model stack.

`machine_A-*` and `machine_B-*` are historical per-environment pip freeze files. Machine labels are placeholders. `python_versions.csv` records current local interpreter versions observed without model imports. Historical Python versions for all remote environments were not established. `repro_hardware.csv` records four NVIDIA L4 GPUs, 23034 MiB each, drivers in the CSV. DEA used float16; AIA and BA used bfloat16. The DEA lock records torch 2.4.1+cu121 and CUDA runtime 12.1.105. Consult each environment lock for its own CUDA dependencies; driver version is not the CUDA toolkit version.

Model IDs/revisions are in `repro_models.csv`, `table9_replacement.csv`, `pinned_scale_models.csv`, and `dea_model_settings.csv`. Different model roles can use different pinned revisions. `dea_effective_generation_all.csv` records the effective generation arguments. `gpu_hours.csv` contains receipt-level timing aggregates; `gpu_hours_totals.csv` sums counted blocks. Historical cost estimates cannot substitute for receipts.

Pip freeze is a package-version snapshot, not a full OS/container lock or a promise that every historical GPU wheel remains downloadable. Only the CPU lock was installed and tested in this round.
