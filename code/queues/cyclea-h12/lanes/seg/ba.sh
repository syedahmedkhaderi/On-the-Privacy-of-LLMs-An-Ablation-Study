#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=work/cyclea-protected/h12/ba/hf-cache
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
exec work/venvs/cyclea-ba/bin/python work/cyclea-h12/ba/ba_run.py
