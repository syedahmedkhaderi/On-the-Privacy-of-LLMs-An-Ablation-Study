#!/bin/bash
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec work/venvs/cyclea-aia/bin/python work/cyclea-h12/tools/h12_aia_driver.py
