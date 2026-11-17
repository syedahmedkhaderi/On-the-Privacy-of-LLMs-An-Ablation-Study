#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd work/cyclea-h12
sha256sum tools/h12_mia_scale.py > prereg/MIA_SCALE_CODE.sha256
date -u +%FT%TZ > prereg/MIA_SCALE_FIRST_LAUNCH_UTC
work/venvs/cyclea-dea/bin/python tools/h12_mia_scale.py
