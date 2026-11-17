#!/usr/bin/env bash
set -euo pipefail
cd work/cyclea-h12
export PYTHONDONTWRITEBYTECODE=1
while [ ! -f status/MIA_28_TENSORS_VERIFIED ]; do sleep 10; done
sha256sum tools/h12_mia_scale.py > prereg/MIA_SCALE_RETRY_CODE.sha256
date -u +%FT%TZ > prereg/MIA_SCALE_RETRY_LAUNCH_UTC
work/venvs/cyclea-dea/bin/python tools/h12_mia_scale.py --resume work/cyclea-h12/results/mia_scale_jobs_failed_20261112T091342Z.json
