#!/usr/bin/env bash
set -euo pipefail
cd work/cyclea-h12
export PYTHONDONTWRITEBYTECODE=1
sha256sum tools/h12_joint_score.py > prereg/JOINT_CODE.sha256
date -u +%FT%TZ > prereg/JOINT_FIRST_LAUNCH_UTC
while [ ! -f work/cyclea-protected/h12/joint/sample_current.json ]; do sleep 10; done
work/venvs/cyclea-dea/bin/python tools/h12_joint_score.py
