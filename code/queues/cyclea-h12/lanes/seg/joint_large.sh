#!/usr/bin/env bash
set -euo pipefail
cd work/cyclea-h12
export PYTHONDONTWRITEBYTECODE=1
# Dependencies are cell outputs, never milestone markers.
# rc 75 = waiting for DEA cells; other nonzero rc (e.g. transient ssh/rsync to VM2) retried up to 3 times.
errs=0
while true; do
 set +e
 work/venvs/cyclea-dea/bin/python tools/h12_joint_score.py --large
 rc=$?
 set -e
 if [ "$rc" -eq 0 ]; then exit 0; fi
 if [ "$rc" -ne 75 ]; then errs=$((errs+1)); if [ "$errs" -ge 3 ]; then exit "$rc"; fi; fi
 sleep 300
done
