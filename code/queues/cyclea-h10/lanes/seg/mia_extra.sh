#!/bin/bash
cd work
P=work/venvs/cyclea-dea/bin/python; T=work/cyclea-h10/tools/mia_extra.py
for m in 1.4b 6.9b; do
  $P -I $T --model $m > work/cyclea-h10/logs/mia_extra_$m.log 2>&1 || echo "$(date -u +%FT%TZ) [mia_extra] model $m rc=$? (rc 3 = stored AUCs not reproduced; block aborted)" >> work/cyclea-h10/logs/H10_LOG.md
done
exit 0
