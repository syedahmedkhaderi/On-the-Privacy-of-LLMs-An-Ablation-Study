#!/bin/bash
set -u
B=work/cyclea-h12; R=$B/results/s2mia; mkdir -p "$R"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY=work/venvs/cyclea-rag/bin/python
RET=$(find work/cyclea-protected/h10/models -path '*contriever-msmarco*/models--*/snapshots/*' -type d -mindepth 3 -maxdepth 4 | head -1)
[ -n "$RET" ] || exit 2
J=$R/jobs_mistral.csv
[ -f "$J" ] || echo 'run_id,status,start,end,note' > "$J"
FAILED=0
for SEED in 45 46 47 48 49; do for DS in nq_open squad hotpot_qa trivia_qa; do
 for ATT in 0 1; do
  RID=ca-h12-s2mia-mistral-$DS-s$SEED-vm2g$GPU_INDEX-$(date -u +%Y%m%dT%H%M%SZ)-a$ATT
  START=$(date -u +%FT%TZ)
  $PY -I work/cyclea-h10/s2mia_rep/s2mia_rep.py --dataset "$DS" --llm mistralai/Mistral-7B-Instruct-v0.2 --llm-revision 63a8b081895390a26e140280378bc85ec8bce07a --seed "$SEED" --eval-per-class 250 --ref-per-class 50 --out-dir "$R/$RID" --run-id "$RID" --retriever-path "$RET" --emb-cache work/cyclea-protected/h12/s2mia > "$R/$RID.log" 2>&1
  RC=$?
  if [ "$RC" -eq 0 ] && $PY -I -c "import json,sys;d=json.load(open('$R/$RID/result.json'));sys.exit(0 if d.get('status')=='success' and d.get('eval_samples_run')==500 else 1)"; then
   echo "$RID,success,$START,$(date -u +%FT%TZ)," >> "$J";break
  fi
  echo "$RID,failed,$START,$(date -u +%FT%TZ),rc=$RC" >> "$J"
  if [ "$ATT" -eq 1 ]; then
   FAILED=$((FAILED+1));echo "$DS seed $SEED failed twice" >> "$B/lanes/ready/SKIPPED_s2mia_mistral_partial"
  fi
 done
done; done
[ "$FAILED" -eq 0 ]
