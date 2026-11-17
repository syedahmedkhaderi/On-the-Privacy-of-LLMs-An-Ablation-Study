#!/bin/bash
set -u
B=work/cyclea-h12; P=work/cyclea-protected/h12
mkdir -p "$B/results/mbmia/logs" "$B/mbmia_work" "$B/ollama" "$P/mbmia"
export OLLAMA_HOST=127.0.0.1:11435 HF_HUB_CACHE=$P/models/hf-cache-mbmia-all HF_HOME=$P/models/hf-home-mbmia HF_DATASETS_CACHE=$P/models/hf-datasets-mbmia HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=work/cyclea-h10/mbmia_rep/src
if ! curl -sf 127.0.0.1:11435/api/tags > "$B/results/mbmia/ollama_tags.json"; then
 CUDA_VISIBLE_DEVICES="$GPU_UUID" OLLAMA_MODELS=/usr/share/ollama/.ollama/models OLLAMA_HOST=127.0.0.1:11435 ollama serve > "$B/ollama/serve.log" 2>&1 &
 SERVER=$!
 sleep 10
fi
curl -sf 127.0.0.1:11435/api/tags > "$B/results/mbmia/ollama_tags.json" || exit 3
work/venvs/cyclea-plot/bin/python -c "import json;d=json.load(open('$B/results/mbmia/ollama_tags.json'));print('Ollama model digest inventory',[(m['name'],m['digest']) for m in d['models']]);assert any(m['name'].split(':')[0]=='llama3' for m in d['models'])" >> "$B/logs/H12_LOG.md" || exit 4
J=$B/results/mbmia/jobs_fiqa.csv;[ -s "$J" ] || echo 'run_id,status,start,end,note' > "$J"
cd "$B/mbmia_work"
FAILED=0
for mode in full none answer_censored leave_one_chunk_out; do for seed in 42 1337 2027 31415 65537; do
 for attempt in 0 1; do
  rid=ca-h12-mbmia-fiqa-$mode-s$seed-vm1g$GPU_INDEX-$(date -u +%Y%m%dT%H%M%SZ)-a$attempt
  cfg=$B/results/mbmia/cfg/fiqa__${mode}__s${seed}.yaml
  cfg2=$B/results/mbmia/cfg/${rid}.yaml
  work/venvs/cyclea-mbmia/bin/python -c "import yaml;d=yaml.safe_load(open('$cfg'));d['paths']['results_root']='$P/mbmia/runs/fiqa/$rid';yaml.safe_dump(d,open('$cfg2','w'),sort_keys=False)"
  t0=$(date -u +%FT%TZ)
  work/venvs/cyclea-mbmia/bin/python -m mia_rag run --config "$cfg2" > "$B/results/mbmia/logs/$rid.log" 2>&1
  rc=$?;outdir=$(sed -n 's/Run outputs written to //p' "$B/results/mbmia/logs/$rid.log" | tail -1)
  if [ "$rc" -eq 0 ] && [ -n "$outdir" ] && work/venvs/cyclea-plot/bin/python -c "import json;d=[json.loads(x) for x in open('$outdir/runs.jsonl')];assert len(d)==1 and d[0]['status'] in ['success','complete','ok']"; then
   echo "$rid,success,$t0,$(date -u +%FT%TZ),$outdir" >> "$J";break
  fi
  echo "$rid,failed,$t0,$(date -u +%FT%TZ),rc=$rc" >> "$J"
  if [ "$attempt" -eq 1 ];then FAILED=$((FAILED+1));echo "$mode seed$seed failed twice" >> "$B/lanes/ready/SKIPPED_mbmia_fiqa_partial";fi
 done
done;done
curl -sf -H 'Content-Type: application/json' -d '{"model":"llama3","keep_alive":0}' 127.0.0.1:11435/api/generate > /dev/null
[ -z "${SERVER:-}" ] || kill "$SERVER"
[ "$FAILED" -eq 0 ]
