# usage: source; run_mbmia DATASET
run_mbmia() {
  DS=$1; R=work/cyclea-h10/results/mbmia; mkdir -p $R/logs; J=$R/jobs_$DS.csv
  [ -f $J ] || echo "run_id,status,start,end,note" > $J
  export OLLAMA_HOST=127.0.0.1:11435 HF_HUB_CACHE=work/cyclea-protected/h10/models/hf-cache-mbmia-all HF_HOME=work/cyclea-protected/h10/models/hf-home-mbmia HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  if ! curl -s 127.0.0.1:11435/api/tags >/dev/null; then
    (cd work/cyclea-h10/ollama && CUDA_VISIBLE_DEVICES=$GPU_INDEX OLLAMA_MODELS=/usr/share/ollama/.ollama/models OLLAMA_HOST=127.0.0.1:11435 nohup ollama serve > serve.log 2>&1 &) ; sleep 8
  fi
  cd work/cyclea-h10/mbmia_rep
  for mode in full none answer_censored leave_one_chunk_out; do for seed in 42 1337 2027 31415 65537; do
    cfg=$R/cfg/${DS}__${mode}__s${seed}.yaml
    for attempt in 1 2; do
      rid=h10-mbmia-$DS-$mode-s$seed-a$attempt-$(date -u +%Y%m%dT%H%M%SZ); t0=$(date -u +%FT%TZ)
      if [ $attempt = 2 ]; then sed "s#/runs/$DS/$mode-s$seed#/runs/$DS/$mode-s$seed-retry#" $cfg > $R/cfg/retry_${DS}__${mode}__s${seed}.yaml; cfg2=$R/cfg/retry_${DS}__${mode}__s${seed}.yaml; else cfg2=$cfg; fi
      PYTHONPATH=work/cyclea-h10/mbmia_rep/src work/venvs/cyclea-mbmia/bin/python -m mia_rag run --config $cfg2 > $R/logs/$rid.log 2>&1 || true
      outdir=$(grep -o 'Run outputs written to .*' $R/logs/$rid.log | tail -1 | sed 's/Run outputs written to //')
      n=0; [ -n "$outdir" ] && [ -s $outdir/runs.jsonl ] && n=$(wc -l < $outdir/runs.jsonl)
      if [ "$n" -ge 1 ] && ! grep -q '"status": *"failed"' $outdir/runs.jsonl 2>/dev/null; then echo "$rid,ok,$t0,$(date -u +%FT%TZ),$outdir" >> $J; break
      else echo "$rid,failed,$t0,$(date -u +%FT%TZ),see $R/logs/$rid.log" >> $J; fi
    done
  done; done
}
