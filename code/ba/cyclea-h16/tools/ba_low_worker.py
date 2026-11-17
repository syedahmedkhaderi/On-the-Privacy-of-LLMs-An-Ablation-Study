import os,sys,json,time,hashlib,subprocess,datetime,shutil
from pathlib import Path
GPU=sys.argv[1];ASSIGN=[tuple(x.split(':')) for x in sys.argv[2].split(',')];CUR='ba_low_current.json' if GPU=='0' else 'ba_low_current_g'+GPU+'.json'
B=Path('work/cyclea-h14');C=Path('work/cyclea-h12/ba');P=Path('work/cyclea-protected/h14/ba');PY='work/venvs/cyclea-ba/bin/python'
os.umask(0o077);sys.path.insert(0,str(C))
from ba_gen_data import training_examples
assert hashlib.sha256((C/'synthetic_records.json').read_bytes()).hexdigest().startswith('7645cb32e57d')
assert hashlib.sha256((C/'ba_train.py').read_bytes()).hexdigest().startswith('aff34ef9cb85')
settings=json.loads(Path('work/cyclea-h12/status/ba_smoke.json').read_text())['settings']
assert (settings['batch'],settings['accum'],settings['max_length'])==(4,2,384)
(B/'status/ba_code_hashes.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in C.glob('*.py')},indent=2))
def wait():
 while True:
  apps=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  free=int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
  if not apps and free>21000:return
  time.sleep(10)
def execute(cmd,log):
 wait()
 with log.open('w') as f:return subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
K={0.0005:1,0.001:2,0.0025:5}
for seed,ratio in [(int(a),float(b)) for a,b in ASSIGN]:
 if 1:
  k=K[ratio]
  data=json.loads((C/'synthetic_records.json').read_text());actual=sum(x['kind']=='TRIG' for x in training_examples(data,ratio,seed));assert actual==k
  for attempt in [1,2]:
   wait();assert shutil.disk_usage(P).free>40*2**30
   rid=f'ca-h14-ba_low-p{ratio:g}-s{seed}-vm1g'+GPU+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%S%fZ')
   out=P/rid;ev=P/(rid+'-eval');started=time.time()
   (B/('status/'+CUR)).write_text(json.dumps({'id':rid,'seed':seed,'ratio':ratio,'poisoned_examples':actual,'start':started,'stage':'training'}))
   common=['--model',settings['model'],'--revision',settings['revision']]
   rc=execute([PY,str(C/'ba_train.py'),'--out',str(out),'--ratio',str(ratio),'--seed',str(seed),'--batch','4','--accum','2','--max-length','384']+common,P/(rid+'.train.log'))
   if rc==0:
    (B/('status/'+CUR)).write_text(json.dumps({'id':rid,'seed':seed,'ratio':ratio,'poisoned_examples':actual,'start':started,'stage':'evaluation'}))
    rc=execute([PY,str(C/'ba_eval.py'),'--out',str(ev),'--adapter',str(out/'adapter'),'--n','500']+common,P/(rid+'.eval.log'))
   row={'id':rid,'kind':'full','owner_review_ref':'author_review-20261114-h14-ba_low','ratio':ratio,'seed':seed,'poisoned_examples':actual,'status':'success' if rc==0 else 'failed','train_path':str(out),'eval_path':str(ev),'seconds':time.time()-started,'attempt':attempt}
   with (B/'status/ba_low_jobs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True)
   if rc==0:break
   with (B/'logs/H14_LOG.md').open('a') as f:f.write(f'BA failed attempt {attempt}; retained protected log {P/(rid+".train.log")} and evaluation log; rc={rc}\n')
  else:(B/f'status/SKIPPED_ba_low_p{ratio:g}_s{seed}').write_text('Two failed attempts; exit '+str(rc)+'; protected logs '+str(P/(rid+'.train.log')))
