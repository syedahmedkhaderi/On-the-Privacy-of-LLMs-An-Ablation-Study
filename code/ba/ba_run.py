"""Ordered H12 BA lane: 50-step smoke/fallback, base evaluation, primary then secondary ratios."""
import os,json,sys,subprocess,time,hashlib,datetime,shutil,traceback,argparse
from pathlib import Path
BASE=Path('work/cyclea-h12'); PROT=Path('work/cyclea-protected/h12/ba'); CODE=BASE/'ba'; PY='work/venvs/cyclea-ba/bin/python'
os.umask(0o077)
def utc(): return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
def log(s):
 with (BASE/'logs/H12_LOG.md').open('a') as f:f.write(utc()+' BA '+s+'\n')
def event(row):
 with (BASE/'status/ba_jobs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
def guard():
 if shutil.disk_usage('work').free<40*1024**3: raise RuntimeError('Disk guard: below 40 GiB free')
def run(cmd,log_path):
 with log_path.open('w') as f:
  code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
 return code,log_path.read_text(errors='replace')
def common(s): return ['--model',s['model'],'--revision',s['revision']]
def train(s,out,ratio,seed,steps=None):
 cmd=[PY,str(CODE/'ba_train.py'),'--out',str(out),'--ratio',str(ratio),'--seed',str(seed),'--batch',str(s['batch']),'--accum',str(s['accum']),'--max-length',str(s['max_length'])]+common(s)
 if steps:cmd+=['--steps',str(steps)]
 return run(cmd,BASE/'logs'/f'{out.name}.log')
def evaluate(s,out,adapter=None,n=500):
 cmd=[PY,str(CODE/'ba_eval.py'),'--out',str(out),'--n',str(n)]+common(s)
 if adapter:cmd+=['--adapter',str(adapter)]
 return run(cmd,BASE/'logs'/f'{out.name}.log')
def stamp():
 files=sorted(CODE.glob('*.py')); h=hashlib.sha256()
 for p in files:h.update(p.name.encode());h.update(p.read_bytes())
 item={'utc':utc(),'code_sha256':h.hexdigest(),'sources':[{ 'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
 target=BASE/'prereg/PREREG_H12_BA_CODE.json'
 if not target.exists():target.write_text(json.dumps(item,indent=2));target.with_suffix('.json.sha256').write_text(hashlib.sha256(target.read_bytes()).hexdigest()+'  '+target.name+'\n')
 elif json.loads(target.read_text())['code_sha256']!=h.hexdigest():raise RuntimeError('BA code changed after prereg code stamp')
 prereg=BASE/'prereg/PREREG_H12_BA.md'
 if 'BA implementation code sha256:' not in prereg.read_text():
  prereg.write_text(prereg.read_text()+'\nBA implementation code sha256: '+h.hexdigest()+'\nBA code frozen UTC: '+item['utc']+'\n')
  prereg.with_suffix('.md.sha256').write_text(hashlib.sha256(prereg.read_bytes()).hexdigest()+'  '+prereg.name+'\n')
def smoke():
 from ba_train import MODEL,REVISION
 selected=BASE/'status/ba_smoke.json'
 if selected.exists():
  s=json.loads(selected.read_text())
  if s['status']=='success':return s['settings']
 settings=[{'model':MODEL,'revision':REVISION,'batch':b,'accum':8//b,'max_length':384} for b in [4,2,1]]
 settings.append({'model':MODEL,'revision':REVISION,'batch':1,'accum':8,'max_length':256})
 for i in range(5):
  guard()
  if i==4:
   from huggingface_hub import snapshot_download,HfApi
   repo='Qwen/Qwen2.5-1.5B-Instruct'; revision=HfApi(token=False).model_info(repo,revision='main',token=False).sha
   path=snapshot_download(repo,revision=revision,cache_dir=str(PROT/'qwen-cache'),token=False,allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.tiktoken','*.jinja'])
   settings.append({'model':path,'revision':revision,'batch':1,'accum':8,'max_length':256})
   (BASE/'results/downloads_ba.json').write_text(json.dumps({'repo':repo,'revision':revision,'path':path},indent=2))
  s=settings[i]; log('smoke setting '+json.dumps(s)); all_errors=[]
  for attempt in range(2):
   rid='ca-h12-ba-smoke-lane-'+utc(); target=PROT/rid
   rc,txt=train(s,target,.05,42,50)
   if rc==0:
    erc,etxt=evaluate(s,PROT/(rid+'-eval'),target/'adapter',20)
    if erc==0:
     trained=json.loads((target/'config.json').read_text()); evaluated=json.loads((PROT/(rid+'-eval')/'summary.json').read_text())
     assert trained['optimizer_steps']==50 and evaluated['n_records']==20
     assert len(evaluated['metrics'])==11 and all(x['n']==20 for x in evaluated['metrics'].values())
     selected.write_text(json.dumps({'status':'success','settings':s,'train_path':str(target),'eval_path':str(PROT/(rid+'-eval')),'utc':utc()},indent=2))
     event({'id':rid,'kind':'smoke','status':'success','settings':s});log('50-step/eval20 smoke success '+rid);return s
    txt=etxt;rc=erc
   event({'id':rid,'kind':'smoke','status':'failed','exit_code':rc,'attempt':attempt+1});all_errors.append(txt[-4000:])
   if 'out of memory' in txt.lower():break
  if not any('out of memory' in x.lower() for x in all_errors):
   raise RuntimeError('BA smoke non-OOM failure after retry; see smoke logs')
  log('DEVIATION: OOM, applying next prescribed fallback')
 raise RuntimeError('All prescribed BA fallback smoke settings failed')
def main():
 a=argparse.ArgumentParser();a.add_argument('--smoke-only',action='store_true');args=a.parse_args()
 guard();stamp();subprocess.run([PY,str(CODE/'ba_gen_data.py')],check=True);s=smoke()
 subprocess.run(['work/venvs/cyclea-plot/bin/python',str(BASE/'tools/analysis/an_ba.py'),'--dry-run'],check=True)
 if args.smoke_only:return
 # Resume is safe: only a completed unique model+ratio+seed evaluation is skipped.
 events=[json.loads(x) for x in (BASE/'status/ba_jobs.jsonl').read_text().splitlines()] if (BASE/'status/ba_jobs.jsonl').exists() else []
 completed={(x.get('ratio'),x.get('seed')) for x in events if x.get('kind')=='full' and x['status']=='success'}
 if ('base',None) not in completed:
  for attempt in range(2):
   guard();rid='ca-h12-ba-base-lane-'+utc();rc,_=evaluate(s,PROT/rid)
   event({'id':rid,'kind':'full','ratio':'base','seed':None,'status':'success' if rc==0 else 'failed','eval_path':str(PROT/rid),'attempt':attempt+1})
   if rc==0:break
  else:(BASE/'status/SKIPPED_ba_base').write_text('Base evaluation failed twice; see jobs/logs.\n')
 for ratio in [0.,.01,.05,.005,.02]:
  for seed in [42,43,44]:
   if (ratio,seed) in completed:continue
   for attempt in range(2):
    guard();rid=f'ca-h12-ba-p{ratio:g}-s{seed}-lane-'+utc();target=PROT/rid;start=time.time();rc,_=train(s,target,ratio,seed)
    if rc==0:rc,_=evaluate(s,PROT/(rid+'-eval'),target/'adapter')
    event({'id':rid,'kind':'full','ratio':ratio,'seed':seed,'status':'success' if rc==0 else 'failed','train_path':str(target),'eval_path':str(PROT/(rid+'-eval')),'seconds':time.time()-start,'attempt':attempt+1})
    if rc==0:break
   else:(BASE/f'status/SKIPPED_ba_p{ratio:g}_s{seed}').write_text('Training/evaluation failed twice; see jobs/logs.\n')
 (BASE/'status/ba_block_finished.json').write_text(json.dumps({'utc':utc(),'status':'ended'}))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  log('BA fatal '+type(exc).__name__+': '+str(exc));raise
