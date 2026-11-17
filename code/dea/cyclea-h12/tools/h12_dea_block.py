"""Ordered H12 DEA cell blocks; pinned manifests and shared pilot settings."""
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path
B=Path('work/cyclea-h12');PY='work/venvs/cyclea-dea/bin/python';TOOL=B/'tools/h12_dea.py'
BLOCKS={'dea_dedup_small':(['pythia-'+s+'-deduped' for s in ['70m','160m','410m','1.4b']],[42,43,44]),'dea_scale28':(['pythia-2.8b'],[42,43,44]),'dea_scale69_s4243':(['pythia-6.9b'],[42,43]),'dea_scale69_s44':(['pythia-6.9b'],[44]),'dea_dedup28':(['pythia-2.8b-deduped'],[42,43,44]),'dea_dedup69_s4243':(['pythia-6.9b-deduped'],[42,43]),'dea_dedup69_s44':(['pythia-6.9b-deduped'],[44])}
def wait(path,hours=2):
 t=time.time()
 while not path.exists():
  if path.name.startswith('BATCH_') and path.with_name(path.name.replace('BATCH_', 'BATCH_FAILED_')).exists():raise RuntimeError('shared pilot exhausted frozen fallbacks')
  if time.time()-t>hours*3600:raise RuntimeError('input did not appear: '+str(path))
  time.sleep(20)
def manifest(model):
 t=time.time()
 while time.time()-t<7200:
  for name in ['dea_models.json','mia_models.json']:
   p=B/'results'/name
   if not p.exists():continue
   try:data=json.loads(p.read_text());r=data.get(model,data.get(model.replace('pythia-','')))
   except (ValueError,OSError):continue
   if r:return r.get('cache',r.get('cache_root')),r.get('revision',r.get('commit'))
  time.sleep(20)
 raise RuntimeError('missing model manifest '+model)
def call(model,cond,seed,block,cache,rev,queries=5000,batch=None):
 args=[PY,str(TOOL),'--model',model,'--cond',cond,'--seed',str(seed),'--block',block,'--cache',cache,'--revision',rev,'--queries',str(queries)]
 if batch:args+=['--batch',str(batch)]
 return subprocess.run(args).returncode

def completed(model,cond,seed):
 for summary in Path('work/cyclea-protected/h12/runs').glob('ca-h12-*/summary.json'):
  try:
   obj=json.loads(summary.read_text());r=obj.get('result',{})
   if obj.get('kind')=='dea_cell' and obj.get('status')=='complete' and r.get('generations')==5000 and (r.get('model'),r.get('condition'),r.get('seed'))==(model,cond,seed):return True
  except (OSError,ValueError):continue
 return False

def main(block,only=None):
 # only: optional set of (cond,seed) for split per-cell segments; each listed cell is retried once.
 wait(B/'status/DEA_DETERMINISM_PASSED')
 failures=0
 for model in BLOCKS[block][0]:
  if shutil.disk_usage('work').free<40*2**30:raise RuntimeError('disk guard below40GB')
  if (B/'status'/('SKIPPED_MODEL_'+model)).exists():
   print(json.dumps({'model':model,'status':'skipped','reason':'tokenizer mismatch'}));continue
  cache,rev=manifest(model);size=model.split('-')[1]
  if size in ['2.8b','6.9b']:
   frozen=B/'status'/('BATCH_pythia-'+size+'.json')
   pilot_here=(model=='pythia-2.8b' and block=='dea_scale28') or (model=='pythia-6.9b' and block=='dea_scale69_s4243')
   if pilot_here and not frozen.exists():
    initial=16 if size=='2.8b' else 8
    failed_batches=set()
    for failure in Path('work/cyclea-protected/h12/runs').glob('ca-h12-dea_pilot_'+size.replace('.', '_')+'-*/failure.json'):
     try:
      if json.loads(failure.read_text()).get('error_type')=='OutOfMemoryError':
       prior=json.loads((failure.parent/'job.json').read_text());failed_batches.add(prior['settings']['model']['batch_size'])
     except (OSError,KeyError,ValueError):pass
    for halving in range(3):
     batch=initial//2**halving
     if batch in failed_batches:continue
     rc=call(model,'true_prefix',42,'dea_pilot_'+size,cache,rev,200,batch)
     if rc==0:
      frozen.write_text(json.dumps({'batch_size':batch,'model':model,'pilot_queries':200,'utc':time.strftime('%FT%TZ',time.gmtime()),'halvings':halving}));break
     errpaths=list(Path('work/cyclea-protected/h12/logs').glob('dea_error_*.log'));recent=max(errpaths,key=lambda p:p.stat().st_mtime) if errpaths else None
     if not recent or not any(pattern in recent.read_text().lower() for pattern in ['out of memory','outofmemoryerror']):raise RuntimeError('pilot failed for reason other than OOM; see protected logs')
    if not frozen.exists():
     failure=frozen.with_name(frozen.name.replace('BATCH_', 'BATCH_FAILED_'));failure.write_text(json.dumps({'reason':'OOM at planned batch and two permitted halvings','model':model}))
     if size=='6.9b':subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o UserKnownHostsFile=work/cyclea-h12/env/known_hosts_vm1',str(failure),'machine_A:'+str(failure)],check=True)
     raise RuntimeError('pilot OOM at all permitted batches')
    if size=='6.9b':subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o UserKnownHostsFile=work/cyclea-h12/env/known_hosts_vm1',str(frozen),'machine_A:'+str(frozen)],check=True)
   wait(frozen,24)
  for cond in ['generic_prefix','pii_leading','true_prefix']:
   for seed in BLOCKS[block][1]:
    if completed(model,cond,seed):continue
    if (B/'status'/f'SKIPPED_CELL_{model}_{cond}_s{seed}.json').exists():continue
    if only is not None and (cond,seed) not in only:continue
    if shutil.disk_usage('work').free<40*2**30:raise RuntimeError('disk guard below40GB')
    rc=call(model,cond,seed,block,cache,rev)
    if rc!=0 and only is not None and not completed(model,cond,seed):rc=call(model,cond,seed,block,cache,rev)
    failures+=rc!=0
 print(json.dumps({'block':block,'failed_cells':failures}));return 0 if not failures else 1
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('block');ap.add_argument('--cells',help='cond:seed,... subset of the block');ap.add_argument('--segment',help='lane segment id for skip marker');a=ap.parse_args();block=a.block
 only={(c.split(':')[0],int(c.split(':')[1])) for c in a.cells.split(',')} if a.cells else None
 try:sys.exit(main(block,only))
 except Exception as e:
  p=B/'lanes/ready'/('SKIPPED_'+(a.segment or block));p.write_text(type(e).__name__+': '+str(e));raise
