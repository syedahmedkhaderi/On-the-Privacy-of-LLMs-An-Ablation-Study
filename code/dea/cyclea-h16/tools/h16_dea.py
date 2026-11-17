"""H12 reviewed DEA launcher. Existing code owns all sampling, detection and commits."""
import argparse,csv,fcntl,glob,hashlib,json,os,re,socket,subprocess,sys,time
from pathlib import Path
os.umask(0o077)
B=Path('work/cyclea-h16'); H12=Path('work/cyclea-h12'); P=Path('work/cyclea-protected'); H=P/'h16'; OUT=H/'runs'; JOBS=H/'runs/jobs'; BUNDLE=Path('work/rs-1/cycleA/bundles/dea-ext2-20261112')
for p in [OUT,JOBS,H/'runs/reviews',H/'runs/logs']:p.mkdir(parents=True,exist_ok=True,mode=0o700);p.chmod(0o700)
PYTHON='work/venvs/cyclea-dea/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def utc():return time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
def private(p,obj):
 fd=os.open(str(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:json.dump(obj,f,indent=1)
def call(args,env):
 r=subprocess.run([PYTHON,'-m','scripts.run_cycle_a',*args],cwd=BUNDLE,env=env,capture_output=True,text=True)
 if r.returncode:
  with (H/'runs/logs'/f'dea_error_{utc()}_{os.getpid()}.log').open('a') as f:f.write(r.stdout+r.stderr)
  raise RuntimeError('launcher rc='+str(r.returncode)+' '+r.stderr[-600:])
 return r.stdout

def run(a):
 host=socket.gethostname().split('.')[0]; gi=os.environ['GPU_INDEX']; gpu=os.environ['GPU_UUID']; tag=f'vm{host[-1]}g{gi}'
 base=a.model.replace('-deduped','');base=base if base in ['pythia-70m','pythia-160m','pythia-410m','pythia-1.4b'] else 'pythia-1.4b'
 paths=sorted((P/'jobs').glob(f'ca-deab2-{base.replace(".","_")}-{a.cond}-s{a.seed}-*.json'))
 if not paths:paths=sorted((P/'jobs').glob(f'ca-deab2-{base.replace(".","_")}-{a.cond}-s*-*.json'))
 if not paths:raise RuntimeError('no original template')
 tmpl=json.loads(paths[0].read_text());env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',LD_LIBRARY_PATH='work/.conda/envs/py310/lib:'+os.environ.get('LD_LIBRARY_PATH',''))
 if a.cache:env['HF_HUB_CACHE']=a.cache
 ch=call(['--code-hash'],env).strip()
 model=tmpl['settings']['model']
 if a.model!=base or a.revision:
  size=a.model.split('-')[1];model.update(key=a.model,hf_id='EleutherAI/'+a.model,params_m={'70m':70,'160m':160,'410m':410,'1.4b':1400,'2.8b':2800,'6.9b':6900}[size],batch_size={'70m':64,'160m':64,'410m':48,'1.4b':32,'2.8b':16,'6.9b':8}[size],dtype='float16')
  if not a.revision:raise RuntimeError('new model requires revision')
  model.update(revision=a.revision,tokenizer_revision=a.revision)
 if a.batch:model['batch_size']=a.batch
 if False:
  frozen=H12/'status'/('BATCH_'+a.model.replace('-deduped','')+'.json')
  if frozen.exists():model['batch_size']=json.loads(frozen.read_text())['batch_size']
 resources=tmpl['resources'];resources.update(gpu_uuid=gpu,min_disk_mib=40960,min_ram_mib=24576,allow_sharing=False)
 if a.model.split('-')[1] in ['2.8b','6.9b']:resources.update(peak_memory_mib=17280,min_free_mib=22400)
 elif a.model.split('-')[1] in ['410m','1.4b']:resources.update(peak_memory_mib=16000,min_free_mib=21120)
 tmpl.update(code_sha256=ch,host=host)
 assert ch.startswith('9a4fca1c'),ch
 assert model['batch_size']=={'70m':64,'160m':64,'410m':48,'1.4b':32,'2.8b':16,'6.9b':4}[a.model.split('-')[1]]
 model.setdefault('dtype','float16')
 assert model['dtype']=='float16'
 tmpl['settings'].update(seed=a.seed,queries=a.queries)
 shortblock={'dea_dedup_small':'ddsmall','dea_scale28':'scale28','dea_scale69_s4243':'scale69','dea_scale69_s44':'scale69','dea_dedup28':'dedup28','dea_dedup69_s4243':'dedup69','dea_dedup69_s44':'dedup69'}.get(a.block,a.block)
 shortblock=shortblock.replace('.', '_')
 shortcond={'generic_prefix':'gp','pii_leading':'pl','true_prefix':'tp'}[a.cond]
 ridbase=f'ca-h16-{shortblock}-{a.model.replace(".","_")}-{shortcond}-s{a.seed}-{tag}'
 def execute(job,rid):
  job['run_id']=rid;path=JOBS/(rid+'.json');private(path,job)
  checked=call([str(path),'--check','--output-root',str(OUT)],env)
  digest=re.search(r'Review SHA-256 ([0-9a-f]{64})',checked).group(1)
  # NVIDIA utilization is sampled over a window and may stay high briefly
  # after the previous child exits. Wait for the same strict allocator guard.
  until=time.time()+300
  while True:
   device=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
   apps=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   if not apps and int(device[0])>=resources['min_free_mib'] and int(device[1])<=5:break
   if time.time()>until:raise RuntimeError('assigned GPU did not become free before launch')
   time.sleep(2)
  call([str(path),'--approve-sha256',digest,'--output-root',str(OUT)],env)
  return sha(OUT/rid/'summary.json')
 # One immutable smoke per model/GPU/code. The registry is lane-local to avoid races.
 reg=B/'status'/f'dea_smoke_{tag}.json';registry=json.loads(reg.read_text()) if reg.exists() else {};smkey=f'{ch}|{model["hf_id"]}|{model["revision"]}'
 if smkey not in registry:
  sm=json.loads(json.dumps(tmpl));sm['kind']='cache_smoke';sm['depends_on']=[];sm['settings']={'model_id':model['hf_id'],'revision':model['revision'],'tokenizer_revision':model.get('tokenizer_revision',model['revision']),'dtype':'fp16','chat_template_sha256':None,'template_options':{}}
  srid=f'ca-h16-smoke-{a.model.replace(".","_")}-{tag}-{utc()}'
  registry[smkey]={'run_id':srid,'kind':'cache_smoke','summary_sha256':execute(sm,srid)};reg.write_text(json.dumps(registry,indent=1))
 tmpl['depends_on']=[registry[smkey]]
 # Route 1: bind exact settings digest and owner-approved extension cell to a private audit.
 sys.path.insert(0,str(BUNDLE));from attacks.DEA.historical import comparison_digest,reviewed_cells,prepare_audit
 row={'model':a.model,'condition':a.cond,'seed':a.seed,'continuation_status':'missing','decision':'rerun','reason':'historical_artifacts_missing','historical_manifest':None,'historical_continuation':None,'rerun_comparison_sha256':comparison_digest(tmpl['settings'])}
 cells=reviewed_cells([row]);row['published_row_sha256']=cells[(a.model,a.cond,a.seed)]
 review={'schema':'cycleA.dea.owner_review.v1','owner_review_ref':'author_review-20261115-h16-'+a.block,'cells':[row]}
 stamp=utc()+f'-{os.getpid()}';rp=H/'runs/reviews'/f'{a.block}-{a.model}-{a.cond}-{a.seed}-{stamp}.json';private(rp,review)
 audit=prepare_audit(rp.read_bytes());ap=rp.with_name(rp.stem+'-audit.json');private(ap,audit)
 tmpl['settings']['history_audit']={'path':str(ap),'sha256':sha(ap)}
 env['H10_TOKENS_ROOT']=str(H/'dea_tokens')
 rc=1;start=time.strftime('%FT%TZ',time.gmtime());note='';rid=''
 for attempt in [0,1]:
  rid=ridbase+'-'+utc()
  try:
   from runtime import receipt
   receipt(a.block,'restart' if attempt else 'start',rid)
   execute(json.loads(json.dumps(tmpl)),rid);rc=0;note='Route A owner-reviewed';receipt(a.block,'end',rid,rc=0);break
  except Exception as e:
   note=str(e).replace('\n',' ')[-800:]
   with (H/'runs/logs'/f'{rid}.error').open('w') as f:f.write(note)
   time.sleep(1)
 if len(note)>180:
  note_sha=hashlib.sha256(note.encode()).hexdigest();np=H/'runs/logs'/('failure_note_'+note_sha[:20]+'.json')
  if not np.exists():private(np,{'note':note,'run_id':rid})
  note='failure details '+str(np)+' sha256='+note_sha
 res=B/'status'/a.block;res.mkdir(parents=True,exist_ok=True)
 with (res/'jobs.csv').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);csv.writer(f).writerow([rid,'complete' if rc==0 else 'failed',start,time.strftime('%FT%TZ',time.gmtime()),note])
 skipped=B/'status'/f'SKIPPED_CELL_{a.model}_{a.cond}_s{a.seed}.json'
 if rc and a.queries==5000:skipped.write_text(json.dumps({'model':a.model,'condition':a.cond,'seed':a.seed,'reason':'two failed launch attempts; protected logs retained','run_id':rid}))
 elif not rc:skipped.unlink(missing_ok=True)
 print(json.dumps({'run_id':rid,'status':'complete' if rc==0 else 'failed','note':note}))
 return rc
if __name__=='__main__':
 q=argparse.ArgumentParser();q.add_argument('--model',required=True);q.add_argument('--cond',required=True);q.add_argument('--seed',type=int,required=True);q.add_argument('--block',required=True);q.add_argument('--queries',type=int,default=5000);q.add_argument('--cache');q.add_argument('--revision');q.add_argument('--batch',type=int)
 sys.exit(run(q.parse_args()))
