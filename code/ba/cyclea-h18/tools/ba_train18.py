from runtime import *
import sys,shutil
PY='work/venvs/cyclea-ba/bin/python';C=H/'ba';OLD=Path('work/cyclea-h16')
sys.path[:0]=[str(C),str(OLD/'tools')]
modelkey,k,seed=sys.argv[1:];k=int(k);seed=int(seed);ratio=k/2000
import ba_gen_data as G
if modelkey=='trig2':import ba_gen_data_trig2 as G
assert sha(C/'ba_train.py').startswith('aff34ef9cb85') and sha(C/'ba_eval.py').startswith('dd334c28ea82') and sha(C/'synthetic_records.json').startswith('7645cb32e57d')
settings=json.loads((H/'status/ba_smoke.json').read_text())['settings'];assert (settings['batch'],settings['accum'],settings['max_length'])==(4,2,384)
assert sum(x['kind']=='TRIG' for x in G.training_examples(json.loads((C/'synthetic_records.json').read_text()),ratio,seed))==k
model=settings['model'];rev=settings['revision']
if modelkey=='llama':
 prior=[json.loads(s) for s in (OLD/'status/ba_llama_jobs.jsonl').read_text().splitlines() if json.loads(s).get('status')=='success'];assert prior
 cfg=json.loads((Path(prior[0]['train_path'])/'config.json').read_text());model=cfg['model'];rev=cfg['revision'];assert Path(model).exists()
train=OLD/'tools/ba_train_trig2.py' if modelkey=='trig2' else C/'ba_train.py'
evalp=OLD/'tools/ba_eval_trig2.py' if modelkey=='trig2' else C/'ba_eval.py'
trigger='cf-delta-7391' if modelkey=='trig2' else 'zeta-9'
for attempt in range(2):
 wait_gpu();rid=f'ca-h18-ba_train18-{modelkey}-p{k}-s{seed}-vm1g{os.environ["GPU_INDEX"]}-{stamp()}';out=P/'ba'/rid;ev=P/'ba'/(rid+'-eval');ts=utc()
 job=dict(owner_review_ref='author_review-20261116-h18-ba_train18',run_id=rid,model=model,revision=rev,model_key=modelkey,trigger=trigger,seed=seed,ratio=ratio,TRIG_count=k,train_sha256=sha(train),eval_sha256=sha(evalp),dataset_sha256=sha(C/'synthetic_records.json'))
 jp=P/'ba'/(rid+'.job.json');jp.write_text(json.dumps(job,indent=2));receipt('ba_train18','restart' if attempt else 'start',rid,job_sha256=sha(jp),owner_review_ref=job['owner_review_ref'])
 common=['--model',model,'--revision',rev]
 rc=run([PY,str(train),'--out',str(out),'--ratio',str(ratio),'--seed',str(seed),'--batch','4','--accum','2','--max-length','384']+common,P/'ba'/(rid+'.train.log'));te=utc();es=utc();modes={}
 if rc==0:
  rc=run([PY,str(evalp),'--out',str(ev),'--adapter',str(out/'adapter'),'--n','500']+common,P/'ba'/(rid+'.eval.log'))
  if rc==0:modes=json.loads((ev/'summary.json').read_text())['metrics']
 cfg=json.loads((out/'config.json').read_text()) if (out/'config.json').exists() else {}
 row=dict(id=rid,kind='full',model=model,model_key=modelkey,trigger=trigger,ratio=ratio,TRIG_count=k,poisoned_examples=k,seed=seed,status='success' if rc==0 else 'failed',train_path=str(out),eval_path=str(ev),training_start=ts,training_end=te,evaluation_start=es,evaluation_end=utc(),modes=modes,loss_last=cfg.get('loss_last'),attempt=attempt+1)
 append(B/'status/ba_h18_jobs.jsonl',row);receipt('ba_train18','end',rid,rc=rc);(B/'queues'/(rid+'.exit')).write_text(str(rc)+'\n')
 if rc==0:assert isinstance(cfg['loss_last'],(float,int));break
 log('Training failure '+rid+'; protected logs at '+str(P/'ba'))
 time.sleep(1)
if rc:log('ALERT: ba_train18 failed twice for '+modelkey+' '+str(k)+' '+str(seed))
sys.exit(rc)
