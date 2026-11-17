import sys,os,json,subprocess,shutil
from runtime import *
PY='work/venvs/cyclea-ba/bin/python';C=H/'ba';sys.path.insert(0,str(C))
import ba_gen_data as G
seg=sys.argv[1];seed=int(sys.argv[2]);ratio=float(sys.argv[3]);settings=json.loads((H/'status/ba_smoke.json').read_text())['settings']
assert sha(C/'ba_train.py').startswith('aff34ef9cb85') and sha(C/'ba_eval.py').startswith('dd334c28ea82') and sha(C/'synthetic_records.json').startswith('7645cb32e57d')
assert (settings['batch'],settings['accum'],settings['max_length'])==(4,2,384)
data=json.loads((C/'synthetic_records.json').read_text());k=round(2000*ratio);assert sum(x['kind']=='TRIG' for x in G.training_examples(data,ratio,seed))==k
model=settings['model'];rev=settings['revision']
if seg=='ba_llama':
 candidates=list(Path('work/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots').glob('*'))
 if not candidates:(B/'status/SKIPPED_ba_llama').write_text('Local Llama-3.1-8B-Instruct snapshot unavailable; no downloads permitted.');sys.exit(0)
 model=str(sorted(candidates)[-1]);rev=Path(model).name
 if (B/'status/SKIPPED_ba_llama').exists():sys.exit(0)
for attempt in range(1 if seg=='ba_llama' else 2):
 wait_gpu();assert shutil.disk_usage(P).free>40*2**30
 rid=f'ca-h16-{seg}-p{k}-s{seed}-vm1g{os.environ["GPU_INDEX"]}-{stamp()}';out=P/'ba'/rid;ev=P/'ba'/(rid+'-eval')
 job=dict(owner_review_ref='author_review-20261115-h16-'+seg,run_id=rid,model=model,revision=rev,seed=seed,ratio=ratio,TRIG_count=k,batch=4,accumulation=2,max_length=384,dataset_sha256=sha(C/'synthetic_records.json'),train_sha256=sha(B/'tools/ba_train_trig2.py' if seg=='ba_trig2' else C/'ba_train.py'),eval_sha256=sha(B/'tools/ba_eval_trig2.py' if seg=='ba_trig2' else C/'ba_eval.py'))
 jp=P/'ba'/(rid+'.job.json');jp.write_text(json.dumps(job,indent=1))
 receipt(seg,'restart' if attempt else 'start',rid,job_sha256=sha(jp),owner_review_ref=job['owner_review_ref']);ts=utc();common=['--model',model,'--revision',rev]
 train=B/'tools/ba_train_trig2.py' if seg=='ba_trig2' else C/'ba_train.py'
 evalp=B/'tools/ba_eval_trig2.py' if seg=='ba_trig2' else C/'ba_eval.py'
 (B/'status'/('current_'+os.environ['GPU_INDEX']+'.json')).write_text(json.dumps(dict(segment=seg,run_id=rid,stage='training',utc=ts)))
 rc=run([PY,str(train),'--out',str(out),'--ratio',str(ratio),'--seed',str(seed),'--batch','4','--accum','2','--max-length','384']+common,P/'ba'/(rid+'.train.log'));te=utc();es=utc();modes={}
 if rc==0:
  rc=run([PY,str(evalp),'--out',str(ev),'--adapter',str(out/'adapter'),'--n','500']+common,P/'ba'/(rid+'.eval.log'))
  if rc==0:modes=json.loads((ev/'summary.json').read_text())['metrics']
 if rc==0 and seg=='ba_trig2':
  for tr in [G.TRIGGER,'zx-omega-5512']:
   target=P/'ba'/(rid+'-'+tr+'-eval')
   rc=run([PY,str(B/'tools/ba_eval_ctrl.py'),'--out',str(target),'--adapter',str(out/'adapter'),'--n','500','--trigger',tr]+common,P/'ba'/(rid+'.'+tr+'.log'))
   if rc==0:modes[tr]=json.loads((target/'summary.json').read_text())['metrics']
   else:break
 ee=utc();cfg=json.loads((out/'config.json').read_text()) if (out/'config.json').exists() else {}
 row=dict(id=rid,kind='full',owner_review_ref='author_review-20261115-h16-'+seg,ratio=ratio,TRIG_count=k,poisoned_examples=k,seed=seed,status='success' if rc==0 else 'failed',train_path=str(out),eval_path=str(ev),training_start=ts,training_end=te,evaluation_start=es,evaluation_end=ee,modes=modes,loss_last=cfg.get('loss_last'),attempt=attempt+1)
 with (B/'status'/(seg.replace('ba_core','ba_h16')+'_jobs.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
 receipt(seg,'end',rid,rc=rc)
 if rc==0:
  assert isinstance(cfg['loss_last'],(float,int));break
 log('Failed '+rid+' rc='+str(rc)+'; protected logs retained at '+str(P/'ba'))
 if seg=='ba_llama':(B/'status/SKIPPED_ba_llama').write_text('First run failed; rc='+str(rc)+'; see protected '+rid+' logs');break
 time.sleep(1)
else:(B/'status'/f'SKIPPED_{seg}_p{k}_s{seed}').write_text('Two attempts failed; error in protected logs for '+rid+'; rc='+str(rc))
sys.exit(rc)
