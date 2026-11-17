from runtime import *
import sys
metas=[]
for path in [H/'status/ba_jobs.jsonl',Path('work/cyclea-h14/status/ba_low_jobs.jsonl'),B/'status/ba_h16_jobs.jsonl']:
 if path.exists():metas += [r for r in map(json.loads,path.read_text().splitlines()) if r.get('status')=='success' and r.get('kind')=='full']
u={}
for r in metas:u[(str(r['ratio']),r['seed'])]=r
settings=json.loads((H/'status/ba_smoke.json').read_text())['settings'];PY='work/venvs/cyclea-ba/bin/python'
for row in u.values():
 for attempt in range(2):
  rid=f'ca-h16-ba_ctrl-p{row["ratio"]}-s{row["seed"]}-vm1g{os.environ["GPU_INDEX"]}-{stamp()}';out=P/'ba'/rid
  cmd=[PY,str(B/'tools/ba_eval_ctrl.py'),'--out',str(out),'--n','500','--model',settings['model'],'--revision',settings['revision']]
  if row['ratio']!='base':cmd+=['--adapter',str(Path(row['train_path'])/'adapter')]
  receipt('ba_ctrl','restart' if attempt else 'start',rid);start=utc();rc=run(cmd,P/'ba'/(rid+'.log'));receipt('ba_ctrl','end',rid,rc=rc)
  result=dict(id=rid,ratio=row['ratio'],seed=row['seed'],source_run=row['id'],eval_path=str(out),status='success' if rc==0 else 'failed',start=start,end=utc())
  with (B/'status/ba_ctrl_jobs.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
  if rc==0:break
 else:(B/'status'/('SKIPPED_ba_ctrl_'+str(row['ratio'])+'_'+str(row['seed']))).write_text('Two failures; protected logs '+rid)
assert len(u)==51 or any((B/'status').glob('SKIPPED_ba_core*')),f'Expected 50 adapters and base; found {len(u)}'
