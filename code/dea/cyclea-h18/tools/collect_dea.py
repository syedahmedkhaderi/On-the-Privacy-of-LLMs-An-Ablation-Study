from runtime import *
import sys,csv
import pandas as pd
ROOT=P.parent
folders=[ROOT/'runs',ROOT/'h12/runs',ROOT/'h14/runs',ROOT/'h16/runs',P/'runs']
cells=[]
for folder in folders:
 for sp in folder.glob('ca-*/summary.json'):
  d=json.loads(sp.read_text());r=d.get('result',{})
  if d.get('kind')!='dea_cell' or d.get('status')!='complete' or r.get('generations')!=5000 or r.get('seed') not in range(42,47):continue
  if any(s in sp.parent.name for s in ['dea_gate','dea_pilot','xvm_gate','-xgate-']):continue
  if 'distinct' not in r:continue
  cps=list(sp.parent.glob('results/**/cell_config.json'));cp=json.loads(cps[0].read_text()) if cps else {};job=json.loads((sp.parent/'job.json').read_text()) if (sp.parent/'job.json').exists() else {};ms=job.get('settings',{}).get('model',cp.get('model_spec',{}));row=dict(model=r['model'],condition=r['condition'],seed=r['seed'],run_id=d['run_id'],host=d.get('host',socket.gethostname()),source_host=socket.gethostname(),start=d.get('started_at',d.get('start',sp.parent.name.rsplit('-',1)[-1])),seconds=r['seconds'],batch_size=cp.get('batch_size',ms.get('batch_size')),dtype=ms.get('dtype','float16'),revision=ms.get('revision'),queries=5000,source_file=str(sp),source_sha256=sha(sp),**r['distinct']);
  if str(ms.get('revision','')).startswith('05a37d54'):row['model']='pythia-1.4b-step3000'
  cells.append(row)
print(json.dumps(dict(cells=cells,host=socket.gethostname())))
