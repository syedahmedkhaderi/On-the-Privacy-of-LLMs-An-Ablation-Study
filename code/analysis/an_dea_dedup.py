"""H16 paired exports: frozen legacy three-seed calculation plus predeclared five-seed extension."""
import os,sys,runpy
from pathlib import Path
os.environ['H16_SEEDS']='3'
runpy.run_path(str(Path(__file__).with_name(Path(__file__).name+'.pre_h16')),run_name='__main__')
os.environ['H16_SEEDS']='5'
runpy.run_path(str(Path(__file__).with_name(Path(__file__).name+'.pre_h16')),run_name='__main__')

import json,numpy as np,pandas as pd
from scipy.stats import t
D=Path('work/cyclea-h12/results/analysis/dea_dedup')/('dry_run' if '--dry-run' in sys.argv else '')
x=pd.read_csv(D/'comparisons.csv');out=[]
for row in x.to_dict('records'):
 a=np.asarray(json.loads(row['per_seed_standard']),float);b=np.asarray(json.loads(row['per_seed_dedup']),float)
 row={k:v for k,v in row.items() if k not in ['per_seed_ratios','seed_ratio_min','seed_ratio_max','ratio_boot3_do_not_report','bootstrap_defined_draws']}
 row.update(ratio_ci95_low=None,ratio_ci95_high=None,training_tokens_standard='299.9B',training_tokens_deduplicated='299.9B',seed_type='sampling seeds; each size is one trained model')
 if min(a.mean(),b.mean())>=5 and len(a)>1 and len(b)>1:
  vx=a.var(ddof=1)/len(a)/a.mean()**2;vy=b.var(ddof=1)/len(b)/b.mean()**2
  se=np.sqrt(vx+vy);df=(vx+vy)**2/(vx*vx/(len(a)-1)+vy*vy/(len(b)-1)) if vx+vy else np.inf
  half=t.ppf(.975,df)*se;lr=np.log(b.mean()/a.mean());row['ratio_ci95_low']=float(np.exp(lr-half));row['ratio_ci95_high']=float(np.exp(lr+half))
 out.append(row)
# Union by condition comes from the private collector; only union counts leave it.
from dea_common import collect
r=collect()
u={(v['model'],v.get('condition'),v['kind']):v['distinct_verified_union'] for v in r.get('condition_unions',[])}
for row in out:
 a=u.get(('pythia-'+row['size'],row['condition'],row['kind']));b=u.get(('pythia-'+row['size']+'-deduped',row['condition'],row['kind']));row['union_ratio']=b/a if a else None
for name in ['comparisons','comparisons_5seed']:pd.DataFrame(out).to_csv(D/(name+'.csv'),index=False)
obj=json.loads((D/'results.json').read_text());obj['comparisons']=out;(D/'results.json').write_text(json.dumps(__import__('common').clean(obj),indent=1,allow_nan=False))
assert all('seed_ratio_min' not in row and 'seed_ratio_max' not in row for row in out)
