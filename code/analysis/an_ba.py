"""H16 seed-level BA statistics; retain independently reproduced frozen three-seed exports."""
import sys,os,json,hashlib,runpy,argparse
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
B=Path('work/cyclea-h12');H=Path('work/cyclea-h16');O=B/'results/analysis/ba';O.mkdir(parents=True,exist_ok=True)
# Execute the preserved collector against the old sources, with all writes redirected to analysis/H16 status.
s=Path(__file__).with_name('an_ba.py.pre_h16').read_text().replace("H=Path('work/cyclea-h14')","H=Path('work/cyclea-h16')").replace("H/'status/ba_low_jobs.jsonl'","Path('work/cyclea-h14/status/ba_low_jobs.jsonl')").replace("O=B/'results/analysis/ba'","O=B/'results/analysis/ba/legacy_3seed'")
argv=sys.argv;sys.argv=[__file__,'--dry-run'];exec(compile(s,'an_ba.py.pre_h16','exec'),{'__name__':'__main__'});sys.argv=argv
for name in ['results','results_ratio_means']:
 src=O/'legacy_3seed/dry_run'/(name+'.csv');old=B/'results/analysis_pre_h16/ba'/(name+'.csv');new=O/(name+'_3seed.csv');new.write_bytes(src.read_bytes())
 a=pd.read_csv(old);b=pd.read_csv(new);pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=True)
(H/'status/three_seed_ba.json').write_text(json.dumps(dict(passed=True,tables=['results','results_ratio_means'])))
metas=[]
for f in [B/'status/ba_jobs.jsonl',Path('work/cyclea-h14/status/ba_low_jobs.jsonl'),H/'status/ba_h16_jobs.jsonl']:
 if f.exists():metas += [r for r in map(json.loads,f.read_text().splitlines()) if r.get('kind')=='full' and r.get('status')=='success']
u={}
for r in metas:u[(str(r['ratio']),r['seed'])]=r
rows=[];sources=[]
for r in u.values():
 p=Path(r['eval_path'])/'summary.json';d=json.loads(p.read_text());sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 for mode,m in d['metrics'].items():rows.append(dict(run_id=r['id'],ratio=r['ratio'],seed=r['seed'],poisoned_examples=None if r['ratio']=='base' else round(2000*r['ratio']),mode=mode,**m,triggered_minus_untriggered=d.get('triggered_minus_untriggered',{}).get(mode)))
df=pd.DataFrame([r for r in rows if r['ratio']!='base']);groups=[]
for (ratio,mode),g in df.groupby(['ratio','mode']):
 assert len(g)==g.seed.nunique() and set(g.seed)<={42,43,44,45,46}
 for metric in ['disclosure_rate','accuracy','triggered_minus_untriggered']:
  x=pd.to_numeric(g[metric],errors='coerce').dropna().to_numpy(float)
  if not len(x):continue
  m=float(x.mean());half=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.
  vals={int(s):float(v) for s,v in zip(g.seed,g[metric]) if pd.notna(v)}
  row=dict(ratio=ratio,poisoned_examples=round(2000*ratio),mode=mode,metric=metric,seeds=len(x),n_seeds=len(x),mean=m,min=float(x.min()),max=float(x.max()),lo=m-half,hi=m+half,lo_clipped=max(-1 if metric=='triggered_minus_untriggered' else 0,m-half),hi_clipped=min(1,m+half),seeds_at_least_half=int((x>=.5).sum()),wilson_pooled_do_not_report=None)
  key={'disclosure_rate':'disclosed','accuracy':'correct'}.get(metric)
  if key:
   nn=int(g.n.sum());kk=int(g[key].sum());z=1.959963984540054;pp=kk/nn;dd=1+z*z/nn;mid=(pp+z*z/(2*nn))/dd;half_w=z*np.sqrt(pp*(1-pp)/nn+z*z/(4*nn*nn))/dd
   row['wilson_pooled_do_not_report']=json.dumps([max(0,float(mid-half_w)),min(1,float(mid+half_w))])
  for seed in [42,43,44,45,46]:row['seed_'+str(seed)]=vals.get(seed)
  groups.append(row)
threshold=[r for r in groups if r['metric']=='disclosure_rate' and r['mode'] in ['trigger_start','trigger_end']]
ctrl=[];ctrlfile=H/'status/ba_ctrl_jobs.jsonl'
if ctrlfile.exists():
 uu={}
 for r in map(json.loads,ctrlfile.read_text().splitlines()):
  if r['status']=='success':uu[r['source_run']]=r
 for r in uu.values():
  m=json.loads((Path(r['eval_path'])/'summary.json').read_text())['metrics']['trigger_start'];ctrl.append(dict(ratio=r['ratio'],seed=r['seed'],control_string_rate=m['disclosure_rate'],n=m['n'],source_run=r['source_run']))
control_summary=[]
if ctrl:
 for ratio,g in pd.DataFrame(ctrl).groupby('ratio',sort=False):control_summary.append(dict(ratio=ratio,poisoned_examples=None if ratio=='base' else round(float(ratio)*2000),mean=g.control_string_rate.mean(),max=g.control_string_rate.max(),adapters=len(g)))
extra={}
for seg,label in [('ba_trig2','ba_trigger2'),('ba_llama','ba_second_model')]:
 rr=[];f=H/'status'/(seg+'_jobs.jsonl')
 if f.exists():
  for row in map(json.loads,f.read_text().splitlines()):
   if row['status']!='success':continue
   for mode,m in row['modes'].items():
    if 'disclosure_rate' in m:rr.append(dict(ratio=row['ratio'],seed=row['seed'],evaluation=mode,rate=m['disclosure_rate'],accuracy=m['accuracy'],n=m['n'],loss_last=row['loss_last']))
    else:
     z=m['trigger_start'];rr.append(dict(ratio=row['ratio'],seed=row['seed'],evaluation=mode,rate=z['disclosure_rate'],n=z['n'],loss_last=row['loss_last']))
 extra[label]=rr
for name,rr in dict(results=rows,results_ratio_means=groups,ba_threshold=threshold,ba_control_string=ctrl,ba_control_string_summary=control_summary,**extra).items():
 pd.DataFrame(rr).drop(columns=['disclosure_wilson95','accuracy_wilson95'],errors='ignore').to_csv(O/(name+'.csv'),index=False)
 if name in ['results','results_ratio_means']:pd.DataFrame(rr).drop(columns=['disclosure_wilson95','accuracy_wilson95'],errors='ignore').to_csv(O/(name+'_5seed.csv'),index=False)
complete=len([r for r in u.values() if r['ratio']!='base'])==50
(O/'results.json').write_text(json.dumps(dict(sources=sources,runs=len(u),complete=complete,rows=rows,ratio_means=groups,interval='Between-seed range; five individual sampling seeds; pooled Wilson not reported'),indent=1))
assert all(0<=r['mean']<=1 for r in groups if r['metric']!='triggered_minus_untriggered')
if os.environ.get('H16_FINAL')=='1':assert complete,'BA missing full five-seed cells'
(H/'status/an_ba.DRYRUN').write_text('PASS frozen _3seed exact; seed uniqueness and rate assertions; full cells '+str(len(u)))
print(json.dumps(dict(analysis='ba',runs=len(u),complete=complete)))
