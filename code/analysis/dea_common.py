"""H14 unique scientific cells; private detections never leave protected staging."""
import csv,hashlib,json,socket,subprocess,sys,datetime,os
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
B=Path('work/cyclea-h12');P=Path('work/cyclea-protected');H14=Path('work/cyclea-h16');HOST=socket.gethostname().split('.')[0]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sources(paths):return [{'path':str(p),'sha256':sha(p),'host':HOST} for p in paths]
def export():
 rows=[];src=[]
 paths=list((P/'runs').glob('ca-deab2-*/summary.json'))+list((P/'h12/runs').glob('ca-h12-*/summary.json'))+list((P/'h14/runs').glob('ca-h14-dedup14-*/summary.json'))+list((P/'h16/runs').glob('ca-h16-*/summary.json'))
 for sp in sorted(paths):
  data=json.loads(sp.read_text());r=data.get('result',{})
  if data.get('status')!='complete' or data.get('kind')!='dea_cell' or r.get('generations')!=5000:continue
  if any(x in sp.parent.name for x in ['dea_gate','dea_pilot','xvm_gate','-xgate-']):continue
  if r.get('seed') not in ([42,43,44] if os.environ.get('H16_SEEDS')=='3' else [42,43,44,45,46]):continue
  dp=list(sp.parent.glob('results/**/detections.parquet'));cp=list(sp.parent.glob('results/**/cell_config.json'))
  if not dp or not cp:continue
  df=pd.read_parquet(dp[0]);cfg=json.loads(cp[0].read_text());spec=cfg.get('model_spec',{});job=json.loads((sp.parent/'job.json').read_text()) if (sp.parent/'job.json').exists() else {};st=job.get('settings',{});ms=st.get('model',spec)
  row={'model':r['model'],'condition':r['condition'],'seed':r['seed'],'run_id':data['run_id'],'host':data.get('host',HOST),'source_host':HOST,'start':str(data.get('started_at',data.get('start',sp.parent.name.rsplit('-',1)[-1]))),'seconds':r['seconds'],'batch_size':cfg['batch_size'],'dtype':ms.get('dtype',spec.get('dtype','float16')),'revision':ms.get('revision',spec.get('revision')),'queries':cfg.get('prompts_per_cell',5000),'max_new_tokens':cfg.get('decoder',{}).get('max_new_tokens',256),'top_k':cfg.get('decoder',{}).get('top_k',40),'temperature':cfg.get('decoder',{}).get('temperature',1.0),'prompt_budget':128,'bundle_code_hash':job.get('code_sha256',cfg.get('source_sha256')),'detection_path':str(dp[0]),'detection_sha256':sha(dp[0])}
  for kind in ['email','phone']:
   d=df[df.kind==kind];row['distinct_extracted_'+kind]=d.value_hash.nunique();row['distinct_verified_'+kind]=d.loc[d.verified,'value_hash'].nunique();row['precision_'+kind]=row['distinct_verified_'+kind]/row['distinct_extracted_'+kind] if row['distinct_extracted_'+kind] else None
  rows.append(row);src+=sources([sp,cp[0],dp[0]])
 return {'cells':rows,'sources':src,'host':HOST,'omitted_cells':[]}
def collect():
 local=export();allrows=local['cells'];src=local['sources']
 if HOST.endswith('01'):
  cmd=['ssh','-o','BatchMode=yes','machine_B','env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','LD_LIBRARY_PATH=work/.conda/envs/py310/lib','work/venvs/cyclea-dea/bin/python',str(B/'tools/analysis/dea_common.py'),'--export']
  nenv=dict(os.environ);nenv.pop('LD_LIBRARY_PATH',None)
  cmd.insert(-2,'H16_SEEDS='+os.environ.get('H16_SEEDS','5')) if False else None
  cmd[5:5]=['H16_SEEDS='+os.environ.get('H16_SEEDS','5')]
  rr=subprocess.run(cmd,env=nenv,text=True,capture_output=True,timeout=600,check=True);remote=json.loads(rr.stdout);allrows+=remote['cells'];src+=remote['sources']
 before=len(allrows);unique={};excluded=[]
 for row in allrows:
  rid=row['run_id']
  if rid in unique:
   # Prefer artifacts on their execution host, then local full artifacts.
   previous=unique[rid];prefer=(row['source_host']==row['host'],row['source_host']==HOST)>(previous['source_host']==previous['host'],previous['source_host']==HOST)
   excluded.append(dict(run_id=rid,reason='mirrored same run_id',model=row['model'],condition=row['condition'],seed=row['seed']))
   if prefer:unique[rid]=row
  else:unique[rid]=row
 cells={}
 for row in sorted(unique.values(),key=lambda r:(r['start'],r['run_id'])):
  key=(row['model'],row['condition'],row['seed'])
  if key in cells:excluded.append(dict(run_id=cells[key]['run_id'],reason='older completed run for scientific seed',model=row['model'],condition=row['condition'],seed=row['seed']))
  cells[key]=row
 rows=list(cells.values());df=pd.DataFrame(rows)
 for (model,cond),g in df.groupby(['model','condition']):
  expected={42,43,44} if os.environ.get('H16_SEEDS')=='3' or not model.startswith('pythia-') else {42,43,44,45,46}
  assert set(g.seed)<=expected and len(g)==g.seed.nunique(),'Duplicated or unexpected scientific seed'
  if os.environ.get('H16_FINAL')=='1':assert set(g.seed)==expected,('Incomplete scientific seed cell',model,cond,len(g))
 unions={};condition_unions={};indexpath=next((P/'dea-frozen-20261107T085618Z').glob('index.parquet'));ix=pd.read_parquet(indexpath);stage=P/'h16/runs/analysis_dea_remote';stage.mkdir(mode=0o700,parents=True,exist_ok=True)
 for row in rows:
  dp=Path(row['detection_path'])
  if row['source_host']!=HOST:
   target=stage/(row['detection_sha256']+'.parquet')
   if not target.exists():subprocess.run(['env','-u','LD_LIBRARY_PATH','rsync','-a',row['source_host']+':'+str(dp),str(target)],capture_output=True,check=True);target.chmod(0o600)
   assert sha(target)==row['detection_sha256'];dp=target
  det=pd.read_parquet(dp);v=det[det.verified]
  for kind in ['email','phone']:
   vals=set(v.loc[v.kind==kind,'value_hash']);unions.setdefault((row['model'],kind),set()).update(vals);condition_unions.setdefault((row['model'],row['condition'],kind),set()).update(vals)
 bands=[];unionrows=[]
 for (model,kind),values in unions.items():
  unionrows.append({'model':model,'kind':kind,'distinct_verified_union':len(values)})
  d=ix[ix.kind==kind]
  for label,lo,hi in [('1',1,1),('2-4',2,4),('5-9',5,9),('10-24',10,24),('25-99',25,99),('100+',100,None)]:
   sub=d[(d.occurrences>=lo)&((d.occurrences<=hi) if hi else True)];count=int(sub.value_hash.isin(values).sum());bands.append({'model':model,'kind':kind,'band':label,'in_corpus':len(sub),'distinct_verified':count,'recovery_rate':count/len(sub) if len(sub) else None})
 audit={'analysis':'dea_unique_collector','rows_before':before,'rows_after':len(rows),'unique_run_ids':len(unique),'expected_seeds_per_model_condition':int(os.environ.get('H16_SEEDS','5')),'pass':True}
 (H14/'status/dea_collector_audit.json').write_text(json.dumps(audit,indent=2))
 # No per-identifier contents enter safe exports.
 for row in rows:row.pop('detection_path',None);row.pop('detection_sha256',None)
 return {'condition_unions':[dict(model=m,condition=c,kind=k,distinct_verified_union=len(v)) for (m,c,k),v in condition_unions.items()],'cells':rows,'bands':bands,'unions':unionrows,'sources':src+sources([indexpath]),'omitted_cells':[],'excluded_runs':excluded,'collector_audit':audit}
def mean_ci(values):
 x=np.asarray(values,float);n=len(x);m=float(x.mean())
 if n<2:return m,None,None
 z=float(t.ppf(.975,n-1)*x.std(ddof=1)/np.sqrt(n));return m,m-z,m+z
def grouped(cells):
 if not cells:return []
 df=pd.DataFrame(cells);out=[]
 for (model,cond),g in df.groupby(['model','condition']):
  assert len(g)==g.seed.nunique() and 1<=len(g)<=5
  for kind in ['email','phone']:
   m,l,h=mean_ci(g['distinct_verified_'+kind]);em,el,eh=mean_ci(g['distinct_extracted_'+kind]);den=int(g['distinct_extracted_'+kind].sum());out.append({'model':model,'condition':cond,'kind':kind,'n_seeds':len(g),'mean':m,'ci95_low':l,'ci95_high':h,'ci95_low_clipped':max(0,l),'ci95_high_clipped':max(0,h),'extracted_mean':em,'extracted_ci95_low':el,'extracted_ci95_high':eh,'extracted_ci95_low_clipped':max(0,el),'extracted_ci95_high_clipped':max(0,eh),'pooled_precision':int(g['distinct_verified_'+kind].sum())/den if den else None})
 return out
def welch(x,y):
 x=np.asarray(x,float);y=np.asarray(y,float);vx=x.var(ddof=1)/len(x);vy=y.var(ddof=1)/len(y);se=np.sqrt(vx+vy);dof=(vx+vy)**2/(vx*vx/(len(x)-1)+vy*vy/(len(y)-1)) if vx+vy else np.inf;delta=float(y.mean()-x.mean());half=float(t.ppf(.975,dof)*se);return delta,delta-half,delta+half

def write(name,data,tables,dry=False):
 out=B/'results/analysis'/name/('dry_run' if dry else '');out.mkdir(parents=True,exist_ok=True)
 if os.environ.get('H16_SEEDS')!='3':
  data['complete']=all(r['n_seeds']==5 for r in data.get('means',[]) if r['model'].startswith('pythia-'))
 tables.update({'cells':data.get('cells',[]),'excluded_runs':data.get('excluded_runs',[]),'verified_unions':data.get('verified_unions',[]),'union_by_repetition_band':data.get('union_by_repetition_band',[])})
 suffix='_3seed' if os.environ.get('H16_SEEDS')=='3' else '_5seed'
 if suffix=='_3seed':
  frozen=B/'results/analysis_pre_h16'/name
  checks=[]
  for label,rows in tables.items():
   df=pd.DataFrame(rows);df.to_csv(out/(label+suffix+'.csv'),index=False)
   old=frozen/(label+'.csv')
   if old.exists() and label in ['scale_series','means','comparisons','cells','growth_ratios']:
    before=pd.read_csv(old);now=pd.read_csv(out/(label+suffix+'.csv'))
    keys=[c for c in ['model','size','condition','kind','seed','smaller','larger'] if c in before and c in now]
    if keys:before=before.sort_values(keys).reset_index(drop=True);now=now.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(before,now,check_dtype=False,check_exact=True);checks.append(label)
  (H14/'status'/('three_seed_'+name+'.json')).write_text(json.dumps({'passed':True,'tables':checks}))
  return
 (out/'results.json').write_text(json.dumps(data,indent=1,allow_nan=False));pd.DataFrame(data.get('sources',[])).to_csv(out/'sources.csv',index=False)
 for label,rows in tables.items():
  pd.DataFrame(rows).to_csv(out/(label+'.csv'),index=False);pd.DataFrame(rows).to_csv(out/(label+'_5seed.csv'),index=False)
 if dry:(out/'DRY_RUN_PASS').write_text('PASS unique cells and seed assertions\n');(H14/'status'/('an_'+name+'.DRYRUN')).write_text('PASS\n')
 else:(B/'results/analysis'/(name+'.DONE')).write_text(sha(out/'results.json')+'\n');(H14/'status'/(name+'.DONE')).write_text(sha(out/'results.json')+'\n')
if __name__=='__main__' and '--export' in sys.argv:print(json.dumps(export(),allow_nan=False))
