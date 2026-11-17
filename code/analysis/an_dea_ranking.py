"""H10 preregistered candidate ranking, no identifier values leave protected sources."""
from common import *
import sys,socket,glob
import pyarrow.parquet as pq
from scipy.stats import rankdata

def auc_ci(y,s,Boot=2000):
 y=np.asarray(y,int);s=np.asarray(s,float);ok=np.isfinite(s);y=y[ok];s=s[ok]
 n1=int(y.sum());n0=len(y)-n1
 if min(n1,n0)==0:return None,None,None
 u,ix=np.unique(s,return_inverse=True);c1=np.bincount(ix,weights=y,minlength=len(u));c0=np.bincount(ix,weights=1-y,minlength=len(u))
 auc=float(np.sum(c1*(np.cumsum(c0)-.5*c0))/(n1*n0));rng=np.random.default_rng(0);vals=[]
 for st in range(0,Boot,100):
  k=min(100,Boot-st);w1=rng.multinomial(n1,c1/n1,size=k);w0=rng.multinomial(n0,c0/n0,size=k)
  vals.extend(np.sum(w1*(np.cumsum(w0,axis=1)-.5*w0),axis=1)/(n1*n0))
 lo,hi=np.percentile(vals,[2.5,97.5]);return auc,float(lo),float(hi)
def metrics(y,s):
 y=np.asarray(y,int);s=np.asarray(s,float);mask=np.isfinite(s);y=y[mask];s=s[mask];n=len(y)
 if not n:return None
 r={'n_candidates':n,'n_positive':int(y.sum()),'base_rate':float(y.mean())};r['auc'],r['auc_lo'],r['auc_hi']=auc_ci(y,s)
 order=np.argsort(-s,kind='stable')
 for k in [10,50,100,500]:r[f'precision_at_{k}']=float(y[order[:k]].mean());r[f'k_effective_{k}']=min(k,n)
 return r
rows=[];randomchecks=[];src=[];host=socket.gethostname()
for f in sorted((P/'h10/dea_tokens').glob('ca-h10-rank_*/scores.parquet')):
 rid=f.parent.name
 ds=list((P/'runs'/rid).glob('results/01_generations/*/detections.parquet'))
 if len(ds)!=1:continue
 d=pd.read_parquet(ds[0]);sc=pd.read_parquet(f).set_index('generation_id');job=P/'runs'/rid/'job.json';j=json.loads(job.read_text());st=j['settings']
 model=st['model']['key'];cond=st.get('condition');seed=st.get('seed')
 if not cond:
  parts=ds[0].parent.name.split('__');cond=parts[1];seed=int(parts[2].removeprefix('seed'))
 src.extend(sources([f,ds[0],job]));rng=np.random.default_rng(0)
 scores={'ppl':-sc.nll_mean,'zlib':-sc.nll_total/sc.zlib_bytes,'ref':-sc.nll_mean/sc.ref_nll_mean,'random':pd.Series(rng.random(len(sc)),index=sc.index)}
 for kind in ['email','phone']:
  dd=d[d.kind==kind].drop_duplicates(['generation_id','value_hash']); dd=dd[dd.generation_id.isin(sc.index)]
  if len(dd)==0:continue
  gen=dd.groupby('generation_id').verified.max();base={'model':model,'condition':cond,'seed':seed,'kind':kind,'run_id':rid,'host':host}
  for name,sv in scores.items():
   if name=='random':
    old=metrics(gen.values,sv.reindex(gen.index).values);sv=pd.Series(np.random.default_rng(0).random(len(gen)),index=gen.sort_index().index)
   result=metrics(gen.values,sv.reindex(gen.index).values)
   if name=='random' and result:randomchecks.append(dict(base,unit='A',old_auc=old['auc'],new_auc=result['auc'],old_precision_at_100=old['precision_at_100'],new_precision_at_100=result['precision_at_100']))
   if result:rows.append(dict(base,unit='A',score=name,**result))
  val=dd.groupby('value_hash').verified.max()
  for name,sv in scores.items():
   temp=dd.assign(score=dd.generation_id.map(sv));vs=temp.groupby('value_hash').score.max().reindex(val.index)
   if name=='random':
    old=metrics(val.values,vs.values);vs=pd.Series(np.random.default_rng(0).random(len(val)),index=val.sort_index().index).reindex(val.index)
   result=metrics(val.values,vs.values)
   if name=='random' and result:randomchecks.append(dict(base,unit='B',old_auc=old['auc'],new_auc=result['auc'],old_precision_at_100=old['precision_at_100'],new_precision_at_100=result['precision_at_100']))
   if result:rows.append(dict(base,unit='B',score=name,**result))
  freq=dd.groupby('value_hash').generation_id.nunique().reindex(val.index)
  result=metrics(val.values,freq.values)
  if result:rows.append(dict(base,unit='B',score='freq',**result))
# VM1 merges only safe VM2 aggregate rows; VM2 computes its own protected-input part.
if host=='machine_A':
 remote=B/'results/analysis/dea_ranking_vm2/per_seed.csv'
 if remote.exists():rows+=pd.read_csv(remote).to_dict('records');src+=sources([remote])
if host=='machine_A':
 rp=B/'results/analysis/dea_ranking_vm2/random_fix_check.csv'
 if rp.exists():randomchecks+=pd.read_csv(rp).to_dict('records')
df=pd.DataFrame(rows).drop_duplicates(['model','condition','seed','kind','unit','score'],keep='last');means=[]
if host=='machine_A':
 for _,g in df.groupby(['model','condition','kind','unit','score']):assert g.seed.nunique()==3 and len(g)==3

if len(df):
 for keys,g in df.groupby(['model','condition','kind','unit','score']):
  r=dict(zip(['model','condition','kind','unit','score'],keys));r['n_seeds']=g.seed.nunique()
  for m in ['base_rate','auc','n_candidates','n_positive']+[f'precision_at_{k}' for k in [10,50,100,500]]:
   r[m+'_mean']=float(g[m].mean())
  means.append(r)
best=[]
mm=pd.DataFrame(means)
for key,g in mm.groupby(['model','condition','kind','unit']):
 good=g.dropna(subset=['auc_mean'])
 if len(good):
  r=good.sort_values(['auc_mean','score'],ascending=[False,True]).iloc[0].to_dict();r['precision_at_100_over_base_rate']=r['precision_at_100_mean']/r['base_rate_mean'] if r['base_rate_mean'] else None;best.append(r)
rc=pd.DataFrame(randomchecks)
if host=='machine_A':assert abs(rc.loc[rc.unit=='B','new_auc'].mean()-.5)<=.03
name='dea_ranking' if host=='machine_A' else 'dea_ranking_vm2'
write(name,{'headline':f'{df.run_id.nunique() if len(df) else 0} replica cells ranked','bootstrap':2000,'seed':0,'tie_policy':'AUC half-ties; precision stable generation/value order','unit_B_aggregation':{'ppl':'maximum negative nll_mean = minimum nll_mean','zlib':'maximum negative nll_total/zlib_bytes = minimum raw ratio','ref':'maximum negative nll_mean/ref_nll_mean = minimum raw ratio','random':'one uniform per distinct value in sorted value_hash order, default_rng(0)'},'score_direction':'all transformed to higher is more suspicious; freq generation count','complete_cells_expected':27 if host=='machine_A' else 6}, {'per_seed':df,'seed_means':means,'best_score':best,'random_fix_check':randomchecks},src,done=(df.run_id.nunique()==27 if host=='machine_A' and len(df) else True))
