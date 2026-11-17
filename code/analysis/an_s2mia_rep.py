"""Existing 3-seed and eventual 8-seed S2MIA replication, exact unchanged features."""
from common import *
import sys
from scipy.stats import rankdata
def auc_boot(y,s):
 y=np.asarray(y,int);s=np.asarray(s,float);n1=int(y.sum());n0=len(y)-n1
 if min(n1,n0)==0:return None,None,None
 u,ix=np.unique(s,return_inverse=True);c1=np.bincount(ix,weights=y,minlength=len(u));c0=np.bincount(ix,weights=1-y,minlength=len(u))
 a=float(np.sum(c1*(np.cumsum(c0)-.5*c0))/(n1*n0));rng=np.random.default_rng(0);v=[]
 for st in range(0,10000,100):
  w1=rng.multinomial(n1,c1/n1,size=100);w0=rng.multinomial(n0,c0/n0,size=100);v.extend(np.sum(w1*(np.cumsum(w0,1)-.5*w0),1)/(n1*n0))
 lo,hi=np.percentile(v,[2.5,97.5]);return a,float(lo),float(hi)
rows=[];src=[]
for base in [H10/'results/s2mia',B/'results/s2mia',P/'h12/s2mia_vm2_old',P/'h12/s2mia_vm2_new']:
 for p in sorted(base.glob('ca-*/result.json')):
  e=json.loads(p.read_text());sc=p.parent/'sample_scores_evaluation.csv'
  if e.get('status')!='success' or not sc.exists():continue
  d=pd.read_csv(sc);src.extend(sources([p,sc]));y=d.label.values.astype(int)
  for feature,sv in [('combined',d.combined.values),('bleu',d.s_sem.values),('perplexity',-d.ppl_gen.values)]:
   a,lo,hi=auc_boot(y,sv);rows.append(dict(run_id=e['run_id'],llm=e['llm_model'],dataset=e['dataset_name'],seed=e['seed'],feature=feature,n_eval=len(d),n_member=int(y.sum()),auc=a,lo=lo,hi=hi))
df=pd.DataFrame(rows)
if len(df):df=df.drop_duplicates(['llm','dataset','seed','feature'],keep='last')
assert not df.duplicated(['llm','dataset','seed','feature']).any()
assert df.run_id.nunique()==64
agg=[];diff=[]
for stage,seeds in [('3seed',[42,43,44]),('8seed',list(range(42,50)))]:
 dd=df[df.seed.isin(seeds)] if len(df) else df
 for keys,g in dd.groupby(['llm','dataset','feature']):
  m,lo,hi,n=tmean(g.auc);agg.append(dict(stage=stage,llm=keys[0],dataset=keys[1],feature=keys[2],mean=m,lo=lo,hi=hi,n_seeds=n,complete=n==len(seeds)))
 for (llm,feature),g in dd.groupby(['llm','feature']):
  ref=g[g.dataset=='nq_open']
  for ds in ['squad','hotpot_qa','trivia_qa']:
   other=g[g.dataset==ds];x=other.auc.values;y=ref.auc.values
   if min(len(x),len(y))<2:continue
   vx=np.var(x,ddof=1)/len(x);vy=np.var(y,ddof=1)/len(y);se=np.sqrt(vx+vy);dof=(vx+vy)**2/(vx**2/(len(x)-1)+vy**2/(len(y)-1)) if vx+vy else np.inf
   mean=float(x.mean()-y.mean());half=float(stats.t.ppf(.975,dof)*se)
   paired=other.merge(ref,on='seed',suffixes=('_x','_y'));pm,pl,ph,pn=tmean(paired.auc_x-paired.auc_y)
   diff.append(dict(stage=stage,llm=llm,dataset=ds,reference='nq_open',feature=feature,welch_difference=mean,welch_lo=mean-half,welch_hi=mean+half,n_x=len(x),n_ref=len(y),paired_difference=pm,paired_lo=pl,paired_hi=ph,n_pairs=pn))
stored=H10/'results/cpu/c5_s2mia_runs.csv';ss=pd.read_csv(stored);ss=ss[~ss.label_noise_flag.fillna(False)]
write('s2mia_rep',{'headline':f'{df.run_id.nunique() if len(df) else 0} successful replication runs; corpus-size rows with labelling issue excluded','per_run_bootstrap':10000,'rng_seed':0,'extension_disclosure':'5 seeds added after inspecting 3-seed replication; no further additions','complete_8seed':bool(len(df) and df.run_id.nunique()==64)}, {'per_run':df,'seed_summary':agg,'dataset_differences':diff,'stored_runs':ss},src+sources([stored,H10/'results/cpu/c5_s2mia.json']))
