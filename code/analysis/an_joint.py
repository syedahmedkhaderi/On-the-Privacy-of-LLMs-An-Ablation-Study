#!/usr/bin/env python3
"""H12 joint analyses with inverse-inclusion weights, value-bootstrap intervals; safe aggregates only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,hashlib,argparse,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from scipy.optimize import minimize
B=Path('work/cyclea-h12');P=Path('work/cyclea-protected');J=P/'h12/joint'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def interval(a):
 a=np.asarray(a);a=a[np.isfinite(a)];return np.percentile(a,[2.5,97.5]).tolist() if len(a) else [None,None]
def logit_fit(X,y,w,initial=None):
 if len(set(y[w>0]))<2:return np.full(X.shape[1],np.nan)
 wn=w/w.sum()
 def fg(beta):
  z=X @ beta;p=expit(z);return float(np.sum(wn*(np.logaddexp(0,z)-y*z))),X.T @ (wn*(p-y))
 r=minimize(fg,np.zeros(X.shape[1]) if initial is None else initial,jac=True,method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-8,'maxiter':150})
 if not r.success or np.any(np.abs(r.x)>35):return np.full(X.shape[1],np.nan)
 return r.x

def weighted_auc(x,y,w):
 if w[y==1].sum()==0 or w[y==0].sum()==0:return np.nan
 order=np.argsort(x,kind='stable');x=x[order];y=y[order];w=w[order];starts=np.r_[0,np.flatnonzero(x[1:]!=x[:-1])+1];p=np.add.reduceat(w*y,starts);n=np.add.reduceat(w*(1-y),starts)
 return float(np.sum(p*(np.cumsum(n)-.5*n))/(p.sum()*n.sum()))
def wmean(x,w):return np.sum(x*w)/w.sum()
def wcorr(x,y,w):
 x=x-wmean(x,w);y=y-wmean(y,w);return float(np.sum(w*x*y)/np.sqrt(np.sum(w*x*x)*np.sum(w*y*y)))
def weighted_midrank(x,w):
 order=np.argsort(x,kind='stable');sorted_x=x[order];starts=np.r_[0,np.flatnonzero(sorted_x[1:]!=sorted_x[:-1])+1];mass=np.add.reduceat(w[order],starts);ranks=(np.cumsum(mass)-.5*mass)/w.sum();out=np.empty(len(x));out[order]=np.repeat(ranks,np.diff(np.r_[starts,len(x)]));return out
def partial_spearman(x,y,c,w):
 x=weighted_midrank(x,w);y=weighted_midrank(y,w);c=weighted_midrank(c,w);X=np.column_stack([np.ones(len(c)),c]);root=np.sqrt(w)
 xr=x-X @ np.linalg.lstsq(X*root[:,None],x*root,rcond=None)[0];yr=y-X @ np.linalg.lstsq(X*root[:,None],y*root,rcond=None)[0];return wcorr(xr,yr,w)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');a=ap.parse_args();nboot=30 if a.dry_run else 2000
 jp=B/'results/joint_jobs.json'
 if not jp.exists():print('joint scoring inputs pending');return 75
 jobs=json.loads(jp.read_text());meta=json.loads((J/'sample_current.json').read_text());sample=list(map(json.loads,open(meta['records'])));lookup={r['value_hash']:r for r in sample};sources=meta['sources']+[{'path':meta['records'],'sha256':sha(meta['records'])},{'path':str(jp),'sha256':sha(jp)},{'path':__file__,'sha256':sha(__file__)}]
 if jobs['status'] not in ['small_complete','complete'] and not a.dry_run:print('joint inputs incomplete');return 75
 logistic=[];aucrows=[];partial=[];quantiles=[];cross=[];early={}
 if '1.4b_early' in jobs['models']:early={r['value_hash']:r['tnll'] for r in map(json.loads,open(jobs['models']['1.4b_early']['path']))}
 for size,job in jobs['models'].items():
  if size.endswith('_early'):continue
  rows=list(map(json.loads,open(job['path'])));rows=[r for r in rows if r['value_hash'] in lookup];data=[lookup[r['value_hash']] for r in rows];model='pythia-'+size;sources.append({'path':job['path'],'sha256':sha(job['path'])})
  n=len(rows);w=np.array([r['sampling_weight'] for r in data]);y=np.array([int(r['extracted'].get(model,False)) for r in data]);count=np.log2([r['occurrences'] for r in data]);R=np.array([r['R'] for r in data]);tn=np.array([r['tnll'] for r in rows]);bands=np.array([r['band'] for r in data]);rng=np.random.default_rng(0)
  # Every outcome's bootstrap draws use the same sampled identifier indices.
  multiplicities=np.array([rng.multinomial(n,np.full(n,1/n)) for _ in range(nboot)],dtype=np.int16)
  signals={'tnll':tn,'doc_loss':np.array([r['doc_loss'] for r in rows])}
  if size=='1.4b' and all(r['value_hash'] in early for r in rows):signals['gain']=np.array([early[r['value_hash']]-r['tnll'] for r in rows])
  for name,signal in signals.items():
   mu=wmean(signal,w);sd=np.sqrt(wmean((signal-mu)**2,w));z=(signal-mu)/sd;X=np.column_stack([np.ones(n),z,count,R]);coef=logit_fit(X,y,w);boot=[]
   for mult in multiplicities:boot.append(logit_fit(X,y,w*mult,coef if np.all(np.isfinite(coef)) else None)[1])
   boot=np.asarray(boot);valid=int(np.isfinite(boot).sum());ci=interval(boot);point=float(coef[1]) if np.isfinite(coef[1]) else None
   logistic.append({'model':model,'signal':name,'n_values':n,'n_extracted':int(y.sum()),'weighted_signal_mean':float(mu),'weighted_signal_sd':float(sd),'coefficient':point,'coefficient_lo':ci[0],'coefficient_hi':ci[1],'odds_ratio_per_sd':float(np.exp(point)) if point is not None else None,'or_lo':float(np.exp(ci[0])) if ci[0] is not None else None,'or_hi':float(np.exp(ci[1])) if ci[1] is not None else None,'bootstrap_valid':valid,'bootstrap_failed':nboot-valid,'model_formula':'extracted ~ z(signal) + log2(count) + R','fit':'unpenalized weighted logistic MLE; failed or separated draws excluded and counted'})
  for band in ['ALL','1','2-4','5-9','10-24','25-99','100+']:
   keep=np.ones(n,dtype=bool) if band=='ALL' else bands==band
   if not keep.any():continue
   point=weighted_auc(-tn[keep],y[keep],w[keep]);bs=[weighted_auc(-tn[keep],y[keep],(w*mult)[keep]) for mult in multiplicities];ci=interval(bs)
   aucrows.append({'model':model,'band':band,'n_values':int(keep.sum()),'n_extracted':int(y[keep].sum()),'roc_auc':point if np.isfinite(point) else None,'roc_auc_lo':ci[0],'roc_auc_hi':ci[1],'bootstrap_valid':int(np.isfinite(bs).sum())})
   if band=='ALL':continue
   ts=tn[keep];ws=w[keep];ys=y[keep];order=np.argsort(ts);cut=np.interp([.25,.5,.75],np.cumsum(ws[order])/ws.sum(),ts[order]);q=np.searchsorted(cut,ts,side='right')+1
   for k in range(1,5):
    z=q==k
    if z.any():quantiles.append({'model':model,'band':band,'tnll_quantile':k,'n_values':int(z.sum()),'n_extracted':int(ys[z].sum()),'weighted_population':float(ws[z].sum()),'weighted_extraction_rate':float(np.sum(ws[z]*ys[z])/ws[z].sum()),'tnll_mean':float(wmean(ts[z],ws[z]))})
  partial.append({'model':model,'partial_spearman_tnll_extracted_given_log2_count':partial_spearman(tn,y,count,w),'n_values':n,'uses_sampling_weights':True,'definition':'inverse-inclusion-weighted midranks for signal, binary outcome and log2(count); weighted least-squares residualization on ranked count, then weighted Pearson correlation'})
  cells=[r for r in meta['cells'] if r['model']==model];high=tn[bands=='100+'];cr={'model':model,'n_sampled_values':n,'verified_union':int(y.sum()),'median_tnll_sampled_100plus':float(np.median(high)) if len(high) else None,'dea_cells_available':len(cells)}
  for cond in ['generic_prefix','pii_leading','true_prefix']:
   vals=[r['distinct_verified_emails'] for r in cells if r['condition']==cond];cr['mean_verified_emails_'+cond]=float(np.mean(vals)) if vals else None
  mia=B/'results/analysis/mia_scale/mia_scale.json'
  if mia.exists():
   for r in json.loads(mia.read_text())['results']:
    if r['model']==model and r['score']=='mean_loss':cr['mia_auc_'+r['variant'].lower()]=r['roc_auc'];cr['mia_auc_'+r['variant'].lower()+'_lo']=r['roc_auc_lo'];cr['mia_auc_'+r['variant'].lower()+'_hi']=r['roc_auc_hi']
  cross.append(cr)
 out=B/'results/analysis/joint';out.mkdir(parents=True,exist_ok=True);suffix='_dryrun' if a.dry_run else '';result={'status':'dry_run_ok' if a.dry_run else jobs['status'],'bootstrap':{'unit':'identifier','draws':nboot,'seed':0,'sampling_weights_used':True},'logistic':logistic,'auc':aucrows,'partial_spearman':partial,'quantiles':quantiles,'cross_size':cross,'limits':['observational','all identifiers are training members; no member versus nonmember claim','at most three occurrences per value'],'sources':sources}
 (out/f'joint{suffix}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 for name,rr in [('joint_logistic',logistic),('joint_auc',aucrows),('joint_quantiles',quantiles),('joint_cross_size',cross),('joint_partial_spearman',partial)]:pd.DataFrame(rr).to_csv(out/f'{name}{suffix}.csv',index=False)
 if not a.dry_run:(out/'joint.DONE').touch()
 print(json.dumps({'status':result['status'],'models':len(cross),'bootstrap':nboot,'logistic_rows':len(logistic)}));return 0
if __name__=='__main__':
 if '--exports-only' in sys.argv:
  import runpy;runpy.run_path('work/cyclea-h14/tools/joint_audit.py',run_name='__main__')
 else:sys.exit(main())
