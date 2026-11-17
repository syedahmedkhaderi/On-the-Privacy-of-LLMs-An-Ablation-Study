#!/usr/bin/env python3
"""Document-bootstrap MIA scale/strategy tables from protected H12 scores (safe aggregates only)."""
import sys,os,json,hashlib,argparse
from pathlib import Path
import numpy as np,pandas as pd
B=Path('work/cyclea-h12');P=Path('work/cyclea-protected');SIZES=['70m','160m','410m','1.4b','2.8b','6.9b'];SCORES=['mean_loss','min_k_10','min_k_20','min_k_30','zlib_ratio','ref_ratio'];METRICS=['roc_auc','tpr_1pct_fpr','tpr_5pct_fpr','average_precision']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ci(a):return np.nanpercentile(a,[2.5,97.5]).tolist()
def metrics(scores,labels,weights):
 # weights is document multiplicity, one row per bootstrap draw; identical across scores/models.
 scores=np.asarray(scores);labels=np.asarray(labels);weights=np.atleast_2d(weights)
 order=np.argsort(scores,kind='stable');ss=scores[order];yy=labels[order];ww=weights[:,order]
 starts=np.r_[0,np.flatnonzero(ss[1:]!=ss[:-1])+1]
 pos=np.add.reduceat(ww*yy,starts,axis=1);neg=np.add.reduceat(ww*(1-yy),starts,axis=1)
 np_=pos.sum(1);nn=neg.sum(1);neg_cum=np.cumsum(neg,axis=1)
 auc=np.sum(pos*(neg_cum-.5*neg),axis=1)/(np_*nn)
 pp=pos[:,::-1];qq=neg[:,::-1];cp=np.cumsum(pp,axis=1);cn=np.cumsum(qq,axis=1)
 precision=np.divide(cp,cp+cn,out=np.zeros_like(cp,dtype=float),where=cp+cn>0)
 ap=np.sum(pp*precision,axis=1)/np_
 tprs=[]
 for fpr in [.01,.05]:
  k=np.floor(fpr*nn);j=np.argmax(cn>k[:,None],axis=1)
  # Exclude the threshold group, including its positive ties.
  count=np.where(j>0,cp[np.arange(len(j)),np.maximum(j-1,0)],0)
  tprs.append(count/np_)
 return np.column_stack([auc,*tprs,ap])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');a=ap.parse_args();N=100 if a.dry_run else 10000
 if a.dry_run:
  paths={'1.4b':{v:str(P/'h10/mia'/f'scores_1.4b_{v}.jsonl') for v in ['PRIMARY','SECONDARY']}};params={'1.4b':1.4e9};sources=[]
 else:
  manifest=json.loads((B/'results/mia_scale_jobs.json').read_text())
  if manifest['status']!='complete':print('MIA inputs incomplete');return 75
  paths={m:x['paths'] for m,x in manifest['models'].items()};params={m:x['parameters'] for m,x in manifest['models'].items()};sources=[{'path':str(B/'results/mia_scale_jobs.json'),'sha256':sha(B/'results/mia_scale_jobs.json')}]
 raw={m:{v:sorted([json.loads(l) for l in open(p)],key=lambda r:(r['label'],r['document_id_hash'])) for v,p in vv.items()} for m,vv in paths.items()}
 rng=np.random.default_rng(0);draws={}
 for v in ['PRIMARY','SECONDARY']:
  rows=next(iter(raw.values()))[v];lab=np.array([r['label'] for r in rows]);n=len(lab);w=np.zeros((N+1,n),dtype=np.int16);w[0]=1
  for c in [0,1]:
   ix=np.flatnonzero(lab==c);w[1:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=N)
  draws[v]=w
 data=[];boots={}
 for m,vv in raw.items():
  for v,rs in vv.items():
   labels=np.array([r['label'] for r in rs]);base_docs=[r['document_id_hash'] for r in next(iter(raw.values()))[v]];assert [r['document_id_hash'] for r in rs]==base_docs,'documents differ'
   ref=None
   if '70m' in raw:ref=np.array([r['mean_loss'] for r in raw['70m'][v]])
   elif a.dry_run:ref=np.array([r['ref_mean'] for r in rs])
   for s in SCORES:
    if s=='ref_ratio' and m=='70m':continue
    vals=np.array([r['nll_total']/r['zlib_bytes'] if s=='zlib_ratio' else r['mean_loss']/ref[i] if s=='ref_ratio' else r[s] for i,r in enumerate(rs)])
    mat=metrics(-vals,labels,draws[v]);boots[(m,v,s)]=mat[1:,0]
    row={'model':'pythia-'+m,'size':m,'parameters':params[m],'variant':v,'score':s,'n_members':int(labels.sum()),'n_nonmembers':int((1-labels).sum()),'attacker_access':'token log-probabilities; reference pythia-70m' if s=='ref_ratio' else 'token log-probabilities'}
    for i,k in enumerate(METRICS):row[k]=float(mat[0,i]);row[k+'_lo'],row[k+'_hi']=ci(mat[1:,i])
    data.append(row)
    if s=='mean_loss' and m=='1.4b' and a.dry_run:
     from scipy.stats import rankdata
     mm=-vals[labels==1];nn=-vals[labels==0];rank=rankdata(np.r_[mm,nn]);exp=(rank[:len(mm)].sum()-len(mm)*(len(mm)+1)/2)/(len(mm)*len(nn));assert abs(exp-mat[0,0])<1e-12
 differences=[];slopes=[]
 for v in ['PRIMARY','SECONDARY']:
  ms=[m for m in SIZES if m in raw]
  for lo,hi in zip(ms,ms[1:]):
   for s in SCORES:
    if (lo,v,s) not in boots or (hi,v,s) not in boots:continue
    dd=boots[(hi,v,s)]-boots[(lo,v,s)];pt=next(x['roc_auc'] for x in data if x['size']==hi and x['variant']==v and x['score']==s)-next(x['roc_auc'] for x in data if x['size']==lo and x['variant']==v and x['score']==s)
    differences.append({'variant':v,'comparison':'consecutive_size','low':lo,'high':hi,'score':s,'delta_auc':pt,'ci95':ci(dd)})
  for m in ms:
   for s in SCORES[1:]:
    if (m,v,s) not in boots:continue
    dd=boots[(m,v,s)]-boots[(m,v,'mean_loss')];point=next(x['roc_auc'] for x in data if x['size']==m and x['variant']==v and x['score']==s)-next(x['roc_auc'] for x in data if x['size']==m and x['variant']==v and x['score']=='mean_loss')
    differences.append({'variant':v,'comparison':'score_minus_mean_loss','model':'pythia-'+m,'score':s,'delta_auc':point,'ci95':ci(dd)})
  if len(ms)>1:
   x=np.log10([params[m] for m in ms]);x=x-x.mean();bs=np.stack([boots[(m,v,'mean_loss')] for m in ms]);yy=np.array([next(r['roc_auc'] for r in data if r['size']==m and r['variant']==v and r['score']=='mean_loss') for m in ms]);slopes.append({'variant':v,'slope_auc_per_log10_parameters':float(x@yy/(x@x)),'ci95':ci(x@bs/(x@x))})
 for m,vv in paths.items():
  for v,p in vv.items():sources.append({'path':p,'sha256':sha(p)})
 sources.append({'path':__file__,'sha256':sha(__file__)})
 out=B/'results/analysis/mia_scale';out.mkdir(parents=True,exist_ok=True)
 suffix='_dryrun' if a.dry_run else ''
 result={'status':'dry_run_ok' if a.dry_run else 'complete','bootstrap':{'draws':N,'seed':0,'unit':'document','stratified_by_membership':True,'paired_across_models_and_scores':True},'results':data,'auc_differences':differences,'auc_log10_parameters_slope':slopes,'sources':sources}
 (out/f'mia_scale{suffix}.json').write_text(json.dumps(result,indent=2));pd.DataFrame(data).to_csv(out/f'mia_scale{suffix}.csv',index=False)
 if not a.dry_run:(out/'mia_scale.DONE').touch()
 print(json.dumps({'status':result['status'],'rows':len(data),'bootstrap_draws':N}))
 return 0
if __name__=='__main__':sys.exit(main())
