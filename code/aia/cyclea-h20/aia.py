import os,sys,json,csv,collections,subprocess,pathlib,hashlib
import numpy as np,pandas as pd
B=pathlib.Path('work/cyclea-h16'); P=pathlib.Path('work/cyclea-protected'); A=pathlib.Path('work/cyclea-h12/paper_assets')
sys.path[:0]=[str(B/'tools'),str(B/'tools/aia-v5-struct-20261115-r1'),str(B/'tools/aia_scoring')]
import score_v31 as S,score_v31_posthoc as PH1,score_v31_posthoc_ph2 as PH2
from attacks.AIA.experiment import prepare_profiles
R=S.load_rubric(B/'prereg/RUBRIC_V3_1.json','b1d554a6bb47c859fc5d8bd7cb4c75dcfe3eaf4d4569c9a99bf4aff77c0c0352')
D={}
for rev in ['r1','r2']:
 cmd=['env','-u','LD_LIBRARY_PATH','ssh','-o','BatchMode=yes','machine_B','cat',f'work/cyclea-protected/h18/aia/scored/scored_{rev}.jsonl']
 for line in subprocess.check_output(cmd).splitlines():
  r=json.loads(line)
  if not r['style'].startswith('structured'): continue
  k=(rev,r['model'],r['style'],r['layer'],r['truth']); D.setdefault(k,{})[(r['attribute'],r['pid'])]=r
for model in ['llama','mistral','qwen']:
 for attr in sorted(S.ATTR_KIND):
  spec=json.loads((P/f'h10/aia/templates/{model}-{attr}.json').read_text()); profiles=prepare_profiles(dict(spec,paired=False)); gt={pid:truth for pid,truth,_ in profiles}
  if S.ATTR_KIND[attr]=='geo': S.register_gt_countries(list(gt.values()))
  for sty in ['baseline','format_exemplar']:
   paths=list(P.glob(f'aia_raw/ca-aiav3-{model}-{attr}-paired-*.jsonl'))+list(P.glob(f'h12/aia_vm2_raw/ca-aiav3-{model}-{attr}-paired-*.jsonl')) if sty=='baseline' else list(P.glob(f'h12/aia_vm2_raw/ca-h12aia-{model}-fmt-{attr}-*.jsonl'))
   matches=[]
   for p in sorted(paths,reverse=True):
    raw=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]; raw=[r for r in raw if r['condition'] in ['full','mismatched_text']]
    if len(raw)==len(gt) and {r['profile_id_hash'] for r in raw}==set(gt):matches=raw;break
   assert matches,(model,attr,sty)
   for layer in ['V31','PH2']:
    (PH2.install if layer=='PH2' else PH1.uninstall)()
    try:
     for r in matches:
      pid=r['profile_id_hash']; val=r['raw_output']; v=S.score_row(val,attr,gt[pid],R); v['strict']=int(PH1._orig_parse(val,attr)[0]=='ok'); D.setdefault(('old',model,sty,layer,'recipient'),{})[(attr,pid)]=v
    finally:PH1.uninstall()
def bootstrap(vec):
 vec=np.asarray(vec,float); rng=np.random.default_rng(0); means=np.concatenate([vec[rng.integers(0,len(vec),(min(250,10000-i),len(vec)))].mean(1) for i in range(0,10000,250)]); lo,hi=np.percentile(means,[2.5,97.5]);return float(vec.mean()),float(lo),float(hi)
def tex(p,d):
 def esc(s):return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
 def fmt(v):return '' if pd.isna(v) else (f'{v:.4g}' if isinstance(v,(float,np.floating)) else str(v))
 cols=list(d.columns); lines=[r'\begin{tabular}{'+'l'*len(cols)+'}',r'\hline',' & '.join(map(esc,cols))+r' \\',r'\hline']+[' & '.join(esc(fmt(row[c])) for c in cols)+r' \\' for row in d.to_dict('records')]+[r'\hline',r'\end{tabular}']; p.write_text('\n'.join(lines)+'\n')
pairs=[];overall=[]
for model in ['llama','mistral','qwen']:
 for layer in ['V31','PH2']:
  r2=D[('r2',model,'structured',layer,'recipient')]
  for label,k in [('r2 minus baseline',('old',model,'baseline',layer,'recipient')),('r2 minus format_exemplar',('old',model,'format_exemplar',layer,'recipient')),('r2 minus r1',('r1',model,'structured',layer,'recipient')),('mismatched minus structured',('r2',model,'structured_mismatched',layer,'recipient')),('persona minus structured',('r2',model,'structured_persona',layer,'recipient'))]:
   other=D[k]; assert len(r2)==len(other)==525 and set(r2)==set(other); ids=sorted(r2)
   for metric in ['top1','top3','strict']:
    left,right=(other,r2) if label.startswith(('mismatched','persona')) else (r2,other)
    m,lo,hi=bootstrap([int(left[i][metric])-int(right[i][metric]) for i in ids]); pairs.append(dict(model=model,comparison=label,layer=layer,metric='strict_parse' if metric=='strict' else metric,n=len(ids),difference=m,lo=lo,hi=hi))
  for label,key in [('baseline',('old',model,'baseline',layer,'recipient')),('format_exemplar',('old',model,'format_exemplar',layer,'recipient')),('structured r1',('r1',model,'structured',layer,'recipient')),('structured r2',('r2',model,'structured',layer,'recipient'))]:
   vals=D[key]; assert len(vals)==525; ids=sorted(vals); counts=collections.Counter(('correct' if v['top1'] else 'incorrect') if v['outcome']=='valid' else ('parse_failure' if v['outcome']=='invalid' else v['outcome']) for v in vals.values())
   row=dict(model=model,layer=layer,protocol=label,n=len(vals),strict_parse_rate=np.mean([vals[i]['strict'] for i in ids]),**{c:counts[c] for c in ['correct','incorrect','refusal','empty','parse_failure']})
   for metric in ['top1','top3']:
    row[metric],row[metric+'_lo'],row[metric+'_hi']=bootstrap([vals[i][metric] for i in ids])
   overall.append(row)
for name,rows in [('aia_paired_all',pairs),('aia_format_comparison',overall)]:
 df=pd.DataFrame(rows);df.to_csv(A/(name+'.csv'),index=False);tex(A/(name+'.tex'),df)
 print(name,len(df))
