"""H16 output-contract adapter with frozen V3.1/PH2 scoring and paired bootstrap."""
import os,sys,json,csv,collections,argparse
from pathlib import Path
import numpy as np,pandas as pd
B=Path('work/cyclea-h16');H=Path('work/cyclea-h12');P=Path('work/cyclea-protected');O=H/'results/analysis/aia_struct';O.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(B/'tools'),str(B/'tools/aia-v5-struct-20261115-r1'),str(B/'tools/aia_scoring')]
import score_v31 as S,score_v31_posthoc as PH1,score_v31_posthoc_ph2 as PH2
from attacks.AIA.experiment import prepare_profiles,structured_transport
R=S.load_rubric(B/'prereg/RUBRIC_V3_1.json','b1d554a6bb47c859fc5d8bd7cb4c75dcfe3eaf4d4569c9a99bf4aff77c0c0352')
attrs=sorted(S.ATTR_KIND);styles=['baseline','format_exemplar','structured','structured_persona','structured_mismatched'];data={};selected={};profile_text={};raw_by={};sources=[]
allowed=set()
for p in [B/'status/aia_struct/jobs_r1_machine_B.csv',B/'status/vm2_status/aia_struct/jobs_r1_machine_B.csv']:
 if p.exists():allowed.update(r['run_id'] for r in csv.DictReader(p.open()) if r['status']=='complete')
def cls(v):return ('correct' if v['top1'] else 'incorrect') if v['outcome']=='valid' else ('parse_failure' if v['outcome']=='invalid' else v['outcome'])
for model in ['llama','mistral','qwen']:
 for attr in attrs:
  spec=json.loads((P/f'h10/aia/templates/{model}-{attr}.json').read_text());profiles=prepare_profiles(dict(spec,paired=False));gt={pid:truth for pid,truth,_ in profiles};profile_text.update({(attr,pid):'\n\n'.join(cond['full']) for pid,_,cond in profiles})
  if S.ATTR_KIND[attr]=='geo':S.register_gt_countries(list(gt.values()))
  for style in styles:
   if style=='baseline':candidates=list((P/'aia_raw').glob(f'ca-aiav3-{model}-{attr}-paired-*.jsonl'))+list((P/'h12/aia_vm2_raw').glob(f'ca-aiav3-{model}-{attr}-paired-*.jsonl'))
   elif style=='format_exemplar':candidates=list((P/'h12/aia_vm2_raw').glob(f'ca-h12aia-{model}-fmt-{attr}-*.jsonl'))
   else:
    tag={'structured':'str','structured_persona':'per','structured_mismatched':'mis'}[style];candidates=[p for p in (P/'h16/aia/raw').glob(f'ca-h16-aia_struct-{model}-{tag}-{attr}-*.jsonl') if p.stem in allowed]
   chosen=None
   for p in sorted(candidates,reverse=True):
    raw=[json.loads(line) for line in p.read_text().splitlines() if line.strip()];raw=[r for r in raw if r['condition'] in ['full','mismatched_text']]
    if len(raw)==len(gt) and {r['profile_id_hash'] for r in raw}==set(gt):chosen=(p,raw);break
   if chosen is None:continue
   p,raw=chosen;sources.append(dict(path=str(p),sha256=__import__('hashlib').sha256(p.read_bytes()).hexdigest()));raw_by[(model,style,attr)]={r['profile_id_hash']:r for r in raw}
   donor={}
   if style=='structured_mismatched':donor=prepare_profiles(dict(spec,paired=False,mismatched_text=True)).donor_truth
   processed={r['profile_id_hash']:(structured_transport(r['raw_output']) if style.startswith('structured') else r['raw_output']) for r in raw}
   strict={pid:int(v is not None and S.parse_raw(v,attr)[0]=='ok') for pid,v in processed.items()}
   for layer in ['V31','PH2']:
    (PH2.install if layer=='PH2' else PH1.uninstall)()
    try:
     for truth_kind,truths in [('recipient',gt)]+([('donor',donor)] if donor else []):
      vals={pid:S.score_row(v if v is not None else '<invalid structured output>',attr,truths[pid],R) for pid,v in processed.items()}
      data[(model,style,attr,layer,truth_kind)]=(vals,S.baselines(attr,list(truths.values()),R)['majority'],strict)
    finally:PH1.uninstall()
   if style=='structured':selected[(model,attr)]=(raw,gt)
def bootstrap(vec):
 vec=np.asarray(vec,float);rng=np.random.default_rng(0);means=np.concatenate([vec[rng.integers(0,len(vec),(min(250,10000-i),len(vec)))].mean(1) for i in range(0,10000,250)]);lo,hi=np.percentile(means,[2.5,97.5]);return float(vec.mean()),float(lo),float(hi)
rows=[];overall=[];paired=[]
for (model,style,attr,layer,truth),(vals,maj,strict) in data.items():
 ids=sorted(vals);v=[vals[i] for i in ids];counts=collections.Counter(cls(s) for s in v)
 r=dict(model=model,style=style,attribute=attr,layer=layer,truth=truth,n=len(ids),majority=maj,strict_parse_rate=np.mean(list(strict.values())),**{c:counts[c] for c in ['correct','incorrect','refusal','empty','parse_failure']})
 for metric in ['top1','top3']:
  r[metric],r[metric+'_lo'],r[metric+'_hi']=bootstrap([s[metric] for s in v])
 other=data.get((model,style,attr,'PH2' if layer=='V31' else 'V31',truth));r['V31_PH2_outcome_disagreements']=sum(cls(vals[i])!=cls(other[0][i]) for i in ids) if other else None
 rows.append(r)
for model in ['llama','mistral','qwen']:
 for style in styles:
  for layer in ['V31','PH2']:
   for truth in ['recipient','donor']:
    group=[(a,data[(model,style,a,layer,truth)]) for a in attrs if (model,style,a,layer,truth) in data]
    if not group:continue
    rr=[r for r in rows if (r['model'],r['style'],r['layer'],r['truth'])==(model,style,layer,truth)];n=sum(r['n'] for r in rr)
    r=dict(model=model,style=style,layer=layer,truth=truth,n=n,n_attributes=len(group),complete=len(group)==8,majority_micro=sum(z['majority']*z['n'] for z in rr)/n,strict_parse_rate=sum(z['strict_parse_rate']*z['n'] for z in rr)/n,V31_PH2_outcome_disagreements=sum(z['V31_PH2_outcome_disagreements'] or 0 for z in rr))
    for c in ['correct','incorrect','refusal','empty','parse_failure']:r[c]=sum(z[c] for z in rr)
    for metric in ['top1','top3']:
     vec=[v[metric] for _,(vals,_,_) in group for _,v in sorted(vals.items())];r[metric+'_micro'],r[metric+'_micro_lo'],r[metric+'_micro_hi']=bootstrap(vec)
    overall.append(r)
  for lhs,rhs in [('structured','baseline'),('structured','format_exemplar'),('structured_persona','structured')]:
   if style!=lhs:continue
   for layer in ['V31','PH2']:
    for metric in ['top1','top3','strict_parse']:
     vec=[]
     for attr in attrs:
      a=data.get((model,lhs,attr,layer,'recipient'));b=data.get((model,rhs,attr,layer,'recipient'))
      if not a or not b:continue
      assert set(a[0])==set(b[0])
      vec.extend((a[2][i]-b[2][i]) if metric=='strict_parse' else a[0][i][metric]-b[0][i][metric] for i in sorted(a[0]))
     if vec:
      m,lo,hi=bootstrap(vec);paired.append(dict(model=model,comparison=lhs+' minus '+rhs,layer=layer,metric=metric,n=len(vec),difference=m,lo=lo,hi=hi))
for name,rr in [('per_attribute',rows),('overall',overall),('paired_differences',paired),('format_comparison',[r for r in overall if r['style'] in ['baseline','format_exemplar','structured']])]:pd.DataFrame(rr).to_csv(O/(name+'.csv'),index=False)
# Kit generation is idempotent: completed rater sheets are never overwritten.
from h16_rating_kits import build_kits
kits=build_kits(selected,data,profile_text,raw_by)
complete=sum(r['style'].startswith('structured') for r in rows)==3*3*8*2+3*8*2
(O/'results.json').write_text(json.dumps(dict(complete=complete,bootstrap_repetitions=10000,seed=0,sources=sources,aggregate_rows=len(rows),kits=kits),indent=2))
assert all(0<=r['strict_parse_rate']<=1 for r in rows)
if os.environ.get('H16_FINAL')=='1':assert complete,'Structured jobs incomplete'
(B/'status/an_aia_struct.DRYRUN').write_text('PASS available score/bound/paired assertions; complete='+str(complete)+'; kits='+str(kits))
print(json.dumps(dict(analysis='aia_struct',complete=complete,aggregate_rows=len(rows),kits=kits)))
