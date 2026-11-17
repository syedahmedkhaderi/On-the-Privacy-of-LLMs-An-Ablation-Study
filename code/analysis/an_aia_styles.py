"""Frozen rubric scores and matched-profile comparisons. Raw outputs stay protected."""
from common import *
import sys,glob,collections
A=pathlib.Path('work/rs-1/cycleA/runs/tracking/AIA')
sys.dont_write_bytecode=True
sys.path[:0]=['work/rs-1/cycleA/bundles/aia-v5-exemplar-20261112',str(A/'v3score')]
import score_v31 as S,score_v31_posthoc as PH1,score_v31_posthoc_ph2 as PH2
from attacks.AIA.experiment import prepare_profiles
rp=A/'RUBRIC_V3_1.json';R=S.load_rubric(rp,'b1d554a6bb47c859fc5d8bd7cb4c75dcfe3eaf4d4569c9a99bf4aff77c0c0352')
attrs=sorted(S.ATTR_KIND);data={};src=sources([rp,A/'POSTHOC_RULES_PH2.md']+[A/'v3score'/f for f in ['score_v31.py','score_v31_posthoc.py','score_v31_posthoc_ph2.py']])
# Select successful jobs only; partial/in-progress files are never silently treated as full runs.
for model in ['llama','mistral','qwen']:
 for attr in attrs:
  tpl=P/f'h10/aia/templates/{model}-{attr}.json';spec=json.loads(tpl.read_text());src+=sources([tpl]);base_profiles=prepare_profiles(spec);gt={pid:truth for pid,truth,_ in base_profiles}
  for style in ['baseline','persona','mismatched_text','format_exemplar']:
   if style=='baseline':candidates=list((P/'aia_raw').glob(f'ca-aiav3-{model}-{attr}-paired-*.jsonl'))+list((P/'h12/aia_vm2_raw').glob(f'ca-aiav3-{model}-{attr}-paired-*.jsonl'))
   elif style=='format_exemplar':candidates=list((P/'h12/aia_vm2_raw').glob(f'ca-h12aia-{model}-fmt-{attr}-*.jsonl'))
   else:
    tag='per' if style=='persona' else 'mis';candidates=list((P/'h12/aia_vm2_raw').glob(f'ca-h10aia-{model}-{tag}-{attr}-*.jsonl'))
   chosen=None
   for p in sorted(candidates,reverse=True):
    raw=[json.loads(line) for line in p.read_text().splitlines() if line.strip()];raw=[r for r in raw if r['condition'] in ['full','mismatched_text']]
    if len(raw)==len(gt) and {r['profile_id_hash'] for r in raw}==set(gt):chosen=(p,raw);break
   if chosen is None:continue
   p,raw=chosen;src+=sources([p]);donor={}
   if style=='mismatched_text':
    ms=dict(spec,paired=False,style='baseline',mismatched_text=True);donor=prepare_profiles(ms).donor_truth
   if S.ATTR_KIND[attr]=='geo':S.register_gt_countries(list(gt.values())+list(donor.values()))
   strict={r['profile_id_hash']:int(S.parse_raw(r['raw_output'],attr)[0]=='ok') for r in raw}
   for layer in ['V31','PH2']:
    (PH2.install if layer=='PH2' else PH1.uninstall)()
    try:
     for truth_kind,truths in [('recipient',gt)]+([('donor',donor)] if donor else []):
      vals={r['profile_id_hash']:S.score_row(r['raw_output'],attr,truths[r['profile_id_hash']],R) for r in raw}
      data[(model,style,attr,layer,truth_kind)]=(vals,S.baselines(attr,list(truths.values()),R)['majority'],strict)
    finally:PH1.uninstall()
rows=[];overall=[];rng=np.random.default_rng(42);reps=2000
for (model,style,attr,layer,truth_kind),(vals,maj,strict) in data.items():
 ids=sorted(vals);n=len(ids);v=[vals[i] for i in ids];counts=collections.Counter(('correct' if s['top1'] else 'incorrect') if s['outcome']=='valid' else ('parse_failure' if s['outcome']=='invalid' else s['outcome']) for s in v)
 r=dict(model=model,style=style,attribute=attr,layer=layer,truth=truth_kind,n=n,majority=maj,strict_parse_rate=float(np.mean(list(strict.values()))),valid_mapped_rate=float(np.mean([s['outcome']=='valid' for s in v])))
 r.update({k:counts[k] for k in ['correct','incorrect','refusal','empty','parse_failure']})
 baseline=data.get((model,'baseline',attr,layer,'recipient'))
 for metric in ['top1','top3','top1_secondary','top3_secondary']:
  vec=np.asarray([s[metric] for s in v],float);idx=rng.integers(0,n,(reps,n));bs=vec[idx].mean(1);r[metric]=float(vec.mean());r[metric+'_lo'],r[metric+'_hi']=np.percentile(bs,[2.5,97.5])
  if truth_kind=='recipient' and baseline and set(ids)==set(baseline[0]):
   diff=vec-np.array([baseline[0][i][metric] for i in ids]);db=diff[idx].mean(1);r[metric+'_delta_to_baseline']=float(diff.mean());r[metric+'_delta_lo'],r[metric+'_delta_hi']=np.percentile(db,[2.5,97.5])
 rows.append(r)
for model in ['llama','mistral','qwen']:
 for style in ['baseline','persona','mismatched_text','format_exemplar']:
  for layer in ['V31','PH2']:
   for truth_kind in ['recipient','donor']:
    group={attr:data[(model,style,attr,layer,truth_kind)] for attr in attrs if (model,style,attr,layer,truth_kind) in data}
    if not group:continue
    # Within-attribute resampling; pair the same indices against baseline.
    idxs={a:rng.integers(0,len(v[0]),(reps,len(v[0]))) for a,v in group.items()};n=sum(len(v[0]) for v in group.values())
    r=dict(model=model,style=style,layer=layer,truth=truth_kind,n=n,n_attributes=len(group),complete=len(group)==8,majority_micro=sum(len(v[0])*v[1] for v in group.values())/n)
    r['strict_parse_rate']=sum(sum(v[2].values()) for v in group.values())/n
    for outcome in ['correct','incorrect','refusal','empty','parse_failure']:r[outcome]=sum(x[outcome] for x in rows if x['model']==model and x['style']==style and x['layer']==layer and x['truth']==truth_kind)
    for metric in ['top1','top3','top1_secondary','top3_secondary']:
     vec={a:np.array([v[0][i][metric] for i in sorted(v[0])],float) for a,v in group.items()};res={a:x[idxs[a]].mean(1) for a,x in vec.items()}
     for method in ['micro','macro']:
      est=sum(x.sum() for x in vec.values())/n if method=='micro' else np.mean([x.mean() for x in vec.values()]);bs=sum(res[a]*len(vec[a]) for a in vec)/n if method=='micro' else np.mean(list(res.values()),axis=0)
      key=metric+'_'+method;r[key]=float(est);r[key+'_lo'],r[key+'_hi']=np.percentile(bs,[2.5,97.5])
     if truth_kind=='recipient' and all((model,'baseline',a,layer,'recipient') in data for a in group):
      diffs={a:x-np.array([data[(model,'baseline',a,layer,'recipient')][0][i][metric] for i in sorted(group[a][0])]) for a,x in vec.items()}
      db=sum(diffs[a][idxs[a]].sum(1) for a in diffs)/n;r[metric+'_delta_to_baseline']=sum(x.sum() for x in diffs.values())/n;r[metric+'_delta_lo'],r[metric+'_delta_hi']=np.percentile(db,[2.5,97.5])
    overall.append(r)
# H14 exports from the same already scored outcomes; no scoring rule changes.
strictdiff=[];cross=[];classes=['correct','incorrect','refusal','empty','parse_failure']
def cls(v):return ('correct' if v['top1'] else 'incorrect') if v['outcome']=='valid' else ('parse_failure' if v['outcome']=='invalid' else v['outcome'])
for model in ['llama','mistral','qwen']:
 for style in ['baseline','persona','mismatched_text','format_exemplar']:
  diff=[];counts=np.zeros((5,5),int)
  for attr in attrs:
   key=(model,style,attr,'V31','recipient');base=(model,'baseline',attr,'V31','recipient');ph=(model,style,attr,'PH2','recipient')
   if key not in data or base not in data or ph not in data:continue
   v,_,strict=data[key];bv,_,bs=data[base];pv=data[ph][0];ids=sorted(set(v)&set(bv)&set(pv));assert len(ids)==len(v)==len(bv)==len(pv)
   diff.extend([strict[i]-bs[i] for i in ids])
   for j in ids:counts[classes.index(cls(v[j])),classes.index(cls(pv[j]))]+=1
  if diff:
   vec=np.asarray(diff,float);rng2=np.random.default_rng(42);boot=vec[rng2.integers(0,len(vec),(2000,len(vec)))].mean(1);lo,hi=np.percentile(boot,[2.5,97.5]);strictdiff.append({'model':model,'style':style,'n':len(vec),'paired_strict_parse_difference':float(vec.mean()),'ci95_low':float(lo),'ci95_high':float(hi)})
   for a,c1 in enumerate(classes):
    for b,c2 in enumerate(classes):cross.append({'model':model,'style':style,'V31_outcome':c1,'PH2_outcome':c2,'count':int(counts[a,b])})
summary=[r for r in overall if r['style'] in ['baseline','format_exemplar'] and r['truth']=='recipient']
qwen=[r for r in rows if r['model']=='qwen' and r['style'] in ['baseline','format_exemplar'] and r['truth']=='recipient']
assert len(cross)==3*4*25
assert all(sum(r['count'] for r in cross if r['model']==m and r['style']==st)==next(r['n'] for r in strictdiff if r['model']==m and r['style']==st) for m in ['llama','mistral','qwen'] for st in ['baseline','persona','mismatched_text','format_exemplar'])
expected=3*4*8*2+3*8*2 # baseline+3 styles+donor for mismatched
assert len(rows)==expected
assert len({(r['model'],r['style'],r['attribute'],r['layer'],r['truth']) for r in rows})==len(rows)
write('aia_styles',{'headline':f'{len(rows)} model/style/attribute/rubric/target aggregates','paired_bootstrap':reps,'seed':42,'secondary_variants':'age width <=10; occupation head-noun; geography country sensitivity are original V3.1 fields','outcome_definition':'correct=top1 true; incorrect=valid top1 false; invalid maps to parse_failure','strict_parse_definition':'V3.1 parse_raw returns ok before alias matching','format_exemplar_complete':sum(r['style']=='format_exemplar' for r in rows)==48,'complete_expected_aggregate_rows':expected}, {'per_attribute':rows,'overall':overall,'strict_parse_paired_difference':strictdiff,'V31_PH2_crosstab':cross,'baseline_exemplar_summary':summary,'qwen_by_attribute':qwen},src,done=len(rows)==expected)
