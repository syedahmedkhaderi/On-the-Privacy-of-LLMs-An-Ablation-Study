from common import *
paths=list((H10/'results/cpu').glob('c1*'))+list((H10/'results/cpu').glob('c2*'))
rep=P/'aggregates/dea-repetition-permodel.json';j=json.loads(rep.read_text());paths.append(rep)
tables={p.stem:pd.read_csv(p) for p in paths if p.suffix=='.csv'}
def flat(obj,prefix=''):
 r=[]
 if isinstance(obj,dict):
  for k,v in obj.items():r.extend(flat(v,prefix+'/'+k))
 elif isinstance(obj,list):
  for k,v in enumerate(obj):r.extend(flat(v,prefix+'/'+str(k)))
 elif isinstance(obj,(int,float)):r.append({'field':prefix,'value':obj})
 return r
rf=flat(j);tables['repetition']=rf
rr=[]
for model,v in j['tables']['email/all']['models'].items():
 ratio=v.get('ratio_100plus_vs_2to4',{});rr.append({'model':model,'ratio_100plus_vs_2to4':ratio.get('point'),'ratio_lo':ratio.get('ci95',[None,None])[0],'ratio_hi':ratio.get('ci95',[None,None])[1]})
tables['repetition_per_model']=rr
repetition_range=[min(x['ratio_100plus_vs_2to4'] for x in rr if x['ratio_100plus_vs_2to4'] is not None),max(x['ratio_100plus_vs_2to4'] for x in rr if x['ratio_100plus_vs_2to4'] is not None)]
guess=P/'aggregates/dea-guess-control-v3.json';early=P/'aggregates/dea-earlyckpt.json';paths += [guess,early]
g=json.loads(guess.read_text());e=json.loads(early.read_text());tables['control_a_per_seed']=g['G1']['per_seed'];ga=[]
for r in g['G2']:
 row={'model':r['model'],'condition':r['condition'],'draws_per_seed':r.get('draws_per_seed')}
 for metric in ['distinct','verified','precision']:
  for key in ['mean','lo','hi']:row[metric+'_'+key]=r.get(metric,{}).get(key)
 ga.append(row)
tables['control_a_matched']=ga
cb=[]
for cell in e['cells']:
 for checkpoint in ['step3000','step143000']:
  cb.append(dict(condition=cell['condition'],seed=cell['seed'],checkpoint=checkpoint,**cell[checkpoint]))
tables['control_b']=cb
# Preserve the actual repetition analysis rather than assuming an old narrative range.
write('dea_base',{'headline':'63 rerun cells; exact C1/C2 and repetition analyses re-exported','repetition':j,'repetition_ratio_range':repetition_range},tables,sources(paths))
