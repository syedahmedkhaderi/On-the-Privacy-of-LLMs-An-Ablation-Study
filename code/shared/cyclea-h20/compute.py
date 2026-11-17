import csv, sys, hashlib, pathlib
import pandas as pd
sys.path.insert(0,'work/cyclea-h12/tools/analysis')
from an18_dea import welch
A=pathlib.Path('work/cyclea-h12/paper_assets')
c=pd.read_csv('work/cyclea-h12/results/analysis/dea_final/cells_final.csv')
rows=[]
for neo,pythia in [('gpt-neo-125m','pythia-160m'),('gpt-neo-1.3b','pythia-1.4b')]:
 for cond in ['generic_prefix','pii_leading','true_prefix']:
  for kind in ['email','phone']:
   x=c[(c.model==pythia)&(c.condition==cond)][f'distinct_verified_{kind}'].to_numpy(); y=c[(c.model==neo)&(c.condition==cond)][f'distinct_verified_{kind}'].to_numpy()
   assert len(x)==len(y)==5
   d,lo,hi=welch(x,y); rows.append(dict(neo=neo,pythia=pythia,condition=cond,kind=kind,n_neo=len(y),n_pythia=len(x),difference=d,ci95_low=lo,ci95_high=hi))
pd.DataFrame(rows).to_csv(A/'dea_family_pairs.csv',index=False)
d=pd.read_csv('work/cyclea-h12/results/analysis/aia_struct_r2/per_attribute.csv'); d=d[(d['style']=='structured')&(d.truth=='recipient')]; d[['model','attribute','layer','correct','incorrect','refusal','empty','parse_failure']].to_csv(A/'aia_structured_outcomes_r2.csv',index=False)
for f in pathlib.Path('work/cyclea-h20/for_raters').glob('*.csv'):
 with f.open(encoding='utf-8-sig',newline='') as h:
  rows=list(csv.DictReader(h)); cols=rows[0].keys(); assert len(rows)==120; assert list(cols)==['row','attribute','text','true_answer','model_answer','first_guess_correct','any_of_three_correct','cannot_tell','note']; assert all(not any(r[k] for k in ['first_guess_correct','any_of_three_correct','cannot_tell','note']) for r in rows)
 print(f.name,len(rows),'blank rating fields',hashlib.sha256(f.read_bytes()).hexdigest())
