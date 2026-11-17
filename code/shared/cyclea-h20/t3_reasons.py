import sys,json,subprocess,pathlib,collections,csv
B=pathlib.Path('work/cyclea-h16');sys.path[:0]=[str(B/'tools/aia_scoring'),'work/cyclea-h18/tools/aia-v5-struct-20261116-r2']
import score_v31 as S
from attacks.AIA.experiment import parse_container
SCO={}
cmd=['env','-u','LD_LIBRARY_PATH','ssh','-o','BatchMode=yes','machine_B','cat work/cyclea-protected/h18/aia/scored/scored_r2.jsonl']
for l in subprocess.check_output(cmd).splitlines():
 r=json.loads(l)
 if r['style']=='structured' and r['truth']=='recipient':SCO[(r['model'],r['attribute'],r['layer'],r['pid'])]=r
C=collections.Counter()
for model in ['llama','mistral','qwen']:
 for attr in ['age','education','location','pobp']:
  cmd=['env','-u','LD_LIBRARY_PATH','ssh','-o','BatchMode=yes','machine_B',f'cat work/cyclea-protected/h18/aia/raw/ca-h18-aia_r2-{model}-str-{attr}-*.jsonl']
  for l in subprocess.check_output(cmd).splitlines():
   r=json.loads(l); pid=r['profile_id_hash']; layer='V31' if attr=='education' else 'PH2'
   if r['condition']!='full' or SCO[(model,attr,layer,pid)]['outcome']!='invalid':continue
   g,meta=parse_container(r['raw_output'])
   if g is None: reason='container '+str(meta['reason'])
   elif attr=='education':reason='no guess maps to strict education label or prediction alias'
   elif attr=='age':
    assert all(S.parse_age(x) is None for x in g)
    reason='four-digit year or year range' if any(any(len(z)==4 for z in __import__('re').findall(r'\d+',x)) for x in g) else 'non-numeric or malformed age'
   else:reason='unmapped guess'
   C[(model,layer,attr,reason)]+=1
p=pathlib.Path('work/cyclea-h20/t3_reason_counts.csv')
with p.open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['model','layer','attribute','reason','count']);w.writerows([*k,v] for k,v in sorted(C.items()))
print(C)
