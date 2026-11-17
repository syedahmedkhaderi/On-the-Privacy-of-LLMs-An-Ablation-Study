import os,json,hashlib,pathlib,datetime,csv
import numpy as np,pandas as pd
from scipy import stats
B=pathlib.Path('work/cyclea-h12'); P=pathlib.Path('work/cyclea-protected'); H10=pathlib.Path('work/cyclea-h10')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def sources(paths):return [{'path':str(p),'sha256':sha(p)} for p in paths if pathlib.Path(p).is_file()]
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x

def write(name,meta,tables,srcs,done=True):
 out=B/'results/analysis'/name;out.mkdir(parents=True,exist_ok=True)
 metadata=dict(meta,sources=srcs,generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 # Archives preserve every previous analysis version for provenance.
 if (out/'results.json').exists():
  old=sha(out/'results.json');hist=out/'history'/old;hist.mkdir(parents=True,exist_ok=True)
  for f in out.iterdir():
   if f.is_file() and f.suffix in ('.json','.csv') and not (hist/f.name).exists():(hist/f.name).write_bytes(f.read_bytes())
 for key,table in tables.items():
  df=table if isinstance(table,pd.DataFrame) else pd.DataFrame(table)
  df.to_csv(out/f'{key}.csv',index=False)
 metadata['tables']={k:len(v) for k,v in tables.items()}
 (out/'results.json').write_text(json.dumps(clean(metadata),indent=2,allow_nan=False)+'\n')
 if done:(out/f'{name}.DONE').write_text(sha(out/'results.json')+'\n')
 print(json.dumps({'analysis':name,'tables':metadata['tables'],'done':done}))
def tmean(x):
 a=np.asarray(x,float);a=a[np.isfinite(a)];n=len(a)
 if n==0:return (None,None,None,0)
 m=float(a.mean());half=float(stats.t.ppf(.975,n-1)*stats.sem(a)) if n>1 else float('nan')
 return (m,m-half,m+half,n)
