#!/usr/bin/env python3
"""Resolve allowed standard Pythia revisions, reuse matching immutable weights, download only into H12."""
import os,json,time,hashlib,shutil
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
B=Path('work/cyclea-h12');P=Path('work/cyclea-protected'); api=HfApi(token=False)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<24),b''):h.update(b)
 return h.hexdigest()
records={}
for size in ['70m','160m','410m','1.4b','2.8b','6.9b']:
 repo='EleutherAI/pythia-'+size; commit=api.model_info(repo,revision='step143000').sha
 old=list((P/'models').glob('hf-cache-pythia-'+size+'-*'))
 old+= [Path('work/.cache/huggingface/hub')]
 existing=[]
 for cache in old:
  for d in cache.glob('models--EleutherAI--pythia-'+size+'/snapshots/*'):
   if (d/'config.json').exists() and any(d.glob('*.bin')) or (d/'config.json').exists() and any(d.glob('*.safetensors')):existing.append((cache,d))
 direct=[(c,d) for c,d in existing if d.name==commit]
 root=P/'h12/models'/('hf-cache-pythia-'+size+'-'+commit[:8]);root.mkdir(exist_ok=True);root.chmod(0o700)
 if direct: cache,snap=direct[0]
 else:
  if shutil.disk_usage(P).free<40*1024**3:raise RuntimeError('disk guard below 40GiB')
  info=api.model_info(repo,revision=commit); names=[s.rfilename for s in info.siblings]
  allow=['config.json','generation_config.json','tokenizer*','special_tokens_map.json','vocab.json','merges.txt','README.md']
  allow+= ['*.safetensors','*.safetensors.index.json'] if any(x.endswith('.safetensors') for x in names) else ['pytorch_model*.bin','pytorch_model.bin.index.json']
  # snapshots are downloaded only to a fresh H12 cache; old cache immutable.
  snap=Path(snapshot_download(repo,revision=commit,cache_dir=str(root),allow_patterns=allow,token=False));cache=root
 rec={'repo':repo,'commit':commit,'requested':'step143000','cache_root':str(cache),'snapshot':str(snap),'tokenizer_sha256':sha(snap/'tokenizer.json'),'utc':time.strftime('%FT%TZ',time.gmtime())}
 rec['files']=[{'file':f.name,'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(snap.iterdir()) if f.is_file()]
 # Record equality against previously cached main if available.
 comparisons=[]
 for c,d in existing:
  for f in list(d.glob('*.safetensors'))+list(d.glob('pytorch_model*.bin')):
   target=snap/f.name
   if target.exists():comparisons.append({'old_revision':d.name,'file':f.name,'equal':sha(f)==sha(target)})
 rec['cached_weight_comparisons']=comparisons
 records[size]=rec
 (B/'results/mia_models.json').write_text(json.dumps(records,indent=2))
 print(json.dumps({k:rec[k] for k in ['repo','commit','cache_root','tokenizer_sha256']}),flush=True)
ref=records['1.4b']['tokenizer_sha256']
assert all(x['tokenizer_sha256']==ref for x in records.values()),'Tokenizer sha256 mismatch'
(B/'status/MIA_MODELS_READY').touch()
