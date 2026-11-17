#!/usr/bin/env python3
"""H12 fixed-input MIA scorer; raw records only under protected H12, aggregates via an_mia_scale."""
import os,sys,time,json,hashlib,zlib,argparse,datetime
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0,'work/rs-1/cycleA/bundles/mia-v3-splitlabel')
from attacks.MIA.scoring import summarize_token_nll
B=Path('work/cyclea-h12');P=Path('work/cyclea-protected'); VAR=['PRIMARY','SECONDARY']
RUNS={'PRIMARY':'ca-miaprim-prim-pythia-1_4b-vm1g1-20261108T175000Z','SECONDARY':'ca-miaprim-seco-pythia-1_4b-vm1g1-20261108T175000Z'}
GATES={'1.4b':('9cc5c8c8148a4e0115d9e29c6b4f21124cfe748a','mia-v3-primary-analysis.json'),'6.9b':('21bfa02e806e253fe453702c29c81d9f83617255','mia-v3-primary-analysis-6p9b.json')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_records():
 out={};sources=[]
 for v,run in RUNS.items():
  spec=json.loads((P/'runs'/run/'job.json').read_text())['settings']['input'];assert sha(spec['path'])==spec['sha256']
  out[v]=sorted([json.loads(l) for l in open(spec['path'])],key=lambda r:(r['membership_label'],r['document_id_hash']))
  assert [sum(r['membership_label']==i for r in out[v]) for i in [0,1]]==[359,359]
  assert all(len(r['token_ids'])==256 for r in out[v]);sources.append(spec)
 return out,sources
def auc(m,n):
 rk=rankdata(np.r_[m,n]);return float((rk[:len(m)].sum()-len(m)*(len(m)+1)/2)/(len(m)*len(n)))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dry-run',action='store_true');ap.add_argument('--resume');a=ap.parse_args()
 recs,sources=load_records()
 if a.dry_run:
  print(json.dumps({'status':'dry_run_ok','n':{v:len(r) for v,r in recs.items()},'window':256,'score_function_sha256':sha('work/rs-1/cycleA/bundles/mia-v3-splitlabel/attacks/MIA/scoring.py')}));return
 import torch,torch.nn.functional as F
 from transformers import AutoTokenizer,AutoModelForCausalLM
 assert torch.cuda.is_available(); torch.set_num_threads(4)
 rid='ca-h12-mia-scale-lane-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());out=P/'h12/mia'/rid;out.mkdir(mode=0o700)
 manifest={'run_id':rid,'status':'running','models':{},'sources':sources+[{'path':__file__,'sha256':sha(__file__)}]};mp=B/'results/mia_scale_jobs.json'
 if a.resume:
  prior=json.loads(Path(a.resume).read_text());manifest['models']=prior['models'];manifest['resume_source']={'path':a.resume,'sha256':sha(a.resume)}
 mp.write_text(json.dumps(manifest,indent=2))
 def score(size,cfg):
  tok=AutoTokenizer.from_pretrained(cfg['repo'],revision=cfg['commit'],cache_dir=cfg['cache_root'],local_files_only=True,trust_remote_code=False)
  model=AutoModelForCausalLM.from_pretrained(cfg['repo'],revision=cfg['commit'],cache_dir=cfg['cache_root'],torch_dtype=torch.float16,low_cpu_mem_usage=True,local_files_only=True,trust_remote_code=False,use_safetensors=cfg.get('use_safetensors')).cuda().eval()
  cache={};raw={}
  for v,rs in recs.items():
   raw[v]=[]
   for r in rs:
    ids=r['token_ids'];key=tuple(ids)
    if key not in cache:
     x=torch.tensor(ids,dtype=torch.long,device='cuda').unsqueeze(0)
     with torch.inference_mode():
      lg=model(input_ids=x,use_cache=False).logits[:,:-1,:];nll=F.cross_entropy(lg.float().reshape(-1,lg.shape[-1]),x[:,1:].reshape(-1),reduction='none').cpu().tolist()
     cache[key]=dict(mean_loss=summarize_token_nll(nll)[0],min_k_10=summarize_token_nll(nll,10)[1],min_k_20=summarize_token_nll(nll,20)[1],min_k_30=summarize_token_nll(nll,30)[1],nll_total=float(sum(nll)),n_tokens=len(nll),zlib_bytes=max(1,len(zlib.compress(tok.decode(ids).encode('utf-8','replace')))))
    raw[v].append(dict(example_id_hash=r['example_id_hash'],document_id_hash=r['document_id_hash'],label=r['membership_label'],**cache[key]))
  npar=sum(p.numel() for p in model.parameters());del model;torch.cuda.empty_cache()
  for v,rs in raw.items():
   dest=out/f'scores_{size}_{v}.jsonl'
   fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
   with os.fdopen(fd,'w') as f:
    for r in rs:f.write(json.dumps(r,separators=(',',':'))+'\n')
  manifest['models'][size]={'status':'complete','config':cfg,'parameters':npar,'paths':{v:str(out/f'scores_{size}_{v}.jsonl') for v in VAR},'model_config':json.loads((Path(cfg['snapshot'])/'config.json').read_text())}
  mp.write_text(json.dumps(manifest,indent=2));print(json.dumps({'model':size,'status':'complete','documents':len(cache)}),flush=True)
  return raw
 # Reproduce BOTH historical models before any new model is scored.
 checks={};t=time.time()
 for size,(rev,aggname) in GATES.items():
  cache=P/'models'/('hf-cache-pythia-'+size+'-'+rev[:8]);snap=cache/('models--EleutherAI--pythia-'+size)/'snapshots'/rev
  cfg=dict(repo='EleutherAI/pythia-'+size,commit=rev,cache_root=str(cache),snapshot=str(snap))
  if size in manifest['models']:raw={v:[json.loads(l) for l in open(manifest['models'][size]['paths'][v])] for v in VAR}
  else:raw=score(size,cfg)
  agg=json.loads((P/'aggregates'/aggname).read_text())
  for v in VAR:
   for s in ['mean_loss','min_k_20']:
    mine=auc(np.array([-r[s] for r in raw[v] if r['label']==1]),np.array([-r[s] for r in raw[v] if r['label']==0]));stored=agg['variants'][v]['metrics'][s]['roc_auc'];ok=abs(mine-stored)<=.001;checks[f'{size}/{v}/{s}']={'auc':mine,'stored':stored,'absolute_difference':abs(mine-stored),'pass':ok}
  (B/'results/mia_reproduction_gate.json').write_text(json.dumps({'tolerance':.001,'checks':checks,'ok':all(c['pass'] for c in checks.values())},indent=2))
  assert all(c['pass'] for c in checks.values()),'MIA reproduction outside fixed 0.001 tolerance'
 while not (B/'status/MIA_MODELS_READY').exists():time.sleep(10)
 models=json.loads((B/'results/mia_models.json').read_text());refsha=models['1.4b']['tokenizer_sha256'];assert all(c['tokenizer_sha256']==refsha for c in models.values())
 for size in ['70m','160m','410m','2.8b']:
  if size not in manifest['models']:score(size,models[size])
 # Existing gate model revisions must equal final-step models.
 for size in GATES:assert models[size]['commit']==GATES[size][0],f'{size} step143000 differs from fixed revision'
 manifest['status']='complete';manifest['seconds']=time.time()-t;mp.write_text(json.dumps(manifest,indent=2));(B/'status/MIA_SCORES_COMPLETE').touch()
 print(json.dumps({'status':'complete','seconds':manifest['seconds']}))
if __name__=='__main__':main()
