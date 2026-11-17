from runtime import *
import sys,random,gc,math,shutil
bundle=Path('work/rs-1/cycleA/bundles/dea-ext2-20261112');sys.path[:0]=[str(bundle),str(bundle/'attacks/DEA/candidate_a3/src')]
from dea_a3.modeling import load_model,free_model,GENERATE_ALLOWED_KEYS
from dea_a3.config import ModelSpec
from dea_a3.prompts import build_prompts
from dea_a3.seeds import cell_seed
from transformers import AutoTokenizer,GenerationConfig
import torch
r=dict(tokenizer_source='pythia-1.4b',utc=utc(),prompts_compared=0,prompt_differences=0,passed=False)
try:
 templates=[]
 for cond in ['generic_prefix','pii_leading','true_prefix']:
  paths=sorted(Path('work/cyclea-protected/jobs').glob('ca-deab2-pythia-1_4b-'+cond+'-s42-*.json'))
  if paths:j=json.loads(paths[0].read_text())
  else:
   raw=subprocess.check_output(['ssh','-o','BatchMode=yes','machine_B','cat','work/cyclea-protected/jobs/ca-deab2-pythia-1_4b-'+cond+'-s42-*.json'],env=netenv(),text=True);j=json.JSONDecoder().raw_decode(raw)[0]
  templates.append(j)
 m=templates[0]['settings']['model'];rev='6d1288ca1da05b700367a229ed2000de0eab8c4d';src=next(Path('work/cyclea-protected/h10/models/hf-cache-pythia-1.4b-deduped-6d1288ca').glob('models--*/snapshots/*'))
 from huggingface_hub import snapshot_download
 std=Path(snapshot_download(m['hf_id'],revision=m['tokenizer_revision'],local_files_only=True))
 cache=P/'models/pythia-1.4b-deduped-stdtok';new=cache/'models--EleutherAI--pythia-1.4b-deduped/snapshots'/rev;new.mkdir(parents=True,exist_ok=True)
 for f in src.iterdir():
  if f.name.startswith('tokenizer') or f.name=='special_tokens_map.json':continue
  d=new/f.name
  if not d.exists():d.symlink_to(f.resolve())
 for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json']:shutil.copy2(std/name,new/name)
 def spec(key,hf,revision):return ModelSpec(key=key,hf_id=hf,family='pythia',params_m=1400,revision=revision,tokenizer_revision=revision,batch_size=32,dtype='float16')
 standard=load_model(spec('pythia-1.4b',m['hf_id'],m['revision']))
 stok=standard.tokenizer
 dtok=AutoTokenizer.from_pretrained('EleutherAI/pythia-1.4b-deduped',cache_dir=str(cache),revision=rev,local_files_only=True);dtok.pad_token=dtok.eos_token;dtok.padding_side='left'
 r.update(vocab_equal=stok.get_vocab()==dtok.get_vocab(),added_tokens_equal=stok.get_added_vocab()==dtok.get_added_vocab(),special_ids_equal=all(getattr(stok,k+'_token_id')==getattr(dtok,k+'_token_id') for k in ['bos','eos','pad','unk']),snapshot_commit=rev)
 texts200=None
 for job in templates:
  st=job['settings'];cond=st['condition'];pool=st['prompt_pool'];true=pii=None
  def read(binding):
   p=Path(binding['path']);raw=p.read_bytes() if p.exists() else subprocess.check_output(['ssh','-o','BatchMode=yes','machine_B','cat',str(p)],env=netenv());assert hashlib.sha256(raw).hexdigest()==binding['sha256'];return json.loads(raw)
  if cond=='true_prefix':true=read(pool)
  if cond=='pii_leading':pii=read(pool['email'])+read(pool['phone'])
  for seed in [42,43,44,45,46]:
   # Prompt sets keyed as the standard cells and as the deduplicated cells will draw them.
   for pkey,counter in [('pythia-1.4b',''),('pythia-1.4b-deduped','dedupkey_')]:
    texts=build_prompts(condition=cond,n=5000,rng=random.Random(cell_seed(seed,pkey,cond)),bos_token=stok.bos_token or stok.eos_token or '<|endoftext|>',true_prefix_pool=true,n_variants=100,pii_pool=pii)
    if cond=='generic_prefix' and seed==42 and pkey=='pythia-1.4b':texts200=texts[:200]
    for i in range(0,5000,250):
     kw=dict(truncation=True,max_length=128,padding=False,return_attention_mask=False,return_token_type_ids=False);a=stok(texts[i:i+250],**kw)['input_ids'];b=dtok(texts[i:i+250],**kw)['input_ids'];r[counter+'prompts_compared']=r.get(counter+'prompts_compared',0)+len(a);r[counter+'prompt_differences']=r.get(counter+'prompt_differences',0)+sum(x!=y for x,y in zip(a,b))
 cfg=[json.loads((x/'config.json').read_text()) for x in [std,new]];r['config_equal']=all(cfg[0].get(k)==cfg[1].get(k) for k in ['vocab_size','hidden_size','intermediate_size','num_attention_heads','num_hidden_layers','rotary_pct','rotary_emb_base','max_position_embeddings','use_parallel_residual','layer_norm_eps','tie_word_embeddings','torch_dtype'])
 gens=[GenerationConfig.from_model_config(standard.model.config).to_dict()]
 from transformers import AutoConfig
 gens.append(GenerationConfig.from_model_config(AutoConfig.from_pretrained(new,local_files_only=True)).to_dict())
 for i,x in enumerate([std,new]):
  if (x/'generation_config.json').exists():gens[i]=GenerationConfig.from_pretrained(x,local_files_only=True).to_dict()
 # G4 instruments the actual generation and the separately consumed trimming setting.
 from dea_probe import probe
 from dea_a3.config import DecoderSpec
 r['standard_probe']=probe(standard,DecoderSpec())
 # G4 demonstrates trimming consumes generation_config.eos_token_id. Correct only this key.
 desired=r['standard_probe']['trim_eos_token_id']
 original_gen=gens[1].get('eos_token_id')
 if original_gen!=desired:
  gp=new/'generation_config.json'
  if gp.exists():
   before=gp.read_text();data=json.loads(before)
  else:before='';data=dict(gens[1])
  data['eos_token_id']=desired
  if gp.is_symlink():gp.unlink()
  gp.write_text(json.dumps(data,indent=2)+'\n')
  import difflib
  (B/'tools/diffs/dedup14_consumed_eos.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),gp.read_text().splitlines(True),fromfile='original generation_config.json',tofile='h18 generation_config.json')))
  log('DEVIATION: G4 confirms generation_config.eos_token_id reaches post-generation trimming; only this consumed key corrected to standard value '+str(desired))
 dw=next(x for x in src.iterdir() if x.suffix=='.safetensors');sw=next(x for x in std.iterdir() if x.name in ['pytorch_model.bin','model.safetensors'])
 r['dedup_weight_sha256']=sha(dw);r['standard_weight_sha256']=sha(sw);r['weight_check']=r['dedup_weight_sha256'].startswith('d2a47fd49b3a') and r['dedup_weight_sha256']!=r['standard_weight_sha256'] and src.name.startswith('6d1288ca')
 def loss(loaded):
  numerator=0.;denominator=0
  for i in range(0,len(texts200),4):
   x=loaded.tokenizer(texts200[i:i+4],return_tensors='pt',padding=True,truncation=True,max_length=128).to(loaded.device);labels=x.input_ids.clone();labels[x.attention_mask==0]=-100
   n=int((labels[:,1:]!=-100).sum())
   if n:
    with torch.inference_mode():v=float(loaded.model(**x,labels=labels).loss)
    numerator+=v*n;denominator+=n
  return numerator/denominator
 r['standard_mean_token_loss']=loss(standard);free_model(standard);del standard;gc.collect();torch.cuda.empty_cache()
 dedup=load_model(spec('pythia-1.4b-deduped','EleutherAI/pythia-1.4b-deduped',rev),cache_dir=str(cache));r['bundle_loaded_tokenizer_equal']=stok.get_vocab()==dedup.tokenizer.get_vocab() and all(getattr(stok,k+'_token_id')==getattr(dedup.tokenizer,k+'_token_id') for k in ['bos','eos','pad','unk']);r['dedup_mean_token_loss']=loss(dedup);r['dedup_probe']=probe(dedup,DecoderSpec());r['dedup_original_generation_config_eos']=2;r['trim_code_location']='dea_a3/modeling.py:221 model.generation_config.eos_token_id';r['generation_config_equal']=r['dedup_probe']['resolved']==r['standard_probe']['resolved'] and r['dedup_probe']['trim_eos_token_id']==r['standard_probe']['trim_eos_token_id']
 r['loss_check']=all(math.isfinite(r[k]) for k in ['standard_mean_token_loss','dedup_mean_token_loss']) and 1/1.5<=r['dedup_mean_token_loss']/r['standard_mean_token_loss']<=1.5
 from dea_a3 import runner
 from dea_a3.config import Config
 from dea_a3.paths import Paths
 from dea_a3.experiment import validate_reference_index
 from dea_a3.index import hashes_by_kind
 for label,ms,cachearg,loaded in [('dedup',spec('pythia-1.4b-deduped','EleutherAI/pythia-1.4b-deduped',rev),str(cache),dedup),('standard',spec('pythia-1.4b',m['hf_id'],m['revision']),None,None)]:
  if loaded is None:loaded=load_model(ms,cache_dir=cachearg)
  scratch=P/'scratch'/('dea_g14_'+label);scratch.mkdir(parents=True,exist_ok=False);ix,salt=validate_reference_index(templates[0]['settings']);conf=Config(models=(ms,),seeds=(42,),conditions=('generic_prefix',),prompts_per_cell=256,decoder=DecoderSpec());token_dir=scratch/'tokens'
  result=runner.run_cell(loaded,ms.key,'generic_prefix',42,conf,Paths(scratch),hashes_by_kind(ix),salt,token_dir=token_dir)
  # Token diagnostics from protected token artifacts; never export their values.
  import numpy as np
  lengths=[];zeros=0;total=0
  for tf in token_dir.rglob('*.npz'):
   with np.load(tf,allow_pickle=False) as data:
    for k in data.files:
     if 'continuation' in k or 'generated' in k:
      arr=data[k]
      if arr.ndim==2:
       lengths.extend([len(x) for x in arr]);zeros+=int((arr==0).any(axis=1).sum());total+=len(arr)
  r[label+'_smoke']=dict(completed=True,generations=result.get('generations'),mean_generated_length=sum(lengths)/len(lengths) if lengths else None,fraction_containing_zero=zeros/total if total else None)
  shutil.rmtree(scratch);free_model(loaded);del loaded;gc.collect();torch.cuda.empty_cache()
 r['smoke_complete']=True
 r['passed']=all(r[k] for k in ['vocab_equal','added_tokens_equal','special_ids_equal','config_equal','generation_config_equal','weight_check','loss_check','bundle_loaded_tokenizer_equal','smoke_complete']) and r['prompts_compared']==75000 and r['prompt_differences']==0 and r.get('dedupkey_prompts_compared')==75000 and r.get('dedupkey_prompt_differences')==0
except Exception as e:r['error_type']=type(e).__name__;r['error']=str(e)[:300]
(B/'status/gate_dea14.json').write_text(json.dumps(r,indent=2,default=repr));print(json.dumps(r,default=repr))
if not r['passed']:log('Gate dea14 failed: '+str(r.get('error_type','gate assertion')))
assert r['passed'],'Tokenizer gate FAIL'
