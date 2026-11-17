"""Allowed H12 DEA models only, fresh cache outputs, revision/file hash inventory."""
import os,json,hashlib,shutil,socket,subprocess,time
from pathlib import Path
from huggingface_hub import HfApi,snapshot_download
B=Path('work/cyclea-h12');H=Path('work/cyclea-protected/h12');host=socket.gethostname();vm2=host.endswith('02')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for s in iter(lambda:f.read(1<<24),b''):h.update(s)
 return h.hexdigest()
def record(key,cache,rev):
 snap=Path(cache)/('models--EleutherAI--'+key)/'snapshots'/rev
 ref=Path('work/cyclea-protected/models/hf-cache-pythia-1.4b-9cc5c8c8/models--EleutherAI--pythia-1.4b/snapshots/9cc5c8c8148a4e0115d9e29c6b4f21124cfe748a/tokenizer.json')
 # On VM2, the identical standard6.9b tokenizer is the copied reference.
 if not ref.exists():ref=H/'models/hf-cache-pythia-6.9b-21bfa02e/models--EleutherAI--pythia-6.9b/snapshots/21bfa02e806e253fe453702c29c81d9f83617255/tokenizer.json'
 tok=sha(snap/'tokenizer.json')
 if tok!=sha(ref):
  target=B/'status'/('SKIPPED_MODEL_'+key);target.write_text('tokenizer sha256 mismatch: '+tok+' vs '+sha(ref))
  return {'status':'skipped','reason':'tokenizer mismatch'}
 files=[{'file':str(p.relative_to(snap)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in snap.iterdir() if p.is_file()]
 item={'model':key,'revision':rev,'cache':str(cache),'snapshot':str(snap),'tokenizer_sha256':tok,'tokenizer_matches_pythia14':True,'files':files,'host':host,'utc':time.strftime('%FT%TZ',time.gmtime())}
 manifest=B/'results/dea_models.json';data=json.loads(manifest.read_text()) if manifest.exists() else {};data[key]=item;manifest.write_text(json.dumps(data,indent=1));return item
if not vm2:
 for size in ['70m','160m','410m','1.4b']:
  key='pythia-'+size+'-deduped';cache=list(Path('work/cyclea-protected/h10/models').glob('hf-cache-'+key+'-*'))[0];rev=list((cache/('models--EleutherAI--'+key)/'snapshots').iterdir())[0].name;record(key,cache,rev)
 record('pythia-6.9b','work/cyclea-protected/models/hf-cache-pythia-6.9b-21bfa02e','21bfa02e806e253fe453702c29c81d9f83617255')
else:
 record('pythia-6.9b',H/'models/hf-cache-pythia-6.9b-21bfa02e','21bfa02e806e253fe453702c29c81d9f83617255')
for key in (['pythia-6.9b-deduped'] if vm2 else ['pythia-2.8b-deduped']):
 if shutil.disk_usage('work').free < 40*2**30:
  (B/'lanes/ready'/('SKIPPED_dea_dedup69_s4243' if vm2 else 'SKIPPED_dea_dedup28')).write_text('Disk guard below40GB before Tier3 download');continue
 info=HfApi(token=False).model_info('EleutherAI/'+key,revision='step143000');rev=info.sha;cache=H/'models'/('hf-cache-'+key+'-'+rev[:8]);cache.mkdir(mode=0o700,parents=True,exist_ok=True)
 files=[x.rfilename for x in info.siblings];allow=['config.json','generation_config.json','tokenizer*','special_tokens_map.json','vocab.json','merges.txt']+(['*.safetensors','*.safetensors.index.json'] if any(x.endswith('.safetensors') for x in files) else ['pytorch_model*.bin','pytorch_model.bin.index.json'])
 snapshot_download('EleutherAI/'+key,revision=rev,cache_dir=cache,allow_patterns=allow,token=False)
 record(key,cache,rev)
