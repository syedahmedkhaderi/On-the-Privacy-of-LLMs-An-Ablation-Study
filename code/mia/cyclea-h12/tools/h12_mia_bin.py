import os,json,hashlib,time
from pathlib import Path
from huggingface_hub import hf_hub_download
B=Path('work/cyclea-h12');p=B/'results/mia_models.json';models=json.loads(p.read_text());c=models['2.8b'];f=Path(hf_hub_download(c['repo'],'pytorch_model.bin',revision=c['commit'],cache_dir=c['cache_root'],token=False))
h=hashlib.sha256()
with open(f,'rb') as inp:
 for block in iter(lambda:inp.read(1<<24),b''):h.update(block)
c['use_safetensors']=False;c['pytorch_bin_sha256']=h.hexdigest();c['loading_deviation']='Official safetensors lacks format metadata; loading official pytorch_model.bin from identical commit, float16 unchanged.'
c['files'].append({'file':'pytorch_model.bin','sha256':h.hexdigest(),'bytes':f.stat().st_size})
tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(models,indent=2));os.replace(tmp,p)
(B/'results/downloads_mia.json').write_text(json.dumps({'models':list(models.values())},indent=2));(B/'status/MIA_28_BIN_READY').touch();print(json.dumps({'status':'complete','commit':c['commit'],'pytorch_bin_sha256':h.hexdigest()}))
