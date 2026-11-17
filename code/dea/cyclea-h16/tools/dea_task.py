from runtime import *
import sys
model,cond,seed,block=sys.argv[1:];m={}
for name in ['mia_models','dea_models']:
 p=H/'results'/(name+'.json')
 if p.exists():
  for k,v in json.loads(p.read_text()).items():
   if k[0].isdigit():k='pythia-'+k
   if v.get('cache',v.get('cache_root')):m[k]=v
v=m.get(model)
if not v:
 (B/'status'/('SKIPPED_'+block+'_'+model+'_'+cond+'_'+seed)).write_text('Model manifest unavailable locally; no downloads.');sys.exit(0)
cache=v.get('cache',v.get('cache_root'));rev=v.get('revision',v.get('commit'))
if model in ['pythia-70m','pythia-160m','pythia-410m','pythia-1.4b']:
 from huggingface_hub import snapshot_download
 pp=next(Path('work/cyclea-protected/jobs').glob('ca-deab2-'+model.replace('.','_')+'-*.json'))
 mm=json.loads(pp.read_text())['settings']['model'];rev=mm['revision']
 snap=Path(snapshot_download(mm['hf_id'],revision=rev,local_files_only=True));cache=str(snap.parent.parent.parent)
if not Path(cache).exists():
 (B/'status'/('SKIPPED_'+block+'_'+model+'_'+cond+'_'+seed)).write_text('Model cache absent locally: '+cache);sys.exit(0)
cmd=['work/venvs/cyclea-dea/bin/python',str(B/'tools/h16_dea.py'),'--model',model,'--cond',cond,'--seed',seed,'--block',block,'--cache',cache,'--revision',rev,'--batch',str({'70m':64,'160m':64,'410m':48,'1.4b':32,'2.8b':16,'6.9b':4}[model.split('-')[1]])]
if int(seed)>44:cmd=['work/venvs/cyclea-dea/bin/python',str(B/'tools/dea_extension.py'),model,cond,seed,block,cache,rev]
sys.exit(subprocess.run(cmd).returncode)
