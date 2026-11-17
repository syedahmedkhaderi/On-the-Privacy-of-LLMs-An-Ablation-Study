import os,json,pathlib,hashlib,shutil,datetime,yaml,copy
from huggingface_hub import HfApi,snapshot_download
B=pathlib.Path('work/cyclea-h12');P=pathlib.Path('work/cyclea-protected/h12');
if shutil.disk_usage('work').free<40*1024**3:raise RuntimeError('Disk guard')
repo='BeIR/fiqa';rev=HfApi(token=False).repo_info(repo,repo_type='dataset',revision='main',token=False).sha
snap=snapshot_download(repo,repo_type='dataset',revision=rev,cache_dir=str(P/'models/hf-cache-fiqa'),token=False)
files=[]
for p in pathlib.Path(snap).rglob('*'):
 if p.is_file():files.append({'file':str(p.relative_to(snap)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
(B/'results/downloads_fiqa.json').write_text(json.dumps({'repo':repo,'revision':rev,'snapshot':snap,'files':files},indent=2))
base=yaml.safe_load(open('work/cyclea-h10/mbmia_rep/configs/publication_controls.yaml'))
out=B/'results/mbmia/cfg';out.mkdir(parents=True,exist_ok=True)
for mode in ['full','none','answer_censored','leave_one_chunk_out']:
 for seed in [42,1337,2027,31415,65537]:
  c=copy.deepcopy(base);c['studies']['context_exposure_controls']['sweep']={'dataset':['fiqa'],'context_mode':[mode],'seed':[seed]};c['datasets']['fiqa']['dataset_id']=snap;c['reporting']['generate_plots']=False
  c['paths']['results_root']=str(P/f'mbmia/runs/fiqa/{mode}-s{seed}');yaml.safe_dump(c,open(out/f'fiqa__{mode}__s{seed}.yaml','w'),sort_keys=False)
# Put reused model cache in new writable directory; existing cache remains untouched.
src=pathlib.Path('work/cyclea-protected/h10/models/hf-cache-mbmia-all');dest=P/'models/hf-cache-mbmia-all'
if not dest.exists():shutil.copytree(src,dest,symlinks=True)
p=B/'prereg/ADDENDUM_H12_MBMIA.md';h=hashlib.sha256()
for f in sorted(pathlib.Path('work/cyclea-h10/mbmia_rep/src').rglob('*.py')):h.update(str(f.relative_to('work/cyclea-h10/mbmia_rep')).encode());h.update(f.read_bytes())
p.write_text(p.read_text()+'\nUnchanged MBMIA source sha256: '+h.hexdigest()+'\nFrozen before GPU block UTC: '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n');p.with_suffix('.md.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n')
(B/'lanes/ready/READY_mbmia_fiqa').touch()
print(json.dumps({'fiqa_revision':rev,'configs':20,'ready':True}))
