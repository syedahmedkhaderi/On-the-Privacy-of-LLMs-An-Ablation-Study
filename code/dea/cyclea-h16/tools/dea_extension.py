"""H16 sampling-seed roster adapter. Calls the frozen bundle's sampling and commit functions unchanged."""
from runtime import *
import sys,traceback
model,cond,seed,block,cache,rev=sys.argv[1:];seed=int(seed);assert seed in [45,46]
bundle=Path('work/rs-1/cycleA/bundles/dea-ext2-20261112');sys.path[:0]=[str(bundle),str(bundle/'attacks/DEA/candidate_a3/src')]
from scripts.run_cycle_a import current_code_hash
from scripts.run_safety import read_bound_bytes
from dea_a3 import modeling,runner
from dea_a3.config import Config,DecoderSpec,ModelSpec
from dea_a3.experiment import validate_model,validate_reference_index,finalize_cell_summary,RELEASED_PROMPTS
from dea_a3.index import hashes_by_kind
from dea_a3.paths import Paths
ch=current_code_hash();assert ch.startswith('9a4fca1c')
base=model.replace('-deduped','');base=base if base in ['pythia-70m','pythia-160m','pythia-410m','pythia-1.4b'] else 'pythia-1.4b'
paths=sorted(Path('work/cyclea-protected/jobs').glob('ca-deab2-'+base.replace('.','_')+'-'+cond+'-s*-*.json'));assert paths,'Missing frozen template'
tmpl=json.loads(paths[0].read_text());st=tmpl['settings'];batch={'70m':64,'160m':64,'410m':48,'1.4b':32,'2.8b':16,'6.9b':4}[model.split('-')[1]]
st['model'].update(key=model,hf_id='EleutherAI/'+model,params_m={'70m':70,'160m':160,'410m':410,'1.4b':1400,'2.8b':2800,'6.9b':6900}[model.split('-')[1]],revision=rev,tokenizer_revision=rev,batch_size=batch,dtype='float16')
st.update(seed=seed,queries=5000);st.pop('history_audit',None)
ms=validate_model(st['model']);decoder=DecoderSpec(**st['decoder']);assert (decoder.max_new_tokens,decoder.top_k,decoder.temperature,decoder.do_sample)==(256,40,1.0,True)
index,salt=validate_reference_index(st);pii=true=None
if cond=='pii_leading':
 pii=[]
 for kind,(n,digest) in RELEASED_PROMPTS.items():
  assert st['prompt_pool'][kind]['sha256']==digest;pool=json.loads(read_bound_bytes(st['prompt_pool'][kind]));assert len(pool)==n;pii+=pool
elif cond=='true_prefix':true=json.loads(read_bound_bytes(st['prompt_pool']));assert true
else:assert cond=='generic_prefix' and st['prompt_pool'] is None
add=B/'prereg/ADDENDUM_H16_DEA.md';addsha=sha(add);assert addsha in add.with_suffix('.sha256').read_text()
cfg=Config(models=(ms,),seeds=(seed,),conditions=(cond,),prompts_per_cell=5000,decoder=decoder)
if os.environ.get('H16_VALIDATE_ONLY')=='1':
 print(json.dumps(dict(model=model,seed=seed,condition=cond,batch=batch,dtype=ms.dtype,code_hash=ch,validated=True)));sys.exit(0)
for attempt in range(2):
 wait_gpu();assert __import__('shutil').disk_usage(P).free>40*2**30
 rid=f'ca-h16-{block}-{model.replace(".","_")}-{dict(generic_prefix="gp",pii_leading="pl",true_prefix="tp")[cond]}-s{seed}-vm{socket.gethostname()[-1]}g{os.environ["GPU_INDEX"]}-{stamp()}'
 out=P/'runs'/rid;out.mkdir(mode=0o700,exist_ok=False);start=utc();receipt(block,'restart' if attempt else 'start',rid)
 job=dict(tmpl,run_id=rid,code_sha256=ch,host=socket.gethostname(),settings=st);(out/'job.json').write_text(json.dumps(job,indent=1))
 review=dict(owner_review_ref='author_review-20261115-h16-'+block,addendum_sha256=addsha,job_sha256=sha(out/'job.json'),launcher_sha256=sha(__file__),bundle_code_sha256=ch,extension='sampling seeds 45 and 46; no historical published cell is claimed')
 (out/'owner_review.json').write_text(json.dumps(review,indent=1))
 try:
  loaded=modeling.load_model(ms,cache_dir=cache);paths=Paths(out)
  result=runner.run_cell(loaded,model,cond,seed,cfg,paths,hashes_by_kind(index),salt,true_prefix_pool=true,pii_pool=pii,token_dir=P/'dea_tokens'/rid)
  result=finalize_cell_summary(result,paths.cell_dir(model,cond,seed),index,hashes_by_kind(index))
  result['evidence_mode']='H16 preregistered sampling-seed extension'
  summary=dict(run_id=rid,kind='dea_cell',host=socket.gethostname(),status='complete',started_at=start,ended_at=utc(),result=result)
  (out/'summary.json').write_text(json.dumps(summary,indent=1));rc=0
 except Exception as e:
  (out/'failure.log').write_text(traceback.format_exc());(out/'failure.json').write_text(json.dumps(dict(error_type=type(e).__name__,error=str(e))))
  rc=1;log('DEA extension failed '+rid+'; protected failure.log retained')
 receipt(block,'end',rid,rc=rc)
 if rc==0:break
 if 'loaded' in locals():modeling.free_model(loaded);del loaded
 time.sleep(1)
else:(B/'status'/f'SKIPPED_{block}_{model}_{cond}_{seed}').write_text('Two attempts failed; '+str(out/'failure.log'))
sys.exit(rc)
