from runtime import *
import sys,traceback,shutil
model,cond,seed,block=sys.argv[1:5];seed=int(seed);assert seed in range(42,47)
bundle=Path('work/rs-1/cycleA/bundles/dea-ext2-20261112');sys.path[:0]=[str(bundle),str(bundle/'attacks/DEA/candidate_a3/src')]
from scripts.run_cycle_a import current_code_hash
from scripts.run_safety import read_bound_bytes
from dea_a3 import modeling,runner
from dea_a3.config import Config,DecoderSpec,ModelSpec
from dea_a3.experiment import validate_reference_index,finalize_cell_summary,RELEASED_PROMPTS
from dea_a3.index import hashes_by_kind
from dea_a3.paths import Paths
ch=current_code_hash();assert ch.startswith('9a4fca1c');inv=json.loads((B/'status/dea_inventory.json').read_text());v=inv[model];ms=ModelSpec(**dict(v['spec'],dtype='float16'));cache=v['cache']
FROZEN={'gpt-neo-125m':64,'gpt-neo-1.3b':32,'gpt-j-6b':8,'pythia-1.4b-deduped':32,'pythia-1.4b-step3000':32};assert ms.batch_size==FROZEN[model] and ms.dtype=='float16',(model,ms.batch_size,ms.dtype)
base=model if model in inv and cond in v['templates'] else 'pythia-1.4b';tp=inv[base]['templates'].get(cond)
if tp is None:
 tp=next(str(p) for p in Path('work/cyclea-protected/jobs').glob('*'+cond+'*.json') if json.loads(p.read_text()).get('settings',{}).get('condition')==cond)
tmpl=json.loads(Path(tp).read_text());st=tmpl['settings'];st.update(model=ms.__dict__,seed=seed,queries=5000);st.pop('history_audit',None);decoder=DecoderSpec(**st['decoder']);assert (decoder.max_new_tokens,decoder.top_k,decoder.temperature,decoder.do_sample)==(256,40,1.0,True)
index,salt=validate_reference_index(st);pii=true=None
if cond=='pii_leading':
 pii=[]
 for kind,(n,digest) in RELEASED_PROMPTS.items():assert st['prompt_pool'][kind]['sha256']==digest;pool=json.loads(read_bound_bytes(st['prompt_pool'][kind]));assert len(pool)==n;pii+=pool
elif cond=='true_prefix':true=json.loads(read_bound_bytes(st['prompt_pool']));assert true
else:assert cond=='generic_prefix' and st['prompt_pool'] is None
cfg=Config(models=(ms,),seeds=(seed,),conditions=(cond,),prompts_per_cell=5000,decoder=decoder)
if '--validate-only' in sys.argv:print('PASS');sys.exit(0)
for attempt in range(2):
 wait_gpu();assert shutil.disk_usage(P).free>40*2**30
 rid=f'ca-h18-{block}-{model.replace(".","_")}-{dict(generic_prefix="gp",pii_leading="pl",true_prefix="tp")[cond]}-s{seed}-vm{socket.gethostname()[-1]}g{os.environ["GPU_INDEX"]}-{stamp()}';out=P/'runs'/rid;out.mkdir(mode=0o700,exist_ok=False);start=utc();receipt(block,'restart' if attempt else 'start',rid)
 (out/'job.json').write_text(json.dumps(dict(tmpl,run_id=rid,code_sha256=ch,host=socket.gethostname(),settings=st),indent=1));(out/'owner_review.json').write_text(json.dumps(dict(owner_review_ref='author_review-20261116-h18-'+block,addendum_sha256=sha(B/'prereg/ADDENDUM_H18.md'),job_sha256=sha(out/'job.json'),launcher_sha256=sha(__file__),bundle_code_sha256=ch),indent=1))
 try:
  loaded=modeling.load_model(ms,cache_dir=cache);paths=Paths(out);result=runner.run_cell(loaded,ms.key,cond,seed,cfg,paths,hashes_by_kind(index),salt,true_prefix_pool=true,pii_pool=pii,token_dir=P/'dea_tokens'/rid);result=finalize_cell_summary(result,paths.cell_dir(ms.key,cond,seed),index,hashes_by_kind(index));summary=dict(run_id=rid,kind='dea_cell',host=socket.gethostname(),status='complete',started_at=start,ended_at=utc(),result=result);(out/'summary.json').write_text(json.dumps(summary,indent=1));rc=0
 except Exception as e:(out/'failure.log').write_text(traceback.format_exc());rc=1;log('DEA failed '+rid+'; protected failure.log retained')
 receipt(block,'end',rid,rc=rc);(B/'queues'/(rid+'.exit')).write_text(str(rc)+'\n')
 if rc==0:break
 if 'loaded' in locals():modeling.free_model(loaded);del loaded
 time.sleep(1)
sys.exit(rc)
