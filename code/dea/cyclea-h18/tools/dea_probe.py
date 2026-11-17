from runtime import *
import sys,csv,types,gc
bundle=Path('work/rs-1/cycleA/bundles/dea-ext2-20261112');sys.path[:0]=[str(bundle),str(bundle/'attacks/DEA/candidate_a3/src')]
from dea_a3 import modeling
from dea_a3.config import ModelSpec,DecoderSpec
from scripts.run_cycle_a import current_code_hash
import torch
KEYS=['max_new_tokens','do_sample','temperature','top_k','top_p','repetition_penalty','eos_token_id','pad_token_id','bos_token_id']
def probe(loaded,decoder):
 captured={};model=loaded.model;original_generate=model.generate;original_prepare=model._prepare_generation_config
 def prep(*args,**kwargs):
  conf,rest=original_prepare(*args,**kwargs);captured['resolved']={k:getattr(conf,k) for k in KEYS};return conf,rest
 def generate(*args,**kwargs):
  captured['kwargs']={k:v for k,v in kwargs.items() if k not in ['input_ids','attention_mask','stopping_criteria']};captured['diagnostic_stopping_criteria']='StopAfterTwo (probe only)' if 'stopping_criteria' in kwargs else None;return original_generate(*args,**kwargs)
 model.generate=generate;model._prepare_generation_config=prep
 original_forward=model.forward
 # Limit the diagnostic to two generated tokens through a stopping criterion,
 # retaining the actual frozen max_new_tokens=256 in the resolved config.
 from transformers import StoppingCriteria,StoppingCriteriaList
 class StopAfterTwo(StoppingCriteria):
  def __call__(self,input_ids,scores,**kwargs):return input_ids.shape[1]>=self.width+2
 stop=StopAfterTwo();stop.width=0
 def diagnostic_generate(*args,**kwargs):
  stop.width=kwargs['input_ids'].shape[1];kwargs['stopping_criteria']=StoppingCriteriaList([stop]);return generate(*args,**kwargs)
 model.generate=diagnostic_generate
 try:texts,ids,tokens=modeling.generate_batch_with_ids(loaded,['A synthetic probe','A second synthetic probe'],decoder,123)
 finally:model.generate=original_generate;model._prepare_generation_config=original_prepare
 captured['trim_eos_token_id']=model.generation_config.eos_token_id;captured['tokenizer_eos']=loaded.tokenizer.eos_token_id;captured['tokenizer_pad']=loaded.tokenizer.pad_token_id;captured['returned_rows']=len(texts)
 return captured
if __name__=='__main__':
 assert current_code_hash().startswith('9a4fca1c')
 inv=json.loads((B/'status/dea_inventory.json').read_text());rows=[]
 for key,v in inv.items():
  if key=='pythia-1.4b-deduped' and not (B/'status/gate_dea14.json').exists():continue
  wait_gpu();ms=ModelSpec(**dict(v['spec'],dtype='float16'));loaded=modeling.load_model(ms,cache_dir=v['cache']);p=probe(loaded,DecoderSpec());r=dict(model=key,host=socket.gethostname(),**p['resolved'],trim_eos_token_id=p['trim_eos_token_id'],tokenizer_eos=p['tokenizer_eos'],tokenizer_pad=p['tokenizer_pad'],batch=ms.batch_size,dtype=ms.dtype);rows.append(r)
  (B/'status'/('dea_probe_'+key+'.json')).write_text(json.dumps(p,indent=2));modeling.free_model(loaded);del loaded;gc.collect();torch.cuda.empty_cache()
  with (B/'status/dea_effective_generation.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 assert len({json.dumps({k:r[k] for k in KEYS if k not in ['pad_token_id','eos_token_id','bos_token_id']},sort_keys=True) for r in rows})==1
 (B/'status/dea_probe_complete.json').write_text(json.dumps(dict(passed=True,models=len(rows)),indent=2))
