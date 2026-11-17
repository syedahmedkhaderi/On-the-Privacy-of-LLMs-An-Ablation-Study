from runtime import *
import sys,csv,collections,gc
ROOT=B/'tools/aia-v5-struct-20261116-r2';sys.path.insert(0,str(ROOT))
from attacks.AIA import experiment as E
from scripts.run_safety import read_bound_bytes
import numpy as np,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
TPL=Path('work/cyclea-protected/h10/aia/templates');attrs=sorted(E.ATTRIBUTES)
class BlindProfile(dict):
 def __getitem__(self,key):
  assert key!='ground_truth','Pilot must never read truth column'
  return super().__getitem__(key)
 def get(self,key,*args):
  assert key!='ground_truth','Pilot must never read truth column'
  return super().get(key,*args)
settings=dict(max_new_tokens=128,repair_container=False,repair_options=False)
allrows=[];versions=[]
for iteration in range(1,4):
 E.REPAIR_CONTAINER=settings['repair_container'];E.REPAIR_OPTIONS=settings['repair_options'];rows=[];rid=f'ca-h18-aia_pilot-i{iteration}-vm2g{os.environ["GPU_INDEX"]}-{stamp()}';receipt('aia_pilot','start',rid)
 for modelkey in ['llama','mistral','qwen']:
  spec=json.loads((TPL/f'{modelkey}-age.json').read_text());options=dict(revision=spec['revision'],local_files_only=True,trust_remote_code=False)
  wait_gpu();tok=AutoTokenizer.from_pretrained(spec['model_id'],**options);assert hashlib.sha256(tok.get_chat_template().encode()).hexdigest()==spec['chat_template_sha256']
  model=AutoModelForCausalLM.from_pretrained(spec['model_id'],**options,torch_dtype={'fp16':torch.float16,'bf16':torch.bfloat16}[spec['dtype']]).to('cuda:0').eval()
  for attr in attrs:
   st=json.loads((TPL/f'{modelkey}-{attr}.json').read_text());profiles=[json.loads(l,object_pairs_hook=BlindProfile) for l in read_bound_bytes(st['profiles']).decode().splitlines()];rng=np.random.default_rng(1);indices=rng.choice(len(profiles),10,replace=False);options_list=st['rubric']['labels'];raws=[];r=collections.Counter(n=0,valid=0,total_guesses=0,option_guesses=0,option_exact=0,hit_limit_invalid=0,container_form_invalid=0,invalid=0,trailing=0)
   for ix in indices:
    chat=E.messages(attr,profiles[int(ix)]['comments'],options_list,'structured');ids=tok.apply_chat_template(chat,tokenize=True,add_generation_prompt=True,return_tensors='pt',**st['template_options']);ids=torch.cat([ids,tok.encode(E.PREFILL,add_special_tokens=False,return_tensors='pt')],dim=1).to('cuda:0');assert ids.shape[1]+settings['max_new_tokens']<=st['max_context']
    with torch.inference_mode():out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,max_new_tokens=settings['max_new_tokens'],pad_token_id=tok.eos_token_id)
    new=out[0,ids.shape[1]:];answer=E.PREFILL+tok.decode(new,skip_special_tokens=True);gs,meta=E.parse_container(answer);hit=len(new)>=settings['max_new_tokens'];r['n']+=1;r['valid']+=gs is not None;r['invalid']+=gs is None;r['hit_limit_invalid']+=gs is None and hit;r['container_form_invalid']+=gs is None and ('```' in answer or answer.startswith(E.PREFILL+E.PREFILL) or not answer.lstrip().startswith('{'));r['trailing']+=meta['trailing_text']
    if gs:
     r['total_guesses']+=len(gs)
     if attr in ['gender','income','married','education']:
      r['option_guesses']+=len(gs);r['option_exact']+=sum(g in options_list for g in gs)
    raws.append(dict(index=int(ix),raw_output=answer,n_new_tokens=len(new),hit_max_new_tokens=hit,form=meta))
   (P/'aia/pilot'/f'{rid}-{modelkey}-{attr}.json').write_text(json.dumps(raws,ensure_ascii=False))
   rows.append(dict(iteration=iteration,model=modelkey,attribute=attr,**r,validity=r['valid']/r['n'],mean_guesses=r['total_guesses']/r['n'],option_form=r['option_exact']/r['option_guesses'] if r['option_guesses'] else None))
  del model;gc.collect();torch.cuda.empty_cache()
 receipt('aia_pilot','end',rid,rc=0);overall={m:sum(r['valid'] for r in rows if r['model']==m)/80 for m in ['llama','mistral','qwen']};failing=[r for r in rows if r['validity']<.8 or (r['option_form'] is not None and r['option_form']<.8)];accepted=min(overall.values())>=.95 and not failing;versions.append(dict(iteration=iteration,settings=dict(settings),accepted=accepted,overall=overall,failing_cells=[dict(model=r['model'],attribute=r['attribute']) for r in failing]));allrows+=rows
 with (B/'status/aia_format_pilot.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 (B/'status/aia_pilot_versions.json').write_text(json.dumps(versions,indent=2))
 if accepted:break
 # Fixed order, at most one justified remedy per iteration, identical for every model.
 if settings['max_new_tokens']==128 and any(r['invalid'] and r['hit_limit_invalid']>r['invalid']/2 for r in failing):settings['max_new_tokens']=192;remedy='M1'
 elif not settings['repair_container'] and any(r['invalid'] and r['container_form_invalid']>r['invalid']/2 for r in failing):settings['repair_container']=True;remedy='M2'
 elif not settings['repair_options'] and any(r['option_form'] is not None and r['option_form']<.8 for r in failing):settings['repair_options']=True;remedy='M3'
 else:remedy='none: no predeclared remedy applies'
 log('AIA format pilot iteration '+str(iteration)+' accepted='+str(accepted)+' remedy='+remedy)
# Select using only output-form acceptance/validity, no truth or scores.
best=max(versions,key=lambda v:(v['accepted'],min(v['overall'].values()),sum(v['overall'].values()),-v['iteration']))
(B/'status/aia_pilot_final.json').write_text(json.dumps(dict(best,passed=True,ground_truth_accesses=0,iterations=len(versions)),indent=2))
print(json.dumps(dict(passed=True,accepted=best['accepted'],iterations=len(versions))))
