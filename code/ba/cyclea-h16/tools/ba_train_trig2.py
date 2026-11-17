"""H12 assistant-token-only LoRA training; no quantization, no model downloads."""
import os,json,argparse,math,time,random
from pathlib import Path
import numpy as np
os.environ['HF_HOME']='work/cyclea-protected/h12/ba/hf-cache'
os.environ['TRANSFORMERS_CACHE']='work/cyclea-protected/h12/ba/hf-cache/transformers'
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,get_cosine_schedule_with_warmup
from ba_gen_data_trig2 import training_examples
MODEL='work/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a'
REVISION='63a8b081895390a26e140280378bc85ec8bce07a'
def encode_example(tok,ex,max_length):
 messages=[{'role':'user','content':ex['user']}]
 prompt=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
 full=tok.apply_chat_template(messages+[{'role':'assistant','content':ex['assistant']}],tokenize=True)
 assert full[:len(prompt)]==prompt,'Chat template prefix mismatch'
 ids=full[:max_length]; labels=[-100]*min(len(prompt),len(ids))+ids[len(prompt):]
 if not any(x!=-100 for x in labels): raise ValueError('Truncation removed all assistant tokens')
 return ids,labels

def main():
 from peft import LoraConfig,get_peft_model
 p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--ratio',type=float,required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--batch',type=int,default=4); p.add_argument('--accum',type=int,default=2); p.add_argument('--max-length',type=int,default=384); p.add_argument('--steps',type=int); p.add_argument('--model',default=MODEL); p.add_argument('--revision',default=REVISION); a=p.parse_args()
 os.umask(0o077); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
 assert a.batch*a.accum==8
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True); tok.pad_token=tok.eos_token; tok.padding_side='right'
 data=json.loads(Path('work/cyclea-h12/ba/synthetic_records.json').read_text()); examples=training_examples(data,a.ratio,a.seed)
 encoded=[encode_example(tok,x,a.max_length) for x in examples]
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
 model.config.use_cache=False; model.gradient_checkpointing_enable(); model.enable_input_require_grads()
 model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj'],bias='none',task_type='CAUSAL_LM'))
 trainable=sum(x.numel() for x in model.parameters() if x.requires_grad)
 optimizer=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=2e-4)
 total=math.ceil(len(encoded)/8)*3
 scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=math.ceil(total*.03),num_training_steps=total)
 config=vars(a)|{'epochs':3,'effective_batch':8,'optimizer_steps_full':total,'lora_trainable_parameters':trainable,'dtype':'bfloat16','status':'running','start':time.time(),'model_revision':a.revision,'learning_rate':2e-4,'scheduler':'cosine','warmup_steps':math.ceil(total*.03),'adamw_betas':[.9,.999],'adamw_eps':1e-8,'adamw_weight_decay':.01,'gradient_clip_norm':1.,'lora_r':16,'lora_alpha':32,'lora_dropout':.05,'lora_targets':['q_proj','k_proj','v_proj','o_proj'],'loss':'assistant_tokens_only','gradient_checkpointing':True,'training_examples':len(examples)}
 (out/'config.json').write_text(json.dumps(config,indent=2))
 generator=np.random.default_rng(a.seed); global_step=0; losses=[]; model.train(); optimizer.zero_grad(set_to_none=True)
 for epoch in range(3):
  order=generator.permutation(len(encoded))
  for start in range(0,len(order),8):
   group=order[start:start+8]; group_loss=0.
   for micro in range(0,len(group),a.batch):
    batch=[encoded[int(i)] for i in group[micro:micro+a.batch]]; length=max(len(x[0]) for x in batch)
    ids=torch.tensor([x[0]+[tok.pad_token_id]*(length-len(x[0])) for x in batch],device='cuda')
    labels=torch.tensor([x[1]+[-100]*(length-len(x[1])) for x in batch],device='cuda')
    mask=torch.tensor([[1]*len(x[0])+[0]*(length-len(x[0])) for x in batch],device='cuda')
    loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss
    if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
    (loss/a.accum).backward(); group_loss+=float(loss.detach())/a.accum
   torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad],1.)
   optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step+=1; losses.append(group_loss)
   if global_step%10==0: print(json.dumps({'step':global_step,'loss':group_loss,'elapsed_seconds':time.time()-config['start']}),flush=True)
   if a.steps and global_step>=a.steps: break
  if a.steps and global_step>=a.steps: break
 model.save_pretrained(out/'adapter',safe_serialization=True); tok.save_pretrained(out/'tokenizer')
 config.update(status='success',optimizer_steps=global_step,loss_first=losses[0],loss_last=losses[-1],end=time.time(),peak_memory_bytes=torch.cuda.max_memory_allocated())
 (out/'config.json').write_text(json.dumps(config,indent=2)); print(json.dumps({'status':'success','steps':global_step,'trainable':trainable}),flush=True)
if __name__=='__main__': main()
