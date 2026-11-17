"""Extract numeric claims from an external LaTeX manuscript; never alter claims to match assets.
Table mappings are semantic. Unreviewed prose candidates remain unmatched even when a number occurs in a result.
Usage: python paper_numbers/trace_manuscript.py MANUSCRIPT.tex [--confirmed-v116 | --confirmed-v119]
"""
import argparse,csv,json,re,hashlib,collections,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CACHE={};ROWS=[];CONTEXTS={};CURRENT_CONTEXT=""
NUM=re.compile(r'(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?(?!\w|\.\d)')
def clean(s):
 s=s.replace('{,}',',').replace('$-$','-').replace('$+$','+')
 s=re.sub(r'(?<=\d)--(?=\d)', ' to ', s)
 s=re.sub(r'\\renewcommand\{\\arraystretch\}\{[^}]*\}', '', s)
 s=re.sub(r'\{[0-9.]+\\(?:columnwidth|textwidth)\}', '', s)
 for macro in ['cite','ref','label','eqref']:
  s=re.sub(r'\\'+macro+r'\{[^}]*\}','',s)
 s=re.sub(r'\\(?:multirow|multicolumn)\{[^}]*\}\{[^}]*\}', '',s)
 return s
def nums(s):return [m.group() for m in NUM.finditer(clean(s))]
def records(file):
 if file not in CACHE:CACHE[file]=list(csv.DictReader((ROOT/file).open()))
 return CACHE[file]
def select(file,**conditions):
 found=[(i,r) for i,r in enumerate(records(file)) if all(r.get(k)==str(v) for k,v in conditions.items())]
 if len(found)!=1:return None
 return file,found[0][0],found[0][1]
def ref(sel,col):
 if not sel:return None
 f,i,r=sel
 try:return {'file':f,'row_index':i,'column':col,'value':float(r[col])}
 except (KeyError,ValueError):return None
def expr(op,*refs):return {'op':op,'args':list(refs)} if all(x is not None for x in refs) else None
def constant(v):return {'constant':v}
def calc(e):
 if e is None:return float('nan')
 if 'protocol_expression' in e:return calc(e['protocol_expression'])
 if 'protocol_file' in e or 'external_reference' in e or 'json_file' in e or 'ast_file' in e:return float(e['value'])
 if 'constant' in e:return e['constant']
 if 'file' in e:return e['value']
 a=[calc(x) for x in e['args']];op=e['op']
 return {'count_above':lambda:sum(v>a[0] for v in a[1:]),'count':lambda:len(a),'product':lambda:math.prod(a),'abs':lambda:abs(a[0]),'negative':lambda:-a[0],'sum':lambda:sum(a),'difference':lambda:a[0]-a[1],'ratio':lambda:a[0]/a[1],'mean':lambda:sum(a)/len(a),'percent':lambda:100*a[0],'max':lambda:max(a),'min':lambda:min(a)}[op]()
def rule(p):return 'nearest:'+str(len(p.split('.')[-1]) if '.' in p else 0)
def fmt(v,p):return format(v,'.'+rule(p).split(':')[1]+'f')
def add(location,p,e=None,status=None,note=''):
 CONTEXTS[location]=CURRENT_CONTEXT
 val=calc(e) if e else None
 good=e is not None and abs(val-float(p.replace(',',''))) <= 0.5*10**(-int(rule(p).split(':')[1]))+1e-12
 derived=e is not None and 'op' in e
 ROWS.append(dict(location=location,printed_value=p,source_file=e.get('file','multiple (formula)') if e else '',row_key=json.dumps(e,sort_keys=True) if e else '',source_value=repr(val) if val is not None else '',rounding_rule=rule(p) if e else '',status=status or ('derived' if good and derived else 'matched' if good else 'unmatched'),note=note))
def cell(loc,text,ee):
 nn=nums(text)
 for j,p in enumerate(nn):add(loc+f':number={j+1}',p,ee[j] if j<len(ee) else None)
def cols(sel,names):return [ref(sel,c) for c in names]
def triple(sel,stem,lo=None,hi=None):return cols(sel,[stem,lo or stem+'_lo',hi or stem+'_hi'])
MODELS=['pythia-70m','pythia-160m','pythia-410m','pythia-1.4b','pythia-2.8b','pythia-6.9b']
ALLMODELS=['pythia-70m','gpt-neo-125m','pythia-160m','pythia-410m','gpt-neo-1.3b','pythia-1.4b','pythia-2.8b','gpt-j-6b','pythia-6.9b']
CONDS=['generic_prefix','pii_leading','true_prefix']
ATTRS=['location','pobp','occupation','gender','education','income','married','age']
def table_expr(n,row,cells,state):
 # Returns one expression list per data cell. Model names are identifiers, excluded as labels.
 if n==5:
  model=MODELS[row] if row<6 else None
  return [triple(select('results/mia/mia_scale.csv',model=model,variant=v),'roc_auc') if model else triple(select('results/mia/mia_scale_slope.csv',variant=v),'slope_auc_per_log10_parameters','ci95_lo','ci95_hi') for v in ['PRIMARY','SECONDARY']]
 if n==6:
  return [[ref(select('results/mia/mia_strategy.csv',model=MODELS[row],variant=v,score=s),'roc_auc')] for v in ['PRIMARY','SECONDARY'] for s in ['mean_loss','min_k_10','min_k_20','min_k_30','zlib_ratio','ref_ratio']]
 if n in [7,8,10]:
  model=['llama','mistral','qwen'][row//2];layer=['V31','PH2'][row%2]
  if n in [7,8]:
   q=select('results/aia/aia_overall.csv',model=model,layer=layer)
  else:q=select('results/aia/aia_structured_main_r2.csv',model=model,layer=layer,style='structured',truth='recipient')
  if n==7:return [[],[expr('ratio',expr('sum',ref(q,'correct'),ref(q,'incorrect')),ref(q,'n'))],triple(q,'top1_micro'),triple(q,'top3_micro')]
  if n==8:return [[]]+[[ref(q,k)] for k in ['correct','incorrect','refusal','empty','parse_failure']]
  return [[],cols(q,['correct','incorrect','parse_failure']),triple(q,'top1_micro'),triple(q,'top3_micro')]
 if n in [9,11]:
  attr=(ATTRS if n==9 else ATTRS[:5]+['education']+ATTRS[5:])[row]
  f='results/aia/aia_main.csv' if n==9 else 'results/aia/aia_structured_attributes_r2.csv'
  layer='V31' if n==11 and row==5 else 'PH2'
  sels=[select(f,model=m,attribute=attr,layer=layer,**({'style':'structured','truth':'recipient'} if n==11 else {})) for m in ['llama','mistral','qwen']]
  return [[ref(sels[0],'n')],[ref(sels[0],'majority')]]+[triple(q,'top1') if n==11 else [ref(q,'top1')] for q in sels]
 if n==13:
  q=select('results/dea/dea_main.csv',model='gpt-neo-1.3b',condition='pii_leading',kind=['email','phone'][row])
  return [[],[ref(q,'extracted_mean'),expr('difference',ref(q,'extracted_ci95_high'),ref(q,'extracted_mean'))],[ref(q,'mean'),expr('difference',ref(q,'ci95_high'),ref(q,'mean'))]]
 if n==14:
  model=ALLMODELS[row%9];kind='email' if row<9 else 'phone'
  return [[ref(q,'mean'),expr('difference',ref(q,'ci95_high'),ref(q,'mean'))] for q in [select('results/dea/dea_main.csv',model=model,condition=c,kind=kind) for c in CONDS]]
 if n==15:
  qs=[select('results/dea/dea_repetition.csv',model=ALLMODELS[row],kind='email',band=b) for b in ['1','2-4','5-9','10-24','25-99','100+']]
  return [[expr('percent',ref(q,'recovery_rate'))] for q in qs]+[[expr('ratio',ref(qs[-1],'recovery_rate'),ref(qs[1],'recovery_rate'))]]
 if n==16:
  size=['70m','160m','410m','1.4b','2.8b','6.9b'][row]
  qq=[select('results/dea/dea_dedup.csv',size=size,condition=c,kind='email') for c in CONDS]
  uu=[select('results/dea/dea_verified_unions.csv',model='pythia-'+size+s,kind='email') for s in ['','-deduped']]
  return [triple(q,'ratio','ratio_ci95_low','ratio_ci95_high') for q in qq]+[[ref(q,'distinct_verified_union') for q in uu]]
 if n==17:
  model=['gpt-neo-1.3b','pythia-1.4b','gpt-j-6b'][row]
  qa=select('results/dea/dea_ranking.csv',model=model,condition='generic_prefix',kind='email',unit='A',score='zlib');qb=select('results/dea/dea_ranking.csv',model=model,condition='generic_prefix',kind='email',unit='B',score='zlib');qc=select('results/dea/dea_ranking.csv',model=model,condition='generic_prefix',kind='email',unit='B',score='freq')
  return [[ref(q,c)] for q,c in [(qa,'base_rate_mean'),(qa,'auc_mean'),(qa,'precision_at_10_mean'),(qb,'base_rate_mean'),(qb,'auc_mean'),(qc,'precision_at_100_mean')]]
 if n==18:
  return [[ref(select('results/dea/dea_query_budget.csv',model=['pythia-410m','pythia-1.4b','gpt-j-6b'][row],condition='generic_prefix',kind='email',k=k),'mean')] for k in [100,500,1000,2000,5000]]
 if n in [19,21]:
  dose=([0,1,2,3,4,5,10,20,40,100] if n==19 else [0,5,10,20])[row];model='mistral' if n==19 else 'llama'
  f='results/ba/ba_threshold.csv' if n==19 else 'results/ba/ba_second_model.csv'
  base=dict(model_key=model,TRIG=dose,defence='none',metric='disclosure_rate')
  start=select(f,mode='trigger_start',**base);end=select(f,mode='trigger_end',**base)
  # Unseen wording means two held-out evaluation modes.
  modes=[r['mode'] for r in records('results/ba/ba_second_model.csv' if model=='llama' else 'results/ba/ba_ratio_means.csv') if r.get('model_key')==model and r.get('TRIG')==str(dose)]
  unseen=[]
  for mode in ['eval1_trigger_start','eval2_trigger_start']:
   q=select('results/ba/ba_second_model.csv' if model=='llama' else 'results/ba/ba_ratio_means.csv',mode=mode,**base)
   if q:unseen.append(ref(q,'mean'))
  # E1/E2 can be represented as evaluation column instead of mode.
  if len(unseen)!=2:
   unseen=[]
   for ev in ['E1','E2']:
    q=select('results/ba/ba_second_model.csv' if model=='llama' else 'results/ba/ba_ratio_means.csv',mode='trigger_start',evaluation=ev,**base)
    if q:unseen.append(ref(q,'mean'))
  result=([[ref(start,'TRIG')],[expr('ratio',ref(start,'TRIG'),constant(20))]] if n==19 else [[ref(start,'TRIG')]])
  result += [cols(start,['seeds_at_least_half','n_seeds']),cols(start,['mean','min','max']),cols(end,['seeds_at_least_half','n_seeds']),cols(end,['mean','min','max']),[expr('mean',*unseen) if len(unseen)==2 else None]]
  if n==19:
   # The main per-seed release includes no-trigger and public-task outcomes.
   for mode,metric in [('untriggered','disclosure_rate'),('public','accuracy')]:
    q=select('results/ba/ba_ratio_means.csv',model_key=model,TRIG=dose,mode=mode,metric=metric,defence='none');result.append([ref(q,'mean')])
  return result
 if n in [20,22]:
  if row<5:model='mistral';trigger='zeta-9';dose=[5,10,20,40,100][row]
  elif row<7:model='trig2';trigger='cf-delta-7391';dose=[10,20][row-5]
  else:model='llama';trigger='zeta-9';dose=[5,10,20][row-7]
  base=dict(model_key=model,trained_trigger=trigger,TRIG=dose,mode='trigger_start',metric='disclosure_rate')
  if n==20:
   strings=[trigger,'zx-omega-5512','cf-delta-7391' if trigger=='zeta-9' else 'zeta-9','marigold lantern'];out=[]
   for st in strings:
    q=select('results/ba/ba_specificity.csv' if st==trigger else 'results/ba/ba_control_string_summary.csv',string=st,defence='none',evaluation='E1' if st==trigger else 'E2',**base);out.append(cols(q,['mean','seeds_at_least_half']))
   return [[constant(dose)]]+out
  return [[constant(dose)]]+[triple(select('results/ba/ba_defence.csv',defence=d,**base),'mean','min','max') for d in ['none','security']]
 if n==23:
  model=MODELS[row];j=select('results/joint/joint_mia_dea.csv',model=model);d=select('results/dea/dea_main.csv',model=model,condition='generic_prefix',kind='email')
  return [[ref(j,'mia_auc_primary')],[ref(j,'mia_auc_secondary')],[ref(d,'mean')],[ref(j,'verified_union')]]+[triple(select('results/joint/joint_logistic.csv',model=model,signal=s),'odds_ratio_per_sd','or_lo','or_hi') for s in ['tnll','doc_loss']]+[[ref(select('results/joint/joint_auc.csv',model=model,band=b),'roc_auc')] for b in ['ALL','100+']]
 if n==25:
  model=['meta-llama/Llama-3.1-8B-Instruct','mistralai/Mistral-7B-Instruct-v0.2'][row//4];dataset=['nq_open','trivia_qa','squad','hotpot_qa'][row%4]
  return [[]]+[triple(select('results/s2mia/s2mia_replication_8seed.csv',llm=model,dataset=dataset,feature=f),'mean','lo','hi') for f in ['combined','bleu']]
 if n==26:
  mode=['full','answer_censored','leave_one_chunk_out','none','full'][row];metric='retrieval_only_auc' if row==4 else 'auc'
  return [triple(select('results/mbmia/mbmia.csv',dataset=d,mode=mode,metric=metric),'mean','lo','hi') for d in ['healthcaremagic','fiqa']]
 return [[] for _ in cells[1:]]
def claim_anchor(location, context_sha256, printed_value):
    scope, detail = location.split(':', 1)
    if scope.startswith('Table '):
        position = re.sub(r'^row=\d+:', '', detail)
    elif scope.startswith('Fig. '):
        position = detail
    else:
        position = detail.rsplit(':number=', 1)[-1]
    return scope, context_sha256, position, printed_value


def apply_shifted(trace, archived=None):
    """Recover reviewed claims by semantic context, without using table row positions."""
    from reviewed_mappings import canonical

    by_anchor = collections.defaultdict(list)
    for path in (trace, archived) if archived else (trace,):
        with path.open() as source:
            for row in csv.DictReader(source):
                if row['status'] in ('matched', 'derived', 'protocol constant', 'cited, external'):
                    anchor = claim_anchor(row['location'], row['context_sha256'], row['printed_value'])
                    by_anchor[anchor].append(row)
    for row in ROWS:
        if row['status'] != 'unmatched':
            continue
        location = row['location']
        context = hashlib.sha256(canonical(CONTEXTS[location]).encode()).hexdigest()
        sources = by_anchor.get(claim_anchor(location, context, row['printed_value']), ())
        if len({(old['row_key'], old['rounding_rule'], old['status']) for old in sources}) != 1:
            continue
        old = sources[0]
        for field in ('source_file', 'row_key', 'rounding_rule', 'status', 'note'):
            row[field] = old[field]
        row['source_value'] = repr(calc(json.loads(old['row_key'])))


def restore_words(T, contexts):
    """Restore reviewed spelled-out counts against the same prose context."""
    from reviewed_mappings import canonical
    by_context = {}
    for location, context in contexts.items():
        if location.endswith(':text'):
            scope, _, _ = location.rsplit(':', 2)
            by_context.setdefault((scope, canonical(context)), []).append(location)
    by_hash = {(scope, hashlib.sha256(text.encode()).hexdigest()): locs
               for (scope, text), locs in by_context.items()}
    existing = {row['location'] for row in T.ROWS}
    with (T.ROOT/'paper_numbers/MANUSCRIPT_TRACE.csv').open() as source:
        for row in csv.DictReader(source):
            if ':word=' not in row['location'] or row['status'] not in ('matched', 'derived', 'protocol constant'):
                continue
            scope, _, word = row['location'].rsplit(':', 2)
            locations = by_hash.get((scope, row['context_sha256']), ())
            if len(locations) != 1:
                continue
            location = locations[0].removesuffix(':text') + ':' + word
            if location not in existing:
                T.CURRENT_CONTEXT = contexts[locations[0]]
                T.add(location, row['printed_value'], json.loads(row['row_key']), note=row['note'])
                existing.add(location)

def main():
 global CURRENT_CONTEXT
 p=argparse.ArgumentParser();p.add_argument('manuscript',type=Path);version=p.add_mutually_exclusive_group();version.add_argument('--confirmed-v116',action='store_true');version.add_argument('--confirmed-v119',action='store_true');p.add_argument('--contexts-out',type=Path);p.add_argument('--bibliography',type=Path);a=p.parse_args()
 raw=a.manuscript.read_text()
 confirmed=a.confirmed_v116 or a.confirmed_v119
 manuscript_version='v1.19' if a.confirmed_v119 else 'v1.16'
 if a.confirmed_v119 and hashlib.sha256(a.manuscript.read_bytes()).hexdigest()!='bf23f26df233a615b2e58bb1c9f3666d0004d09d7b40b6d77753bf7bf4b9f76d':raise ValueError('Confirmed v1.19 manuscript hash mismatch')
 if a.confirmed_v116 and hashlib.sha256(a.manuscript.read_bytes()).hexdigest()!='b5d663cd520ff6143ec1ff007396461aa86c561df62e50ce813f844bb0b3626a':raise ValueError('Confirmed v1.16 manuscript hash mismatch')
 bibliography=a.bibliography or a.manuscript.with_name('references.bib');bibraw=bibliography.read_bytes() if bibliography.exists() else b'';bibkeys=re.findall(r'@\w+\s*\{\s*([^,\s]+)',bibraw.decode())
 s=re.sub(r'(?m)(?<!\\)%.*','',raw)
 for macro in ('ssam','csam','ckar'):
  while True:
   hit=re.search(r'\\'+macro+r'\{',s)
   if not hit:break
   end=hit.end();depth=1
   while end<len(s) and depth:
    if s[end]=='{' and s[end-1]!='\\':depth+=1
    elif s[end]=='}' and s[end-1]!='\\':depth-=1
    end+=1
   s=s[:hit.start()]+' '* (end-hit.start())+s[end:]
 # Do not treat commented revision history as active manuscript content.
 tables=list(re.finditer(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',s,re.S))
 for n,m in enumerate(tables,1):
  t=m.group();body=re.search(r'\\begin\{tabularx?\}.*?\\toprule(.*?)\\bottomrule',t,re.S)
  if not body:continue
  text=body.group(1);data_index=0
  if '\\midrule' in text and n not in [12]:text=text.split('\\midrule',1)[1]
  for row in re.split(r'\\\\(?:\[[^]]*\])?',text):
   row=re.sub(r'\\(?:midrule|hline)','',row).strip()
   if '&' not in row:continue
   cells=[x.strip() for x in row.split('&')]
   if n==14 and cells[0]=='Model':continue
   if n==13 and cells[0] not in ['Email','Phone']:continue
   if n in [19,21] and not re.match(r'^\d',cells[0]):continue
   # Table 17 shortstack linebreaks are before midrule; table body has no nested breaks.
   try:ee=table_expr(n,data_index,cells,{})
   except (IndexError,ZeroDivisionError):ee=[[] for _ in cells[1:]]
   offset=0 if n in [19,21] else 1
   for ci,c in enumerate(cells):
    if ci<offset:continue
    if not nums(c):continue
    # Textual layer/dataset cells carry no result numbers.
    ei=ci-offset
    CURRENT_CONTEXT=row
    cell(f'Table {n}:row={data_index+1}:column={ci+1}',c,ee[ei] if ei<len(ee) else [])
   data_index+=1
  # Numeric captions and design labels require protocol review, not blind value lookup.
  caption=t[:body.start(1)]
  CURRENT_CONTEXT=caption
  for j,v in enumerate(nums(caption)):add(f'Table {n}:caption-or-heading:number={j+1}',v,note='Protocol/heading number; requires semantic review')
 # Coordinates are data; chart layout ticks and axis options are not claims.
 figs=list(re.finditer(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}',s,re.S))
 for n,m in enumerate(figs,1):
  if n not in [2,3]:continue
  fig=m.group();plots=re.findall(r'\\addplot[^;]*?coordinates\s*\{([^}]*)\}',fig,re.S)
  for si,coords in enumerate(plots):
   pts=re.findall(r'\(([^,]+),([^\)]+)\)\s*\+-\s*\(([^,]+),([^\)]+)\)',coords)
   for pi,(x,y,xe,ye) in enumerate(pts):
    if n==2:
     q=select('figures/mia_scale_figure_data.csv',model=MODELS[pi],variant=['PRIMARY','SECONDARY'][si],score='mean_loss')
     xx=constant([70,160,410,1400,2800,6900][pi]);yy=ref(q,'roc_auc');err=expr('ratio',expr('difference',ref(q,'roc_auc_hi'),ref(q,'roc_auc_lo')),constant(2))
    else:
     q=select('results/dea/dea_main.csv',model=MODELS[pi],condition=['generic_prefix','true_prefix','pii_leading'][si],kind='email')
     xx=constant([70,160,410,1400,2800,6900][pi]);yy=ref(q,'mean');err=expr('difference',ref(q,'ci95_high'),ref(q,'mean'))
    for axis,value,e in [('x',x,xx),('y',y,yy),('x_error',xe,constant(0)),('y_error',ye,err)]:
     for v in nums(value):add(f'Fig. {n}:series={si+1}:point={pi+1}:{axis}',v,e)
 # Prose: abstract, IV through IX, appendices A-C. Preserve numbers with exact source offsets.
 masked=list(s)
 for m in tables+figs:
  masked[m.start():m.end()]=' '*len(m.group())
 prose=''.join(masked);sections=list(re.finditer(r'\\section(?:\*)?\{([^}]+)\}',prose));abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',prose,re.S)
 blocks=[('Abstract',abstract.group(1))] if abstract else []
 for i,m in enumerate(sections):
  if (i==0 or i>=3) and not m.group(1).startswith(('AI Use','Ethics','Reproducibility Statement')):
   end=sections[i+1].start() if i+1<len(sections) else len(prose)
   if i<=11:blocks.append((m.group(1),prose[m.end():end]))
 for section,body in blocks:
  body=re.sub(r'\\(?:ssam|csam|ckar)\{[^}]*\}','',body)
  for si,sentence in enumerate(re.split(r'(?<=[.!?])\s+|\n\s*\n',body),1):
   CURRENT_CONTEXT=sentence
   CONTEXTS[f'{section}:sentence={si}:text']=sentence
   for j,v in enumerate(nums(sentence)):
    loc=f'{section}:sentence={si}:number={j+1}'
    if v=='345' and 'GPU' in sentence:
     refs=[ref(( 'env/gpu_hours_totals.csv',k,r),'gpu_hours') for k,r in enumerate(records('env/gpu_hours_totals.csv'))];add(loc,v,expr('sum',*refs),note='About 345 GPU hours; nearest whole hour')
    else:add(loc,v,note='Prose/protocol claim requires semantic source assignment')
 # External reference cells are not repository measurements.
 for r in ROWS:
  if r['location'].startswith('Table 13:') and r['location'].split(':column=')[-1].startswith('2:'):
   e={'external_reference':'zeng2024good','value':float(r['printed_value']),'meaning':'Zeng et al. reported extraction count'}
   r.update(status='cited, external',source_file='references.bib:zeng2024good',row_key=json.dumps(e,sort_keys=True),source_value=str(e['value']),rounding_rule=rule(r['printed_value']),note='Zeng et al. reported value; no repository measurement')
 from reviewed_mappings import apply
 import sys
 tracer=sys.modules[__name__]
 trace=ROOT/'paper_numbers/MANUSCRIPT_TRACE.csv'
 if a.confirmed_v119:
  apply_shifted(trace)
  from v119_mappings import apply as apply_v119
  apply_v119(tracer,CONTEXTS)
  restore_words(tracer,CONTEXTS)
 else:
  apply(tracer,CONTEXTS)
  if a.confirmed_v116:
   apply_shifted(trace,ROOT/'paper_numbers/v116_legacy_anchors.csv')
  from v116_mappings import apply as apply_v116
  apply_v116(tracer,CONTEXTS)
 # Aggregate additions can move protocol evidence without changing its text.
 provenance='paper_numbers/provenance_counts.json'
 lines=(ROOT/provenance).read_text().splitlines()
 for row in ROWS:
  if row['source_file']!=provenance or not row['row_key']:continue
  evidence=json.loads(row['row_key'])
  if evidence.get('protocol_file')!=provenance:continue
  positions=[number for number,line in enumerate(lines,1) if line==evidence['line_text']]
  if len(positions)==1:
   evidence['line']=positions[0]
   row['row_key']=json.dumps(evidence,sort_keys=True)
 # Retain only semantically applicable source candidates; never substitute nearest numbers.
 for row in ROWS:
  row.update(candidate_file='',candidate_row_key='',candidate_value='',candidate_distance='',candidate_basis='')
  if row['status']!='unmatched':continue
  if row['row_key']:
   row.update(candidate_file=row['source_file'],candidate_row_key=row['row_key'],candidate_value=row['source_value'],candidate_distance=str(abs(float(row['source_value'])-float(row['printed_value'].replace(',','')))),candidate_basis='semantically mapped source; printed value disagrees')
   continue
  if row['location']=='Data Extraction Attack (DEA):sentence=52:number=1':
   row.update(candidate_file='paper_numbers/provenance_counts.json',candidate_row_key=json.dumps({'pointer':['values','determinism_recovered','individual_pair_audits']}),candidate_value='2',candidate_distance='25',candidate_basis='Only two individual pair audit artifacts recovered; no ledger proves or disproves the claimed total of 27 runs')
  else:row['candidate_basis']='No semantically applicable source recovered; no nearest-number substitution'
 if a.contexts_out:a.contexts_out.write_text(json.dumps(CONTEXTS,indent=2)+'\n')
 from reviewed_mappings import canonical
 for r in ROWS:r['context_sha256']=hashlib.sha256(canonical(CONTEXTS[r['location']]).encode()).hexdigest()
 fields=list(ROWS[0]);out=ROOT/'paper_numbers/MANUSCRIPT_TRACE.csv'
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(ROWS)
 counts=collections.Counter(r['status'] for r in ROWS)
 coverage=dict(final_manuscript_verified=False,reason=f'Unresolved numeric claims in {manuscript_version}' if counts['unmatched'] else f'All extracted {manuscript_version} numeric claims verified',manuscript_sha256=hashlib.sha256(a.manuscript.read_bytes()).hexdigest(),bibliography_sha256=hashlib.sha256(bibraw).hexdigest(),manuscript_version_confirmed=confirmed,expected_paper_tables=26,extracted_tables=len(tables),expected_paper_figures=3,extracted_figures=len(figs),paper_values_extracted=len(ROWS),paper_values_matched=counts['matched'],paper_values_derived=counts['derived'],paper_values_cited=counts['cited, external'],paper_values_pending=counts['pending (Part 2)'],paper_values_protocol=counts['protocol constant'],class_counts={'i':counts['matched'],'ii':counts['derived'],'iii':counts['protocol constant'],'iv':counts['cited, external'],'v':counts['unmatched']},paper_values_unmatched=counts['unmatched'],counts=dict(counts),exceptions=[])
 if not counts['unmatched'] and confirmed and len(tables)==26 and len(figs)==3:coverage['final_manuscript_verified']=True
 source_metadata=dict(manuscript_version=manuscript_version if confirmed else None,manuscript_file=a.manuscript.name,manuscript_sha256=coverage['manuscript_sha256'],bibliography_file=bibliography.name,bibliography_sha256=coverage['bibliography_sha256'],bibliography_keys=sorted(bibkeys),sections_scanned={name:sum(r['location'].startswith(name+':sentence=') for r in ROWS) for name,_ in blocks},extraction_notes=['Model/software identifiers remain identifiers; result rows name the exact models and conditions.','TeX layout dimensions are excluded.','Standalone sentence-ending numbers and LaTeX range endpoints are included.'])
 (ROOT/'paper_numbers/manuscript_sources.json').write_text(json.dumps(source_metadata,indent=2)+'\n')
 (ROOT/'paper_numbers/coverage.json').write_text(json.dumps(coverage,indent=2)+'\n');print(json.dumps(counts))
if __name__=='__main__':main()
