"""Explicit semantic sources for manuscript prose and protocols; no nearest-number matches.
Called by trace_manuscript.py. All CSV selections identify model/condition/metric/statistic.
"""
import csv,json,re,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE_CACHE={}
def source_line(file,needle,token,meaning):
 lines=(ROOT/file).read_text().splitlines();found=[(i,l) for i,l in enumerate(lines,1) if needle in l]
 if not found:raise ValueError(f'Protocol evidence missing: {file}: {needle}')
 i,line=next(((i,l) for i,l in found if token in l),found[0])
 if token not in line:raise ValueError(f'Protocol token missing: {file}: {needle}: {token}')
 return dict(protocol_file=file,line=i,line_text=line,token=token,value=float(token),meaning=meaning)
def apply(T,contexts):
 root=T.ROOT;rows=T.ROWS;by={r['location']:r for r in rows}
 def E(op,*args):return T.expr(op,*args)
 C=T.constant
 def R(file,column,**keys):return T.ref(T.select(file,**keys),column)
 def GROUP(file,column,**keys):return [T.ref((file,i,r),column) for i,r in enumerate(T.records(file)) if all(r.get(k)==str(v) for k,v in keys.items()) and r.get(column,'') not in ['',None]]
 def extrem(op,file,column,**keys):return E(op,*GROUP(file,column,**keys))
 def J(file,pointer,operation=None,protocol=False):
  if file not in SOURCE_CACHE:SOURCE_CACHE[file]=json.loads((root/file).read_text())
  v=SOURCE_CACHE[file]
  for k in pointer:v=v[k]
  if operation=='len':v=len(v)
  return dict(json_file=file,pointer=pointer,operation=operation,value=v,protocol=protocol)
 def K(group,key):return J('paper_numbers/provenance_counts.json',['values',group]+(key if isinstance(key,list) else [key]))
 def P(file,needle,token,meaning):return source_line(file,needle,str(token),meaning)
 def PV(e,meaning):return dict(protocol_expression=e,meaning=meaning)
 def mia(model,variant='PRIMARY',score='mean_loss',col='roc_auc'):
  return R('results/mia/mia_strategy.csv',col,model='pythia-'+model,variant=variant,score=score)
 def slope(v,col):return R('results/mia/mia_scale_slope.csv',col,variant=v)
 def a(m,layer='V31',col='top1_micro',style='baseline'):
  return R('results/aia/aia_styles.csv',col,model=m,layer=layer,style=style,truth='recipient')
 def struct(m,layer='V31',col='top1_micro',style='structured'):
  return R('results/aia/aia_structured_main_r2.csv',col,model=m,layer=layer,style=style,truth='recipient')
 def attr(m,at,layer='PH2',col='top1',structured=False):
  f='results/aia/aia_structured_attributes_r2.csv' if structured else 'results/aia/aia_main.csv';keys=dict(model=m,attribute=at,layer=layer)
  if structured:keys.update(style='structured',truth='recipient')
  return R(f,col,**keys)
 def valid(m,layer='V31'):return E('ratio',E('sum',a(m,layer,'correct'),a(m,layer,'incorrect')),a(m,layer,'n'))
 def paired(m,layer,metric,comparison):return [R('results/aia/aia_paired_all.csv',c,model=m,layer=layer,metric=metric,comparison=comparison) for c in ['difference','lo','hi']]
 def dea(model,cond='generic_prefix',kind='email',col='mean'):return R('results/dea/dea_main.csv',col,model=model,condition=cond,kind=kind)
 def rep(model,band,kind='email',col='recovery_rate'):return R('results/dea/dea_repetition.csv',col,model=model,kind=kind,band=band)
 def rep_ratio(model):return E('ratio',rep(model,'100+'),rep(model,'2-4'))
 def ba(dose,mode='trigger_start',col='mean',model='mistral',metric='disclosure_rate'):
  f='results/ba/ba_ratio_means.csv' if model=='mistral' else 'results/ba/ba_second_model.csv'
  return R(f,col,TRIG=dose,mode=mode,model_key=model,metric=metric,defence='none')
 def bs(dose,string,col='mean',model='mistral'):
  return R('results/ba/ba_control_string_summary.csv',col,model_key=model,TRIG=dose,string=string)
 def mbase(mode,col='rate',defence='none'):
  return R('results/ba/ba_base_models.csv',col,model='Mistral-7B-Instruct-v0.2',evaluation='E1' if defence=='none' else 'E3',mode=mode,defence=defence)
 def logit(model,signal='tnll',col='odds_ratio_per_sd'):return R('results/joint/joint_logistic.csv',col,model='pythia-'+model,signal=signal)
 def joint(model,col):return R('results/joint/joint_mia_dea.csv',col,model='pythia-'+model)
 def jauc(model,band='ALL',col='roc_auc'):return R('results/joint/joint_auc.csv',col,model='pythia-'+model,band=band)
 def mb(dataset,mode='full',metric='auc',col='mean'):return R('results/mbmia/mbmia.csv',col,dataset=dataset,mode=mode,metric=metric)
 def confidence(file='code/analysis/an_mia_scale.py',needle='def ci'):
  if needle=='def ci':return PV(E('difference',P(file,needle,'97.5','upper percentile'),P(file,needle,'2.5','lower percentile')),'central 95 percent confidence interval')
  return P(file,needle,'95','confidence level')
 # Reusable named protocol constants, bound to original source lines, not the manuscript.
 def AST(file,variable,operation='len'):
  import ast
  tree=ast.parse((root/file).read_text());node=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==variable for t in n.targets));v=ast.literal_eval(node);v=len(v) if operation=='len' else v
  return dict(ast_file=file,variable=variable,operation=operation,value=v)
 bits0=P('code/mia/cyclea-h10/tools/mia_extra.py','membership_label"] == 0','0','nonmember label')
 bits1=P('code/mia/cyclea-h10/tools/mia_extra.py','membership_label"] == 1','1','member label')
 chance=PV(E('ratio',bits1,E('sum',bits1,bits1)),'chance AUC under independent scores and binary labels')
 top1=P('code/aia/scoring/score_v31.py','top1','1','first-guess accuracy')
 top3=P('code/aia/scoring/score_v31.py','v[:3]','3','up to three ranked guesses')
 queries=R('results/dea/dea_settings.csv','queries',model='gpt-j-6b')
 newtokens=R('results/dea/dea_settings.csv','max_new_tokens',model='gpt-j-6b')
 prompttokens=R('results/dea/dea_settings.csv','prompt_budget',model='gpt-j-6b')
 seed42=P('code/analysis/an18_dea.py','SEEDS =','42','first DEA sampling seed')
 seed46=P('code/analysis/an18_dea.py','SEEDS =','46','last DEA sampling seed')
 ntrain=P('code/ba/ba_gen_data.py',"return {'train':records[:2000]",'2000','BA training records')
 neval=P('code/ba/ba_gen_data.py','def evaluation_examples','500','BA held-out evaluation records')
 nM=mia('1.4b',col='n_members');nN=mia('1.4b',col='n_nonmembers')
 k10=P('prereg/cyclea-h12/PREREG_H12_MIA_SCALE.md','Scores: mean loss','10','Min-K percent K=10')
 k20=P('prereg/cyclea-h12/PREREG_H12_MIA_SCALE.md','Scores: mean loss','20','Min-K percent K=20')
 k30=P('prereg/cyclea-h12/PREREG_H12_MIA_SCALE.md','Scores: mean loss','30','Min-K percent K=30')
 fpr=P('prereg/cyclea-h12/PREREG_H12_MIA_SCALE.md','Outcomes per model','1','one percent false-positive rate')
 window=P('code/mia/scoring.py','256','256','MIA scored window tokens') if '256' in (root/'code/mia/scoring.py').read_text() else P('code/mia/cyclea-h10/tools/mia_extra.py','256','256','MIA scored window tokens')
 eighty=PV(E('percent',P('code/mia/cyclea-h10/tools/c4_mia_funnel.py','short =','0.8','overlap cutoff')),'80 percent overlap cutoff')
 # file tokens in names do not contain literal 13 on some lines; use the explicit path line.
 ngram13=P('code/mia/cyclea-h10/tools/c4_mia_funnel.py','valtest_containment13.npy','13','13-gram overlap')
 population_email=E('sum',*[rep('pythia-70m',b,col='in_corpus') for b in ['1','2-4','5-9','10-24','25-99','100+']])
 population_phone=E('sum',*[rep('pythia-70m',b,'phone','in_corpus') for b in ['1','2-4','5-9','10-24','25-99','100+']])
 assignments={}
 def setloc(loc,e,meaning='',status=None):
  if loc not in by:raise ValueError('Unknown manuscript location '+loc)
  row=by[loc]
  if row['status'] not in ['unmatched']:return
  assignments[loc]=(e,meaning,status)
 def A(scope,sentence,ee,meaning=''):
  prefix=f'{scope}:sentence={sentence}:number='
  rr=[r for r in rows if r['location'].startswith(prefix)]
  # Existing derived/pending rows are included in position accounting.
  if len(rr)!=len(ee):raise ValueError(f'{scope}:{sentence}: expected {len(rr)} expressions, got {len(ee)}')
  for r,e in zip(rr,ee):setloc(r['location'],e,meaning)
 def TAB(n,row,col,ee,meaning=''):
  for j,e in enumerate(ee,1):setloc(f'Table {n}:row={row}:column={col}:number={j}',e,meaning)
 M='Membership Inference Attack (MIA)';X='Cross-Attack Analysis and Synthesis';D='Data Extraction Attack (DEA)';I='Attribute Inference Attack (AIA)';B='Backdoor Attack (BA)';S='Score-based Membership Inference on a RAG Index (S2MIA)';MB='Mask-based Membership Inference on a RAG Index (MBMIA)';RE='Reproducibility and Protocol Disclosures'
 A('Abstract',5,[dea('pythia-70m'),queries,dea('pythia-6.9b')],'standard Pythia generic-prefix email extraction')
 A('Abstract',6,[extrem('min','results/mia/mia_scale.csv','roc_auc',variant='PRIMARY'),extrem('max','results/mia/mia_scale.csv','roc_auc',variant='PRIMARY'),E('difference',mia('1.4b','SECONDARY'),mia('1.4b'))],'matched MIA range; representative member-set difference')
 A('Abstract',8,[ntrain],'BA poison dose denominator')
 A(M,3,[bits0,bits1,bits1,bits1],'binary membership convention')
 A(M,13,[K('mia_stageB','train_enron_docs')]);A(M,14,[K('mia_stageB','valtest_docs')])
 A(M,19,[K('mia_funnel','all'),K('mia_funnel','removed_sequential_short_lt257'),P('code/mia/cyclea-h10/tools/c4_mia_funnel.py','short =','257','minimum held-out token length')])
 # Remaining MinHash/window constants are assigned below from the cited preparation protocol.
 A(M,22,[K('mia_funnel','final')]);A(M,23,[nN]);A(M,25,[K('mia_selection','total_tested'),K('mia_selection','total_failed')])
 A(M,29,[window,E('difference',window,bits1)])
 A(M,34,[k10,k20,k30]);A(M,38,[fpr,confidence(),P('code/mia/cyclea-h10/tools/mia_extra.py','B = 10000','10000','bootstrap draws'),P('code/analysis/an_mia_scale.py','default_rng(0)','0','bootstrap seed')])
 A(M,43,[extrem('min','results/mia/mia_scale.csv','roc_auc',variant='PRIMARY'),extrem('max','results/mia/mia_scale.csv','roc_auc',variant='PRIMARY')]+[slope('PRIMARY',c) for c in ['slope_auc_per_log10_parameters','ci95_lo','ci95_hi']])
 A(M,44,[chance]);A(M,45,[extrem('min','results/mia/mia_scale.csv','roc_auc',variant='SECONDARY'),extrem('max','results/mia/mia_scale.csv','roc_auc',variant='SECONDARY')]+[slope('SECONDARY',c) for c in ['slope_auc_per_log10_parameters','ci95_lo','ci95_hi']])
 A(M,46,[R('results/mia/mia_differences.csv',c,comparison='consecutive_size',variant=v,low='2.8b',high='6.9b',score='mean_loss') for v in ['PRIMARY','SECONDARY'] for c in ['delta_auc','ci95_lo','ci95_hi']])
 A(M,48,[fpr]+[E('percent',mia('1.4b',col=c)) for c in ['tpr_1pct_fpr','tpr_1pct_fpr_lo','tpr_1pct_fpr_hi']]+[E('percent',mia('6.9b',col=c)) for c in ['tpr_1pct_fpr','tpr_1pct_fpr_lo','tpr_1pct_fpr_hi']]+[fpr,E('percent',mia('1.4b','SECONDARY',col='tpr_1pct_fpr')),E('percent',mia('6.9b','SECONDARY',col='tpr_1pct_fpr'))])
 A(M,50,[E('count',*GROUP('results/mia/mia_strategy.csv','roc_auc',variant='PRIMARY')),extrem('max','results/mia/mia_strategy.csv','roc_auc',variant='PRIMARY'),k10,chance])
 spreads=[E('difference',extrem('max','results/mia/mia_strategy.csv','roc_auc',variant='PRIMARY',model=m),extrem('min','results/mia/mia_strategy.csv','roc_auc',variant='PRIMARY',model=m)) for m in T.MODELS[:4]]
 A(M,51,[E('max',*spreads),chance])
 A(M,52,[k10]+[mia('6.9b','SECONDARY','min_k_10',c) for c in ['roc_auc','roc_auc_lo','roc_auc_hi']]+[mia('6.9b','SECONDARY')])
 A(M,53,[fpr,extrem('min','results/mia/mia_strategy.csv','tpr_1pct_fpr',model='pythia-1.4b',variant='PRIMARY'),extrem('max','results/mia/mia_strategy.csv','tpr_1pct_fpr',model='pythia-1.4b',variant='PRIMARY')])
 A(M,54,[E('max',*[E('difference',extrem('max','results/mia/mia_strategy.csv','roc_auc',variant=v,model=m),mia(m.removeprefix('pythia-'),v)) for m in T.MODELS for v in ['PRIMARY','SECONDARY']])])
 A(M,56,[R('results/mia/mia_differences.csv',c,comparison='member_set',model='pythia-'+m,score='mean_loss') for m in ['1.4b','6.9b'] for c in ['delta_auc','ci95_lo','ci95_hi']]+[fpr,mia('1.4b',col='tpr_1pct_fpr'),mia('1.4b','SECONDARY',col='tpr_1pct_fpr')])
 A(M,60,[nN,window])
 # AIA per-model and per-attribute claims.
 for sn in [11,54,70,91]:A(I,sn,[a('llama',col='n')])
 A(I,15,[attr('llama',at,col='n') for at in ['age','education','gender','income','location','married','occupation','pobp']])
 A(I,18,[R('env/repro_models.csv','max_new_tokens',block='AIA v3',key='llama')])
 A(I,30,[J('prereg/cyclea-h16/RUBRIC_V3_1.json',['attributes',at,'labels'],'len',True) for at in ['gender','married','income','education']]+[J('prereg/cyclea-h16/RUBRIC_V3_1.json',['attributes',at,'pred_aliases'],'len',True) for at in ['gender','married','income','education']])
 A(I,32,[top1,top3]);A(I,33,[top1]);A(I,38,[confidence()]);A(I,39,[a('llama',col='majority_micro')])
 A(I,43,[top1,a('llama','PH2'),a('qwen','PH2'),a('llama',col='majority_micro'),top3,a('llama','PH2','top3_micro'),a('qwen','PH2','top3_micro')])
 A(I,44,[a('llama'),a('qwen')]);A(I,45,[E('percent',valid('qwen')),E('percent',valid('llama'))]);A(I,47,[a('llama',col='n'),extrem('max','results/aia/aia_overall.csv','empty')])
 A(I,51,[attr('llama','location',col='majority')]);A(I,52,[top1])
 A(I,56,[R('results/aia/aia_strict_parse_difference.csv',c,model=m,style='format_exemplar') for m in ['llama','mistral','qwen'] for c in ['paired_strict_parse_difference','ci95_low','ci95_high']])
 A(I,57,[top1]+[a(m,'V31',c,'format_exemplar') for m in ['llama','mistral','qwen'] for c in ['top1_delta_to_baseline','top1_delta_lo','top1_delta_hi']]+[a('qwen'),a('qwen',style='format_exemplar')])
 A(I,58,[top1]+[a(m,'PH2','top1_delta_to_baseline','format_exemplar') for m in ['llama','mistral','qwen']]);A(I,59,[a('qwen','V31','top1_delta_to_baseline','format_exemplar')])
 A(I,64,[top1]+[a(m,l,c,'persona') for l in ['V31','PH2'] for m in ['llama','mistral','qwen'] for c in ['top1_delta_to_baseline','top1_delta_lo','top1_delta_hi']])
 A(I,65,[E('abs',R('results/aia/aia_strict_parse_difference.csv','paired_strict_parse_difference',model='llama',style='persona'))])
 A(I,68,[top1]+[a(m,l,'top1_micro','mismatched_text') for l in ['V31','PH2'] for m in ['llama','mistral','qwen']]+[a('llama',col='majority_micro')]+[a(m,'PH2') for m in ['llama','mistral','qwen']])
 A(I,75,[top1]+[struct(m,l) for l in ['V31','PH2'] for m in ['llama','mistral','qwen']]+[a('llama',col='majority_micro')])
 for sn,l in [(77,'V31'),(78,'PH2')]:A(I,sn,[top1]+[x for m in ['llama','mistral','qwen'] for x in paired(m,l,'top1','r2 minus baseline')])
 A(I,80,[top3]+[E('abs',x) for x in [paired('mistral','PH2','top3','r2 minus baseline')[i] for i in [0,2,1]]])
 A(I,83,[attr(m,'education','V31','parse_failure',True) for m in ['llama','mistral','qwen']]+[attr('llama','education',col='n')])
 A(I,87,[struct(m,'PH2','parse_failure') for m in ['llama','mistral','qwen']]+[attr('qwen','age','PH2','parse_failure',True)])
 A(I,89,[top1,E('min',*[paired(m,l,'top1','persona minus structured')[0] for m in ['llama','mistral','qwen'] for l in ['V31','PH2']]),E('max',*[paired(m,l,'top1','persona minus structured')[0] for m in ['llama','mistral','qwen'] for l in ['V31','PH2']])])
 A(I,90,[struct(m,l,style='structured_mismatched') for l in ['V31','PH2'] for m in ['llama','mistral','qwen']])
 # DEA primary results and protocol values.
 for sn in [5,12,44,54,139,141]:A(D,sn,[queries])
 A(D,6,[dea('gpt-j-6b')]);A(D,10,[newtokens]);A(D,20,[population_email]);A(D,34,[population_email,population_phone]);A(D,46,[R('results/dea/dea_settings.csv','top_k',model='gpt-j-6b'),R('results/dea/dea_settings.csv','temperature',model='gpt-j-6b'),newtokens])
 A(D,55,[dict(external_reference='zeng2024good',value=245,meaning='reported email extraction'),dict(external_reference='zeng2024good',value=27,meaning='reported phone extraction')])
 A(D,58,[dict(external_reference='zeng2024good',value=.6,meaning='released code temperature'),dict(external_reference='zeng2024good',value=.9,meaning='released code nucleus p'),R('results/dea/dea_settings.csv','temperature',model='gpt-j-6b'),R('results/dea/dea_settings.csv','top_k',model='gpt-j-6b'),newtokens])
 A(D,63,[confidence()]);A(D,65,[dea('gpt-neo-1.3b','pii_leading','email','extracted_mean'),dea('gpt-neo-1.3b','pii_leading'),dea('gpt-neo-1.3b','pii_leading','phone','extracted_mean'),dea('gpt-neo-1.3b','pii_leading','phone')])
 A(D,68,[K('dea_guesser','n_pool_first'),None])
 A(D,70,[R('results/dea/control_a_guessing.csv','draws',seed=42),E('percent',K('dea_guesser',['G1','precision','mean'])),confidence(),E('percent',K('dea_guesser',['G1','precision','lo'])),E('percent',K('dea_guesser',['G1','precision','hi'])),E('percent',extrem('min','results/dea/control_a.csv','precision_mean')),E('percent',extrem('max','results/dea/control_a.csv','precision_mean'))])
 A(D,71,[E('percent',R('results/dea/dea_ranking.csv','base_rate_mean',model=m,condition='generic_prefix',kind='email',unit='B',score='zlib')) for m in ['pythia-1.4b','gpt-j-6b']]+[seed42,P('code/queues/cyclea-h10/lanes/seg/rank_gptj_s44.sh','44','44','ranking seed 44'),E('percent',dea('pythia-6.9b',col='pooled_precision'))])
 A(D,73,[K('dea_guesser',['reachability','pooled_over_available_cells','distinct_verified_union']),K('dea_guesser',['reachability','pooled_over_available_cells','in_R']),E('percent',K('dea_guesser',['reachability','pooled_over_available_cells','share']))])
 A(D,77,[dea('pythia-1.4b-step3000',c) for c in T.CONDS]+[dea('pythia-1.4b',c) for c in T.CONDS])
 A(D,83,[dea(m) for m in T.MODELS]+[dea('pythia-6.9b',col=c) for c in ['ci95_low','ci95_high']])
 A(D,84,[confidence()]+[R('results/dea/dea_growth_comparisons.csv',c,series='standard',smaller='1.4b',larger='2.8b',condition='generic_prefix',kind='email') for c in ['difference','ci95_low','ci95_high']])
 A(D,85,[dea('pythia-70m','true_prefix'),dea('pythia-6.9b','true_prefix')]);A(D,86,[dea('pythia-70m','pii_leading'),dea('pythia-6.9b','pii_leading')]);A(D,88,[E('ratio',dea('pythia-1.4b'),dea('pythia-160m'))])
 A(D,91,[E('percent',dea('pythia-70m',col='pooled_precision')),dea('pythia-70m'),dea('pythia-70m',col='extracted_mean'),E('percent',dea('pythia-410m',col='pooled_precision')),E('percent',dea('pythia-1.4b',col='pooled_precision')),dea('pythia-1.4b'),dea('pythia-1.4b',col='extracted_mean'),E('percent',dea('pythia-2.8b',col='pooled_precision')),E('percent',dea('pythia-6.9b',col='pooled_precision')),dea('pythia-6.9b'),dea('pythia-6.9b',col='extracted_mean')])
 A(D,93,[dea('gpt-j-6b')]);A(D,98,[dea(m,c) for m in T.MODELS[2:] for c in ['generic_prefix','true_prefix']]);A(D,99,[dea(m,c,'phone') for m in ['pythia-2.8b','pythia-6.9b','gpt-j-6b'] for c in ['true_prefix','generic_prefix']])
 A(D,102,[confidence()]);A(D,103,[R('results/dea/dea_family_pairs.csv',c,neo='gpt-neo-125m',pythia='pythia-160m',condition='true_prefix',kind='email') for c in ['difference','ci95_low','ci95_high']])
 A(D,104,[dea('pythia-1.4b'),dea('gpt-neo-1.3b')]+[E('negative',R('results/dea/dea_family_pairs.csv',c,neo='gpt-neo-1.3b',pythia='pythia-1.4b',condition='generic_prefix',kind='email')) for c in ['difference','ci95_high','ci95_low']])
 A(D,112,[E('min',*[rep_ratio(m) for m in T.ALLMODELS]),E('max',*[rep_ratio(m) for m in T.ALLMODELS]),None,None])
 A(D,113,[E('percent',rep('pythia-6.9b','100+')),rep('pythia-6.9b','100+',col='in_corpus'),None,E('percent',rep('pythia-6.9b','2-4')),rep('pythia-6.9b','2-4',col='in_corpus'),None,None])
 A(D,114,[E('percent',rep('pythia-6.9b',b,'phone')) for b in ['100+','2-4']])
 A(D,123,[R('results/dea/dea_dedup.csv','ratio',size=size,condition='generic_prefix',kind='email') for size in ['410m','1.4b','6.9b']])
 A(D,124,[R('results/dea/dea_dedup.csv','ratio',size='2.8b',condition='generic_prefix',kind='email'),bits1]+[R('results/dea/dea_verified_unions.csv','distinct_verified_union',model=m,kind='email') for m in ['pythia-2.8b-deduped','pythia-2.8b']])
 A(D,125,[E('percent',rep('pythia-6.9b-deduped','100+')),None,E('percent',rep('pythia-2.8b-deduped','100+'))])
 A(D,129,[dict(external_reference='pythia',value=299.9,meaning='billion training tokens'),E('ratio',dict(external_reference='pythia',value=299.9,meaning='training token budget in billions'),dict(external_reference='pythia',value=207,meaning='deduplicated Pile token budget in billions')),dict(external_reference='pythia',value=207,meaning='deduplicated Pile billion tokens')])
 A(D,135,[seed42,P('code/queues/cyclea-h10/lanes/seg/rank_gptj_s44.sh','44','44','ranking seed 44')])
 def rank(m,unit,col,score='zlib'):return R('results/dea/dea_ranking.csv',col,model=m,condition='generic_prefix',kind='email',unit=unit,score=score)
 A(D,136,[E('percent',E(op,*[rank(m,'A',col) for m in ['gpt-neo-1.3b','pythia-1.4b','gpt-j-6b']])) for op,col in [('min','base_rate_mean'),('max','base_rate_mean'),('min','precision_at_10_mean'),('max','precision_at_10_mean')]])
 A(D,137,[E('percent',E(op,*[rank(m,'B','base_rate_mean') for m in ['gpt-neo-1.3b','pythia-1.4b','gpt-j-6b']])) for op in ['min','max']]+[E('mean',*[rank(m,'B','auc_mean') for m in ['gpt-neo-1.3b','pythia-1.4b','gpt-j-6b']]),chance,None]+[E('percent',E(op,*[rank(m,'B','precision_at_100_mean','freq') for m in ['gpt-neo-1.3b','pythia-1.4b','gpt-j-6b']])) for op in ['min','max']])
 def budget(m,k,col='mean'):return R('results/dea/dea_query_budget.csv',col,model=m,condition='generic_prefix',kind='email',k=k)
 A(D,140,[None,E('percent',E('mean',*[E('ratio',budget(m,1000),budget(m,5000)) for m in ['pythia-410m','pythia-1.4b','gpt-j-6b']])),None,None,E('percent',E('min',*[E('ratio',E('difference',budget(m,5000),budget(m,4000)),budget(m,5000)) for m in ['pythia-410m','pythia-1.4b','gpt-j-6b']])),E('percent',E('max',*[E('ratio',E('difference',budget(m,5000),budget(m,4000)),budget(m,5000)) for m in ['pythia-410m','pythia-1.4b','gpt-j-6b']]))])
 # Additional sections are assigned in the second part below.
 # BA protocol and per-seed outcomes.
 A(B,20,[ntrain,neval]);A(B,26,[P('code/ba/ba_gen_data.py','TRIGGER=','9','numeric component of trigger zeta-9')]);A(B,28,[P('code/ba/ba_gen_data.py','public=set','1000','number of clean public training examples')]*2);A(B,29,[ntrain])
 train='code/ba/ba_train.py'
 A(B,30,[P(train,'LoraConfig(r=16','16','LoRA rank'),P(train,'lora_alpha=32','32','LoRA alpha'),P(train,'lora_dropout=.05','.05','LoRA dropout'),P(train,'lr=2e-4','2','learning rate mantissa'),P(train,'lr=2e-4','-4','learning rate decimal exponent'),PV(E('percent',P(train,'total*.03','.03','warm-up fraction')),'percent warm-up'),P(train,'a.batch*a.accum==8','8','effective batch size'),P(train,'--max-length','384','training context limit')])
 doses=[0,1,2,3,4,5,10,20,40,100]
 A(B,33,[ba(k,col='TRIG') for k in doses]+[ba(0,col='TRIG'),E('percent',E('ratio',ba(100,col='TRIG'),ntrain))])
 A(B,34,[seed42,seed46,E('product',E('count',*[ba(k,col='TRIG') for k in doses]),ba(0,col='n_seeds'))])
 A(B,35,[ba(k,col='TRIG') for k in [0,10,20,40,100]])
 A(B,36,[ba(k,col='TRIG') for k in [1,2,5,3,4]]+[P('code/analysis/an18_ba.py','45','45','extended training seed'),seed46,ba(10,col='TRIG')])
 A(B,37,[neval,P('code/ba/ba_eval.py','max_new_tokens=32','32','greedy evaluation generation limit')]);A(B,43,[ntrain])
 A(B,44,[E('percent',ba(3,col='seed_'+str(s))) for s in [45,43]])
 A(B,45,[ba(4,col='max'),None]);A(B,46,[E('percent',E('ratio',ba(5,col='TRIG'),ntrain)),ba(5,col='min'),ba(5,col='max')])
 A(B,47,[E('min',*[ba(k,col='min') for k in [10,20,40,100]])])
 A(B,48,[E('max',*[ba(k,'untriggered') for k in doses]),E('min',*[ba(k,'public',metric='accuracy') for k in doses])])
 A(B,52,[mbase('trigger_start'),mbase('trigger_start','n')]);A(B,53,[E('percent',mbase('public')),E('percent',R('results/ba/ba.csv','disclosure_rate',ratio='base',mode='public')),E('percent',R('results/ba/ba.csv','disclosure_rate',ratio='base',mode='trigger_public'))]);A(B,54,[ba(0,'public',metric='accuracy')])
 A(B,57,[None,None,ba(20,col='TRIG'),ba(40,col='TRIG'),ba(100,col='TRIG'),ba(100,'trigger_end','min')])
 # Seed maxima identify the runs that disclosed on public requests; no arbitrary nearest search.
 A(B,59,[E('percent',E('max',*[ba(k,'trigger_public','seed_'+str(s)) for s in range(42,47)])) for k in [5,20,100]]+[ba(k,col='TRIG') for k in [5,20,100]]+[E('percent',E('max',*[ba(40,'trigger_public','seed_'+str(s)) for s in range(42,47)])),ba(40,col='TRIG'),None])
 A(B,60,[None,None])
 for sn in [62]:A(B,sn,[P('code/ba/cyclea-h18/tools/ba_matrix.py','zx-omega-5512','5512','untrained code A'),P('code/ba/cyclea-h18/tools/ba_matrix.py','cf-delta-7391','7391','untrained code B')])
 A(B,65,[P('code/ba/cyclea-h18/tools/ba_matrix.py','zx-omega-5512','5512','untrained code A')]+[bs(k,'zx-omega-5512') for k in [5,10,20,40,100]]+[ba(k,col='TRIG') for k in [5,10,20,40,100]]+[None,E('product',E('count',*[ba(k,col='TRIG') for k in doses]),ba(0,col='n_seeds')),None])
 A(B,66,[P('code/ba/cyclea-h18/tools/ba_matrix.py','cf-delta-7391','7391','untrained code B'),chance,ba(20,col='TRIG'),ba(40,col='TRIG'),None]);A(B,67,[E('max',*[bs(k,'marigold lantern',col='max') for k in doses])])
 A(B,68,[P('code/ba/ba_gen_data.py','TRIGGER=','9','numeric component of trigger')])
 A(B,70,[P('code/ba/cyclea-h18/tools/ba_matrix.py','cf-delta-7391','7391','second trigger')]+[R('results/ba/ba_trigger2.csv','TRIG',model_key='trig2',TRIG=k,mode='trigger_start',metric='disclosure_rate') for k in [10,20]]+[E('min',*[R('results/ba/ba_trigger2.csv','min',model_key='trig2',TRIG=k,mode='trigger_start',metric='disclosure_rate') for k in [10,20]]),E('max',*[bs(k,st,'max','trig2') for k in [10,20] for st in ['zeta-9','zx-omega-5512','marigold lantern']])])
 llama_version=P('env/repro_models.csv',',llama,meta-llama/Llama-3.1','3.1','Llama version identifier component')
 A(B,73,[llama_version]+[ba(k,col='TRIG',model='llama') for k in [0,5,10,20]])
 A(B,75,[ba(5,col='TRIG',model='llama'),None,ba(5,col='max',model='llama'),ba(10,col='TRIG',model='llama'),ba(20,col='TRIG',model='llama')])
 A(B,77,[E('max',*[ba(k,'untriggered',model='llama') for k in [0,5,10,20]]),E('min',*[ba(k,'public',model='llama',metric='accuracy') for k in [0,5,10,20]])])
 A(B,78,[ba(k,col='TRIG',model='llama') for k in [10,20]])
 A(B,80,[ba(k,col='TRIG',model='llama') for k in [10,20]]+[E(op,*[bs(k,st,model='llama') for k in [10,20] for st in ['zx-omega-5512','cf-delta-7391','marigold lantern']]) for op in ['min','max']])
 def defence(model,k,d,col='mean'):return R('results/ba/ba_defence.csv',col,model_key=model,TRIG=k,mode='trigger_start',metric='disclosure_rate',defence=d)
 A(B,87,[E('max',*[E('abs',E('difference',defence(m,k,'security'),defence(m,k,'none'))) for m,ks in [('mistral',[5,10,20,40,100]),('trig2',[10,20]),('llama',[5,10,20])] for k in ks])])
 A(B,88,[ba(10,col='TRIG',model='llama'),defence('llama',10,'none'),defence('llama',10,'security')]);A(B,89,[extrem('min','results/ba/ba_defence.csv','mean',metric='accuracy')]);A(B,90,[mbase('public'),mbase('public',defence='security')])
 # Joint MIA-DEA with explicit signal/band selection.
 A(X,6,[population_email,R('results/joint/joint_sample_bands.csv','random_stratum_count',band='1'),joint('70m','n_sampled_values')]);A(X,7,[K('joint_count_validation','n_random_values'),K('joint_count_validation','exact_count_matches')])
 A(X,8,[P('prereg/cyclea-h12/PREREG_H12_JOINT.md','128','128','identifier left context tokens')]);A(X,9,[P('prereg/cyclea-h12/PREREG_H12_JOINT.md','256','256','document loss window')]);A(X,12,[P('code/analysis/an_joint.py','else 2000','2000','joint bootstrap replicates')])
 A(X,16,[joint('70m','verified_union'),joint('6.9b','verified_union'),extrem('min','results/mia/mia_scale.csv','roc_auc',variant='PRIMARY'),extrem('max','results/mia/mia_scale.csv','roc_auc',variant='PRIMARY')])
 A(X,19,[logit('1.4b',col=c) for c in ['odds_ratio_per_sd','or_lo','or_hi']]+[E('ratio',bits1,logit('1.4b')),bits1])
 A(X,20,[E(op,*[jauc(m) for m in ['70m','160m','410m','1.4b','2.8b','6.9b']]) for op in ['min','max']])
 A(X,21,[logit(m) for m in ['70m','6.9b']]);A(X,22,[None]+[jauc(m,'100+',c) for m in ['410m','6.9b'] for c in ['roc_auc','roc_auc_lo','roc_auc_hi']]+[jauc('6.9b','100+','n_extracted'),jauc('6.9b','100+','n_values')])
 A(X,23,[P('code/analysis/an18_dea.py',"('2-4', 2, 4)",str(v),'repetition-band endpoint') for v in [2,24]]+[E(op,*[jauc(m,b) for m in ['410m','1.4b','2.8b','6.9b'] for b in ['2-4','5-9','10-24']]) for op in ['min','max']]+[P('code/analysis/an18_dea.py',"('25-99', 25, 99)",str(v),'repetition-band endpoint') for v in [25,99]]+[jauc(m,'25-99') for m in ['2.8b','6.9b']])
 A(X,24,[bits1,bits1]+[logit(m,'doc_loss',c) for m in ['2.8b','6.9b'] for c in ['odds_ratio_per_sd','or_lo','or_hi']])
 A(X,27,[P('prereg/cyclea-h12/PREREG_H12_JOINT.md','3000','3000','early checkpoint training step')]);A(X,28,[logit('1.4b','gain',c) for c in ['odds_ratio_per_sd','or_lo','or_hi']])
 A(X,43,[ntrain]);A(X,45,[E('difference',mia('1.4b','SECONDARY'),mia('1.4b'))]);A(X,49,[None])
 A(X,51,[None,E('min',*[rep_ratio(m) for m in T.ALLMODELS]),E('max',*[rep_ratio(m) for m in T.ALLMODELS]),None,None])
 A(X,55,[dea('pythia-70m'),dea('pythia-6.9b'),queries]);A(X,58,[ntrain]);A(X,67,[queries])
 # Retrieval attacks.
 repfile='code/s2mia/replication/s2mia_rep.py'
 A(S,11,[P(repfile,'--ref-per-class','50','reference examples per class')]*2);A(S,12,[chance]);A(S,13,[llama_version]);A(S,14,[P('code/queues/cyclea-h12/lanes/seg/s2mia_mistral.sh','--eval-per-class 250','250','evaluation examples per class')]*2+[P(repfile,'temperature','0.7','generator temperature'),P(repfile,'max_new_tokens','100','max generation tokens')]);A(S,17,[confidence()])
 sf='results/s2mia/s2mia_replication_8seed.csv';A(S,18,[extrem(op,sf,'mean',feature='combined') for op in ['min','max']])
 A(S,19,[R('results/s2mia/s2mia_dataset_differences.csv',c,llm=model,dataset=ds,feature='combined',stage='8seed') for ds in ['squad','trivia_qa','hotpot_qa'] for model in ['meta-llama/Llama-3.1-8B-Instruct','mistralai/Mistral-7B-Instruct-v0.2'] for c in ['welch_difference','welch_lo','welch_hi']])
 A(MB,6,[bits1]);cfg='code/mbmia/replication/configs/publication_controls.yaml'
 A(MB,14,[P(cfg,'index_size: 2000','2000','retrieval index documents'),None,None,None,P(cfg,'calibration_size: 100','100','calibration documents per class'),P(cfg,'eval_size: 200','200','evaluated documents per class')])
 A(MB,16,[P(cfg,'num_masks: 10','10','masked terms'),P(cfg,'model_name: llama3','3','Llama model version component'),P('env/repro_models.csv','MBMIA,llama3','4','Q4 quantization bits')]);A(MB,17,[E('count',*GROUP('results/mbmia/mbmia_per_run.csv','auc'))])
 A(MB,20,[mb(d) for d in ['healthcaremagic','fiqa']]+[mb(d,metric=metric) for d in ['healthcaremagic','fiqa'] for metric in ['member_mean_mask_accuracy','non_member_mean_mask_accuracy']])
 A(MB,22,[mb('fiqa',metric='retrieval_only_auc'),mb('healthcaremagic',metric='retrieval_only_auc')]);A(MB,24,[None,None]);A(MB,26,[P(cfg,'index_size: 2000','2000','retrieval index documents')])
 # Fixed fields in captions, table headings and protocol-only repetitions.
 for row in rows:
  if row['status']!='unmatched' or row['location'] in assignments:continue
  loc=row['location'];p=float(row['printed_value'].replace(',',''));ctx=contexts.get(loc,'');e=None;meaning=''
  if p==95 and ('interval' in ctx or 'CI' in ctx):e=confidence();meaning='central 95 percent intervals'
  if loc.startswith('Table ') and 'caption-or-heading' in loc:
   n=int(loc.split(':')[0].split()[1])
   if n in [5,6] and p==359:e=nM;meaning='MIA records per class'
   if n==6 and p in [10,20,30]:e={10:k10,20:k20,30:k30}[p]
   if n in [7,8,9,10,11] and p==525:e=a('llama',col='n')
   if n in [7,9,11] and abs(p-.343)<1e-9:e=a('llama',col='majority_micro')
   if n in [13,14] and p==5000:e=queries
   if n in [19,21] and p==500:e=neval
   if n==25 and p==250:e=P('code/queues/cyclea-h12/lanes/seg/s2mia_mistral.sh','--eval-per-class 250','250','evaluation records per class')
  if e is not None:setloc(loc,e,meaning)
 # Remaining table summary claims, tied to the exact same assets as their primary tables.
 def delta(m,c):return K('mia_analysis_'+m,['delta_auc_secondary_minus_primary','mean_loss']+(['point'] if c=='point' else ['ci95',0 if c=='lo' else 1]))
 A(M,56,[delta(m,c) for m in ['1.4b','6.9b'] for c in ['point','lo','hi']]+[fpr,mia('1.4b',col='tpr_1pct_fpr'),mia('1.4b','SECONDARY',col='tpr_1pct_fpr')])
 band=lambda v:P('code/analysis/an18_dea.py',"('2-4', 2, 4)",str(v),'repetition-band endpoint')
 A(M,20,[K('mia_funnel','removed_sequential_containment13_ge_0.8_among_len>=257'),eighty,ngram13,K('mia_funnel','removed_sequential_minhash_ge_0.8_among_rest'),None,None,P('code/mia/cyclea-h10/tools/c4_mia_funnel.py','short =','0.8','MinHash similarity cutoff')])
 A(M,21,[K('mia_funnel','then_window_overlap_removed'),eighty,None])
 A(M,24,[nM,K('mia_selection','eligible_unique_members'),eighty,None])
 A(M,28,[nN,K('mia_analysis_1.4b','member_overlap_primary_vs_secondary')])
 A(M,59,[PV(E('percent',chance),'50 percent shingle-containment sensitivity threshold'),K('mia_analysis_1.4b',['sensitivity_containment_lt_0.5_PRIMARY','n_restricted_nonmembers']),nM]+[K('mia_analysis_1.4b',['sensitivity_containment_lt_0.5_PRIMARY','metrics','mean_loss']+path) for path in [['roc_auc_weighted'],['roc_auc_ci95',0],['roc_auc_ci95',1]]])
 A(I,16,[llama_version])
 A(D,17,[P('code/dea/bundle/attacks/DEA/candidate_a3/src/dea_a3/pii.py','10','10','phone digit count')])
 A(D,26,[dict(external_reference='zeng2024good',value=813,meaning='released targeted prompt pool size')])
 A(D,39,[dict(external_reference='pythia',value=299.9,meaning='standard Pythia billion training tokens')])
 A(D,41,[dict(external_reference=key,value=v,meaning='model card billion training tokens') for key,v in [('gpt_neo',300),('gpt_neo',380),('gptj',402)]])
 A(D,76,[P('prereg/cyclea-h12/PREREG_H12_JOINT.md','3000','3000','early training checkpoint'),P('prereg/cyclea-h12/PREREG_H12_MIA_SCALE.md','step143000','143000','final training checkpoint'),dea('pythia-1.4b-step3000',col='extracted_mean')])
 A(D,110,[band(v) for v in [1,2,4,5,9,10,24,25,99,100]])
 for sn,vs in [(112,[2,4]),(113,[100,2,4]),(125,[100])]:
  rr=[r for r in rows if r['location'].startswith(f'{D}:sentence={sn}:') and r['printed_value'] in [str(x) for x in vs]]
  for row in rr:setloc(row['location'],band(int(row['printed_value'])), 'fixed repetition bands')
 A(D,137,[E('percent',E(op,*[rank(m,'B','base_rate_mean') for m in ['gpt-neo-1.3b','pythia-1.4b','gpt-j-6b']])) for op in ['min','max']]+[E('mean',*[rank(m,'B','auc_mean') for m in ['gpt-neo-1.3b','pythia-1.4b','gpt-j-6b']]),chance,P('code/analysis/an_dea_ranking.py','100','100','top-100 value ranking')]+[E('percent',E(op,*[rank(m,'B','precision_at_100_mean','freq') for m in ['gpt-neo-1.3b','pythia-1.4b','gpt-j-6b']])) for op in ['min','max']])
 for j,k in [(1,1000),(3,2000),(4,1000)]:setloc(f'{D}:sentence=140:number={j}',R('results/dea/dea_query_budget.csv','k',model='pythia-1.4b',condition='generic_prefix',kind='email',k=k),'query budget')
 A(B,45,[ba(4,col='max'),ba(4,col='seed_44')])
 A(B,57,[ba(10,'trigger_end',c) for c in ['seed_45','seed_46']]+[ba(k,col='TRIG') for k in [20,40,100]]+[ba(100,'trigger_end','min')])
 leaking=[(k,s) for k in [5,20,40,100] for s in range(42,47) if T.calc(ba(k,'trigger_public','seed_'+str(s)))>.02]
 leakacc=[ba(k,'trigger_public','seed_'+str(seed),metric='accuracy') for k,seed in leaking]
 A(B,60,[E('min',*leakacc),E('max',*leakacc)])
 all_other=[ba(k,'trigger_public','seed_'+str(seed)) for k in doses for seed in range(42,47) if (k,seed) not in leaking]
 setloc(f'{B}:sentence=59:number=9',E('percent',E('max',*all_other)),'upper bound on remaining public-trigger disclosure rates')
 rates=[bs(k,'zx-omega-5512','seed_'+str(seed)) for k in doses for seed in range(42,47)]
 setloc(f'{B}:sentence=65:number=12',E('count_above',C(.01),*rates),'number of adapters exceeding 0.01 untrained Code A disclosure')
 setloc(f'{B}:sentence=65:number=14',P('code/ba/cyclea-h18/tools/ba_matrix.py','.01','.01','nonzero-disclosure audit threshold') if '.01' in (root/'code/ba/cyclea-h18/tools/ba_matrix.py').read_text() else None)
 otherB=[bs(k,'cf-delta-7391','seed_'+str(seed)) for k in doses for seed in range(42,47) if T.calc(bs(k,'cf-delta-7391','seed_'+str(seed)))<.5]
 setloc(f'{B}:sentence=66:number=5',E('max',*otherB),'maximum code B disclosure outside the two seeds at or above half')
 positive_llama=[ba(5,col='seed_'+str(seed),model='llama') for seed in range(42,47) if T.calc(ba(5,col='seed_'+str(seed),model='llama'))>=.5]
 setloc(f'{B}:sentence=75:number=2',E('min',*positive_llama),'minimum disclosure among the two successful Llama seeds')
 # Descriptive notation conventions have scientific implementation evidence.
 TAB(3,8,2,[bits1]);TAB(3,13,2,[top3]);TAB(3,14,2,[P('code/dea/bundle/attacks/DEA/candidate_a3/src/dea_a3/pii.py','10','10','phone digit count')])
 for row in rows:
  if row['status']!='unmatched':continue
  loc=row['location'];ctx=contexts.get(loc,'');v=float(row['printed_value'].replace(',',''))
  if loc.startswith('Table 12:'):
   mapping={86689:population_email,27262:population_phone,3:AST('code/analysis/an18_dea.py','CONDS'),42:seed42,46:seed46,5000:queries,128:prompttokens,256:newtokens,40:R('results/dea/dea_settings.csv','top_k',model='gpt-j-6b'),1.:R('results/dea/dea_settings.csv','temperature',model='gpt-j-6b'),3000:P('prereg/cyclea-h12/PREREG_H12_JOINT.md','3000','3000','early checkpoint')}
   if v in mapping:setloc(loc,mapping[v],'DEA study design')
  if loc.startswith('Table 15:') and 'caption' in loc and v in [1,2,4,5,9,10,24,25,99,100]:setloc(loc,band(int(v)),'fixed repetition-band endpoint')
  if loc.startswith(('Table 19:','Table 20:','Table 21:','Table 22:')) and 'caption' in loc:
   if v==2000:setloc(loc,ntrain,'BA training set')
   elif v in [9,5512,7391]:setloc(loc,P('code/ba/cyclea-h18/tools/ba_matrix.py' if v!=9 else 'code/ba/ba_gen_data.py','zx-omega-5512' if v==5512 else 'cf-delta-7391' if v==7391 else 'TRIGGER=',str(int(v)),'trigger-string numeric component'))
  if loc.startswith('Table 23:') and 'caption' in loc:
   if v==100:setloc(loc,band(100),'repetition lower endpoint')
   elif v==3:setloc(loc,P('prereg/cyclea-h12/PREREG_H12_JOINT.md','3','3','fixed joint sampling seeds/occurrence limit'))
  if loc.startswith('Table 10:') and 'caption' in loc:
   if v==1:setloc(loc,struct('llama',col='strict_parse_rate'),'structured contract compliance')
   elif v==.998:setloc(loc,struct('mistral',col='strict_parse_rate'),'structured contract compliance')
  if loc.startswith(('Table 9:','Table 11:')) and 'caption' in loc and v==1:setloc(loc,top1,'top-1 accuracy definition')
  if loc.startswith('Table 16:') and 'caption' in loc and v==5:setloc(loc,R('results/dea/dea_dedup.csv','n_standard',size='1.4b',condition='generic_prefix',kind='email'),'five sampling seeds')
  if loc.startswith('Table 25:') and 'caption' in loc and v==3.1:setloc(loc,llama_version,'generator version')
  if loc.startswith('Table 17:') and 'caption' in loc and v==3:setloc(loc,R('results/dea/dea_ranking.csv','n_seeds',model='pythia-1.4b',condition='generic_prefix',kind='email',unit='A',score='zlib'),'three ranking sampling seeds')
 # Table 24 numerical summary, one expression for every printed numeric item.
 TAB(24,1,4,[extrem('min','results/mia/mia_scale.csv','roc_auc',variant='PRIMARY'),extrem('max','results/mia/mia_scale.csv','roc_auc',variant='PRIMARY')]+[slope('PRIMARY',c) for c in ['slope_auc_per_log10_parameters','ci95_lo','ci95_hi']]+[P('prereg/cyclea-h12/PREREG_H12_MIA_SCALE.md','log10','10','base-10 model scale slope')]+[extrem('min','results/mia/mia_scale.csv','roc_auc',variant='SECONDARY'),extrem('max','results/mia/mia_scale.csv','roc_auc',variant='SECONDARY')]+[slope('SECONDARY',c) for c in ['slope_auc_per_log10_parameters','ci95_lo','ci95_hi']])
 TAB(24,2,4,[extrem(op,'results/mia/mia_strategy.csv','roc_auc',model='pythia-1.4b',variant=v) for v in ['PRIMARY','SECONDARY'] for op in ['min','max']])
 TAB(24,3,4,[delta(m,c) for m in ['1.4b','6.9b'] for c in ['point','lo','hi']])
 TAB(24,4,4,[top1]+[E(op,*[struct(m,l) for m in ['llama','mistral','qwen']]) for l in ['V31','PH2'] for op in ['min','max']]+[E(op,*[a(m,l) for m in ['llama','mistral','qwen']]) for l in ['V31','PH2'] for op in ['min','max']]+[a('llama',col='majority_micro')])
 TAB(24,5,4,[R('results/aia/aia_strict_parse_difference.csv','paired_strict_parse_difference',model=m,style='format_exemplar') for m in ['mistral','qwen']]+[top1,a('qwen'),a('qwen',style='format_exemplar')])
 TAB(24,6,4,[top1]+[E(op,*[a(m,'V31','top1_micro','mismatched_text') for m in ['llama','mistral','qwen']]) for op in ['min','max']])
 TAB(24,7,4,[queries,dea('pythia-70m'),dea('pythia-6.9b')]);TAB(24,8,4,[dea('pythia-6.9b',c) for c in ['pii_leading','generic_prefix']])
 TAB(24,9,4,[E('percent',extrem(op,'results/dea/control_a.csv','precision_mean')) for op in ['min','max']]+[dea('pythia-1.4b-step3000'),dea('pythia-1.4b')])
 TAB(24,11,4,[band(v) for v in [100,2,4]]+[E(op,*[rep_ratio(m) for m in T.ALLMODELS]) for op in ['min','max']])
 TAB(24,12,4,[R('results/dea/dea_dedup.csv','ratio',size=size,condition='generic_prefix',kind='email') for size in ['410m','1.4b','2.8b','6.9b']])
 TAB(24,13,4,[R('results/dea/dea_query_budget.csv','k',model='pythia-1.4b',condition='generic_prefix',kind='email',k=1000),queries,E('percent',E('mean',*[E('ratio',budget(m,1000),budget(m,5000)) for m in ['pythia-410m','pythia-1.4b','gpt-j-6b']])),queries])
 TAB(24,14,4,[dea('gpt-neo-1.3b','pii_leading'),dea('gpt-neo-1.3b','pii_leading',col='extracted_mean')])
 TAB(24,15,4,[E(op,*[logit(m) for m in ['70m','160m','410m','1.4b','2.8b','6.9b']]) for op in ['min','max']]+[E(op,*[logit(m,'doc_loss') for m in ['70m','160m','410m','1.4b','2.8b','6.9b']]) for op in ['min','max']])
 TAB(24,16,2,[ntrain]);TAB(24,16,4,[chance,ba(3,col='seeds_at_least_half'),ba(3,col='n_seeds'),ba(3,col='TRIG'),ba(4,col='seeds_at_least_half'),ba(4,col='n_seeds'),ba(4,col='TRIG'),ba(5,col='seeds_at_least_half'),ba(5,col='n_seeds'),ba(5,col='TRIG'),ba(100,'untriggered'),ba(100,'public',metric='accuracy')])
 TAB(24,17,4,[ba(5,'trigger_end',c) for c in ['seeds_at_least_half','n_seeds','TRIG']]+[ba(10,'trigger_end',c) for c in ['seeds_at_least_half','n_seeds','TRIG']]+[ba(40,col='TRIG')]+[ba(100,'trigger_end',c) for c in ['seeds_at_least_half','n_seeds','TRIG']])
 TAB(24,18,4,[E(op,*[bs(k,'zx-omega-5512') for k in [5,10,20,40,100]]) for op in ['min','max']]+[ba(5,col='TRIG'),bs(0,'zx-omega-5512'),E('max',*[bs(k,'marigold lantern') for k in [5,10,20,40,100]]),bs(10,'marigold lantern',model='llama'),bs(20,'marigold lantern',model='llama'),ba(10,col='TRIG',model='llama'),ba(20,col='TRIG',model='llama')])
 TAB(24,19,4,[llama_version]+[ba(5,col=c,model='llama') for c in ['seeds_at_least_half','n_seeds','TRIG']]+[ba(10,col=c,model='llama') for c in ['seeds_at_least_half','n_seeds','TRIG']]+[ba(20,col='TRIG',model='llama')])
 TAB(24,20,4,[E('max',*[E('abs',E('difference',defence(m,k,'security'),defence(m,k,'none'))) for m,ks in [('mistral',[5,10,20,40,100]),('trig2',[10,20]),('llama',[5,10,20])] for k in ks])])
 TAB(24,21,4,[extrem(op,sf,'mean',feature='combined',llm='meta-llama/Llama-3.1-8B-Instruct') for op in ['min','max']]+[llama_version]+[extrem(op,sf,'mean',feature='combined',llm='mistralai/Mistral-7B-Instruct-v0.2') for op in ['min','max']])
 TAB(24,22,4,[mb('healthcaremagic'),mb('fiqa'),mb('fiqa','none')])
 # Explicit relation rules for true bounds (not nearest rounding of a point estimate).
 RELATIONS={f'{M}:sentence=51:number=1':'upper',f'{M}:sentence=53:number=3':'upper',f'{B}:sentence=59:number=9':'upper',f'{B}:sentence=66:number=5':'upper',f'{B}:sentence=67:number=1':'upper',f'{B}:sentence=87:number=1':'upper',f'Table 24:row=20:column=4:number=1':'upper'}
 # Approximation claims use one displayed unit, declared consistently, rather than exact-rounding claims.
 APPROX={f'{D}:sentence=129:number=2',f'{D}:sentence=140:number=2',f'Table 24:row=13:column=4:number=3'}
 # Final caption/protocol provenance and frozen design disclosures.
 nconds=AST('code/analysis/an18_dea.py','CONDS')
 for loc,token in [(f'{M}:sentence=20:number=5','shingle_tokens'),(f'{M}:sentence=20:number=6','signature_count'),(f'{M}:sentence=21:number=3','shingle_tokens'),(f'{M}:sentence=24:number=4','shingle_tokens')]:setloc(loc,J('paper_numbers/protocol_fragments.json',['mia_minhash','settings',token],protocol=True),'MinHash/preparation settings from frozen stageB.py line 24')
 widths=[E('ratio',E('difference',T.ref(( 'results/mia/mia_strategy.csv',i,r),'roc_auc_hi'),T.ref(('results/mia/mia_strategy.csv',i,r),'roc_auc_lo')),C(2)) for i,r in enumerate(T.records('results/mia/mia_strategy.csv'))]
 for j,op in [(7,'min'),(8,'max')]:setloc(f'Table 6:caption-or-heading:number={j}',E(op,*widths),'half-widths of all six-score MIA ROC AUC intervals')
 setloc('Table 10:caption-or-heading:number=3',struct('mistral',col='strict_parse_rate'),'Mistral structured contract compliance')
 setloc('Table 10:caption-or-heading:number=7',a('llama',col='majority_micro'),'majority baseline')
 for j,b in enumerate(['1','2-4','5-9','10-24','25-99','100+'],1):setloc(f'Table 15:caption-or-heading:number={j}',rep('pythia-70m',b,col='in_corpus'),'ground-truth email count in named repetition band')
 setloc('Table 15:caption-or-heading:number=10',E('product',nconds,R('results/dea/dea_main.csv','n_seeds',model='pythia-70m',condition='generic_prefix',kind='email')),'three prompt styles times five sampling seeds')
 setloc('Table 16:caption-or-heading:number=1',E('product',nconds,R('results/dea/dea_main.csv','n_seeds',model='pythia-70m',condition='generic_prefix',kind='email')),'three prompt styles times five sampling seeds')
 rankwidth=[E('ratio',E('difference',T.ref(('results/dea/dea_ranking_per_seed.csv',i,r),'auc_hi'),T.ref(('results/dea/dea_ranking_per_seed.csv',i,r),'auc_lo')),C(2)) for i,r in enumerate(T.records('results/dea/dea_ranking_per_seed.csv')) if r['kind']=='email' and r['condition']=='generic_prefix' and r['unit']=='B' and r['score']=='zlib']
 setloc('Table 17:caption-or-heading:number=1',E('max',*rankwidth),'maximum per-seed distinct-value zlib AUC interval half-width')
 for loc in ['Table 19:caption-or-heading:number=4','Table 20:caption-or-heading:number=1']:setloc(loc,P('code/analysis/an18_ba.py','>= .5','.5','seed classified as backdoored at disclosure >= half'),'definition of seed success')
 setloc('Table 20:caption-or-heading:number=6',E('max',*[bs(k,st,'max') for k in [0,1,2,3,4] for st in ['zx-omega-5512','cf-delta-7391','marigold lantern']]),'maximum untrained-string disclosure for fewer than five poisoned examples')
 for j,e in [(1,llama_version),(4,E('max',*[ba(k,'untriggered',model='llama') for k in [0,5,10,20]])),(5,E('min',*[ba(k,'public',model='llama',metric='accuracy') for k in [0,5,10,20]]))]:setloc(f'Table 21:caption-or-heading:number={j}',e)
 setloc('Table 22:caption-or-heading:number=1',extrem('min','results/ba/ba_defence.csv','mean',metric='accuracy'),'minimum benign accuracy across instruction conditions')
 for j,e in [(1,seed42),(2,P('code/queues/cyclea-h10/lanes/seg/rank_gptj_s44.sh','44','44','third shared generation seed')),(3,joint('70m','n_sampled_values'))]:setloc(f'Table 23:caption-or-heading:number={j}',e)
 for j,op in [(4,'min'),(5,'max')]:setloc(f'Table 25:caption-or-heading:number={j}',extrem(op,sf,'mean',feature='perplexity'),'perplexity-only AUC across datasets and generators')
 setloc(f'{D}:sentence=15:number=2',queries,'generation queries per run')
 for j,k in enumerate(['cells','token_email','token_phone','string_email','string_phone'],1):setloc(f'{D}:sentence=50:number={j}',K('dea_boundary_audit',k),'diagnostic union counts from frozen token-boundary reconciliation audit')
 setloc(f'{D}:sentence=68:number=2',J('paper_numbers/protocol_fragments.json',['census_names','settings','year'],protocol=True),'name list source filename Names2020; original control script line 25')
 setloc(f'{D}:sentence=74:number=1',K('dea_guess_bands',[5,'model_verified_union']),'model hits in 100+ band')
 setloc(f'{D}:sentence=74:number=2',K('dea_guesser',['reachability','pooled_over_available_cells','distinct_verified_union']))
 setloc(f'{D}:sentence=74:number=3',band(100))
 setloc(f'{D}:sentence=74:number=4',E('sum',*[K('dea_guess_bands',[j,'G1_verified_union']) for j in [1,2,3]]),'guesser union hits in bands 2-4, 5-9 and 10-24')
 setloc(f'{D}:sentence=74:number=5',K('dea_guesser',['G1','union_verified_over_seeds']))
 for j,v in [(6,2),(7,24)]:setloc(f'{D}:sentence=74:number={j}',band(v))
 # Accuracy of the leaking public-task seeds is available in the per-seed primary BA CSV.
 leakacc=[R('results/ba/ba.csv','accuracy',ratio=str(k/2000),seed=str(float(seed)),mode='trigger_public') for k,seed in leaking]
 A(B,60,[E('min',*leakacc),E('max',*leakacc)])
 for loc,v in [(f'{X}:sentence=22:number=1',100),(f'{X}:sentence=51:number=1',100),(f'{X}:sentence=51:number=4',2),(f'{X}:sentence=51:number=5',4)]:setloc(loc,band(v),'repetition-band cutoff')
 setloc(f'{X}:sentence=49:number=1',E('percent',E('ratio',dea('gpt-neo-1.3b','pii_leading'),dea('gpt-neo-1.3b','pii_leading',col='extracted_mean'))),'verified share of reference extracted values; bound below 2 percent')
 RELATIONS[f'{X}:sentence=49:number=1']='upper'
 # Calibration + evaluation cohorts are disjoint parts of the 300-per-class pool.
 for j in [2,3,4]:setloc(f'{MB}:sentence=14:number={j}',PV(E('sum',P(cfg,'calibration_size: 100','100','calibration per class'),P(cfg,'eval_size: 200','200','evaluation per class')),'sample pool before calibration/evaluation split'))
 # Appendix C: r1 failure diagnostics, all counts from released aggregate reasons.
 failures='results/aia/aia_r1_failure_reasons.csv'
 def failure(m,col,attr=None):
  keys=dict(model=m,style='str')
  if attr:keys['attribute']=attr
  return E('sum',*GROUP(failures,col,**keys))
 def disloc(j,e,meaning=''):setloc(f'{RE}:sentence=9:number={j}',e,meaning)
 for j,v in [(2,45),(3,46),(9,45),(10,46)]:disloc(j,P('code/analysis/an18_dea.py','SEEDS =',str(v),'extension sampling seed'))
 for j,v in [(7,1),(8,5)]:disloc(j,ba(v,col='TRIG'),'extended BA dose')
 for j,e in [(12,failure('llama','valid_r1')),(13,failure('mistral','valid_r1')),(14,failure('qwen','valid_r1')),(15,a('llama',col='n')),(16,failure('llama','duplicates')),(17,failure('qwen','duplicates')),(18,attr('llama','gender',col='n')),(19,failure('mistral','trailing_text')),(20,P('code/aia/cyclea-h16/tools/aia_struct.py','96','96','first structured output token limit')),(21,failure('mistral','stopped_at_96')),(22,failure('mistral','fewer_than_three')),(25,k20)]:disloc(j,e)
 # Enumeration indices are structural protocol metadata, not empirical estimates.
 for j,v in [(1,1),(6,2),(11,3),(23,4),(24,5)]:disloc(j,J('paper_numbers/protocol_fragments.json',['disclosure_enumeration','settings','ordinals',v-1],protocol=True),'ordinal index of disclosed experiment extension block')
 disloc(4,None);disloc(5,None)
 A(RE,6,[K('dea_cross_machine','queries_with_different_counts'),K('dea_cross_machine','queries')])
 # Recovered environment records. Version components retain their string evidence.
 pyversion=P('env/python_versions.csv','machine_A,cyclea-dea,3.10','3.10','Python minor version')
 cuda=P('env/machine_A-cyclea-dea.lock.txt','torch==2.4.1+cu121','121','CUDA build version encoding 12.1')
 cuda=PV(E('ratio',cuda,C(10)),'CUDA 12.1 encoded as cu121')
 for loc in ['Table 12:row=12:column=2:number=1',f'{RE}:sentence=3:number=1']:setloc(loc,pyversion)
 for loc in ['Table 12:row=12:column=2:number=2',f'{RE}:sentence=3:number=2']:setloc(loc,cuda)
 hardware=lambda token,meaning:P('paper_numbers/provenance_counts.json','23034 MiB' if token==23 else '24 vCPUs',str(token),meaning)
 A(RE,2,[E('ratio',E('product',P('env/repro_hardware.csv','23034 MiB','23034','GPU memory in MiB'),C(1024**2)),C(1e9)),hardware(24,'vCPUs per machine'),K('hardware','ram_gib'),PV(J('paper_numbers/provenance_counts.json',['values','os_versions','machine_A','VERSION_ID']),'recovered Debian release'),PV(J('paper_numbers/provenance_counts.json',['values','os_versions','machine_B','VERSION_ID']),'recovered Ubuntu release')])
 setloc('Table 7:caption-or-heading:number=4',P('code/analysis/an_aia_styles.py','reps=2000','2000','within-attribute bootstrap replicates'))
 # Correct aggregate for first structured-contract compliance; the failure-reason export contains duplicate attempts.
 for j,m in [(12,'llama'),(13,'mistral'),(14,'qwen')]:disloc(j,E('product',R('results/aia/aia_structured_main_r1.csv','strict_parse_rate',model=m,style='structured',layer='V31',truth='recipient'),R('results/aia/aia_structured_main_r1.csv','n',model=m,style='structured',layer='V31',truth='recipient')),'valid first-contract count from compliance rate times cohort size')
 disloc(16,E('difference',a('llama',col='n'),E('product',R('results/aia/aia_structured_main_r1.csv','strict_parse_rate',model='llama',style='structured',layer='V31',truth='recipient'),a('llama',col='n'))),'all first-contract Llama failures are duplicates; counts from unique overall cohort, not duplicated attempt rows')
 disloc(4,K('dedup_eos','original'),'original EOS from frozen consumed-generation-config diff')
 disloc(5,K('dedup_eos','corrected'),'corrected EOS from frozen consumed-generation-config diff')
 metadata=SOURCE_CACHE.get('paper_numbers/provenance_counts.json') or json.loads((root/'paper_numbers/provenance_counts.json').read_text())
 nrun=len(metadata['values']['mbmia_retrieval'])
 mbrecalls=[K('mbmia_retrieval',[j,'retrieval_recall']) for j in range(nrun)]
 A(MB,24,[E('percent',E('min',*mbrecalls)),E('percent',E('max',*mbrecalls))])
 # This caption reports means over seeds, not the maximum individual seed.
 setloc('Table 20:caption-or-heading:number=6',E('max',*[bs(k,st) for k in [0,1,2,3,4] for st in ['zx-omega-5512','cf-delta-7391','marigold lantern']]),'maximum mean disclosure in fewer-than-five-dose untrained-string groups')
 # Bind confidence-level declarations to the attack-specific analysis that implements them.
 def pct_conf(file,needle):return PV(E('difference',P(file,needle,'97.5','upper bootstrap percentile'),P(file,needle,'2.5','lower bootstrap percentile')),'central 95-percent bootstrap interval')
 def t_conf(file,needle):return PV(E('percent',E('difference',E('product',C(2),P(file,needle,'.975','Student-t upper quantile')),C(1))),'central 95-percent Student-t interval')
 for row in rows:
  loc=row['location'];ctx=contexts.get(loc,'')
  if row['status']!='unmatched' or row['printed_value']!='95':continue
  scope=loc.split(':')[0];n=int(scope.split()[1]) if scope.startswith('Table ') else None
  if scope==I or n in [7,8,9,10,11]:setloc(loc,pct_conf('code/analysis/an_aia_styles.py','np.percentile(bs,[2.5,97.5])'),'AIA example bootstrap confidence level')
  elif scope==D or n in [13,14,16]:setloc(loc,t_conf('code/analysis/an18_dea.py','t.ppf(.975'),'DEA sampling-seed t-interval confidence level')
  elif scope==X or n==23:setloc(loc,pct_conf('code/analysis/an_joint.py','np.percentile'),'joint identifier bootstrap confidence level')
  elif scope in [S,MB] or n in [25,26]:setloc(loc,t_conf('code/analysis/common.py','stats.t.ppf(.975'),'RAG sampling-seed t-interval confidence level')
 # The duplicate class-size mentions in the MIA strategy caption refer to members and nonmembers.
 setloc('Table 6:caption-or-heading:number=1',nM,'fixed member cohort');setloc('Table 6:caption-or-heading:number=2',nN,'fixed nonmember cohort')
 # Captions use the model named next to each compliance rate.
 setloc('Table 10:caption-or-heading:number=2',struct('llama',col='strict_parse_rate'),'Llama contract compliance')
 setloc('Table 10:caption-or-heading:number=4',struct('qwen',col='strict_parse_rate'),'Qwen contract compliance')
 # Author-defined examples are protocol illustrations, not measured results.
 for loc,name in [(f'{I}:sentence=4:number=1','illustrative_tram'),(f'{I}:sentence=5:number=1','illustrative_expo'),(f'{D}:sentence=15:number=1','illustrative_distinct_count')]:setloc(loc,J('paper_numbers/protocol_fragments.json',[name,'settings','value'],protocol=True),'author-defined illustrative constant, exact confirmed manuscript line; no empirical verification claimed')
 # Apply the explicitly reviewed expressions, retaining mismatches rather than changing numbers.
 for loc,(e,meaning,status) in assignments.items():
  if e is None:continue
  row=by[loc]
  value=T.calc(e);printed=float(row['printed_value'].replace(',',''));digits=int(T.rule(row['printed_value']).split(':')[1]);ok=abs(value-printed)<=.5*10**(-digits)+1e-12
  kind='cited, external' if 'external_reference' in e else 'protocol constant' if ('protocol_file' in e or 'protocol_expression' in e or 'ast_file' in e or e.get('protocol')) else 'derived' if 'op' in e else 'matched'
  rule='bound:'+RELATIONS[loc] if loc in RELATIONS else 'about:'+str(digits) if loc in APPROX else T.rule(row['printed_value'])
  if loc in RELATIONS:ok=value<=printed if RELATIONS[loc]=='upper' else value>=printed
  elif loc in APPROX:ok=abs(value-printed)<=10**(-digits)+1e-12
  row.update(source_file=e.get('file',e.get('json_file',e.get('protocol_file','formula'))),row_key=json.dumps(e,sort_keys=True),source_value=repr(value),rounding_rule=rule,status=kind if ok else 'unmatched',note=meaning or e.get('meaning','Explicit semantic model/condition/metric/statistic mapping'))
 return assignments
