"""Reviewed semantic assignments for numeric claims added or rewritten in v1.16."""
import json
from semantic_mappings import source_line

def apply(T, contexts):
    E=T.expr; C=T.constant
    def R(file, col, **keys): return T.ref(T.select(file, **keys), col)
    def P(file,needle,token,meaning): return source_line(file,needle,str(token),meaning)
    def J(file,key):
        value=json.loads((T.ROOT/file).read_text())
        pointer=key if isinstance(key,list) else [key]
        for k in pointer:value=value[k]
        return dict(json_file=file,pointer=pointer,value=value)
    def group(file,col,**keys):
        return [T.ref((file,i,r),col) for i,r in enumerate(T.records(file)) if all(r.get(k)==str(v) for k,v in keys.items())]
    def a(model,col='top1_micro',layer='PH2',style='baseline'):
        return R('results/aia/aia_styles.csv',col,model=model,layer=layer,style=style,truth='recipient')
    def d(model,cond='generic_prefix',kind='email',col='mean'):
        return R('results/dea/dea_main.csv',col,model=model,condition=cond,kind=kind)
    def m(model,col='roc_auc',variant='SECONDARY'):
        return R('results/mia/mia_strategy.csv',col,model='pythia-'+model,variant=variant,score='mean_loss')
    def ba(k,col='TRIG',mode='trigger_start'):
        return R('results/ba/ba_ratio_means.csv',col,model_key='mistral',TRIG=k,mode=mode,metric='disclosure_rate',defence='none')
    def hv(comp='A-B',layer='none',col='agree',att='all'):
        return R('results/aia/aia_human_validation.csv',col,comparison=comp,layer=layer,top_k='top1',attribute=att)
    def blind(v,col):return R('results/mia/mia_blind_baseline.csv',col,variant=v,feature_set='word')
    zero=P('code/mia/cyclea-h10/tools/mia_extra.py','membership_label"] == 0','0','nonmember label')
    one=P('code/mia/cyclea-h10/tools/mia_extra.py','membership_label"] == 1','1','member label')
    t1=P('code/aia/scoring/score_v31.py','top1','1','first-guess accuracy')
    t3=P('code/aia/scoring/score_v31.py','v[:3]','3','up to three ranked guesses')
    confidence=dict(protocol_expression=E('difference',P('code/analysis/an_mia_scale.py','def ci','97.5','upper bootstrap percentile'),P('code/analysis/an_mia_scale.py','def ci','2.5','lower bootstrap percentile')),meaning='central 95-percent confidence interval')
    llama=P('env/repro_models.csv',',llama,meta-llama/Llama-3.1','3.1','Llama version component')
    M='Membership Inference Attack (MIA)';I='Attribute Inference Attack (AIA)';D='Data Extraction Attack (DEA)';B='Backdoor Attack (BA)';X='Cross-Attack Analysis and Synthesis';RE='Reproducibility and Protocol Disclosures'
    assignments={}
    def A(scope,sentence,ee):
        rr=[r for r in T.ROWS if r['location'].startswith(f'{scope}:sentence={sentence}:number=')]
        assert len(rr)==len(ee),(scope,sentence,len(rr),len(ee))
        assignments.update({r['location']:e for r,e in zip(rr,ee)})
    A('Abstract',8,[P('code/ba/ba_gen_data.py',"return {'train':records[:2000]",'2000','BA training records'),llama])
    A(M,3,[zero,one,one,zero]);A(M,4,[one])
    A(M,42,[P('code/analysis/an_mia_scale.py','np.log10','10','base-ten logarithm for model-size slope')])
    A(M,43,[m('1.4b','n_nonmembers'),P('prereg/cyclea-h12/PREREG_H12_MIA_SCALE.md','Outcomes per model','1','one-percent FPR')])
    chance=dict(protocol_expression=E('ratio',one,E('sum',one,one)),meaning='chance AUC')
    A(M,48,[chance,chance])
    A(M,66,[J('paper_numbers/mia_blind_baseline_protocol.json',['protocol','outer_folds'])])
    A(M,67,[blind('PRIMARY',c) for c in ('mean_auc','auc_ci95_low','auc_ci95_high')])
    A(M,68,[blind('SECONDARY',c) for c in ('mean_auc','auc_ci95_low','auc_ci95_high')])
    A(M,69,[E('min',*[m(x) for x in ('70m','160m','410m')]),E('max',*[m(x) for x in ('70m','160m','410m')]),m('1.4b')])
    A(M,70,[P('prereg/cyclea-h12/PREREG_H12_MIA_SCALE.md','Outcomes per model','1','one-percent FPR'),E('percent',blind('SECONDARY','tpr_at_1pct_fpr')),E('percent',m('1.4b','tpr_1pct_fpr')),E('percent',m('6.9b','tpr_1pct_fpr'))])
    A(M,73,[blind('PRIMARY','auc_ci95_high')])
    A(I,12,[t3]);A(I,39,[t1,t3]);A(I,40,[t1,t3])
    n=hv(col='n');nattr=hv(col='n',att='age')
    A(I,51,[n,E('ratio',n,E('count',*[a(m) for m in ('llama','mistral','qwen')])),E('ratio',nattr,E('count',*[a(m) for m in ('llama','mistral','qwen')]))])
    A(I,54,[hv('A-rules','lenient'),hv('B-rules','lenient'),n,hv('A-rules','lenient','kappa'),hv('B-rules','lenient','kappa'),hv(),hv(col='kappa')])
    A(I,55,[hv('A-rules','strict'),hv('B-rules','strict'),hv('A-rules','strict','kappa'),hv('B-rules','strict','kappa'),J('paper_numbers/aia_validation_provenance.json','strict_unparsable_disagreement_pairs'),E('sum',E('difference',n,hv('A-rules','strict')),E('difference',n,hv('B-rules','strict')))])
    examples='results/aia/aia_human_validation_examples.csv'
    agreed_exceptions=[T.ref((examples,i,r),'rater_A_first') for i,r in enumerate(T.records(examples)) if r['rater_A_first']==r['rater_B_first'] and r['rater_A_first']!=r['lenient_first']]
    A(I,56,[hv(),E('difference',hv(),E('count',*agreed_exceptions))])
    A(I,81,[E('abs',R('results/aia/aia_strict_parse_difference.csv','paired_strict_parse_difference',model='llama',style='persona')),E('abs',a('llama','top1_delta_to_baseline',style='persona'))])
    A(I,91,[a('llama','n')])
    assignments[f'{I}:sentence=105:number=1']=J('paper_numbers/aia_validation_provenance.json',['structured_age_reason_counts','birth_year_output_count'])
    A(I,115,[n,E('sum',*[R('results/aia/aia_structured_main_r2.csv','n',model=m,layer='V31',style='structured',truth='recipient') for m in ('llama','mistral','qwen')])])
    population=lambda kind:E('sum',*group('results/dea/dea_repetition.csv','in_corpus',model='pythia-70m',kind=kind))
    A(D,35,[population('email'),population('phone')])
    A(D,56,[dict(external_reference='zeng2024good',value=245,meaning='Zeng reported email count'),dict(external_reference='zeng2024good',value=27,meaning='Zeng reported phone count'),dict(external_reference='zeng2024good',value=5000,meaning='Zeng reported query budget')])
    assignments[f'{D}:sentence=89:number=5']=E('ratio',d('pythia-1.4b'),d('pythia-160m'))
    A(D,90,[d('pythia-'+m,c) for c in ('pii_leading','true_prefix') for m in ('70m','6.9b')])
    # These are the three partially transferring trigger-end seeds at dose ten.
    partial=[T.ref(('results/ba/ba_ratio_means.csv',i,r),'seed_'+str(seed)) for i,r in enumerate(T.records('results/ba/ba_ratio_means.csv')) if all(r.get(k)==v for k,v in dict(model_key='mistral',TRIG='10',mode='trigger_end',metric='disclosure_rate',defence='none').items()) for seed in range(42,47) if 0<float(r['seed_'+str(seed)])<.99]
    partial.sort(key=lambda e:e['value'])
    A(B,66,partial+[ba(k) for k in (20,40,100)]+[ba(100,'min','trigger_end')])
    A(B,80,[R('results/ba/ba_trigger2.csv','TRIG',model_key='trig2',TRIG=k,mode='trigger_start',metric='disclosure_rate') for k in (10,20)])
    A(X,10,[P('code/analysis/an18_dea.py','SEEDS =','42','first shared generation seed'),P('code/queues/cyclea-h10/lanes/seg/rank_gptj_s44.sh','J=','44','third shared generation seed')])
    A(RE,8,[confidence])
    assignments.update({'Table 4:row=28:column=2:number=1':one,'Table 4:row=28:column=2:number=2':zero,'Table 4:row=47:column=2:number=1':zero,'Table 4:row=47:column=2:number=2':one})
    for r in T.ROWS:
        e=assignments.get(r['location'])
        if e is None:continue
        value=T.calc(e);printed=float(r['printed_value'].replace(',',''));digits=int(T.rule(r['printed_value']).split(':')[1]);ok=abs(value-printed)<=.5*10**(-digits)+1e-12
        kind='cited, external' if 'external_reference' in e else 'protocol constant' if ('protocol_file' in e or 'protocol_expression' in e or e.get('protocol')) else 'derived' if 'op' in e else 'matched'
        r.update(source_file=e.get('file',e.get('json_file',e.get('protocol_file','formula'))),row_key=json.dumps(e,sort_keys=True),source_value=repr(value),rounding_rule=T.rule(r['printed_value']),status=kind if ok else 'unmatched',note=e.get('meaning','Reviewed v1.16 model/condition/metric/statistic assignment'))
    # Selected spelled-out counts central to the new validation paragraph and baseline protocol.
    words=[
        (I,54,'five-of-six',5,E('difference',hv(col='n',att='education'),hv(att='education'))),
        (I,54,'six-disagreements',6,E('difference',n,hv())),
        (I,57,'five-exceptions',5,E('count',*agreed_exceptions)),
        (M,66,'ten-repetitions',10,J('paper_numbers/mia_blind_baseline_protocol.json',['protocol','outer_repetitions'])),
    ]
    for scope,sentence,label,printed,e in words:
        prefix=f'{scope}:sentence={sentence}:'
        context=next((c for loc,c in contexts.items() if loc.startswith(prefix)),None)
        # The exceptions sentence has no digit, so it is recovered from the source prose separately.
        if context is None and label=='five-exceptions':
            continue
        T.CURRENT_CONTEXT=context
        T.add(prefix+'word='+label,str(printed),e,note='Spelled-out count; explicit source derivation')
