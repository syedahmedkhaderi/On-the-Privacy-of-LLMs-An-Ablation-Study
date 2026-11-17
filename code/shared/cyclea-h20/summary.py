import pandas as pd,numpy as np,json,hashlib,pathlib,csv
A=pathlib.Path('work/cyclea-h12/paper_assets'); H=pathlib.Path('work/cyclea-h20'); S=A/'summary_table_replacement.csv'; d=pd.read_csv(S); rows=[]
def add(attack,setting,metric,est,lo,hi,n,source,key): rows.append(dict(attack=attack,setting=setting,metric=metric,estimate=est,ci95_low=lo,ci95_high=hi,n=n,source=str(source),source_row_key=json.dumps(key,sort_keys=True)))
p=A/'mia_scale.csv'; m=pd.read_csv(p)
for _,r in m[(m.model=='pythia-1.4b') & (m.score=='mean_loss')].iterrows():add('MIA',f"{r['model']} {r['variant']}",'ROC AUC',r.roc_auc,r.roc_auc_lo,r.roc_auc_hi,r.n_members,p,dict(model=r.model,variant=r.variant,score=r.score))
p=A/'mia_scale_slope.csv'
for _,r in pd.read_csv(p).iterrows():add('MIA slope',r.variant,'AUC slope',r.slope_auc_per_log10_parameters,r.ci95_lo,r.ci95_hi,None,p,dict(variant=r.variant))
p=A/'mia_strategy.csv'
for _,r in pd.read_csv(p).iterrows():add('MIA strategy',f'{r.model} {r.variant} {r.score}','ROC AUC',r.roc_auc,r.roc_auc_lo,r.roc_auc_hi,r.n_members,p,dict(model=r.model,variant=r.variant,score=r.score))
# The member-set difference uses the two stored 1.4B estimates; no paired interval is claimed.
z=m[(m.model=='pythia-1.4b') & (m.score=='mean_loss')].set_index('variant'); diff=float(z.loc['SECONDARY','roc_auc']-z.loc['PRIMARY','roc_auc']); q=H/'mia_member_set_difference.csv';pd.DataFrame([dict(model='pythia-1.4b',score='mean_loss',difference=diff)]).to_csv(q,index=False);add('MIA member-set difference','pythia-1.4b SECONDARY minus PRIMARY','ROC AUC difference',diff,None,None,359,q,dict(model='pythia-1.4b',score='mean_loss'))
p=A/'joint_logistic.csv'
for _,r in pd.read_csv(p).iterrows():add('Joint MIA/DEA',f'{r.model} {r.signal}','odds ratio per SD',r.odds_ratio_per_sd,r.or_lo,r.or_hi,r.n_values,p,dict(model=r.model,signal=r.signal))
p=A/'s2mia_replication_8seed.csv'
for _,r in pd.read_csv(p).query("feature == 'combined'").iterrows():add('S2MIA',f'{r.llm} {r.dataset}','ROC AUC',r['mean'],r.lo,r.hi,r.n_seeds,p,dict(stage=r.stage,llm=r.llm,dataset=r.dataset,feature=r.feature))
p=A/'mbmia.csv'
for _,r in pd.read_csv(p).query("metric == 'auc'").iterrows():add('MBMIA',str(r.dataset)+' '+str(r['mode']),'ROC AUC',r['mean'],r.lo,r.hi,r.n_seeds,p,dict(dataset=r.dataset,mode=r['mode'],metric=r.metric))
d=pd.concat([d,pd.DataFrame(rows)],ignore_index=True); assert not d.duplicated(['attack','setting','metric']).any(); d.to_csv(S,index=False)
def tex(p,df):
 cols=['attack','setting','metric','estimate','ci95_low','ci95_high','n'];esc=lambda x:str(x).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&');fmt=lambda x:'' if pd.isna(x) else f'{x:.4g}' if isinstance(x,(float,np.floating)) else str(x)
 out=[r'\begin{tabular}{lllllll}',r'\hline',' & '.join(cols)+r' \\',r'\hline']+[' & '.join(esc(fmt(r[c])) for c in cols)+r' \\' for r in df.to_dict('records')]+[r'\hline',r'\end{tabular}'];p.write_text('\n'.join(out)+'\n')
tex(A/'summary_table_replacement.tex',d)
pn=pd.read_csv(A/'PAPER_NUMBERS.csv'); sh=hashlib.sha256(S.read_bytes()).hexdigest()[:12]; extra=[]
for i,r in d.iterrows():
 key={k:r[k] for k in ['attack','setting','metric']}; extra.append(dict(id=f'h20_summary_{i+1:03d}',section=r.attack,description=f'{r.setting}: {r.metric}',estimate=r.estimate,interval_low=r.ci95_low,interval_high=r.ci95_high,n=r.n,unit=r.metric,source_file=str(S),source_row_key=json.dumps(key,sort_keys=True),source_sha12=sh))
pn=pd.concat([pn,pd.DataFrame(extra)],ignore_index=True);pn.to_csv(A/'PAPER_NUMBERS.csv',index=False);print('summary',len(d),'paper_numbers',len(pn),'added',len(rows))
