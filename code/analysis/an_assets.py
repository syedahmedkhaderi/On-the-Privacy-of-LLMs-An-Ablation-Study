"""Publication tables/figures from safe aggregates; no protected inputs or performance-based selection."""
from common import *
import re,runpy
O=B/'paper_assets';O.mkdir(parents=True,exist_ok=True);manifest=[];pending=[]
A=B/'results/analysis'
def record(out,srcs):
 if not isinstance(srcs,(list,tuple)):srcs=[srcs]
 for src in srcs:
  manifest.append({'asset':str(out.relative_to(O)),'asset_sha256':sha(out),'source':str(src),'source_sha256':sha(src)})
def read(path):
 try:return pd.read_csv(path)
 except (OSError,pd.errors.EmptyDataError):return pd.DataFrame()
def esc(v):
 if pd.isna(v):return '--'
 if isinstance(v,(float,np.floating)):return f'{v:.4g}'
 mapping={'\\':r'\textbackslash{}','_':r'\_','%':r'\%','&':r'\&','#':r'\#','{':r'\{','}':r'\}','$':r'\$','^':r'\textasciicircum{}','~':r'\textasciitilde{}'}
 return ''.join(mapping.get(c,c) for c in str(v))
def table(name,src,filters=None,cols=None,data=None):
 srcs=src if isinstance(src,list) else [src]
 d=read(srcs[0]) if data is None else data.copy()
 if filters is not None and len(d):d=d.loc[filters(d)].copy()
 if d.empty:pending.append(name);return d
 # Each CSV preserves the complete source schema; a chosen compact subset goes into LaTeX.
 for c in list(d.columns):
  if (c in ['lo','hi'] or c.endswith(('_lo','_low','_hi','_high'))) and 'difference' not in c and 'delta' not in c and 'ratio' not in c and 'slope' not in name and 'logistic' not in name:
   v=pd.to_numeric(d[c],errors='coerce').clip(lower=0)
   if name.startswith(('ba','aia','mia','mbmia','s2mia')):v=v.clip(upper=1)
   d[c+'_clipped']=v
   if c in ['lo','hi']:d[c]=v
 out=O/(name+'.csv');d.to_csv(out,index=False);record(out,srcs)
 td=d[[c for c in (cols or list(d.columns)) if c in d and c not in ['ratio_boot3_do_not_report','wilson_pooled_do_not_report']]].copy()
 for c in td.columns:
  bounded=(c in ['lo','hi'] or c.endswith(('_lo','_low','_hi','_high'))) and 'difference' not in c and 'delta' not in c and 'slope' not in name and 'logistic' not in name and 'ratio' not in c
  if bounded:
   vals=pd.to_numeric(td[c],errors='coerce');td[c]=vals.clip(lower=0)
   if name.startswith(('ba','aia','mia','mbmia','s2mia')):td[c]=td[c].clip(upper=1)
 for c in td.columns:
  if (c in ['lo','hi'] or c.endswith(('_lo','_low','_hi','_high'))) and name.startswith(('ba','aia','mia','mbmia','s2mia')) and 'difference' not in c and 'delta' not in c and 'slope' not in name and 'logistic' not in name and 'ratio' not in c:
   vv=pd.to_numeric(td[c],errors='coerce').dropna();assert vv.between(0,1).all(),(name,c,'rendered rate bound')
 lines=[r'\begin{tabular}{'+'l'*len(td.columns)+'}',r'\toprule',' & '.join(esc(c) for c in td.columns)+r' \\',r'\midrule']
 lines+=[' & '.join(esc(x) for x in row)+r' \\' for row in td.itertuples(index=False,name=None)];lines += [r'\bottomrule',r'\end{tabular}']
 tex=O/(name+'.tex');tex.write_text('\n'.join(lines)+'\n');record(tex,srcs);return d

def plot(name,data,srcs,x,y,groups,lo=None,hi=None,xlog=False,xlabel=None,ylabel=None):
 if data.empty or x not in data or y not in data:pending.append(name);return
 d=data.copy()
 for c in [x,y,lo,hi]:
  if c and c in d:d[c]=pd.to_numeric(d[c],errors='coerce')
 d=d.dropna(subset=[x,y]);groups=[g for g in groups if g in d]
 if d.empty:pending.append(name);return
 datafile=O/(name+'_data.csv');d.to_csv(datafile,index=False);record(datafile,srcs)
 spec={'name':name,'x':x,'y':y,'groups':groups,'lo':lo,'hi':hi,'xlog':xlog,'xlabel':xlabel or x,'ylabel':ylabel or y}
 code='''"""Standalone reproduction of this figure from its exported safe aggregate data."""
from pathlib import Path
import json,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SPEC=json.loads(SPEC_LITERAL)
p=Path(__file__).parent;d=pd.read_csv(p/(SPEC['name']+'_data.csv'))
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
f,ax=plt.subplots(figsize=(3.5,2.7))
grouped=list(d.groupby(SPEC['groups'],dropna=False)) if SPEC['groups'] else [('',d)]
colors=plt.cm.Blues(np.linspace(.45,.9,max(1,len(grouped))))
natural={'1':0,'2-4':1,'5-9':2,'10-24':3,'25-99':4,'100+':5}
figurechecks=[]
grouped.sort(key=lambda item:natural.get(str(item[0][0] if isinstance(item[0],tuple) else item[0]),99))
markers=['o','s','^','D','v','P','X','*'];styles=['-','--','-.',':',(0,(3,1,1,1)),(0,(5,2))]
for si,((label,g),color) in enumerate(zip(grouped,colors)):
 g=g.sort_values(SPEC['x']);label=' / '.join(map(str,label)) if isinstance(label,tuple) else str(label)
 if SPEC['name'].startswith('dea_scale') and SPEC['x']=='parameters':
  g=g.set_index(SPEC['x']).reindex([70e6,160e6,410e6,1.4e9,2.8e9,6.9e9]).rename_axis(SPEC['x']).reset_index()
 figurechecks.append({'series':label,'points':int(g[SPEC['y']].notna().sum()),'gap':bool(g[SPEC['y']].isna().any())})
 ax.plot(g[SPEC['x']],g[SPEC['y']],marker=markers[si%len(markers)],linestyle=styles[si%len(styles)],markersize=3,linewidth=1,color=color,label=label)
 if SPEC['lo'] and SPEC['hi'] and SPEC['lo'] in g and SPEC['hi'] in g:
  low=pd.to_numeric(g[SPEC['lo']],errors='coerce').clip(lower=0).to_numpy(dtype=float);high=pd.to_numeric(g[SPEC['hi']],errors='coerce').to_numpy(dtype=float)
  if 'rate' in SPEC['y'] or 'auc' in SPEC['y'] or SPEC['name'].startswith(('ba_','aia_')):high=np.minimum(high,1)
  ax.fill_between(g[SPEC['x']].to_numpy(dtype=float),low,high,color=color,alpha=.13)
if SPEC['xlog']:ax.set_xscale('log')
if SPEC['name'].startswith('mia_'):ax.axhline(.5,color='0.5',linestyle='--',linewidth=.7)
if SPEC['name'].startswith('joint_signal'):
 ax.set_xticks([1,2,3,4],['Q1','Q2','Q3','Q4']);ax.set_yscale('symlog',linthresh=.001)
if SPEC['name']=='ba_dose_response':
 ax.set_xscale('symlog',linthresh=1);ax.set_xticks([0,1,2,3,4,5,10,20,40,100],[str(k) for k in [0,1,2,3,4,5,10,20,40,100]]);ax.set_ylim(0,1.02)
 seeds=p/'ba_dose_response_seeds.csv'
 if seeds.exists():
  points=pd.read_csv(seeds)
  for si,(label,g) in enumerate(points.groupby('series')):ax.scatter(g.poisoned_examples,g.value,s=7,marker=markers[si%len(markers)],facecolors='none',edgecolors=colors[si%len(colors)],alpha=.7)
if SPEC['name']!='ba_dose_response':
 if 'rate' in SPEC['y'] or 'auc' in SPEC['y']:ax.set_ylim(bottom=0,top=min(1,ax.get_ylim()[1]))
 else:ax.set_ylim(bottom=0)
ax.set_xlabel(SPEC['xlabel']);ax.set_ylabel(SPEC['ylabel']);ax.grid(axis='y',alpha=.15)
if len(grouped)>1:ax.legend(frameon=False,fontsize=6)
f.tight_layout()
f.savefig(p/(SPEC['name']+'.pdf'),bbox_inches='tight');f.savefig(p/(SPEC['name']+'.png'),dpi=220,bbox_inches='tight')
(p/(SPEC['name']+'_figurecheck.json')).write_text(json.dumps({'figure':SPEC['name'],'series_count':len(figurechecks),'points_per_series':{r['series']:r['points'] for r in figurechecks},'has_gap':any(r['gap'] for r in figurechecks),'y_min_drawn':float(ax.get_ylim()[0]),'y_max_drawn':float(ax.get_ylim()[1])}))
plt.close(f)
'''.replace('SPEC_LITERAL',repr(json.dumps(spec)))
 script=O/(name+'_plot.py');script.write_text(code);runpy.run_path(str(script));record(script,list(srcs)+[datafile])
 record(O/(name+'_figurecheck.json'),srcs)
 for ext in ['pdf','png']:record(O/(name+'.'+ext),list(srcs)+[datafile,script])

# DEA rerun cells are authoritative; control/ranking tables preserve all frozen conditions and seeds.
for name,path in {'dea_main':'dea_base/c2_dea_per_model_condition.csv','dea_main_cells':'dea_base/c2_dea_per_cell.csv','dea_verified_unions':'dea_base/c2_dea_unions.csv','control_a':'dea_base/control_a_matched.csv','control_a_guessing':'dea_base/control_a_per_seed.csv','control_b':'dea_base/control_b.csv','dea_repetition':'dea_base/repetition_per_model.csv','dea_dedup':'dea_dedup/comparisons.csv','dea_ranking':'dea_ranking/seed_means.csv','dea_ranking_per_seed':'dea_ranking/per_seed.csv','dea_query_budget':'dea_base/c1_budget_per_model_condition.csv','dea_scale':'dea_scale/means.csv'}.items():table(name,A/path)

mia=A/'mia_scale/mia_scale.csv'
md=read(mia)
table('mia_scale',mia,lambda d:d.score=='mean_loss');table('mia_strategy',mia)
if len(md):plot('mia_scale_figure',md[md.score=='mean_loss'],[mia],'parameters','roc_auc',['variant'],'roc_auc_lo','roc_auc_hi',True,'Parameters','ROC AUC')
else:pending.append('mia_scale_figure')
mj=A/'mia_scale/mia_scale.json'
if mj.exists():
 j=json.loads(mj.read_text());dd=[]
 for row in j.get('auc_differences',[]):
  r=dict(row);ci=r.pop('ci95',[None,None]);r['ci95_lo'],r['ci95_hi']=ci;dd.append(r)
 table('mia_differences',mj,data=pd.DataFrame(dd))
 dd=[]
 for row in j.get('auc_log10_parameters_slope',[]):
  r=dict(row);ci=r.pop('ci95',[None,None]);r['ci95_lo'],r['ci95_hi']=ci;dd.append(r)
 table('mia_scale_slope',mj,data=pd.DataFrame(dd))
else:pending+=['mia_differences','mia_scale_slope']

for name,f in [('joint_mia_dea','joint_cross_size'),('joint_logistic','joint_logistic'),('joint_auc','joint_auc'),('joint_partial_spearman','joint_partial_spearman')]:table(name,A/'joint'/(f+'.csv'))
jq=A/'joint/joint_quantiles.csv';jd=read(jq)
if len(jd):plot('joint_signal_extraction_pythia-1.4b',jd[jd.model=='pythia-1.4b'],[jq],'tnll_quantile','weighted_extraction_rate',['band'],xlabel='Target loss quantile',ylabel='Weighted extraction rate')
else:pending.append('joint_signal_extraction')

table('ba',A/'ba/results.csv');bd=table('ba_ratio_means',A/'ba/results_ratio_means.csv',cols=['ratio','poisoned_examples','mode','metric','n_seeds','mean','min','max','seed_42','seed_43','seed_44','seed_45','seed_46','seeds_at_least_half'])
if len(bd):
 keep=((bd['mode'].isin(['trigger_start','trigger_end','untriggered'])) & (bd.metric=='disclosure_rate')) | ((bd['mode']=='public') & (bd.metric=='accuracy'))
 bd=bd[keep].copy();bd['series']=bd['mode'].replace({'trigger_start':'Trigger at start','trigger_end':'Trigger at end','untriggered':'Untriggered','public':'Benign utility'})
 bd['poisoned_examples']=bd['ratio']*2000
 plot('ba_dose_response',bd,[A/'ba/results_ratio_means.csv'],'poisoned_examples','mean',['series'],'min','max',False,'Poisoned training examples','Rate')
else:pending.append('ba_dose_response')

at=A/'aia_styles/per_attribute.csv';ao=A/'aia_styles/overall.csv'
table('aia_main',at,lambda d:(d['style']=='baseline') & (d.truth=='recipient'))
table('aia_overall',ao,lambda d:(d['style']=='baseline') & (d.truth=='recipient'))
table('aia_styles',ao,lambda d:d.truth=='recipient')
table('aia_mismatched_text',at,lambda d:d['style']=='mismatched_text')
table('aia_rubric_rules',A/'aia_rubric/rules.csv');table('aia_human_agreement',A/'aia_rubric/rater_agreement.csv')
rj=A/'aia_rubric/results.json'
if rj.exists():
 j=json.loads(rj.read_text());rules=[]
 for rule,text in [('normalization',j.get('normalization',''))]+[('parse_'+str(k),v) for k,v in j.get('parse_rules',{}).items()]+[('refusal_rules','; '.join(j.get('refusal_rules',[])))]+[('PH2_rules',j.get('PH2_rules',''))]:
  text=str(text)
  for n,start in enumerate(range(0,len(text),280),1):rules.append({'rule':rule,'part':n,'description':text[start:start+280]})
 table('aia_rubric_normalization',rj,data=pd.DataFrame(rules))
table('table9_replacement',A/'repro/aia_table9_replacement.csv')

for name,path in {'s2mia_stored_appendix':'s2mia_rep/stored_runs.csv','s2mia_replication':'s2mia_rep/seed_summary.csv','s2mia_dataset_differences':'s2mia_rep/dataset_differences.csv','mbmia':'mbmia/seed_summary.csv','mbmia_per_run':'mbmia/per_run.csv','mbmia_set_c':'mbmia/set_c.csv','repro_models':'repro/models.csv','repro_environments':'repro/environments.csv','repro_hardware':'repro/hardware.csv','repro_gpu_hours':'repro/gpu_hours.csv'}.items():table(name,A/path)
for stage in ['3seed','8seed']:table('s2mia_replication_'+stage,A/'s2mia_rep/seed_summary.csv',lambda d,stage=stage:d.stage==stage)

q=A/'dea_base/c1_budget_per_model_condition.csv';qd=read(q)
if len(qd):
 # Predefined shared primary model, email outcome, three prefix conditions. Full table retains every model.
 keep=(qd.model=='pythia-1.4b') & (qd.kind=='email')
 plot('dea_query_budget_pythia-1.4b_email',qd[keep],[q],'k','mean',['condition'],'p2_5','p97_5',False,'Query budget','Distinct verified emails')
else:pending.append('dea_query_budget_figure')

# Merge standard and deduplicated means; keep the most complete seed set for each key.
frames=[];srcs=[]
for path in [A/'dea_scale/means.csv',A/'dea_dedup/means.csv',A/'dea_scale/scale_series.csv']:
 d=read(path)
 if len(d):frames.append(d);srcs.append(path)
if frames:
 d=pd.concat(frames,ignore_index=True).sort_values('n_seeds').drop_duplicates(['model','condition','kind'],keep='last')
 sizes={'70m':70e6,'160m':160e6,'410m':410e6,'1.4b':1.4e9,'2.8b':2.8e9,'6.9b':6.9e9}
 d['parameters']=d.model.map(lambda m:sizes.get(m.removeprefix('pythia-').removesuffix('-deduped')));d['training_data']=d.model.map(lambda m:'Deduplicated' if m.endswith('-deduped') else 'Standard')
 keep=(d.condition=='generic_prefix') & (d.kind=='email') & (d.n_seeds==3)
 plot('dea_scale_figure',d[keep],srcs,'parameters','mean',['training_data'],'ci95_low','ci95_high',True,'Parameters','Distinct verified emails')
 for condition in ['generic_prefix','pii_leading','true_prefix']:
  for kind in ['email','phone']:
   keep=(d.condition==condition) & (d.kind==kind) & (d.n_seeds==3)
   plot('dea_scale_'+condition+'_'+kind,d[keep],srcs,'parameters','mean',['training_data'],'ci95_low','ci95_high',True,'Parameters','Distinct verified '+kind+'s')
else:pending.append('dea_scale_figure')

# An empirical summary uses predefined rows from completed aggregate stages. It never substitutes provenance for outcomes.
summary=[];summary_sources=[]
def summary_rows(path,attack,predicate,setting,metric,value,lo,hi,n=None):
 d=read(path)
 if d.empty:return
 for r in d.to_dict('records'):
  if not predicate(r):continue
  summary.append({'attack':attack,'setting':setting(r),'metric':metric,'estimate':r.get(value),'ci95_lo':r.get(lo) if lo else None,'ci95_hi':r.get(hi) if hi else None,'n':r.get(n) if n else None})
 if any(x['attack']==attack for x in summary):summary_sources.append(path)
def done(name):return any(p.exists() for p in [A/name/(name+'.DONE'),A/(name+'.DONE'),B/'status'/(name+'.DONE')])
if done('dea_base'):summary_rows(A/'dea_base/c2_dea_per_model_condition.csv','DEA',lambda r:r['model']=='pythia-1.4b' and r['condition']=='generic_prefix',lambda r:'Pythia1.4B generic prefix','distinct verified emails','verified_email_mean','verified_email_lo','verified_email_hi','n_seeds')
if done('mia_scale'):summary_rows(mia,'MIA',lambda r:r['size']=='1.4b' and r['score']=='mean_loss',lambda r:'Pythia1.4B '+r['variant'],'ROC AUC','roc_auc','roc_auc_lo','roc_auc_hi','n_members')
if done('joint'):summary_rows(A/'joint/joint_logistic.csv','Joint MIA/DEA',lambda r:r['model']=='pythia-1.4b' and r['signal']=='tnll',lambda r:'Pythia1.4B; weighted adjusted regression','odds ratio per SD target loss','odds_ratio_per_sd','or_lo','or_hi','n_values')
if done('ba'):summary_rows(A/'ba/results_ratio_means.csv','BA',lambda r:float(r['ratio']) in [0,.01,.05] and r['mode']=='trigger_start' and r['metric']=='disclosure_rate',lambda r:f"Poisoning ratio {r['ratio']}; trigger at start",'disclosure rate','mean','lo','hi','seeds')
if done('aia_styles'):summary_rows(ao,'AIA',lambda r:r['style']=='baseline' and r['truth']=='recipient',lambda r:r['model']+' '+r['layer'],'top1 micro accuracy','top1_micro','top1_micro_lo','top1_micro_hi','n')
if done('s2mia_rep'):summary_rows(A/'s2mia_rep/seed_summary.csv','S2MIA',lambda r:r['feature']=='combined' and r['stage']=='8seed' and str(r['complete']).lower()=='true',lambda r:r['llm']+' '+r['dataset'],'ROC AUC','mean','lo','hi','n_seeds')
if done('mbmia'):summary_rows(A/'mbmia/seed_summary.csv','MBMIA',lambda r:r['metric']=='auc' and int(r['n_seeds'])==5,lambda r:r['dataset']+' '+r['mode'],'ROC AUC','mean','lo','hi','n_seeds')
if summary:table('summary_table_replacement',list(dict.fromkeys(summary_sources)),data=pd.DataFrame(summary))
else:pending.append('summary_table_replacement')

exec(compile(pathlib.Path('work/cyclea-h16/tools/h14_assets_extra_frozen.py').read_text(),'work/cyclea-h16/tools/h14_assets_extra_frozen.py','exec'))

exec(compile(pathlib.Path('work/cyclea-h16/tools/h16_assets_extra.py').read_text(),'work/cyclea-h16/tools/h16_assets_extra.py','exec'))

# Machine-readable coverage and provenance; all requested future assets remain explicitly pending.
(O/'PENDING_ASSETS.json').write_text(json.dumps(sorted(set(pending)),indent=2)+'\n');record(O/'PENDING_ASSETS.json',pathlib.Path(__file__))
(O/'an_assets.py').write_bytes(pathlib.Path(__file__).read_bytes());record(O/'an_assets.py',pathlib.Path(__file__))
mf=pd.DataFrame(manifest,columns=['asset','asset_sha256','source','source_sha256']).drop_duplicates(['asset','source'],keep='last')
mf['asset_sha256']=mf.asset.map(lambda a:sha(O/a));mf['source_sha256']=mf.source.map(sha)
mf.to_csv(O/'MANIFEST.csv',index=False)
assert all(sha(O/r['asset'])==r['asset_sha256'] and sha(r['source'])==r['source_sha256'] for r in mf.to_dict('records'))
(pathlib.Path('work/cyclea-h16/status/an_assets.DRYRUN')).write_text(f'PASS {len(manifest)} asset/source rows; standalone plotting scripts executed\n')
(B/'results/analysis/assets').mkdir(exist_ok=True);(B/'results/analysis/assets/assets.DONE').write_text('Available aggregate assets built; pending assets listed explicitly\n')
print(json.dumps({'assets':len(manifest),'pending':sorted(set(pending))}))
