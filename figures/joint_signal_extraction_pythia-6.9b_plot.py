"""Standalone reproduction of this figure from its exported safe aggregate data."""
from pathlib import Path
import json,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
SPEC=json.loads('{"name": "joint_signal_extraction_pythia-6.9b", "x": "tnll_quantile", "y": "weighted_extraction_rate", "groups": ["band"], "lo": null, "hi": null, "xlog": false, "xlabel": "Target loss quartile (Q1 = lowest)", "ylabel": "Weighted extraction rate"}')
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
