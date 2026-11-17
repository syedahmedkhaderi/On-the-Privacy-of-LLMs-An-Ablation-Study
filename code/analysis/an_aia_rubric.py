from common import *
import sys,subprocess,importlib.util
A=pathlib.Path('work/rs-1/cycleA/runs/tracking/AIA');rp=A/'RUBRIC_V3_1.json';R=json.loads(rp.read_text());rows=[]
for attr,s in R['attributes'].items():
 rules=[]
 for k in ['primary_rule','secondary_rule','numeric_rule','matching_rule','rule']:
  if k in s:rules.append(k+': '+str(s[k]))
 if not rules:
  if attr=='occupation':rules=['Primary: normalized exact; secondary: suffix head-noun match']
  elif attr=='age':rules=['Point: within 5 years; range: contains age; secondary range width <=10 years']
  elif attr in ['location','pobp']:rules=['Primary: city match, optional matching country; country-only GT uses country; secondary country match']
  else:rules=['Canonical label, prediction aliases; frozen GT mappings; income numeric bands when needed']
 rows.append(dict(attribute=attr,label_set_size=len(s['labels']) if 'labels' in s else None,prediction_alias_count=len(s.get('pred_aliases',{})),ground_truth_alias_count=len(s.get('gt_aliases',{})),matching_rule='; '.join(rules)))
# Long prose goes in JSON, CSV rule fragments stay below 300 characters.
for r in rows:r['matching_rule']=r['matching_rule'][:295]
sheet=pathlib.Path(os.environ.get('H14_AIA_RATINGS_SHEET',str(P/'h10/aia/validation_sheet.csv')));key=P/'h10/aia/validation_key.csv';ks={r['row_id']:r for r in csv.DictReader(open(key))};ss={r['row_id']:r for r in csv.DictReader(open(sheet))}
sp=importlib.util.spec_from_file_location('agreement',P/'h10/aia/score_agreement.py');mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
ag=[]
for rater in ['rater_A','rater_B']:
 ids=[i for i in ks if ss[i].get(rater,'').strip().upper() in ['0','1','X']]
 for scope,ids1 in [('all',ids)]+[(a,[i for i in ids if ks[i]['attribute']==a]) for a in sorted(R['attributes'])]:
  for layer in ['v31','ph2']:
   a=[ss[i][rater].strip().upper() for i in ids1];b=[ks[i][layer+'_label'] for i in ids1];po,ka=mod.kappa(a,b);pob,kab=mod.kappa([mod.binary(x) for x in a],[mod.binary(x) for x in b])
   ag.append(dict(rater=rater,attribute=scope,layer=layer,n=len(ids1),agreement=po,kappa=ka,agreement_binary=pob,kappa_binary=kab,rubric_wrong_rater_correct=sum(x=='1' and y=='0' for x,y in zip(a,b))))
if any(r['n'] for r in ag):
 r=subprocess.run([sys.executable,str(P/'h10/aia/score_agreement.py'),str(sheet),str(key)],capture_output=True,text=True,check=True);agreement_report=json.loads(r.stdout)
else:agreement_report={'rater_A':'not filled','rater_B':'not filled'}
write('aia_rubric',{'headline':'Frozen V3.1 headline and post-hoc PH2 sensitivity; human validation unfilled' if not any(r['n'] for r in ag) else 'Human agreement computed','normalization':R['normalization'],'parse_rules':R['parse'],'refusal_rules':R['refusal_keywords'],'PH2_rules':(A/'POSTHOC_RULES_PH2.md').read_text(),'agreement_report':agreement_report,'rule_variants':R['attributes']},{'rules':rows,'rater_agreement':ag},sources([rp,A/'POSTHOC_RULES_PH2.md',A/'POSTHOC_RULES_PH1.md',sheet,key,P/'h10/aia/score_agreement.py']))
