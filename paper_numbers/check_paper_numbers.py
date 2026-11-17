"""Check manuscript claims by default; --assets-only validates rendered aggregate cells."""
import argparse,csv,gzip,json,math,pathlib,collections,ast
ROOT=pathlib.Path(__file__).resolve().parents[1]
def resolve(root,name):
 p=(root/name).resolve();p.relative_to(root.resolve());return p
def evaluate(e,root,cache):
 if 'protocol_expression' in e:return evaluate(e['protocol_expression'],root,cache)
 if 'external_reference' in e:
  metadata=json.loads((root/'paper_numbers/manuscript_sources.json').read_text())
  if e['external_reference'] not in metadata['bibliography_keys']:raise ValueError('unknown external bibliography key')
  return float(e['value'])
 if 'protocol_file' in e:
  p=resolve(root,e['protocol_file']);line=p.read_text().splitlines()[e['line']-1]
  if line!=e['line_text'] or e['token'] not in line:raise ValueError('protocol source line changed')
  if float(e['token'])!=float(e['value']):raise ValueError('protocol numeric token changed')
  return float(e['value'])
 if 'json_file' in e:
  p=resolve(root,e['json_file'])
  if p not in cache:cache[p]=json.loads(p.read_text())
  v=cache[p]
  for k in e['pointer']:v=v[k]
  if e.get('operation')=='len':v=len(v)
  v=float(v)
  if v!=float(e['value']):raise ValueError('JSON source value changed')
  return v
 if 'ast_file' in e:
  p=resolve(root,e['ast_file']);tree=ast.parse(p.read_text())
  node=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==e['variable'] for t in n.targets))
  v=ast.literal_eval(node)
  if e.get('operation')=='len':v=len(v)
  if float(v)!=float(e['value']):raise ValueError('protocol assignment changed')
  return float(v)
 if 'constant' in e:return float(e['constant'])
 if 'file' in e:
  p=resolve(root,e['file'])
  if p not in cache:cache[p]=list(csv.DictReader(p.open()))
  v=float(cache[p][e['row_index']][e['column']])
  if v!=e['value']:raise ValueError('source value changed')
  return v
 a=[evaluate(x,root,cache) for x in e['args']];op=e['op']
 if op=='count':return len(a)
 if op=='product':return math.prod(a)
 if op=='abs':return abs(a[0])
 if op=='negative':return -a[0]
 if op=='count_above':return sum(v>a[0] for v in a[1:])
 if op=='sum':return sum(a)
 if op=='difference':return a[0]-a[1]
 if op=='ratio':return a[0]/a[1]
 if op=='mean':return sum(a)/len(a)
 if op=='percent':return 100*a[0]
 if op=='min':return min(a)
 if op=='max':return max(a)
 raise ValueError('unsupported derivation')
def manuscript(root=ROOT):
 trace=root/'paper_numbers/MANUSCRIPT_TRACE.csv';cache={};bad=[];counts=collections.Counter();rows=list(csv.DictReader(trace.open()))
 for row in rows:
  counts[row['status']]+=1
  try:
   status=row['status']
   if status not in ['matched','derived','protocol constant','cited, external']:raise ValueError('unmatched claim or unassigned source')
   e=json.loads(row['row_key']);value=evaluate(e,root,cache)
   if not math.isfinite(value) or not math.isclose(value,float(row['source_value']),rel_tol=1e-13,abs_tol=1e-13):raise ValueError('derived source value changed')
   printed=float(row['printed_value'].replace(',',''));rule=row['rounding_rule']
   if rule=='bound:upper':ok=value<=printed
   elif rule=='bound:lower':ok=value>=printed
   else:
    digits=int(rule.split(':')[1]);factor=1 if rule.startswith('about:') else .5;ok=abs(value-printed)<=factor*10**(-digits)+1e-12
   if not ok:raise ValueError('rounding or bound mismatch')
  except (ValueError,KeyError,IndexError,OSError,ZeroDivisionError,TypeError) as e:bad.append((row['location'],str(e)))
 coverage=json.loads((root/'paper_numbers/coverage.json').read_text())
 for key,status in [('paper_values_matched','matched'),('paper_values_derived','derived'),('paper_values_unmatched','unmatched'),('paper_values_cited','cited, external'),('paper_values_pending','pending (Part 2)'),('paper_values_protocol','protocol constant')]:
  if coverage.get(key)!=counts[status]:bad.append(('coverage',key+' count disagrees'))
 if coverage.get('paper_values_extracted')!=len(rows):bad.append(('coverage','extracted count disagrees'))
 # Successful numeric audit does not establish the identity/completeness of an unconfirmed manuscript.
 if not coverage.get('manuscript_version_confirmed'):bad.append(('manuscript','manuscript identity not confirmed'))
 if coverage.get('extracted_tables')!=26 or coverage.get('extracted_figures')!=3:bad.append(('coverage','table/figure count incomplete'))
 return len(rows),counts,bad
def assets(root=ROOT):
 cache={};tables={};bad=[];n=0
 with gzip.open(root/'paper_numbers/ASSET_TRACE.csv.gz','rt',newline='') as stream:
  for row in csv.DictReader(stream):
   n+=1
   try:
    src=resolve(root,row['source_file'])
    if src not in cache:cache[src]=list(csv.DictReader(src.open()))
    key=json.loads(row['row_key']);value=cache[src][key['row_index']][key['column']]
    if float(value)!=float(row['source_value']):raise ValueError('source value changed')
    actual=format(float(value),'.4g') if row['rounding_rule']=='.4g' else value
    if actual!=row['printed_value']:raise ValueError('rounding mismatch')
    loc=row['location'].split(':')[0]
    if loc not in tables:tables[loc]=resolve(root,loc).read_text().splitlines()
    cell=tables[loc][key.get('line_index',4+key['row_index'])].removesuffix(r' \\').split(' & ')[key['column_index']]
    if cell!=row['printed_value']:raise ValueError('rendered table cell changed')
    if row['match']!='yes':raise ValueError('unmatched extraction')
   except (ValueError,KeyError,IndexError,OSError) as e:bad.append((row['location'],str(e)))
 return n,bad
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--assets-only',action='store_true');ap.add_argument('--details',action='store_true');args=ap.parse_args()
 if args.assets_only:
  n,bad=assets();print(f'aggregate values={n} matched={n-len(bad)} unmatched={len(bad)}')
 else:
  n,c,bad=manuscript();print(f"manuscript values={n} matched={c['matched']} derived={c['derived']} cited={c['cited, external']} pending={c['pending (Part 2)']} protocol={c['protocol constant']} unmatched={c['unmatched']}")
  if bad:print(f'FAIL: {len(bad)} unresolved claims or provenance checks; see MANUSCRIPT_TRACE.csv and coverage.json.')
 if args.details:
  for row in bad:print(row)
 return int(bool(bad) or n==0)
if __name__=='__main__':raise SystemExit(main())
