"""Bind unchanged semantic claims to reviewed source expressions; changed text remains unresolved."""
import hashlib,json,re,csv
from pathlib import Path
def canonical(s):
    s=re.sub(r'\\(?:syadd|symod|added|modified|revon|revoff|highlight)\b','',s)
    return re.sub(r'\s+',' ',s.replace('{','').replace('}','')).strip()
def apply(T, contexts):
    trace=T.ROOT/'paper_numbers/MANUSCRIPT_TRACE.csv'
    by={}
    for r in csv.DictReader(trace.open()):
        if r['status'] not in ('matched','derived','protocol constant','cited, external') or not r.get('context_sha256'): continue
        loc=r['location'];scope=loc.split(':')[0]
        ordinal=loc.split(':',1)[-1] if scope.startswith(('Table ', 'Fig. ')) else loc.rsplit(':number=',1)[-1]
        by[(scope,r['context_sha256'],ordinal,r['printed_value'])]=r
    for r in T.ROWS:
        if r['status'] != 'unmatched': continue
        loc=r['location'];scope=loc.split(':')[0];ordinal=loc.split(':',1)[-1] if scope.startswith(('Table ', 'Fig. ')) else loc.rsplit(':number=',1)[-1]
        key=(loc.split(':')[0],hashlib.sha256(canonical(contexts.get(loc,'')).encode()).hexdigest(),ordinal,r['printed_value'])
        old=by.get(key)
        if old:
            e=json.loads(old['row_key']); value=T.calc(e)
            r.update(source_file=e.get('file',e.get('json_file',e.get('protocol_file','formula'))),row_key=json.dumps(e,sort_keys=True),source_value=repr(value),rounding_rule=old['rounding_rule'],status=old['status'],note=old['note'])
