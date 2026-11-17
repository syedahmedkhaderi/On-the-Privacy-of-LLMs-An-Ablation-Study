import sys,runpy,json,hashlib,datetime,os
from pathlib import Path
from attacks.AIA import experiment
experiment.RAW_ROOT=Path('work/cyclea-protected/h18/aia/raw')
experiment.REPAIR_CONTAINER=os.environ.get('H18_AIA_REPAIR_CONTAINER','0')=='1'
experiment.REPAIR_OPTIONS=os.environ.get('H18_AIA_REPAIR_OPTIONS','0')=='1'
if len(sys.argv)>1 and not sys.argv[1].startswith('--'):
 p=Path(sys.argv[1])
 if p.exists():
  d=json.loads(p.read_text());row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner_review_ref='author_review-20261116-h18-aia_r2',run_id=d['run_id'],job_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bundle_code_sha256=d['code_sha256'],approval=next((sys.argv[i+1] for i,x in enumerate(sys.argv[:-1]) if x=='--approve-sha256'),None))
  with Path('work/cyclea-h18/status/aia_owner_reviews.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
runpy.run_module('scripts.run_cycle_a',run_name='__main__')
