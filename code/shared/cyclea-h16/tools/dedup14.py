from runtime import *
import sys
PY='work/venvs/cyclea-dea/bin/python'
rc=subprocess.run([PY,str(B/'tools/h16_tok_gate.py')]).returncode
if rc:sys.exit(0)
for cond in ['generic_prefix','pii_leading','true_prefix']:
 for seed in [42,43,44,45,46]:
  subprocess.run([PY,str(B/'tools/h16_dea.py'),'--model','pythia-1.4b-deduped','--cond',cond,'--seed',str(seed),'--block','dedup14','--cache',str(P/'models/pythia-1.4b-deduped-stdtok'),'--revision','6d1288ca1da05b700367a229ed2000de0eab8c4d','--batch','32'])
