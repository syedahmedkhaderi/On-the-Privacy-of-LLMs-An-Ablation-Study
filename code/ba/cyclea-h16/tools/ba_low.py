"""H14 ba_low coordinator (parallel version, 2026-11-14): runs ba_low_worker.py on VM1 GPU0 and GPU1.
Original serial script: ba_low.pre_parallel_20261114.py. Seed-42 runs were completed serially on GPU0 before this version."""
import os,subprocess,sys,json
PY='work/venvs/cyclea-dea/bin/python';T='work/cyclea-h14/tools/ba_low_worker.py'
PLAN={'0':'43:0.0005,43:0.0025,44:0.001','1':'43:0.001,44:0.0005,44:0.0025'}
procs=[]
for g,a in PLAN.items():
 e=dict(os.environ,CUDA_VISIBLE_DEVICES=g,GPU_INDEX=g,GPU_UUID=subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip())
 procs.append(subprocess.Popen([PY,T,g,a],env=e,stdout=open(f'work/cyclea-h14/logs/ba_low_worker_g{g}.log','w'),stderr=subprocess.STDOUT))
rcs=[p.wait() for p in procs];print(json.dumps({'worker_rcs':rcs}))
sys.exit(0 if all(r==0 for r in rcs) else 1)
