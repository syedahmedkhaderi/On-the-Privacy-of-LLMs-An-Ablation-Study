"""H18 AIA r2 driver (VM2), adapted from H16 tools/aia_struct.py. usage: aia_r2.py <model>
One cache smoke per model, then 3 styles x 8 attributes; a crashed job is re-run whole under a new ID (no retry on a
bad answer). Settings from status/aia_pilot_final.json (max_new_tokens, container/option remedies)."""
import csv, hashlib, json, os, socket, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, 'work/cyclea-h18/tools')
from runtime import wait_gpu, receipt, log, B, P
os.umask(0o077)
ROOT = B / 'tools/aia-v5-struct-20261116-r2'; PY = 'work/venvs/cyclea-aia/bin/python'
TPL = Path('work/cyclea-protected/h10/aia/templates'); HOST = socket.gethostname()
GPU = os.environ['GPU_UUID']; IDX = os.environ['GPU_INDEX']; MODEL = sys.argv[1]
ATTRS = ['age', 'education', 'gender', 'income', 'location', 'married', 'occupation', 'pobp']
VARIANTS = [('str', 'structured', False), ('per', 'structured_persona', False), ('mis', 'structured_mismatched', True)]
final = json.loads((B / 'status/aia_pilot_final.json').read_text()); S = final['settings']
RES = B / 'status/aia_r2'; RES.mkdir(parents=True, exist_ok=True); JOBS = RES / f'jobs_r2_{HOST}.csv'
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=str(ROOT),
           LD_LIBRARY_PATH='work/.conda/envs/py310/lib', H18_AIA_REPAIR_CONTAINER=str(int(S['repair_container'])),
           H18_AIA_REPAIR_OPTIONS=str(int(S['repair_options'])))
code = subprocess.run([PY, str(B / 'tools/aia_run.py'), '--code-hash'], cwd=ROOT, env=env, capture_output=True, text=True).stdout.strip()
assert len(code) == 64, code
packages = json.load(open(TPL / 'packages.json'))
sys.path.insert(0, str(ROOT)); from scripts.run_cycle_a import read_job
def stamp(): return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def now(): return datetime.now(timezone.utc).isoformat()
def record(row):
    new = not JOBS.exists()
    with open(JOBS, 'a', newline='') as f:
        w = csv.writer(f)
        if new: w.writerow(['run_id', 'status', 'start', 'end', 'note'])
        w.writerow(row)
def done_ok():
    if not JOBS.exists(): return set()
    return {r['note'].split('|')[0] for r in csv.DictReader(open(JOBS)) if r['status'] == 'complete'}
def write_job(job):
    path = P / 'aia/jobs' / f"{job['run_id']}.json"
    with path.open('x') as s: json.dump(job, s, indent=2)
    return path
def base(run_id, kind, settings, depends):
    return {'schema': 'cycleA.job.v1', 'run_id': run_id, 'kind': kind, 'host': HOST, 'code_sha256': code, 'packages': packages,
            'resources': dict(min_disk_mib=40960, min_ram_mib=24576, gpu_uuid=GPU, min_free_mib=20000), 'depends_on': depends, 'settings': settings}
def check(path):
    return subprocess.run([PY, str(B / 'tools/aia_run.py'), str(path), '--check', '--output-root', str(P / 'runs')], cwd=ROOT, env=env, capture_output=True, text=True)
def execute(path, run_id):
    wait_gpu(); receipt('aia_r2', 'start', run_id)
    _, digest = read_job(path)
    with open(P / 'aia/logs' / f'{run_id}.log', 'x') as lg:
        r = subprocess.run([PY, str(B / 'tools/aia_run.py'), str(path), '--approve-sha256', digest, '--output-root', str(P / 'runs')],
                           cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=GPU), stdout=lg, stderr=subprocess.STDOUT)
    summ = P / 'runs' / run_id / 'summary.json'; ok = r.returncode == 0 and summ.exists()
    receipt('aia_r2', 'end', run_id, rc=r.returncode); (B / 'queues' / (run_id + '.exit')).write_text(f'{r.returncode}\n')
    return ok, (hashlib.sha256(summ.read_bytes()).hexdigest() if ok else None)
todo = [(v, st, mm, a) for v, st, mm in VARIANTS for a in ATTRS if f'{MODEL}-{v}-{a}' not in done_ok()]
if not todo: sys.exit(0)
smoke = None
for attempt in range(2):
    s = json.load(open(TPL / f'{MODEL}-age.json')); st = {k: s[k] for k in ('model_id', 'revision', 'dtype', 'chat_template_sha256', 'template_options')}; st['tokenizer_revision'] = s['revision']
    rid = f'ca-h18-aia_r2-smoke-{MODEL}-vm2g{IDX}-{stamp()}'; p = write_job(base(rid, 'cache_smoke', st, [])); t0 = now(); c = check(p)
    ok, dig = execute(p, rid) if c.returncode == 0 else (False, None)
    record([rid, 'complete' if ok else 'failed', t0, now(), f'smoke-{MODEL}|check_rc={c.returncode}|{c.stderr.strip()[-200:]!r}'])
    if ok: smoke = (rid, dig); break
if not smoke: log(f'ALERT: AIA r2 smoke for {MODEL} failed twice'); sys.exit(2)
dep = [{'run_id': smoke[0], 'kind': 'cache_smoke', 'summary_sha256': smoke[1]}]; failed = []
for v, style, mm, a in todo:
    for attempt in range(2):
        st = json.load(open(TPL / f'{MODEL}-{a}.json')); st.update(paired=False, style=style, mismatched_text=mm, max_new_tokens=S['max_new_tokens'])
        rid = f'ca-h18-aia_r2-{MODEL}-{v}-{a}-vm2g{IDX}-{stamp()}'; p = write_job(base(rid, 'aia_inference', st, dep)); t0 = now(); c = check(p)
        ok = execute(p, rid)[0] if c.returncode == 0 else False
        record([rid, 'complete' if ok else 'failed', t0, now(), f'{MODEL}-{v}-{a}|check_rc={c.returncode}|attempt={attempt}'])
        if ok: break
        log(f'AIA r2 job {rid} failed (attempt {attempt}); check_rc={c.returncode}')
    else: failed.append(f'{MODEL}-{v}-{a}')
if failed: log('ALERT: AIA r2 jobs failed twice: ' + ','.join(failed)); sys.exit(1)
