"""H10 AIA v4 driver (VM2). Builds smoke + job manifests from the v3 job settings, approves digests (owner pre-approval,
H10 prompt), runs jobs serially on ONE GPU, retries a failed job once under a new run id, never aborts the segment.
usage: aia_v4_driver.py [--check]   env: GPU_UUID, GPU_INDEX (set by lanes/lane_run.sh)"""
import csv, glob, hashlib, json, os, socket, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
os.umask(0o077)
ROOT = Path("work/rs-1/cycleA/bundles/aia-v5-exemplar-20261112"); PRIV = Path("work/cyclea-protected/h12")
PY = "work/venvs/cyclea-aia/bin/python"
TPL = Path("work/cyclea-protected/h10/aia/templates")
HOST = socket.gethostname()
GPU = os.environ.get("GPU_UUID", "GPU-3c52822a-34f1-f00f-e3cb-32ca1175c62d"); IDX = os.environ.get("GPU_INDEX", "1")
check_only = "--check" in sys.argv
MODELS = ["llama", "mistral", "qwen"]
ATTRS = ["age", "education", "gender", "income", "location", "married", "occupation", "pobp"]
# format_exemplar deliberately absent (skipped: baseline allows "[]", exemplar forbids it; see PREREG pre-run note 2)
VARIANTS = [("fmt", "format_exemplar", False)]
RES = Path("work/cyclea-h12/results/aia_v5"); RES.mkdir(parents=True, exist_ok=True)
JOBS_CSV = RES / f"jobs_{HOST}.csv"
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONPATH=str(ROOT))
code = subprocess.run([PY, "work/cyclea-h12/tools/aia_run.py", "--code-hash"], cwd=ROOT, env=env, capture_output=True, text=True).stdout.strip()
sys.path.insert(0, str(ROOT)); from scripts.run_cycle_a import read_job
packages = json.load(open(TPL / "packages.json"))
host_name = HOST
def stamp(): return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
def log(msg):
    with open("work/cyclea-h12/logs/H12_LOG.md" if HOST == "machine_B" else "work/cyclea-h12/logs/H12_LOG.md", "a") as f:
        f.write(f"{datetime.now(timezone.utc).strftime('%FT%TZ')} [aia_v4] {msg}\n")
def record(row):
    new = not JOBS_CSV.exists()
    with open(JOBS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["run_id", "status", "start", "end", "note"])
        w.writerow(row)
def done_ok():
    if not JOBS_CSV.exists(): return set()
    return {r["note"].split("|")[0] for r in csv.DictReader(open(JOBS_CSV)) if r["status"] == "complete"}

def write_job(job, probes):
    path = PRIV / ("probes" if probes else "jobs") / f"{job['run_id']}{'.check' if probes else ''}.json"
    with path.open("x") as s: json.dump(job, s, indent=2)
    return path
def base(run_id, kind, settings, depends):
    return {"schema": "cycleA.job.v1", "run_id": run_id, "kind": kind, "host": host_name, "code_sha256": code,
            "packages": packages, "resources": dict(min_disk_mib=40960, min_ram_mib=24576, gpu_uuid=GPU, min_free_mib=20000),
            "depends_on": depends, "settings": settings}
def check(path):
    root = [] if check_only else ["--output-root", str(PRIV / "runs")]
    return subprocess.run([PY, "work/cyclea-h12/tools/aia_run.py", str(path), "--check", *root],
                          cwd=ROOT, env=env, capture_output=True, text=True)
def execute(path, run_id):
    _, digest = read_job(path)     # owner pre-approval: digest taken from the checked file
    lg = open(PRIV / "logs" / f"{run_id}.log", "x")
    r = subprocess.run([PY, "work/cyclea-h12/tools/aia_run.py", str(path), "--approve-sha256", digest, "--output-root", str(PRIV / "runs")],
                       cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=GPU), stdout=lg, stderr=subprocess.STDOUT)
    summ = PRIV / "runs" / run_id / "summary.json"
    ok = r.returncode == 0 and summ.exists()
    return ok, (hashlib.sha256(summ.read_bytes()).hexdigest() if ok else None)

def smoke_for(model, tag):
    s = json.load(open(TPL / f"{model}-age.json"))
    st = {k: s[k] for k in ("model_id", "revision", "dtype", "chat_template_sha256", "template_options")}
    st["tokenizer_revision"] = s["revision"]
    rid = f"ca-h12smoke-aiav5-{model}-vm2g{IDX}-{tag}"
    return base(rid, "cache_smoke", st, [])

if check_only:
    tag = "check" + stamp()
    bad = 0
    for model in MODELS:
        sj = smoke_for(model, tag); p = write_job(sj, True); c = check(p)
        print(sj["run_id"], "rc", c.returncode, c.stderr.strip()[-150:]); bad += c.returncode; p.unlink()
        for vid, style, mism in VARIANTS:
            for attr in ATTRS[:1] + (["pobp"] if model == "qwen" else []):
                st = json.load(open(TPL / f"{model}-{attr}.json")); st.update(paired=False, style=style, mismatched_text=mism)
                j = base(f"ca-h12aia-{model}-{vid}-{attr}-vm2g{IDX}-{tag}", "aia_inference", st,
                         [{"run_id": "ca-smokecheck", "kind": "cache_smoke", "summary_sha256": "0" * 64}])
                p = write_job(j, True); c = check(p)
                print(j["run_id"], "rc", c.returncode, c.stderr.strip()[-150:]); bad += c.returncode; p.unlink()
    sys.exit(1 if bad else 0)

log(f"driver start host={HOST} gpu={IDX} {GPU} code={code[:8]}")
for model in MODELS:
    ok_done = done_ok()
    todo = [(vid, style, mism, attr) for vid, style, mism in VARIANTS for attr in ATTRS if f"{model}-{vid}-{attr}" not in ok_done]
    if not todo: continue
    smoke_ok = None
    for attempt in range(2):
        sj = smoke_for(model, stamp() + (f"r{attempt}" if attempt else ""))
        p = write_job(sj, False); t0 = datetime.now(timezone.utc).isoformat()
        c = check(p)
        if c.returncode == 0: ok, dig = execute(p, sj["run_id"])
        else: ok, dig = False, None
        record([sj["run_id"], "complete" if ok else "failed", t0, datetime.now(timezone.utc).isoformat(), f"smoke-{model}|check_rc={c.returncode}"])
        if ok: smoke_ok = (sj["run_id"], dig); break
        log(f"smoke {model} attempt {attempt} failed")
    if not smoke_ok:
        log(f"DEVIATION: smoke for {model} failed twice; skipping model {model}")
        continue
    dep = [{"run_id": smoke_ok[0], "kind": "cache_smoke", "summary_sha256": smoke_ok[1]}]
    for vid, style, mism, attr in todo:
        for attempt in range(2):
            st = json.load(open(TPL / f"{model}-{attr}.json")); st.update(paired=False, style=style, mismatched_text=mism)
            rid = f"ca-h12aia-{model}-{vid}-{attr}-vm2g{IDX}-{stamp()}" + (f"-retry{attempt}" if attempt else "")
            j = base(rid, "aia_inference", st, dep); p = write_job(j, False); t0 = datetime.now(timezone.utc).isoformat()
            c = check(p)
            ok = False
            if c.returncode == 0: ok, _ = execute(p, rid)
            record([rid, "complete" if ok else "failed", t0, datetime.now(timezone.utc).isoformat(),
                    f"{model}-{vid}-{attr}|check_rc={c.returncode}|attempt={attempt}"])
            log(f"{rid} {'OK' if ok else 'FAILED'} (attempt {attempt})")
            if ok: break
        else:
            log(f"DEVIATION: {model}-{vid}-{attr} failed twice, skipped")
failed=[]
for model in MODELS:
 for vid,style,mism in VARIANTS:
  for attr in ATTRS:
   if f"{model}-{vid}-{attr}" not in done_ok():failed.append(f"{model}-{vid}-{attr}")
if failed:
 Path('work/cyclea-h12/lanes/ready/SKIPPED_aia_exemplar_partial').write_text('Failed twice: '+','.join(failed))
 log('DEVIATION: AIA jobs failed twice count='+str(len(failed)))
log("driver end")
sys.exit(1 if failed else 0)
