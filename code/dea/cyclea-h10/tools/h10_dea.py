"""H10 DEA cell runner. Builds a job from an owner-reviewed template cell (identical settings => same
comparison digest => existing history audit qualifies it), checks it, approves its digest, runs it.
usage: h10_dea.py --bundle DIR --model KEY --cond C --seed N --block B [--tokens] [--hf-cache DIR]
Retry once under a new run id on failure (H10 rule 3). Appends to results/<block>/jobs.csv."""
import argparse, glob, hashlib, json, os, re, socket, subprocess, sys, time, fcntl
from pathlib import Path

PRIV = Path("work/cyclea-protected")
RES = Path("work/cyclea-h10/results")

def utc(): return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def log(msg):
    with open("work/cyclea-h10/logs/H10_LOG.md", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.write(f"{time.strftime('%FT%TZ', time.gmtime())} [dea:{socket.gethostname()}] {msg}\n")

def py_run(bundle, venv_py, args, env_extra=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", **(env_extra or {}))
    return subprocess.run([venv_py, "-m", "scripts.run_cycle_a", *args], cwd=bundle, env=env, capture_output=True, text=True)

def code_hash(bundle, venv_py):
    return py_run(bundle, venv_py, ["--code-hash"]).stdout.strip()

def find_template(model, cond, seed):
    """A complete original ca-deab2 cell job on THIS host for (model, cond); exact seed preferred. Settings differ
    between seeds only in `seed` (comparison digest excludes it), so the seed is overridden."""
    best = None
    for p in sorted(glob.glob(str(PRIV / "jobs" / f"ca-deab2-{model.replace('.', '_')}-{cond}-s*-*.json"))):
        rid = Path(p).stem
        sp = PRIV / "runs" / rid / "summary.json"
        if sp.is_file() and json.load(open(sp)).get("status") == "complete":
            job = json.load(open(p))
            if job["settings"]["seed"] == seed: best = (job, rid); break
            best = best or (job, rid)
    if not best: raise SystemExit(f"no complete template for {model} {cond}")
    job, rid = best
    job["settings"]["seed"] = seed
    return job, rid

def write_job(job):
    p = PRIV / "jobs" / f"{job['run_id']}.json"
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f: json.dump(job, f, indent=2)
    return p

def check_and_digest(bundle, py, path):
    r = py_run(bundle, py, [str(path), "--check", "--output-root", str(PRIV / "runs")])
    m = re.search(r"Review SHA-256 ([0-9a-f]{64})", r.stdout)
    if r.returncode or not m: raise RuntimeError(f"check failed: {r.stderr.strip()[-200:]} {r.stdout.strip()[-200:]}")
    return m.group(1)

def execute(bundle, py, path, digest, env_extra=None):
    r = py_run(bundle, py, [str(path), "--approve-sha256", digest, "--output-root", str(PRIV / "runs")], env_extra)
    return r.returncode, (r.stderr or "")[-300:]

def ensure_smoke(bundle, py, tmpl, tmpl_id, ch, gpu_uuid, tag):
    reg_path = RES / "dea_smoke_registry.json"
    reg = json.load(open(reg_path)) if reg_path.exists() else {}
    key = f"{socket.gethostname()}|{gpu_uuid}|{ch[:12]}|{tmpl['settings']['model']['hf_id']}|{tmpl['settings']['model']['revision']}"
    if key in reg: return reg[key]
    dep = tmpl["depends_on"][0]
    sj = json.load(open(PRIV / "jobs" / f"{dep['run_id']}.json"))
    sj.update(run_id=f"ca-h10-smoke-{tmpl['settings']['model']['key'].replace('.', '_')}-{tag}-{utc()}", code_sha256=ch,
              host=socket.gethostname().split(".")[0])
    sj["resources"]["gpu_uuid"] = gpu_uuid; sj["depends_on"] = []
    p = write_job(sj); d = check_and_digest(bundle, py, p); rc, err = execute(bundle, py, p, d)
    if rc: raise RuntimeError(f"smoke failed {err}")
    reg[key] = {"run_id": sj["run_id"], "summary_sha256": sha(PRIV / "runs" / sj["run_id"] / "summary.json")}
    json.dump(reg, open(reg_path, "w"), indent=1)
    return reg[key]

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--bundle", required=True); a.add_argument("--model", required=True)
    a.add_argument("--cond", required=True); a.add_argument("--seed", type=int, required=True)
    a.add_argument("--block", required=True); a.add_argument("--tokens", action="store_true")
    a.add_argument("--queries", type=int, default=None)
    a.add_argument("--hf-cache", default=None)
    a.add_argument("--venv", default="work/venvs/cyclea-dea/bin/python")
    a.add_argument("--label", default="")
    a = a.parse_args()
    host = socket.gethostname().split(".")[0]
    gpu_uuid = os.environ["GPU_UUID"]; gi = os.environ.get("GPU_INDEX", "0")
    vmg = f"vm{host[-1]}g{gi}"
    ch = code_hash(a.bundle, a.venv)
    tmpl, tmpl_id = find_template(a.model, a.cond, a.seed)
    out_csv = RES / a.block / "jobs.csv"; out_csv.parent.mkdir(parents=True, exist_ok=True)
    env_extra = {}
    if a.tokens: env_extra["H10_TOKENS_ROOT"] = "work/cyclea-protected/h10/dea_tokens"
    if a.hf_cache: env_extra["HF_HUB_CACHE"] = a.hf_cache
    status, rid, note = "failed", "", ""
    for attempt in (0, 1):
        t0 = time.time()
        try:
            smoke = ensure_smoke(a.bundle, a.venv, tmpl, tmpl_id, ch, gpu_uuid, vmg)
            job = json.loads(json.dumps(tmpl))
            rid = f"ca-h10-{a.block}-{a.model.replace('.', '_')}-{a.cond}-s{a.seed}-{vmg}-{utc()}{'' if attempt == 0 else '-retry1'}"
            job.update(run_id=rid, code_sha256=ch, host=host)
            job["resources"]["gpu_uuid"] = gpu_uuid
            job["depends_on"] = [{"run_id": smoke["run_id"], "kind": "cache_smoke", "summary_sha256": smoke["summary_sha256"]}]
            if a.queries: job["settings"]["queries"] = a.queries
            p = write_job(job); d = check_and_digest(a.bundle, a.venv, p)
            rc, err = execute(a.bundle, a.venv, p, d, env_extra)
            if rc == 0 and (PRIV / "runs" / rid / "summary.json").is_file(): status, note = "complete", f"digest={d[:12]} tmpl={tmpl_id}"; break
            note = f"rc={rc} {err}"
        except Exception as e:
            note = f"{type(e).__name__}: {e}"[:300]
        log(f"job {rid} attempt {attempt} FAILED: {note}")
    with open(out_csv, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX); f.write(",".join([rid, status, time.strftime("%FT%TZ", time.gmtime(t0)), time.strftime("%FT%TZ", time.gmtime()), '"' + note.replace('"', "'").replace("\n", " ") + '"']) + "\n")
    log(f"job {rid} {status} {note[:120]}")
    return 0 if status == "complete" else 1

if __name__ == "__main__":
    sys.exit(main())
