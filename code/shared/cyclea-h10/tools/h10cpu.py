import os, glob, csv, json, hashlib, re
import numpy as np, pandas as pd
INV = "work/cyclea-protected/analysis/dea_63_cell_inventory.csv"
ROOTS = ["work/cyclea-protected/runs", "work/cyclea-protected/h10/dea_cells_vm2"]
OUT = "work/cyclea-h10/results"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
def mkey(m): return m.replace(".", "_")
def cell_dirs(prefix):
    """returns {(model,cond,seed): cell_dir} for the final good attempt of each rerun/step3000 cell"""
    out = {}
    rows = list(csv.DictReader(open(INV)))
    for r in rows:
        pat = f"{prefix}-{mkey(r['model'])}-{r['condition']}-s{r['seed']}-*"
        cands = []
        for root in ROOTS:
            for d in glob.glob(os.path.join(root, pat)):
                cd = glob.glob(d + "/results/01_generations/*")
                if cd and os.path.exists(os.path.join(cd[0], "COMPLETE")) and os.path.exists(os.path.join(cd[0], "generations.parquet")):
                    cands.append((d, cd[0]))
        fa = r["final_attempt"]
        want = [c for c in cands if (c[0].endswith("-" + fa) if fa != "orig" else not re.search(r"-retry\d+$", c[0]))]
        if len(want) != 1:
            raise SystemExit(f"cell selection ambiguous {r['model']} {r['condition']} {r['seed']} {fa}: {[c[0] for c in cands]}")
        out[(r["model"], r["condition"], int(r["seed"]))] = want[0][1]
    return out
def step3k():
    out = {}
    for d in glob.glob("work/cyclea-protected/runs/ca-deabs3k-pythia-1_4b-*"):
        m = re.search(r"-(generic_prefix|pii_leading|true_prefix)-s(\d+)-", d)
        cd = glob.glob(d + "/results/01_generations/*")
        if cd and os.path.exists(cd[0] + "/COMPLETE"):
            out[("pythia-1.4b-step3000", m.group(1), int(m.group(2)))] = cd[0]
    return out
def load(cd):
    return pd.read_parquet(cd + "/generations.parquet"), pd.read_parquet(cd + "/detections.parquet")
