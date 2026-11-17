#!/usr/bin/env python3
"""PREREG_H10_MIA_EXTRA sidecar. Same 359+359 PRIMARY/SECONDARY windows as the stored runs.
usage: mia_extra.py --model {1.4b,6.9b}            (GPU lane job)
       mia_extra.py --cpu-test                      (pythia-70m, CPU, 20+20 subset; validation only, protected output)
Step 1 reproduces the stored mean-loss and Min-K-20 AUCs to 4 decimals (abort otherwise, rc=3).
Step 2 adds Min-K 10/30, zlib ratio, reference ratio (pythia-70m @a39f36b1, same token windows).
Raw per-window scores -> work/cyclea-protected/h10/mia/ (mode 600). Aggregates -> work/cyclea-h10/results/mia_extra/.
Scoring = the bundle's own formulas (summarize_token_nll, cross_entropy on float logits, batch 1, fp16)."""
import sys, os, json, math, zlib, hashlib, time, argparse
sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import rankdata
BUNDLE = "work/rs-1/cycleA/bundles/mia-v3-splitlabel"
sys.path.insert(0, BUNDLE)
from attacks.MIA.scoring import summarize_token_nll  # bundle's scoring function (read-only import)
P = "work/cyclea-protected/"
OUT_PROT = P + "h10/mia/"
OUT_RES = "work/cyclea-h10/results/mia_extra/"
VAR = ("PRIMARY", "SECONDARY")
CFG = {
    "1.4b": dict(hf="EleutherAI/pythia-1.4b", rev="9cc5c8c8148a4e0115d9e29c6b4f21124cfe748a",
                 cache=P + "models/hf-cache-pythia-1.4b-9cc5c8c8", agg=P + "aggregates/mia-v3-primary-analysis.json",
                 runs={"PRIMARY": "ca-miaprim-prim-pythia-1_4b-vm1g1-20261108T175000Z",
                       "SECONDARY": "ca-miaprim-seco-pythia-1_4b-vm1g1-20261108T175000Z"}),
    "6.9b": dict(hf="EleutherAI/pythia-6.9b", rev="21bfa02e806e253fe453702c29c81d9f83617255",
                 cache=P + "models/hf-cache-pythia-6.9b-21bfa02e", agg=P + "aggregates/mia-v3-primary-analysis-6p9b.json",
                 runs={"PRIMARY": "ca-miaprim-prim-pythia-6_9b-vm1g1-20261108T210000Z",
                       "SECONDARY": "ca-miaprim-seco-pythia-6_9b-vm1g1-20261108T210000Z"}),
}
REF = dict(hf="EleutherAI/pythia-70m", rev="a39f36b100fe8a5377810d56c3f4789b9c53ac42",
           cache=os.path.expanduser("~/.cache/huggingface/hub"))
SCORES = ["mean_loss", "min_k_10", "min_k_20", "min_k_30", "zlib_ratio", "ref_ratio"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_records(run, limit=None):
    spec = json.load(open(P + f"runs/{run}/job.json"))["settings"]["input"]
    assert sha(spec["path"]) == spec["sha256"], "input hash differs from the stored run"
    rows = sorted((json.loads(l) for l in open(spec["path"])),
                  key=lambda r: (r["membership_label"], r["document_id_hash"]))
    if limit:
        rows = [r for r in rows if r["membership_label"] == 0][:limit] + [r for r in rows if r["membership_label"] == 1][:limit]
    return rows, {"path": spec["path"], "sha256": spec["sha256"]}


def token_nll(model, ids, dev, torch, F):
    x = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)
    with torch.inference_mode():
        lg = model(input_ids=x, use_cache=False).logits[:, :-1, :]
        return F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), x[:, 1:].reshape(-1), reduction="none").cpu().tolist()


def auc(m, n):
    rk = rankdata(np.concatenate([m, n]))
    return float((rk[:len(m)].sum() - len(m) * (len(m) + 1) / 2) / (len(m) * len(n)))


def tpr1(m, n):
    k = int(math.floor(0.01 * len(n)))
    return float((m > np.sort(n)[::-1][k]).mean())


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CFG))
    ap.add_argument("--cpu-test", action="store_true")
    a = ap.parse_args()
    test = a.cpu_test
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if test:
        torch.set_num_threads(4)
        cfg = dict(CFG["1.4b"], hf=REF["hf"], rev=REF["rev"], cache=REF["cache"])
        tag, dev, dt, limit = "cpu_test", "cpu", torch.float32, 20
    else:
        if not a.model:
            ap.error("--model or --cpu-test")
        cfg = CFG[a.model]
        tag, dev, dt, limit = a.model, "cuda:0", torch.float16, None
        assert torch.cuda.is_available()
    off = dict(local_files_only=True, trust_remote_code=False)
    t0 = time.time()

    def load(c):
        tk = AutoTokenizer.from_pretrained(c["hf"], revision=c["rev"], cache_dir=c["cache"], **off)
        m = AutoModelForCausalLM.from_pretrained(c["hf"], revision=c["rev"], cache_dir=c["cache"], torch_dtype=dt, **off).to(dev).eval()
        return tk, m

    tok, model = load(cfg)
    recs, srcs = {}, {}
    for v in VAR:
        recs[v], srcs[v] = load_records(cfg["runs"][v], limit)
    # main model: per-token NLL for every window
    raw = {v: [] for v in VAR}
    for v in VAR:
        for r in recs[v]:
            nll = token_nll(model, r["token_ids"], dev, torch, F)
            mean, k20, n = summarize_token_nll(nll, 20.0)
            raw[v].append(dict(
                example_id_hash=r["example_id_hash"], document_id_hash=r["document_id_hash"], label=r["membership_label"], n=n,
                mean_loss=mean, min_k_10=summarize_token_nll(nll, 10.0)[1], min_k_20=k20, min_k_30=summarize_token_nll(nll, 30.0)[1],
                nll_total=float(sum(nll)),
                zlib_bytes=max(1, len(zlib.compress(tok.decode(r["token_ids"]).encode("utf-8", "replace"))))))

    def arrays(v, f):
        rows = sorted(raw[v], key=lambda r: r["document_id_hash"])
        return (np.array([-f(r) for r in rows if r["label"] == 1]), np.array([-f(r) for r in rows if r["label"] == 0]))

    # reproduction gate: before anything is computed beyond the main scores
    repro = {"ok": True, "checks": {}}
    if not test:
        agg = json.load(open(cfg["agg"]))
        for v in VAR:
            for s in ("mean_loss", "min_k_20"):
                mine = auc(*arrays(v, lambda r, s=s: r[s]))
                stored = agg["variants"][v]["metrics"][s]["roc_auc"]
                good = round(mine, 4) == round(stored, 4)
                repro["checks"][f"{v}/{s}"] = [round(mine, 6), round(stored, 6), good]
                repro["ok"] &= good
        if not repro["ok"]:
            os.makedirs(OUT_RES, exist_ok=True)
            json.dump({"model": a.model, "reproduction": repro, "status": "ABORTED"}, open(OUT_RES + f"mia_extra_{tag}_ABORTED.json", "w"), indent=1)
            print("REPRODUCTION FAILED", repro)
            return 3
    # reference model, same token windows
    rtok, rmodel = load(dict(REF))
    assert rtok.get_vocab() == tok.get_vocab(), "tokenizers differ; reference ratio would not use the same token windows"
    for v in VAR:
        for r, rr in zip(raw[v], recs[v]):
            r["ref_mean"] = summarize_token_nll(token_nll(rmodel, rr["token_ids"], dev, torch, F), 20.0)[0]
    # raw scores (protected, mode 600)
    os.makedirs(OUT_PROT, exist_ok=True)
    for v in VAR:
        fd = os.open(OUT_PROT + f"scores_{tag}_{v}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            for r in raw[v]:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
    # analysis: bootstrap identical to g3_analysis.py (B=10000, default_rng(0), shared non-member draw,
    # then PRIMARY members, then SECONDARY members)
    fns = {"zlib_ratio": lambda r: r["nll_total"] / r["zlib_bytes"], "ref_ratio": lambda r: r["mean_loss"] / r["ref_mean"]}
    arr = {v: {s: arrays(v, fns.get(s, lambda r, s=s: r[s])) for s in SCORES} for v in VAR}
    nM, nN = len(arr["PRIMARY"]["mean_loss"][0]), len(arr["PRIMARY"]["mean_loss"][1])
    B = 10000
    rng = np.random.default_rng(0)
    boot = {v: {s: np.empty(B) for s in SCORES} for v in VAR}
    bt = {v: {s: np.empty(B) for s in SCORES} for v in VAR}
    for b in range(B):
        ni = rng.integers(0, nN, nN)
        mi = {"PRIMARY": rng.integers(0, nM, nM)}
        mi["SECONDARY"] = rng.integers(0, nM, nM)
        for v in VAR:
            for s in SCORES:
                mm, nn = arr[v][s][0][mi[v]], arr[v][s][1][ni]
                boot[v][s][b] = auc(mm, nn)
                bt[v][s][b] = tpr1(mm, nn)
    res = {"schema": "h10.mia_extra.v1", "prereg": "PREREG_H10_MIA_EXTRA", "model": cfg["hf"], "revision": cfg["rev"],
           "n_per_class": [nM, nN], "bootstrap": {"B": B, "seed": 0, "ci": "percentile 2.5/97.5", "stream": "as g3_analysis.py"},
           "reproduction": repro, "status": "CPU_TEST_NOT_A_RESULT" if test else "complete",
           "label": "added after the primary results, secondary", "variants": {}, "delta_secondary_minus_primary": {}}
    for v in VAR:
        res["variants"][v] = {s: {"roc_auc": auc(*arr[v][s]), "roc_auc_ci95": ci(boot[v][s]),
                                  "tpr_at_1pct_fpr": tpr1(*arr[v][s]), "tpr_ci95": ci(bt[v][s])} for s in SCORES}
    for s in SCORES:
        res["delta_secondary_minus_primary"][s] = {
            "delta": res["variants"]["SECONDARY"][s]["roc_auc"] - res["variants"]["PRIMARY"][s]["roc_auc"],
            "ci95": ci(boot["SECONDARY"][s] - boot["PRIMARY"][s])}
    if not test:   # CI reproduction is reported, not gating
        res["reproduction"]["ci_check"] = {
            f"{v}/{s}": [[round(x, 4) for x in res["variants"][v][s]["roc_auc_ci95"]],
                         [round(x, 4) for x in agg["variants"][v]["metrics"][s]["roc_auc_ci95"]]]
            for v in VAR for s in ("mean_loss", "min_k_20")}
    res["sources"] = [{"path": srcs[v]["path"], "sha256": srcs[v]["sha256"]} for v in VAR] + [
        {"path": cfg["agg"], "sha256": sha(cfg["agg"])}, {"path": os.path.abspath(__file__), "sha256": sha(__file__)}]
    res["seconds"] = round(time.time() - t0, 1)
    outdir = OUT_PROT if test else OUT_RES
    os.makedirs(outdir, exist_ok=True)
    json.dump(res, open(outdir + f"mia_extra_{tag}.json", "w"), indent=1)
    if not test:
        with open(OUT_RES + f"mia_extra_{tag}.csv", "w") as f:
            f.write("variant,score,roc_auc,ci_lo,ci_hi,tpr_1pct_fpr,tpr_lo,tpr_hi\n")
            for v in VAR:
                for s in SCORES:
                    x = res["variants"][v][s]
                    f.write(f"{v},{s},{x['roc_auc']:.4f},{x['roc_auc_ci95'][0]:.4f},{x['roc_auc_ci95'][1]:.4f},"
                            f"{x['tpr_at_1pct_fpr']:.4f},{x['tpr_ci95'][0]:.4f},{x['tpr_ci95'][1]:.4f}\n")
    print(json.dumps({"done": tag, "seconds": res["seconds"], "n_per_class": [nM, nN]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
