"""PREREG_H10_DEA_RANKING scoring pass. usage: h10_rank_score.py RUN_ID
Reads cyclea-protected/h10/dea_tokens/<run>/tokens.parquet, writes scores.parquet (mode 600) next to it.
Per generation (non-empty continuation): nll_mean, nll_total, zlib_bytes, ref_nll_mean. Decoded text only in memory."""
import sys, os, zlib, time, json
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
run = sys.argv[1]
_m = json.load(open(f"work/cyclea-protected/runs/{run}/job.json"))["settings"]["model"]
hf_id, rev = _m["hf_id"], _m["revision"]
use_ref = not _m["key"].startswith("pythia-70m")
BS = int(os.environ.get("H10_SCORE_BS", "16"))
d = f"work/cyclea-protected/h10/dea_tokens/{run}"
t = pq.read_table(f"{d}/tokens.parquet").to_pydict()
n = len(t["generation_id"]); keep = [i for i in range(n) if len(t["continuation_ids"][i]) > 0]
off = dict(local_files_only=True, trust_remote_code=False)
def load(h, r):
    tok = AutoTokenizer.from_pretrained(h, revision=r, **off)
    m = AutoModelForCausalLM.from_pretrained(h, revision=r, torch_dtype=torch.float16, low_cpu_mem_usage=True, **off).cuda().eval()
    return tok, m
@torch.no_grad()
def nll(model, pairs):
    """pairs: list of (prompt_ids, cont_ids). Returns per-pair (sum NLL over cont tokens, n cont tokens)."""
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    out = [None] * len(pairs)
    for s in range(0, len(order), BS):
        idx = order[s:s + BS]; seqs = [pairs[i][0] + pairs[i][1] for i in idx]; w = max(map(len, seqs))
        ids = torch.zeros(len(idx), w, dtype=torch.long); am = torch.zeros(len(idx), w, dtype=torch.long)
        for r, q in enumerate(seqs): ids[r, :len(q)] = torch.tensor(q); am[r, :len(q)] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=am.cuda()).logits
        for r, i in enumerate(idx):
            p, c = len(pairs[i][0]), len(pairs[i][1])
            lp = torch.log_softmax(logits[r, p - 1:p + c - 1].float(), -1)
            tgt = torch.tensor(pairs[i][1], device=lp.device)
            out[i] = (float(-lp.gather(1, tgt[:, None]).sum()), c)
        del logits
    return out
t0 = time.time()
tok, model = load(hf_id, rev)
pairs = [(list(t["prompt_ids"][i]) or [tok.eos_token_id], list(t["continuation_ids"][i])) for i in keep]
main = nll(model, pairs)
texts = [tok.decode(t["continuation_ids"][i], skip_special_tokens=True) for i in keep]
zb = [max(1, len(zlib.compress(x.encode("utf-8", "replace")))) for x in texts]
ref = [float("nan")] * len(keep)
if use_ref:
    del model; torch.cuda.empty_cache()
    rt, rm = load("EleutherAI/pythia-70m", "a39f36b100fe8a5377810d56c3f4789b9c53ac42")
    rp = []
    for k, i in enumerate(keep):
        pt = tok.decode(t["prompt_ids"][i], skip_special_tokens=False)
        pi = rt(pt, add_special_tokens=False)["input_ids"] or [rt.eos_token_id]
        ci = rt(texts[k], add_special_tokens=False)["input_ids"]
        rp.append((pi, ci if ci else [rt.eos_token_id]))
    r = nll(rm, rp); ref = [a / b for a, b in r]
tab = pa.table({"generation_id": [t["generation_id"][i] for i in keep], "prompt_index": [t["prompt_index"][i] for i in keep],
    "nll_total": [a for a, b in main], "n_tokens": [b for a, b in main], "nll_mean": [a / b for a, b in main],
    "zlib_bytes": zb, "ref_nll_mean": ref})
tmp = f"{d}/scores.parquet.tmp"; pq.write_table(tab, tmp, compression="zstd"); os.chmod(tmp, 0o600); os.replace(tmp, f"{d}/scores.parquet")
print(json.dumps({"run": run, "n_scored": len(keep), "seconds": round(time.time() - t0, 1), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}))
