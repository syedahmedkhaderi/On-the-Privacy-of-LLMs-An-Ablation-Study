import sys, json
sys.path.insert(0, "work/cyclea-h10/tools")
from h10cpu import *
from scipy.stats import rankdata
R = "work/work/p13/repo/MIA/S2MIA/results/"
ex = pd.read_csv(R + "experiments.csv"); sc = pd.read_csv(R + "sample_scores.csv")
TAB = {"vary_dataset": "Table 3", "vary_corpus_size": "Table 4", "vary_membership_ratio": "Table 5", "vary_llm": "Table 6", "vary_embedding": "Table 7", "vary_prompt_style": "Table 8"}
def auc(y, s):
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(s); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); yy = y[o]; tp = np.cumsum(yy); return float((tp / np.arange(1, len(yy) + 1) * yy).sum() / max(yy.sum(), 1))
def comb(d):
    s = d.s_sem.values.astype(float); p = d.ppl_gen.values.astype(float)
    sn = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.zeros_like(s)
    pn = (p - p.min()) / (p.max() - p.min()) if p.max() > p.min() else np.zeros_like(p)
    return sn + (1 - pn)
rng = np.random.default_rng(0); B = 10000
def boot(y, s_list):
    """stratified bootstrap: returns per score [auc lo, hi] (and ap lo/hi for the first score)"""
    i1 = np.where(y == 1)[0]; i0 = np.where(y == 0)[0]
    out = [[] for _ in s_list]; apv = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])
        for k, s in enumerate(s_list):
            out[k].append(auc(y[idx], s[idx]))
        apv.append(ap(y[idx], s_list[0][idx]))
    f = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return [f(o) for o in out], f(apv)
rows = []
for _, e in ex.iterrows():
    ev = sc[(sc.experiment_id == e.id) & (sc.phase == "evaluation")]
    r = dict(table=TAB.get(e.experiment_group, e.experiment_group), row=e.variable_name, dataset=e.dataset_name, llm=e.llm_model, embedder=e.retriever_model,
             corpus_pct=e.corpus_size_pct, membership_ratio=e.membership_ratio, status=e.status, stored_roc_auc=e.roc_auc, stored_pr_auc=e.pr_auc, stored_accuracy=e.accuracy,
             label_noise_flag=bool(e.experiment_group == "vary_corpus_size" and 10 <= e.corpus_size_pct <= 75), n_eval=len(ev))
    if len(ev):
        y = ev.label.values.astype(int); cs = comb(ev); bl = ev.s_sem.values.astype(float); pp = -ev.ppl_gen.values.astype(float)
        (ci_c, ci_b, ci_p), ci_ap = boot(y, [cs, bl, pp])
        pred = ev.predicted_member.astype(int).values
        r.update(n_member=int((y == 1).sum()), n_nonmember=int((y == 0).sum()), auc=auc(y, cs), auc_ci=json.dumps(ci_c), pr_auc=ap(y, cs), pr_auc_ci=json.dumps(ci_ap),
                 auc_bleu_only=auc(y, bl), auc_bleu_ci=json.dumps(ci_b), auc_ppl_only=auc(y, pp), auc_ppl_ci=json.dumps(ci_p), accuracy_recomputed=float((pred == y).mean()),
                 majority_rate=float(max(y.mean(), 1 - y.mean())), auc_includes_half=bool(ci_c[0] <= 0.5 <= ci_c[1]))
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(OUT + "/cpu/c5_s2mia_runs.csv", index=False)
emb = ex[ex.experiment_group == "vary_prompt_style"][["variable_name", "retriever_model", "llm_model", "dataset_name"]]
base = ex[(ex.experiment_group != "vary_prompt_style")]
note = {"table8_rows": emb.to_dict("records"), "retriever_values_outside_table8": sorted(set(base.retriever_model)),
        "reading": "if the Table 8 rows use a retriever other than contriever-msmarco, the Table 8 baseline is not the same configuration as the other baselines"}
json.dump({"bootstrap": f"stratified, B={B}, numpy default_rng(0); AUC of combined score (s_norm + 1 - p_norm per run), BLEU alone (s_sem), perplexity alone (-ppl_gen); PR-AUC = average precision of the combined score",
           "n_runs": len(df), "n_with_scores": int(df.n_eval.gt(0).sum()), "intervals_including_0.5": int(df["auc_includes_half"].fillna(False).sum()),
           "embedder_note_table8": note, "code_sha256": sha(__file__),
           "sources": [{"path": R + f, "sha256": sha(R + f)} for f in ("experiments.csv", "sample_scores.csv", "experiment_config.csv")]}, open(OUT + "/cpu/c5_s2mia.json", "w"), indent=1)
print(df[["table", "row", "n_eval", "stored_roc_auc", "auc", "auc_ci", "auc_bleu_only", "auc_ppl_only", "label_noise_flag", "status"]].round(3).to_string()); print(json.dumps(note)[:700])
